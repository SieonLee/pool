"""
Over-detection audit for pool_real_004, pool_real_025, pool_real_031.

For each image:
  1. Load warped image from training dataset
  2. Parse GT labels (YOLO format)
  3. Run v7 model at conf 0.25 / 0.30 / 0.35 / 0.40
  4. Count TP / FP / FN vs GT
  5. Classify each FP (duplicate, stripe/highlight, rail/pocket, cloth, unlabeled real ball)
  6. Save overlay images: GT only, then each conf level side-by-side
  7. Print structured report

Usage:
  python scripts/audit_overdet.py --ball-ckpt models/checkpoints/ball_yolo_v7_below_baseline.pt
"""
import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

BASE      = Path(__file__).parent.parent
WARP_DIR  = BASE / "datasets" / "pose-warped-balls" / "images" / "train"
LABEL_DIR = BASE / "datasets" / "pose-warped-balls" / "labels" / "train"
OUT_DIR   = BASE / "review" / "overdet_audit"
WARP_W, WARP_H = 900, 450

STEMS   = ["pool_real_004", "pool_real_025", "pool_real_031"]
CONFS   = [0.25, 0.30, 0.35, 0.40]
IOU_MATCH = 0.35   # IoU to count a detection as TP

FONT    = cv2.FONT_HERSHEY_SIMPLEX
CLR_GT  = (0, 220, 0)       # green — GT boxes
CLR_CUE = (255, 255, 0)     # cyan — cue_ball prediction
CLR_OBJ = (255, 100, 0)     # blue — obj_ball prediction
CLR_FP  = (0, 0, 255)       # red — FP marker ring


def parse_gt(label_path: Path, w=WARP_W, h=WARP_H):
    """Return list of (cls, cx, cy, bw, bh) in pixel coords."""
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, xc, yc, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), \
                                float(parts[3]), float(parts[4])
        boxes.append((cls, xc*w, yc*h, bw*w, bh*h))
    return boxes


def box_iou(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw * ih
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / ua if ua > 0 else 0.0


def classify_fp(warped, x1, y1, x2, y2, cls, all_dets, det_idx,
                gt_boxes, gt_w=WARP_W, gt_h=WARP_H):
    """
    Returns a string tag for why this detection is a FP.
    Checks in order:
      1. duplicate — another same-class det overlaps same GT-miss region
      2. rail/pocket — centre near table edge
      3. unlabeled_real — bright round blob with no nearby GT
      4. stripe_highlight — high brightness, low saturation, small
      5. cloth — low brightness
      6. unknown
    """
    cx, cy = (x1+x2)/2, (y1+y2)/2
    bw, bh = x2-x1, y2-y1

    # 1. Duplicate: another det of same class whose centre is very close
    for j, (ocls, ocf, ox1, oy1, ox2, oy2) in enumerate(all_dets):
        if j == det_idx or ocls != cls:
            continue
        iou = box_iou(x1, y1, x2, y2, ox1, oy1, ox2, oy2)
        dist = math.hypot((ox1+ox2)/2 - cx, (oy1+oy2)/2 - cy)
        if iou > 0.1 or dist < max(bw, bh) * 1.2:
            return "duplicate"

    # 2. Rail/pocket — within 30px of edge
    rail = 30
    if cx < rail or cx > WARP_W-rail or cy < rail or cy > WARP_H-rail:
        return "rail_pocket"

    # 3. Appearance analysis
    px1, py1 = max(0, int(x1)), max(0, int(y1))
    px2, py2 = min(WARP_W, int(x2)), min(WARP_H, int(y2))
    patch = warped[py1:py2, px1:px2]
    if patch.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    bright = float(hsv[:, :, 2].mean())
    sat    = float(hsv[:, :, 1].mean())
    aspect = min(bw, bh) / max(max(bw, bh), 1)

    # 4. Stripe/highlight — very bright, low sat, elongated
    if bright > 200 and sat < 40:
        return "stripe_highlight"

    # 5. Cloth — dark
    if bright < 80:
        return "cloth"

    # 6. Round bright blob — possibly unlabeled ball
    if bright > 120 and aspect > 0.55:
        return "possibly_unlabeled_ball"

    return "unknown"


def run_model(model, warped, conf):
    results = model(warped, verbose=False, conf=conf)
    dets = []
    if results and results[0].boxes is not None:
        for i in range(len(results[0].boxes)):
            b = results[0].boxes
            dets.append((int(b.cls[i].item()), float(b.conf[i].item()),
                         *b.xyxy[i].cpu().numpy().tolist()))
    return dets


def draw_gt(img, gt_boxes):
    for cls, cx, cy, bw, bh in gt_boxes:
        x1, y1 = int(cx - bw/2), int(cy - bh/2)
        x2, y2 = int(cx + bw/2), int(cy + bh/2)
        cv2.rectangle(img, (x1, y1), (x2, y2), CLR_GT, 2)
        label = "CUE" if cls == 0 else "OBJ"
        cv2.putText(img, label, (x1, y1-4), FONT, 0.38, CLR_GT, 1)


def draw_preds(img, dets, gt_boxes, conf_thresh, warped_orig):
    """Draw predictions; mark FPs in red."""
    matched_gt = set()
    tp_list, fp_list = [], []

    for i, (cls, cf, x1, y1, x2, y2) in enumerate(dets):
        # try to match against GT of same class
        best_iou, best_j = 0.0, -1
        for j, (gcls, gcx, gcy, gbw, gbh) in enumerate(gt_boxes):
            if gcls != cls or j in matched_gt:
                continue
            gx1, gy1 = gcx - gbw/2, gcy - gbh/2
            gx2, gy2 = gcx + gbw/2, gcy + gbh/2
            iou = box_iou(x1, y1, x2, y2, gx1, gy1, gx2, gy2)
            if iou > best_iou:
                best_iou, best_j = iou, j

        if best_iou >= IOU_MATCH and best_j >= 0:
            matched_gt.add(best_j)
            tp_list.append(i)
            color = CLR_CUE if cls == 0 else CLR_OBJ
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(img, f"{cf:.2f}", (int(x1), int(y1)-4), FONT, 0.35, color, 1)
        else:
            fp_list.append(i)
            tag = classify_fp(warped_orig, x1, y1, x2, y2, cls, dets, i, gt_boxes)
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), CLR_FP, 2)
            cv2.putText(img, f"FP:{tag[:6]} {cf:.2f}", (int(x1), int(y1)-4),
                        FONT, 0.33, CLR_FP, 1)
            # outer ring
            cx_c, cy_c = int((x1+x2)/2), int((y1+y2)/2)
            r = int(max(x2-x1, y2-y1)/2) + 4
            cv2.circle(img, (cx_c, cy_c), r, CLR_FP, 1)

    fn_indices = [j for j in range(len(gt_boxes)) if j not in matched_gt]
    # Draw missed GT in orange
    for j in fn_indices:
        gcls, gcx, gcy, gbw, gbh = gt_boxes[j]
        x1, y1 = int(gcx - gbw/2), int(gcy - gbh/2)
        x2, y2 = int(gcx + gbw/2), int(gcy + gbh/2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 140, 255), 2)
        cv2.putText(img, "FN", (x1, y1-4), FONT, 0.38, (0, 140, 255), 1)

    return len(tp_list), len(fp_list), len(fn_indices)


def audit_stem(model, stem, report_lines):
    warped_path = WARP_DIR / f"{stem}_warped.jpg"
    if not warped_path.exists():
        warped_path = WARP_DIR / f"{stem}_warped.png"
    label_path  = LABEL_DIR / f"{stem}_warped.txt"

    warped = cv2.imread(str(warped_path))
    if warped is None:
        report_lines.append(f"ERROR: warped image not found for {stem}")
        return

    gt_boxes = parse_gt(label_path)
    gt_cue = sum(1 for b in gt_boxes if b[0] == 0)
    gt_obj = sum(1 for b in gt_boxes if b[0] == 1)

    report_lines.append(f"\n{'='*70}")
    report_lines.append(f"  AUDIT: {stem}")
    report_lines.append(f"{'='*70}")
    report_lines.append(f"  GT labels:  {len(gt_boxes)} total  (cue={gt_cue}, obj={gt_obj})")
    if len(gt_boxes) == 0:
        report_lines.append("  WARNING: empty GT label file — no ground truth available")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save GT-only overlay
    gt_img = warped.copy()
    draw_gt(gt_img, gt_boxes)
    cv2.putText(gt_img, f"GT: {len(gt_boxes)} balls (cue={gt_cue} obj={gt_obj})",
                (8, 20), FONT, 0.55, CLR_GT, 1)
    cv2.imwrite(str(OUT_DIR / f"{stem}_gt.jpg"), gt_img)

    # Per-conf analysis
    report_lines.append(f"\n  {'CONF':>6}  {'DET':>4}  {'TP':>4}  {'FP':>4}  {'FN':>4}  FP breakdown")
    report_lines.append(f"  {'-'*65}")

    all_conf_imgs = []

    for conf in CONFS:
        dets = run_model(model, warped, conf)
        pred_cue = [d for d in dets if d[0] == 0]
        pred_obj = [d for d in dets if d[0] == 1]

        img = warped.copy()
        draw_gt(img, gt_boxes)
        tp, fp, fn = draw_preds(img, dets, gt_boxes, conf, warped)

        # Count FP tags
        fp_tags = {}
        for i, (cls, cf, x1, y1, x2, y2) in enumerate(dets):
            # Recheck match status
            best_iou = 0.0
            for gcls, gcx, gcy, gbw, gbh in gt_boxes:
                if gcls != cls:
                    continue
                gx1, gy1 = gcx - gbw/2, gcy - gbh/2
                gx2, gy2 = gcx + gbw/2, gcy + gbh/2
                iou = box_iou(x1, y1, x2, y2, gx1, gy1, gx2, gy2)
                best_iou = max(best_iou, iou)
            if best_iou < IOU_MATCH:
                tag = classify_fp(warped, x1, y1, x2, y2, cls, dets, i, gt_boxes)
                fp_tags[tag] = fp_tags.get(tag, 0) + 1

        tag_str = "  ".join(f"{k}×{v}" for k, v in sorted(fp_tags.items()))

        header = f"conf={conf:.2f}  cue={len(pred_cue)}  obj={len(pred_obj)}"
        cv2.putText(img, header, (8, 20), FONT, 0.50, (200, 200, 200), 1)
        cv2.putText(img, f"TP={tp} FP={fp} FN={fn}", (8, 38), FONT, 0.50, (200, 200, 200), 1)

        conf_path = OUT_DIR / f"{stem}_conf{int(conf*100)}.jpg"
        cv2.imwrite(str(conf_path), img)
        all_conf_imgs.append(img)

        report_lines.append(
            f"  {conf:>6.2f}  {len(dets):>4}  {tp:>4}  {fp:>4}  {fn:>4}  {tag_str or '—'}"
        )

    # Composite: GT + all 4 conf levels in 2×3 grid
    gt_img_resized = cv2.resize(gt_img, (450, 225))
    row1 = np.hstack([gt_img_resized,
                      cv2.resize(all_conf_imgs[0], (450, 225))])
    row2 = np.hstack([cv2.resize(all_conf_imgs[1], (450, 225)),
                      cv2.resize(all_conf_imgs[2], (450, 225))])
    row3 = np.hstack([cv2.resize(all_conf_imgs[3], (450, 225)),
                      np.zeros((225, 450, 3), np.uint8)])
    composite = np.vstack([row1, row2, row3])
    cv2.imwrite(str(OUT_DIR / f"{stem}_composite.jpg"), composite)

    report_lines.append(f"\n  Saved: {OUT_DIR.relative_to(BASE)}/{stem}_*.jpg")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ball-ckpt", required=True)
    args = parser.parse_args()

    ckpt = Path(args.ball_ckpt)
    if not ckpt.exists():
        print(f"ERROR: checkpoint not found: {ckpt}")
        sys.exit(1)

    print(f"Loading model: {ckpt.name}")
    model = YOLO(str(ckpt))

    report_lines = [
        "=" * 70,
        "  OVER-DETECTION AUDIT — pool_real_004, pool_real_025, pool_real_031",
        f"  Model: {ckpt.name}",
        f"  IoU match threshold: {IOU_MATCH}",
        "=" * 70,
    ]

    for stem in STEMS:
        audit_stem(model, stem, report_lines)

    report_lines += [
        "",
        "=" * 70,
        "  LEGEND",
        "=" * 70,
        "  Green box  = GT label",
        "  Cyan box   = TP cue_ball prediction",
        "  Blue box   = TP obj_ball prediction",
        "  Red box    = FP (false positive)",
        "  Orange box = FN (missed GT)",
        "",
        "  FP tags:",
        "    duplicate          — overlaps another det of same class",
        "    rail_pocket        — centre within 30px of image edge",
        "    stripe_highlight   — bright (V>200) low-sat (S<40) region",
        "    cloth              — dark region (V<80)",
        "    possibly_unlabeled_ball — round bright blob, no nearby GT",
        "    unknown            — none of the above",
        "=" * 70,
    ]

    report_text = "\n".join(report_lines)
    report_path = OUT_DIR / "audit_report.txt"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text)
    print(report_text)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
