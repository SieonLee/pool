"""
Evaluate trained YOLO ball detector on pose-warped images.

Runs inference on all labeled warped images (train + val) and reports:
  - precision, recall, mAP50 (from YOLO val)
  - per-image: predicted count, GT count, FP, FN, cue detected
  - debug overlays: GT, prediction, side-by-side

Outputs → review/pose_warped_ball_yolo/
  <stem>_gt.jpg          GT boxes on warped image
  <stem>_pred.jpg        predicted boxes on warped image
  <stem>_compare.jpg     side-by-side GT vs prediction
  <stem>_result.json     per-image metrics
  summary.json           full evaluation summary

Run:
  python scripts/eval_ball_yolo.py
  python scripts/eval_ball_yolo.py --checkpoint models/checkpoints/ball_yolo_v1.pt
  python scripts/eval_ball_yolo.py --conf 0.3
"""
import argparse
import json
import cv2
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent.parent
DATASET = BASE / "datasets" / "pose-warped-balls"
IMG_TRAIN = DATASET / "images" / "train"
IMG_VAL = DATASET / "images" / "val"
LBL_TRAIN = DATASET / "labels" / "train"
LBL_VAL = DATASET / "labels" / "val"
CKPT_DEFAULT = BASE / "models" / "checkpoints" / "ball_yolo_active.pt"
YAML_PATH = DATASET / "dataset.yaml"
OUT_DIR = BASE / "review" / "pose_warped_ball_yolo"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WARP_W, WARP_H = 900, 450
CLASS_NAMES = ["cue_ball", "object_ball"]
CUE_COLOR = (255, 255, 255)
OBJ_COLOR = (0, 165, 255)
GT_ALPHA = 0.85
FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_gt_boxes(lbl_path):
    if not lbl_path.exists():
        return []
    boxes = []
    for line in lbl_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            cls = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])
            boxes.append((cls, cx, cy, bw, bh))
    return boxes


def yolo_to_pixel(cx, cy, bw, bh, W=WARP_W, H=WARP_H):
    x1 = int((cx - bw / 2) * W)
    y1 = int((cy - bh / 2) * H)
    x2 = int((cx + bw / 2) * W)
    y2 = int((cy + bh / 2) * H)
    return x1, y1, x2, y2


def draw_gt(img, boxes):
    canvas = img.copy()
    for cls, cx, cy, bw, bh in boxes:
        x1, y1, x2, y2 = yolo_to_pixel(cx, cy, bw, bh)
        color = CUE_COLOR if cls == 0 else OBJ_COLOR
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = CLASS_NAMES[cls]
        cv2.putText(canvas, label[0].upper(), (x1 + 2, y1 + 14),
                    FONT, 0.45, color, 1)
    n_cue = sum(1 for b in boxes if b[0] == 0)
    n_obj = sum(1 for b in boxes if b[0] == 1)
    cv2.putText(canvas, f"GT  cue={n_cue}  obj={n_obj}  total={len(boxes)}",
                (8, 24), FONT, 0.65, (200, 255, 200), 2)
    return canvas


def draw_pred(img, detections, conf_thresh=0.25):
    canvas = img.copy()
    shown = [(cls, conf, x1, y1, x2, y2)
             for cls, conf, x1, y1, x2, y2 in detections if conf >= conf_thresh]
    for cls, conf, x1, y1, x2, y2 in shown:
        color = CUE_COLOR if cls == 0 else OBJ_COLOR
        cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        tag = f"{CLASS_NAMES[cls][0].upper()} {conf:.2f}"
        cv2.putText(canvas, tag, (int(x1) + 2, int(y1) + 14),
                    FONT, 0.4, color, 1)
    n_cue = sum(1 for d in shown if d[0] == 0)
    n_obj = sum(1 for d in shown if d[0] == 1)
    cv2.putText(canvas, f"PRED  cue={n_cue}  obj={n_obj}  total={len(shown)}",
                (8, 24), FONT, 0.65, (255, 200, 200), 2)
    return canvas


def iou(b1, b2):
    """IoU between two (x1,y1,x2,y2) boxes."""
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def match_boxes(gt_boxes, pred_detections, conf_thresh=0.25, iou_thresh=0.4):
    """
    Returns: tp, fp, fn lists and per-class metrics.
    gt_boxes: [(cls, cx, cy, bw, bh), ...]
    pred_detections: [(cls, conf, x1, y1, x2, y2), ...]
    """
    preds = [(cls, conf, x1, y1, x2, y2)
             for cls, conf, x1, y1, x2, y2 in pred_detections if conf >= conf_thresh]
    preds_sorted = sorted(preds, key=lambda d: -d[1])

    gt_pixel = []
    for cls, cx, cy, bw, bh in gt_boxes:
        x1, y1, x2, y2 = yolo_to_pixel(cx, cy, bw, bh)
        gt_pixel.append((cls, x1, y1, x2, y2))

    matched_gt = set()
    tp = fp = fn = 0
    tp_cue = tp_obj = fn_cue = fn_obj = fp_cue = fp_obj = 0

    for cls_p, conf, px1, py1, px2, py2 in preds_sorted:
        best_iou, best_idx = 0.0, -1
        for gi, (cls_g, gx1, gy1, gx2, gy2) in enumerate(gt_pixel):
            if gi in matched_gt or cls_g != cls_p:
                continue
            v = iou((px1, py1, px2, py2), (gx1, gy1, gx2, gy2))
            if v > best_iou:
                best_iou, best_idx = v, gi
        if best_iou >= iou_thresh and best_idx >= 0:
            matched_gt.add(best_idx)
            tp += 1
            if cls_p == 0:
                tp_cue += 1
            else:
                tp_obj += 1
        else:
            fp += 1
            if cls_p == 0:
                fp_cue += 1
            else:
                fp_obj += 1

    fn = len(gt_pixel) - len(matched_gt)
    for gi, (cls_g, *_) in enumerate(gt_pixel):
        if gi not in matched_gt:
            if cls_g == 0:
                fn_cue += 1
            else:
                fn_obj += 1

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "tp_cue": tp_cue, "fp_cue": fp_cue, "fn_cue": fn_cue,
        "tp_obj": tp_obj, "fp_obj": fp_obj, "fn_obj": fn_obj,
    }


def precision_recall(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return round(prec, 3), round(rec, 3)


def process_image(model, img_path, lbl_path, conf_thresh, img_out_dir):
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    stem = img_path.stem
    gt_boxes = load_gt_boxes(lbl_path)

    # Run inference
    results = model(img, verbose=False, conf=conf_thresh)
    detections = []
    if results and results[0].boxes is not None:
        boxes_data = results[0].boxes
        for i in range(len(boxes_data)):
            cls = int(boxes_data.cls[i].item())
            conf = float(boxes_data.conf[i].item())
            x1, y1, x2, y2 = boxes_data.xyxy[i].cpu().numpy()
            detections.append((cls, conf, float(x1), float(y1), float(x2), float(y2)))

    # Save visuals
    gt_vis = draw_gt(img, gt_boxes)
    pred_vis = draw_pred(img, detections, conf_thresh)
    compare = np.hstack([gt_vis, pred_vis])

    out_stem = img_out_dir / stem
    out_stem.mkdir(exist_ok=True)
    cv2.imwrite(str(out_stem / f"{stem}_gt.jpg"), gt_vis)
    cv2.imwrite(str(out_stem / f"{stem}_pred.jpg"), pred_vis)
    cv2.imwrite(str(out_stem / f"{stem}_compare.jpg"), compare)

    # Metrics
    match = match_boxes(gt_boxes, detections, conf_thresh)
    n_gt = len(gt_boxes)
    n_gt_cue = sum(1 for b in gt_boxes if b[0] == 0)
    n_gt_obj = sum(1 for b in gt_boxes if b[0] == 1)
    n_pred = len([d for d in detections if d[1] >= conf_thresh])
    cue_detected = any(d[0] == 0 and d[1] >= conf_thresh for d in detections)

    prec, rec = precision_recall(match["tp"], match["fp"], match["fn"])
    prec_cue, rec_cue = precision_recall(match["tp_cue"], match["fp_cue"], match["fn_cue"])
    prec_obj, rec_obj = precision_recall(match["tp_obj"], match["fp_obj"], match["fn_obj"])

    result = {
        "image": img_path.name,
        "split": "val" if "val" in str(lbl_path) else "train",
        "gt_total": n_gt,
        "gt_cue": n_gt_cue,
        "gt_obj": n_gt_obj,
        "pred_total": n_pred,
        "tp": match["tp"], "fp": match["fp"], "fn": match["fn"],
        "tp_cue": match["tp_cue"], "fp_cue": match["fp_cue"], "fn_cue": match["fn_cue"],
        "tp_obj": match["tp_obj"], "fp_obj": match["fp_obj"], "fn_obj": match["fn_obj"],
        "precision": prec,
        "recall": rec,
        "precision_cue": prec_cue,
        "recall_cue": rec_cue,
        "precision_obj": prec_obj,
        "recall_obj": rec_obj,
        "cue_detected": cue_detected,
        "detections": [
            {"cls": CLASS_NAMES[d[0]], "conf": round(d[1], 3),
             "x1": round(d[2]), "y1": round(d[3]), "x2": round(d[4]), "y2": round(d[5])}
            for d in detections if d[1] >= conf_thresh
        ],
    }

    with open(out_stem / f"{stem}_result.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def print_report(results, conf_thresh):
    print(f"\n{'=' * 80}")
    print(f"YOLO BALL DETECTOR EVALUATION  (conf≥{conf_thresh})")
    print(f"{'=' * 80}")
    fmt = "{:30s} {:>5} {:>5} {:>4} {:>4} {:>4} {:>7} {:>7} {:>6}"
    print(fmt.format("image", "gt", "pred", "tp", "fp", "fn", "prec", "rec", "cue"))
    print("-" * 80)
    for r in results:
        if r is None:
            continue
        cue = "YES" if r["cue_detected"] else "no"
        print(fmt.format(
            r["image"][:30],
            str(r["gt_total"]),
            str(r["pred_total"]),
            str(r["tp"]),
            str(r["fp"]),
            str(r["fn"]),
            f"{r['precision']:.3f}",
            f"{r['recall']:.3f}",
            cue,
        ))

    valid = [r for r in results if r]
    if not valid:
        return

    tp = sum(r["tp"] for r in valid)
    fp = sum(r["fp"] for r in valid)
    fn = sum(r["fn"] for r in valid)
    tp_c = sum(r["tp_cue"] for r in valid)
    fp_c = sum(r["fp_cue"] for r in valid)
    fn_c = sum(r["fn_cue"] for r in valid)
    tp_o = sum(r["tp_obj"] for r in valid)
    fp_o = sum(r["fp_obj"] for r in valid)
    fn_o = sum(r["fn_obj"] for r in valid)

    overall_prec, overall_rec = precision_recall(tp, fp, fn)
    cue_prec, cue_rec = precision_recall(tp_c, fp_c, fn_c)
    obj_prec, obj_rec = precision_recall(tp_o, fp_o, fn_o)
    n_cue_detected = sum(1 for r in valid if r["cue_detected"])
    n_has_cue = sum(1 for r in valid if r["gt_cue"] > 0)

    print("-" * 80)
    print(f"Overall    precision={overall_prec:.3f}  recall={overall_rec:.3f}")
    print(f"cue_ball   precision={cue_prec:.3f}  recall={cue_rec:.3f}  "
          f"detected={n_cue_detected}/{n_has_cue} images")
    print(f"obj_ball   precision={obj_prec:.3f}  recall={obj_rec:.3f}")
    f1 = 2 * overall_prec * overall_rec / (overall_prec + overall_rec + 1e-9)
    print(f"F1 (overall): {f1:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(CKPT_DEFAULT))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--yolo-val", action="store_true",
                        help="Also run YOLO's built-in val (mAP50 etc.)")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"Checkpoint not found: {ckpt}")
        print("Run train_ball_yolo.py first.")
        return

    from ultralytics import YOLO
    model = YOLO(str(ckpt))

    if args.yolo_val and YAML_PATH.exists():
        print("Running YOLO built-in val (mAP50)...")
        val_results = model.val(data=str(YAML_PATH), verbose=False)
        print(f"  mAP50:    {val_results.box.map50:.3f}")
        print(f"  mAP50-95: {val_results.box.map:.3f}")
        print(f"  Per-class mAP50: {val_results.box.ap50.tolist()}")

    # Collect all labeled images (train + val)
    all_pairs = []
    for img_dir, lbl_dir in [(IMG_TRAIN, LBL_TRAIN), (IMG_VAL, LBL_VAL)]:
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.glob("*.jpg")):
            # Skip aug files for cleaner per-image report
            if img_path.stem.startswith("aug_"):
                continue
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if lbl_path.exists() and lbl_path.stat().st_size > 0:
                all_pairs.append((img_path, lbl_path))

    if not all_pairs:
        print("No labeled images found.")
        return

    print(f"\nEvaluating {len(all_pairs)} image(s)...")
    results = []
    for img_path, lbl_path in all_pairs:
        print(f"  {img_path.name}...", end=" ", flush=True)
        r = process_image(model, img_path, lbl_path, args.conf, OUT_DIR)
        results.append(r)
        if r:
            print(f"gt={r['gt_total']}  pred={r['pred_total']}  "
                  f"tp={r['tp']} fp={r['fp']} fn={r['fn']}  "
                  f"cue={'✓' if r['cue_detected'] else '✗'}")
        else:
            print("FAILED")

    print_report(results, args.conf)

    summary = {
        "checkpoint": str(ckpt),
        "conf_threshold": args.conf,
        "n_images": len(all_pairs),
        "results": [r for r in results if r],
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nOutputs → {OUT_DIR}")
    print(f"Summary → {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
