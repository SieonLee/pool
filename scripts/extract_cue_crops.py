"""
Cue-ball crop extractor — builds the binary cue identity classifier dataset.

Sources:
  1. GT crops  — from warped training images + YOLO label files
                 cls=0 → datasets/cue_crops/cue_ball/gt_<stem>_<i>.jpg
                 cls=1 → datasets/cue_crops/not_cue/gt_<stem>_<i>.jpg

  2. Hard negatives — YOLO v7 false-positive cue predictions on every
                 warped training image. Any cls=0 prediction whose center
                 does not match a GT cue box (IoU < 0.3) is a FP cue that
                 looks cue-like to the model but isn't.
                 → datasets/cue_crops/not_cue/fp_<stem>_<i>.jpg

Quality filters applied to every crop:
  - Laplacian variance < BLUR_THRESH → skip (too blurry)
  - Crop size after padding < MIN_CROP_PX → skip (too small to be useful)

Output:
  datasets/cue_crops/
    cue_ball/   ← true cue crops (GT only)
    not_cue/    ← GT obj crops + FP hard negatives
    manifest.json

Run:
  python scripts/extract_cue_crops.py
  python scripts/extract_cue_crops.py --aug   # also include augmented images
  python scripts/extract_cue_crops.py --show-blur-thresh  # calibrate blur threshold
"""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / "scripts"))

WARP_W, WARP_H  = 900, 450
CROP_SIZE        = 64       # resize all crops to this square
CROP_CONTEXT     = 1.6      # crop radius = GT_radius * CROP_CONTEXT
MIN_CROP_PX      = 20       # skip crop if source region is smaller than this
BLUR_THRESH      = 5.0      # Laplacian variance below this → blurry, skip
# Note: warped crops from small source images (~340px) are inherently soft;
# 5.0 retains them while filtering only purely textureless patches.
BALL_CONF_FP     = 0.10     # run YOLO at this conf to capture all FP cues
                             # (lowered from 0.15 to catch more borderline false cues)

# Whiteness-fallback hard-negative mining thresholds (mirror perceive.py)
CUE_MIN_BRIGHT   = 140
CUE_MAX_SAT      = 120
CUE_MIN_ASPECT   = 0.32
OBJ_CONF_WF      = 0.35     # min YOLO obj conf to consider for whiteness fallback

TRAIN_IMG_DIR    = BASE / "datasets" / "pose-warped-balls" / "images" / "train"
TRAIN_LABEL_DIR  = BASE / "datasets" / "pose-warped-balls" / "labels" / "train"
OUT_DIR          = BASE / "datasets" / "cue_crops"
BALL_CKPT        = BASE / "models" / "checkpoints" / "ball_yolo_v7_below_baseline.pt"


# ─────────────────────────────────────────────────────────────────────────────
# Crop helpers
# ─────────────────────────────────────────────────────────────────────────────

def yolo_to_pixel(cx_n, cy_n, w_n, h_n, img_w=WARP_W, img_h=WARP_H):
    cx = cx_n * img_w
    cy = cy_n * img_h
    w  = w_n  * img_w
    h  = h_n  * img_h
    return cx, cy, w, h


def extract_square_crop(img, cx, cy, r, size=CROP_SIZE):
    """
    Extract a square crop centred at (cx,cy) with half-side r.
    Pads with reflect if the region extends outside the image.
    Returns (crop_bgr, source_wh) or (None, None) if source region is too small.
    """
    r = int(math.ceil(r))
    src_w = src_h = r * 2
    if src_w < MIN_CROP_PX or src_h < MIN_CROP_PX:
        return None, None

    x1 = int(round(cx - r))
    y1 = int(round(cy - r))
    x2 = x1 + src_w
    y2 = y1 + src_h

    # Pad if needed
    pad_top    = max(0, -y1)
    pad_bottom = max(0, y2 - img.shape[0])
    pad_left   = max(0, -x1)
    pad_right  = max(0, x2 - img.shape[1])

    if any([pad_top, pad_bottom, pad_left, pad_right]):
        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right,
                                 cv2.BORDER_REFLECT_101)
        y1 += pad_top;  y2 += pad_top
        x1 += pad_left; x2 += pad_left

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None

    crop_resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    return crop_resized, (src_w, src_h)


def blur_score(crop):
    """Laplacian variance — higher = sharper."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ─────────────────────────────────────────────────────────────────────────────
# IoU helpers (for FP matching)
# ─────────────────────────────────────────────────────────────────────────────

def box_iou(cx1, cy1, r1, cx2, cy2, r2):
    """Approximate IoU for two square bboxes given centre and half-side."""
    x1a, y1a = cx1 - r1, cy1 - r1
    x2a, y2a = cx1 + r1, cy1 + r1
    x1b, y1b = cx2 - r2, cy2 - r2
    x2b, y2b = cx2 + r2, cy2 + r2

    ix1, iy1 = max(x1a, x1b), max(y1a, y1b)
    ix2, iy2 = min(x2a, x2b), min(y2a, y2b)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (2 * r1) ** 2
    area_b = (2 * r2) ** 2
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def is_near_gt_cue(pred_cx, pred_cy, pred_r, gt_cues, iou_thresh=0.30):
    """Return True if the prediction overlaps any GT cue box at ≥ iou_thresh."""
    for (gt_cx, gt_cy, gt_r) in gt_cues:
        if box_iou(pred_cx, pred_cy, pred_r, gt_cx, gt_cy, gt_r) >= iou_thresh:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Main extraction
# ─────────────────────────────────────────────────────────────────────────────

def parse_label_file(label_path):
    """Return list of (cls, cx_px, cy_px, r_px) for each annotation."""
    entries = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        cx_px, cy_px, w_px, h_px = yolo_to_pixel(cx, cy, w, h)
        r_px = max(w_px, h_px) / 2.0 * CROP_CONTEXT
        entries.append((cls, cx_px, cy_px, r_px))
    return entries


def extract_gt_crops(img, entries, stem, out_cue, out_notcue, stats):
    """Extract GT-labeled crops. Returns list of manifest entries."""
    records = []
    for i, (cls, cx, cy, r) in enumerate(entries):
        crop, src_wh = extract_square_crop(img, cx, cy, r)
        if crop is None:
            stats["skipped_small"] += 1
            continue

        bl = blur_score(crop)
        if bl < BLUR_THRESH:
            stats["skipped_blur"] += 1
            continue

        label = "cue_ball" if cls == 0 else "not_cue"
        out_dir = out_cue if cls == 0 else out_notcue
        fname   = f"gt_{stem}_{i:03d}.jpg"
        fpath   = out_dir / fname
        cv2.imwrite(str(fpath), crop)

        records.append({
            "file": fname, "label": label, "source": "gt",
            "stem": stem, "ball_idx": i,
            "cx": round(cx, 1), "cy": round(cy, 1), "r": round(r, 1),
            "blur_score": round(bl, 1),
        })
        stats["gt_cue" if cls == 0 else "gt_obj"] += 1

    return records


def extract_whiteness_hardneg(img, ball_model, entries, stem, out_notcue, stats):
    """
    Mine the whiteness-fallback path as hard negatives.

    Mirrors the production whiteness fallback in perceive.py:
      run YOLO at OBJ_CONF_WF, pick object_ball candidates that meet the
      bright/low-sat/aspect criteria, take the best (highest brightness-sat),
      save it as a not_cue crop IF it doesn't overlap a GT cue.

    This directly targets the pool_real_035-type FP: an object_ball that
    passes the whiteness gate and then fools the classifier.
    """
    gt_cues = [(cx, cy, r / CROP_CONTEXT)
               for cls, cx, cy, r in entries if cls == 0]

    results = ball_model(img, verbose=False, conf=OBJ_CONF_WF)
    if not results or results[0].boxes is None:
        return []

    candidates = []
    for box in results[0].boxes:
        if int(box.cls.item()) != 1:
            continue  # only object_ball predictions

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cx   = (x1 + x2) / 2;  cy  = (y1 + y2) / 2
        r_px = max(x2 - x1, y2 - y1) / 2.0
        w, h = int(x2 - x1), int(y2 - y1)
        asp  = min(w, h) / max(max(w, h), 1)

        patch = img[max(0, int(y1)):min(WARP_H, int(y2)),
                    max(0, int(x1)):min(WARP_W, int(x2))]
        if patch.size == 0:
            continue
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        V   = float(hsv[:, :, 2].mean())
        S   = float(hsv[:, :, 1].mean())

        if V >= CUE_MIN_BRIGHT and S <= CUE_MAX_SAT and asp >= CUE_MIN_ASPECT:
            candidates.append((V - S, cx, cy, r_px))

    if not candidates:
        return []

    # Take the top candidate (mirrors how perceive.py selects the best recovery)
    _, cx, cy, r_px = max(candidates, key=lambda x: x[0])

    if is_near_gt_cue(cx, cy, r_px, gt_cues):
        return []   # it actually overlaps the real cue → skip

    crop, _ = extract_square_crop(img, cx, cy, r_px * CROP_CONTEXT)
    if crop is None:
        stats["skipped_small"] += 1
        return []

    bl = blur_score(crop)
    if bl < BLUR_THRESH:
        stats["skipped_blur"] += 1
        return []

    fname = f"wf_{stem}.jpg"
    fpath = out_notcue / fname
    cv2.imwrite(str(fpath), crop)
    stats["fp_whiteness"] = stats.get("fp_whiteness", 0) + 1

    return [{
        "file": fname, "label": "not_cue", "source": "fp_whiteness",
        "stem": stem,
        "cx": round(float(cx), 1), "cy": round(float(cy), 1),
        "r": round(float(r_px), 1), "blur_score": round(bl, 1),
    }]


def extract_fp_crops(img, ball_model, entries, stem, out_notcue, stats):
    """
    Run YOLO v7 on the warped image, collect cls=0 predictions that do NOT
    match any GT cue box → these are hard-negative false-positive cue crops.
    """
    # GT cue bounding boxes for IoU matching
    gt_cues = [(cx, cy, r / CROP_CONTEXT)   # use tight radius for IoU
               for cls, cx, cy, r in entries if cls == 0]

    results = ball_model(img, verbose=False, conf=BALL_CONF_FP)
    records = []
    fp_idx  = 0
    if not results or results[0].boxes is None:
        return records

    for box in results[0].boxes:
        if int(box.cls.item()) != 0:
            continue   # only inspect cue predictions

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        pred_cx  = (x1 + x2) / 2
        pred_cy  = (y1 + y2) / 2
        pred_r   = max(x2 - x1, y2 - y1) / 2.0
        pred_cf  = float(box.conf.item())

        if is_near_gt_cue(pred_cx, pred_cy, pred_r, gt_cues):
            continue   # matches a real cue → skip

        # This is a false-positive cue prediction → hard negative
        crop, _ = extract_square_crop(img, pred_cx, pred_cy,
                                      pred_r * CROP_CONTEXT)
        if crop is None:
            stats["skipped_small"] += 1
            continue

        bl = blur_score(crop)
        if bl < BLUR_THRESH:
            stats["skipped_blur"] += 1
            continue

        fname = f"fp_{stem}_{fp_idx:03d}.jpg"
        fpath = out_notcue / fname
        cv2.imwrite(str(fpath), crop)
        fp_idx += 1

        records.append({
            "file": fname, "label": "not_cue", "source": "fp_hardneg",
            "stem": stem, "yolo_conf": round(pred_cf, 3),
            "cx": round(pred_cx, 1), "cy": round(pred_cy, 1),
            "r": round(pred_r, 1), "blur_score": round(bl, 1),
        })
        stats["fp_hardneg"] += 1

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aug", action="store_true",
                        help="Also extract from augmented images (aug_*)")
    parser.add_argument("--show-blur-thresh", action="store_true",
                        help="Print blur scores to help calibrate BLUR_THRESH")
    parser.add_argument("--ball-ckpt", default=str(BALL_CKPT))
    args = parser.parse_args()

    from ultralytics import YOLO
    ball_model = YOLO(args.ball_ckpt)
    print(f"Ball model: {Path(args.ball_ckpt).name}")

    # Output dirs
    out_cue     = OUT_DIR / "cue_ball"
    out_notcue  = OUT_DIR / "not_cue"
    out_cue.mkdir(parents=True, exist_ok=True)
    out_notcue.mkdir(parents=True, exist_ok=True)

    # Collect base images (non-augmented unless --aug)
    img_paths = sorted(TRAIN_IMG_DIR.glob("*.jpg"))
    if not args.aug:
        img_paths = [p for p in img_paths if not p.stem.startswith("aug_")]
    print(f"Processing {len(img_paths)} training images "
          f"({'including' if args.aug else 'excluding'} augmented)...")

    stats = {
        "gt_cue": 0, "gt_obj": 0, "fp_hardneg": 0, "fp_whiteness": 0,
        "skipped_blur": 0, "skipped_small": 0,
    }
    manifest = []

    for img_path in img_paths:
        label_path = TRAIN_LABEL_DIR / (img_path.stem + ".txt")

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  SKIP {img_path.name} — could not read")
            continue

        if not label_path.exists():
            # No label file: no GT crops, but still mine FP hard-negatives.
            # This catches YOLO false cues on completely unannotated images
            # (e.g., pool_real_008 which has an empty label file and a YOLO
            # false-positive cls=0 at cx=820 that fools the classifier).
            stem   = img_path.stem
            fp_recs = extract_fp_crops(img, ball_model, [], stem, out_notcue, stats)
            wf_recs = extract_whiteness_hardneg(img, ball_model, [], stem, out_notcue, stats)
            manifest.extend(fp_recs)
            manifest.extend(wf_recs)
            n_fp = len(fp_recs); n_wf = len(wf_recs)
            if n_fp or n_wf:
                print(f"  {stem:45s}  (no GT)  fp={n_fp}  wf={n_wf}")
            continue

        entries = parse_label_file(label_path)
        stem    = img_path.stem

        gt_recs = extract_gt_crops(img, entries, stem, out_cue, out_notcue, stats)
        fp_recs = extract_fp_crops(img, ball_model, entries, stem, out_notcue, stats)
        wf_recs = extract_whiteness_hardneg(img, ball_model, entries, stem, out_notcue, stats)

        if args.show_blur_thresh:
            for r in gt_recs + fp_recs + wf_recs:
                src = r.get("source", "?")
                tag = "CUE" if r["label"] == "cue_ball" else src[:3].upper()
                print(f"  {tag:3s}  blur={r['blur_score']:6.1f}  {r['file']}")

        manifest.extend(gt_recs)
        manifest.extend(fp_recs)
        manifest.extend(wf_recs)

        n_cue = sum(1 for r in gt_recs if r["label"] == "cue_ball")
        n_obj = sum(1 for r in gt_recs if r["label"] == "not_cue")
        n_fp  = len(fp_recs)
        n_wf  = len(wf_recs)
        print(f"  {stem:45s}  cue={n_cue}  obj={n_obj}  fp={n_fp}  wf={n_wf}")

    # Write manifest — convert numpy scalars to native Python types first
    def _native(obj):
        if hasattr(obj, "item"):
            return obj.item()
        return obj

    def _clean(rec):
        return {k: _native(v) for k, v in rec.items()}

    manifest = [_clean(r) for r in manifest]

    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps({
        "ball_ckpt": Path(args.ball_ckpt).name,
        "crop_size": CROP_SIZE,
        "crop_context": CROP_CONTEXT,
        "blur_thresh": BLUR_THRESH,
        "fp_conf_thresh": BALL_CONF_FP,
        "stats": stats,
        "crops": manifest,
    }, indent=2))

    total = (stats["gt_cue"] + stats["gt_obj"]
             + stats["fp_hardneg"] + stats.get("fp_whiteness", 0))
    print(f"\n{'─'*60}")
    print(f"  cue_ball  (GT)          : {stats['gt_cue']:>4}")
    print(f"  not_cue   (GT obj)      : {stats['gt_obj']:>4}")
    print(f"  not_cue   (FP hardneg)  : {stats['fp_hardneg']:>4}")
    print(f"  not_cue   (FP whiteness): {stats.get('fp_whiteness',0):>4}")
    print(f"  ─────────────────────")
    print(f"  total                 : {total:>4}")
    print(f"  skipped (blur)        : {stats['skipped_blur']:>4}")
    print(f"  skipped (small)       : {stats['skipped_small']:>4}")
    print(f"\nOutputs → {OUT_DIR}")
    print(f"Manifest → {manifest_path}")


if __name__ == "__main__":
    main()
