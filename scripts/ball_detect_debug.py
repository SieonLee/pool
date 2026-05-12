"""
Ball detection debug on pose-warped table.

Pipeline:
  1. Pose model → predict 4 corners
  2. Warp to 900×450
  3. HoughCircles on warped image → ball candidates
  4. Filter by size / color / position
  5. Save debug overlays + JSON report

Outputs → review/ball_detect_debug/<stem>/
  <stem>_corners.jpg      original + predicted corners
  <stem>_warp.jpg         pose-warped table (clean)
  <stem>_balls.jpg        warped + ball detections
  <stem>_result.json      per-image metrics

Summary → review/ball_detect_debug/summary.json

Run:
  python scripts/ball_detect_debug.py
  python scripts/ball_detect_debug.py --image picture/pool_real_007.jpg
"""
import argparse
import cv2
import json
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent.parent
CHECKPOINT = BASE / "models" / "checkpoints" / "table_corners_mvp_v1.pt"
PICTURE_DIR = BASE / "picture"
GT_DIR = BASE / "annotations" / "table_corners_gt"
OUT_DIR = BASE / "review" / "ball_detect_debug"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WARP_W, WARP_H = 900, 450
FONT = cv2.FONT_HERSHEY_SIMPLEX
CORNER_COLORS = [(0, 255, 0), (0, 200, 255), (0, 0, 255), (255, 0, 255)]
CORNER_NAMES = ["TL", "TR", "BR", "BL"]

# Ball detection tuning for top-view 900×450
# Billiard ball diameter ~57mm, table ~2700mm long → ~19px diameter at 900px width
BALL_RADIUS_MIN = 8    # px on warped image
BALL_RADIUS_MAX = 28   # px on warped image
BALL_MIN_DIST = 18     # minimum distance between ball centers

# Test images to always include in report
TARGET_STEMS = [
    "pool_real_002", "pool_real_003", "pool_real_007",
    "pool_real_010", "pool_real_023", "pool_real_024", "pool_real_027",
]

SKIP_SUFFIXES = {"new_uploads_contact_sheet", "search_contact_sheet", "thumb"}


def get_pose_corners(model, img):
    results = model(img, verbose=False)
    if not results or results[0].keypoints is None:
        return None, None
    kps = results[0].keypoints
    if len(kps) == 0:
        return None, None
    if kps.conf is not None:
        best = int(kps.conf.mean(dim=1).argmax())
        conf = kps.conf[best].cpu().numpy()
    else:
        best = 0
        conf = None
    xy = kps.xy[best].cpu().numpy()
    return xy, conf


def warp_image(img, corners):
    dst = np.array([[0, 0], [WARP_W, 0], [WARP_W, WARP_H], [0, WARP_H]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (WARP_W, WARP_H)), M


def cloth_mask(warped_img):
    """Binary mask of playable cloth area (green/blue-green felt)."""
    hsv = cv2.cvtColor(warped_img, cv2.COLOR_BGR2HSV)
    # Cover green (H 35-85) and blue-green (H 85-130) cloth colors
    m1 = cv2.inRange(hsv, (35, 40, 40), (130, 255, 255))
    # Morphologically expand to fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(m1, cv2.MORPH_CLOSE, kernel)
    return mask


def detect_balls(warped_img):
    """HoughCircles on warped image. Returns list of (cx, cy, r)."""
    gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 1.5)

    # Adaptive param2: images with high texture need stricter threshold
    cloth_region = gray[int(0.15*WARP_H):int(0.85*WARP_H),
                        int(0.15*WARP_W):int(0.85*WARP_W)]
    texture_std = float(cloth_region.std())
    # Low texture (smooth cloth, few balls) → lower threshold to catch all balls
    # High texture (busy background) → higher threshold to reduce false positives
    param2 = 18 if texture_std < 18 else 24

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=BALL_MIN_DIST,
        param1=60,
        param2=param2,
        minRadius=BALL_RADIUS_MIN,
        maxRadius=BALL_RADIUS_MAX,
    )
    if circles is None:
        return []

    # Edge margin filter
    margin = BALL_RADIUS_MAX + 4
    raw = [(float(cx), float(cy), float(r)) for cx, cy, r in circles[0]
           if margin < cx < WARP_W - margin and margin < cy < WARP_H - margin]
    if not raw:
        return []

    # Radius consistency: billiard balls are uniform size.
    # Keep only circles within ±35% of the median radius.
    radii = np.array([r for _, _, r in raw])
    median_r = float(np.median(radii))
    raw = [(cx, cy, r) for cx, cy, r in raw
           if 0.65 * median_r <= r <= 1.35 * median_r]

    # Cloth mask filter: reject circles whose center lands outside cloth area
    mask = cloth_mask(warped_img)
    raw = [(cx, cy, r) for cx, cy, r in raw
           if mask[min(int(cy), WARP_H-1), min(int(cx), WARP_W-1)] > 0]

    return sorted(raw, key=lambda b: (b[1], b[0]))


def is_likely_cue_ball(warped_img, cx, cy, r):
    """Heuristic: cue ball is white/light colored."""
    x0 = max(0, int(cx - r))
    y0 = max(0, int(cy - r))
    x1 = min(WARP_W, int(cx + r))
    y1 = min(WARP_H, int(cy + r))
    patch = warped_img[y0:y1, x0:x1]
    if patch.size == 0:
        return False
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    mean_v = hsv[:, :, 2].mean()
    mean_s = hsv[:, :, 1].mean()
    return mean_v > 190 and mean_s < 50


def draw_corners_on_img(img, corners, confs=None):
    canvas = img.copy()
    for i, (x, y) in enumerate(corners):
        x, y = int(round(x)), int(round(y))
        cv2.circle(canvas, (x, y), 8, CORNER_COLORS[i], -1)
        label = CORNER_NAMES[i]
        if confs is not None:
            label += f" {confs[i]:.2f}"
        cv2.putText(canvas, label, (x + 10, y - 10), FONT, 0.65, CORNER_COLORS[i], 2)
    pts = np.array([(int(round(x)), int(round(y))) for x, y in corners], np.int32)
    cv2.polylines(canvas, [pts], isClosed=True, color=(255, 255, 255), thickness=2)
    return canvas


def draw_balls_on_warp(warped_img, balls, cue_idx=None):
    canvas = warped_img.copy()
    for i, (cx, cy, r) in enumerate(balls):
        color = (255, 255, 255) if i == cue_idx else (0, 165, 255)
        cv2.circle(canvas, (int(cx), int(cy)), int(r), color, 2)
        cv2.circle(canvas, (int(cx), int(cy)), 2, color, -1)
        cv2.putText(canvas, str(i + 1), (int(cx) - 6, int(cy) + 5),
                    FONT, 0.45, color, 1)
    cv2.putText(canvas, f"Balls: {len(balls)}", (8, 24), FONT, 0.7, (200, 200, 200), 2)
    if cue_idx is not None:
        cv2.putText(canvas, f"Cue: #{cue_idx + 1}", (8, 50), FONT, 0.6, (255, 255, 255), 2)
    return canvas


def process_image(model, img_path: Path, gt_corners=None):
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    # --- Pose warp ---
    pred_corners, confs = get_pose_corners(model, img)
    if pred_corners is None:
        return {
            "image": img_path.name, "status": "no_pose_detection",
            "ball_count": 0, "cue_detected": False,
        }

    avg_conf = float(np.mean(confs)) if confs is not None else None
    warped, M = warp_image(img, pred_corners)
    balls = detect_balls(warped)

    # Identify cue ball
    cue_idx = None
    for i, (cx, cy, r) in enumerate(balls):
        if is_likely_cue_ball(warped, cx, cy, r):
            cue_idx = i
            break

    # --- Stem and output dir ---
    stem = img_path.stem
    img_out = OUT_DIR / stem
    img_out.mkdir(exist_ok=True)

    # Save: original + corners
    corners_vis = draw_corners_on_img(img, pred_corners, confs)
    cv2.imwrite(str(img_out / f"{stem}_corners.jpg"), corners_vis)

    # Save: clean warp
    cv2.imwrite(str(img_out / f"{stem}_warp.jpg"), warped)

    # Save: warp + balls
    balls_vis = draw_balls_on_warp(warped, balls, cue_idx)
    cv2.imwrite(str(img_out / f"{stem}_balls.jpg"), balls_vis)

    # --- GT corner error (if available) ---
    corner_error = None
    if gt_corners is not None:
        errs = np.linalg.norm(pred_corners - gt_corners, axis=1)
        corner_error = {
            "mean_px": round(float(errs.mean()), 2),
            "max_px": round(float(errs.max()), 2),
            "per_corner": {n: round(float(e), 2) for n, e in zip(CORNER_NAMES, errs)},
        }

    result = {
        "image": img_path.name,
        "status": "ok",
        "corner_confidence": round(avg_conf, 3) if avg_conf else None,
        "corner_error": corner_error,
        "ball_count": len(balls),
        "cue_detected": cue_idx is not None,
        "cue_ball_index": cue_idx,
        "ball_centers": [{"x": round(cx, 1), "y": round(cy, 1), "r": round(r, 1)}
                         for cx, cy, r in balls],
        "predicted_corners": pred_corners.tolist(),
    }

    # Save per-image JSON
    with open(img_out / f"{stem}_result.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def load_gt(stem):
    jf = GT_DIR / (stem + ".json")
    if not jf.exists():
        return None
    with open(jf) as f:
        data = json.load(f)
    c = data["corners"]
    return np.array([c["TL"], c["TR"], c["BR"], c["BL"]], dtype=np.float32)


def print_report(results):
    print("\n" + "=" * 72)
    print("BALL DETECTION REPORT — Pose-Warped Table")
    print("=" * 72)
    fmt = "{:22s} {:>6} {:>8} {:>8} {:>8} {:>6}"
    print(fmt.format("image", "balls", "cue", "c_mean", "c_max", "status"))
    print("-" * 72)
    for r in results:
        if r is None:
            continue
        ce = r.get("corner_error") or {}
        c_mean = f"{ce['mean_px']:.1f}px" if ce else "N/A"
        c_max  = f"{ce['max_px']:.1f}px" if ce else "N/A"
        cue    = "YES" if r.get("cue_detected") else "no"
        status = r.get("status", "?")
        print(fmt.format(
            r["image"][:22],
            str(r.get("ball_count", "-")),
            cue, c_mean, c_max, status
        ))
    total = [r for r in results if r and r.get("status") == "ok"]
    if total:
        avg_balls = np.mean([r["ball_count"] for r in total])
        n_cue = sum(1 for r in total if r["cue_detected"])
        print("-" * 72)
        print(f"Avg balls detected: {avg_balls:.1f}  |  Cue found in {n_cue}/{len(total)} images")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None, help="Single image path (default: all target images)")
    parser.add_argument("--all", action="store_true", help="Run on all annotated images")
    args = parser.parse_args()

    from ultralytics import YOLO
    model = YOLO(str(CHECKPOINT))

    if args.image:
        imgs = [Path(args.image)]
    elif args.all:
        exts = {".jpg", ".jpeg", ".png"}
        imgs = sorted(p for p in PICTURE_DIR.iterdir()
                      if p.suffix.lower() in exts and p.stem not in SKIP_SUFFIXES
                      and not p.stem.startswith("aug_") and not p.stem.startswith("tgt_"))
    else:
        # Default: target stems + any image with GT annotation
        stems = set(TARGET_STEMS)
        for jf in GT_DIR.glob("*.json"):
            if not jf.stem.startswith("aug_") and not jf.stem.startswith("tgt_"):
                stems.add(jf.stem)
        imgs = []
        for stem in sorted(stems):
            for ext in [".jpg", ".jpeg", ".png"]:
                p = PICTURE_DIR / (stem + ext)
                if p.exists():
                    imgs.append(p)
                    break

    print(f"Processing {len(imgs)} image(s)...")
    results = []
    for img_path in imgs:
        print(f"  {img_path.name}...", end=" ", flush=True)
        gt = load_gt(img_path.stem)
        r = process_image(model, img_path, gt)
        results.append(r)
        if r:
            balls = r.get("ball_count", "?")
            ce = r.get("corner_error") or {}
            c_mean = f"{ce['mean_px']:.1f}px" if ce else "?"
            print(f"balls={balls}  corner_err={c_mean}")
        else:
            print("FAILED")

    print_report(results)

    summary = {
        "n_images": len(imgs),
        "results": [r for r in results if r],
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nOutputs → {OUT_DIR}")
    print(f"Summary → {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
