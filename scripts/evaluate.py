"""
Evaluate predicted corners vs GT annotations.

Metrics per image:
  - Per-corner Euclidean error (pixels)
  - Mean corner error
  - Max corner error
  - PCK@10  (% corners within 10px of GT)
  - PCK@20  (% corners within 20px of GT)

Visual outputs (review/table_corner_mvp/eval/):
  - <stem>_gt.jpg          original + GT corners
  - <stem>_pred.jpg        original + predicted corners
  - <stem>_warp_gt.jpg     warped using GT corners
  - <stem>_warp_pred.jpg   warped using predicted corners
  - <stem>_diff.jpg        side-by-side warp comparison

Summary JSON + printed table saved to review/table_corner_mvp/eval_report.json

Run:
  python scripts/evaluate.py [--weights models/table_corners_pose/weights/best.pt]
"""
import argparse
import json
import cv2
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent.parent
GT_DIR = BASE / "annotations" / "table_corners_gt"
PICTURE_DIR = BASE / "picture"
EVAL_DIR = BASE / "review" / "table_corner_mvp" / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_WEIGHTS = BASE / "models" / "table_corners_pose" / "weights" / "best.pt"

WARP_W, WARP_H = 900, 450
CORNER_COLORS = [(0, 255, 0), (0, 200, 255), (0, 0, 255), (255, 0, 255)]
CORNER_NAMES = ["TL", "TR", "BR", "BL"]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_gt(json_path: Path) -> np.ndarray:
    with open(json_path) as f:
        data = json.load(f)
    c = data["corners"]
    return np.array([c["TL"], c["TR"], c["BR"], c["BL"]], dtype=np.float32)


def warp(img, corners):
    dst = np.array([[0, 0], [WARP_W, 0], [WARP_W, WARP_H], [0, WARP_H]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (WARP_W, WARP_H))


def draw_corners(img, corners, label_prefix=""):
    canvas = img.copy()
    pts = []
    for i, (x, y) in enumerate(corners):
        x, y = int(round(x)), int(round(y))
        pts.append((x, y))
        cv2.circle(canvas, (x, y), 8, CORNER_COLORS[i], -1)
        cv2.putText(canvas, label_prefix + CORNER_NAMES[i],
                    (x + 10, y - 10), FONT, 0.65, CORNER_COLORS[i], 2)
    cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], isClosed=True,
                  color=(255, 255, 255), thickness=2)
    return canvas


def get_prediction(model, img: np.ndarray) -> np.ndarray | None:
    results = model(img, verbose=False)
    if not results or results[0].keypoints is None:
        return None
    kps_data = results[0].keypoints
    if len(kps_data) == 0:
        return None
    if kps_data.conf is not None:
        best = int(kps_data.conf.mean(dim=1).argmax())
    else:
        best = 0
    xy = kps_data.xy[best].cpu().numpy()  # (4,2)
    return xy


def side_by_side(img_a, img_b, label_a="GT", label_b="PRED"):
    h = max(img_a.shape[0], img_b.shape[0])
    w = img_a.shape[1] + img_b.shape[1] + 4
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:img_a.shape[0], :img_a.shape[1]] = img_a
    canvas[:img_b.shape[0], img_a.shape[1]+4:img_a.shape[1]+4+img_b.shape[1]] = img_b
    cv2.putText(canvas, label_a, (8, 24), FONT, 0.8, (200, 200, 200), 2)
    cv2.putText(canvas, label_b, (img_a.shape[1] + 12, 24), FONT, 0.8, (200, 200, 200), 2)
    return canvas


def pck(errors_px, threshold):
    return float(np.mean(errors_px < threshold)) * 100


def print_table(rows, headers):
    col_w = [max(len(h), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in col_w))
    for r in rows:
        print(fmt.format(*r))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"ERROR: weights not found at {weights_path}")
        print("Run: python scripts/train.py")
        return

    gt_files = sorted(GT_DIR.glob("*.json"))
    if not gt_files:
        print("No GT annotations found. Run annotate.py first.")
        return

    from ultralytics import YOLO
    model = YOLO(str(weights_path))

    all_errors = []   # flat list of per-corner errors across all images
    per_image = []
    table_rows = []
    missing = []

    for jf in gt_files:
        stem = jf.stem
        # Find image
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            c = PICTURE_DIR / (stem + ext)
            if c.exists():
                img_path = c
                break
        if img_path is None:
            missing.append(stem)
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            missing.append(stem)
            continue

        gt_corners = load_gt(jf)
        pred_corners = get_prediction(model, img)

        # GT overlay
        gt_vis = draw_corners(img, gt_corners, "GT:")
        cv2.imwrite(str(EVAL_DIR / f"{stem}_gt.jpg"), gt_vis)

        # Warped GT
        warp_gt = warp(img, gt_corners)
        cv2.imwrite(str(EVAL_DIR / f"{stem}_warp_gt.jpg"), warp_gt)

        if pred_corners is None:
            print(f"  {stem}: NO DETECTION")
            table_rows.append((stem, "NO DET", "-", "-", "-", "-"))
            per_image.append({"image": stem, "status": "no_detection"})
            # Save blank pred
            blank = img.copy()
            cv2.putText(blank, "NO DETECTION", (30, 60), FONT, 1.5, (0, 0, 255), 3)
            cv2.imwrite(str(EVAL_DIR / f"{stem}_pred.jpg"), blank)
            continue

        # Pred overlay
        pred_vis = draw_corners(img, pred_corners, "P:")
        cv2.imwrite(str(EVAL_DIR / f"{stem}_pred.jpg"), pred_vis)

        # Warped pred
        warp_pred = warp(img, pred_corners)
        cv2.imwrite(str(EVAL_DIR / f"{stem}_warp_pred.jpg"), warp_pred)

        # Side-by-side warp diff
        diff = side_by_side(warp_gt, warp_pred, "GT warp", "Pred warp")
        cv2.imwrite(str(EVAL_DIR / f"{stem}_diff.jpg"), diff)

        # Metrics
        errors = np.linalg.norm(pred_corners - gt_corners, axis=1)   # (4,)
        mean_err = float(errors.mean())
        max_err = float(errors.max())
        all_errors.extend(errors.tolist())

        per_image.append({
            "image": stem,
            "status": "ok",
            "corner_errors_px": errors.tolist(),
            "mean_error_px": round(mean_err, 2),
            "max_error_px": round(max_err, 2),
            "pck10": round(pck(errors, 10), 1),
            "pck20": round(pck(errors, 20), 1),
        })
        table_rows.append((
            stem,
            f"{mean_err:.1f}",
            f"{max_err:.1f}",
            f"{pck(errors, 10):.0f}%",
            f"{pck(errors, 20):.0f}%",
            "OK",
        ))
        print(f"  {stem}: mean={mean_err:.1f}px  max={max_err:.1f}px")

    # Overall metrics
    print("\n" + "=" * 60)
    print("PER-IMAGE RESULTS")
    print("=" * 60)
    print_table(table_rows, ["image", "mean_err", "max_err", "PCK@10", "PCK@20", "status"])

    if all_errors:
        ae = np.array(all_errors)
        summary = {
            "n_images": len(gt_files),
            "n_evaluated": sum(1 for r in per_image if r.get("status") == "ok"),
            "n_no_detection": sum(1 for r in per_image if r.get("status") == "no_detection"),
            "overall_mean_error_px": round(float(ae.mean()), 2),
            "overall_max_error_px": round(float(ae.max()), 2),
            "pck10": round(pck(ae, 10), 1),
            "pck20": round(pck(ae, 20), 1),
            "per_image": per_image,
        }
        print(f"\nOVERALL ({summary['n_evaluated']}/{summary['n_images']} detected)")
        print(f"  Mean corner error : {summary['overall_mean_error_px']:.2f} px")
        print(f"  Max corner error  : {summary['overall_max_error_px']:.2f} px")
        print(f"  PCK@10            : {summary['pck10']:.1f}%")
        print(f"  PCK@20            : {summary['pck20']:.1f}%")
    else:
        summary = {"n_images": len(gt_files), "n_evaluated": 0, "per_image": per_image}
        print("No valid predictions to score.")

    report_path = BASE / "review" / "table_corner_mvp" / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nReport → {report_path}")
    print(f"Overlays → {EVAL_DIR}")

    if missing:
        print(f"\nImages not found for: {missing}")


if __name__ == "__main__":
    main()
