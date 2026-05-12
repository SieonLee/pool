"""
Run inference on one image or a folder, visualize predicted corners + warp.

Run:
  python scripts/infer_debug.py --source picture/pool_real_001.png
  python scripts/infer_debug.py --source picture/          # whole folder
  python scripts/infer_debug.py --source picture/ --weights models/table_corners_pose/weights/best.pt

Output saved to review/table_corner_mvp/infer/
"""
import argparse
import cv2
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent.parent
DEFAULT_WEIGHTS = BASE / "models" / "table_corners_pose" / "weights" / "best.pt"
OUT_DIR = BASE / "review" / "table_corner_mvp" / "infer"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WARP_W, WARP_H = 900, 450
CORNER_COLORS = [(0, 255, 0), (0, 200, 255), (0, 0, 255), (255, 0, 255)]
CORNER_NAMES = ["TL", "TR", "BR", "BL"]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def warp_to_rect(img, corners_xy: np.ndarray) -> np.ndarray:
    """corners_xy: shape (4,2) order TL TR BR BL"""
    dst = np.array([[0, 0], [WARP_W, 0], [WARP_W, WARP_H], [0, WARP_H]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners_xy.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (WARP_W, WARP_H))


def draw_corners(img, corners, conf=None):
    canvas = img.copy()
    for i, (x, y) in enumerate(corners):
        x, y = int(round(x)), int(round(y))
        cv2.circle(canvas, (x, y), 8, CORNER_COLORS[i], -1)
        label = CORNER_NAMES[i]
        if conf is not None:
            label += f" {conf[i]:.2f}"
        cv2.putText(canvas, label, (x + 10, y - 10), FONT, 0.7, CORNER_COLORS[i], 2)
    # Draw polygon
    pts = np.array([(int(round(x)), int(round(y))) for x, y in corners], dtype=np.int32)
    cv2.polylines(canvas, [pts], isClosed=True, color=(255, 255, 255), thickness=2)
    return canvas


def infer_image(model, img_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  ERROR: cannot read {img_path}")
        return

    results = model(img, verbose=False)

    if not results or results[0].keypoints is None:
        print(f"  {img_path.name}: no detection")
        # Save blank
        blank = img.copy()
        cv2.putText(blank, "NO DETECTION", (30, 60), FONT, 1.5, (0, 0, 255), 3)
        cv2.imwrite(str(OUT_DIR / f"{img_path.stem}_no_det.jpg"), blank)
        return

    kps_data = results[0].keypoints
    # Pick highest-confidence detection
    if kps_data.conf is not None:
        # conf shape: (N_instances, N_kpts)
        instance_conf = kps_data.conf.mean(dim=1)
        best = int(instance_conf.argmax())
    else:
        best = 0

    xy = kps_data.xy[best].cpu().numpy()   # (4, 2)
    conf = kps_data.conf[best].cpu().numpy() if kps_data.conf is not None else None

    print(f"  {img_path.name}: corners {xy.tolist()}")

    # Visualize
    vis = draw_corners(img, xy, conf)
    warped = warp_to_rect(img, xy)

    cv2.imwrite(str(OUT_DIR / f"{img_path.stem}_pred.jpg"), vis)
    cv2.imwrite(str(OUT_DIR / f"{img_path.stem}_warp.jpg"), warped)
    print(f"    → {img_path.stem}_pred.jpg  {img_path.stem}_warp.jpg")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Image file or folder")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"ERROR: weights not found at {weights_path}")
        print("Run: python scripts/train.py  first")
        return

    from ultralytics import YOLO
    model = YOLO(str(weights_path))

    source = Path(args.source)
    if source.is_dir():
        imgs = sorted(p for p in source.iterdir()
                      if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    else:
        imgs = [source]

    print(f"Running inference on {len(imgs)} image(s)...")
    for p in imgs:
        infer_image(model, p)

    print(f"\nOutputs → {OUT_DIR}")


if __name__ == "__main__":
    main()
