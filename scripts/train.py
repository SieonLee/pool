"""
Train YOLO pose model on billiards table corners.

Requires: pip install ultralytics

Run:
  python scripts/train.py [--epochs 100] [--model yolo11n-pose.pt] [--imgsz 640]

Output: models/table_corners_pose/weights/best.pt
"""
import argparse
from pathlib import Path

BASE = Path(__file__).parent.parent
DATASET_YAML = BASE / "datasets" / "table-corners-pose" / "dataset.yaml"
MODEL_DIR = BASE / "models"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11n-pose.pt",
                        help="Pretrained YOLO pose checkpoint (downloaded automatically)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (0 = disabled)")
    parser.add_argument("--device", default="",
                        help="'' = auto, 'cpu', '0', 'mps'")
    args = parser.parse_args()

    if not DATASET_YAML.exists():
        print(f"ERROR: dataset.yaml not found at {DATASET_YAML}")
        print("Run: python scripts/convert_to_yolo_pose.py")
        return

    from ultralytics import YOLO

    model = YOLO(args.model)

    results = model.train(
        data=str(DATASET_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device if args.device else None,
        project=str(MODEL_DIR),
        name="table_corners_pose",
        exist_ok=True,
        # Pose-specific
        pose=12.0,          # pose loss weight
        # Augmentation — keep moderate, table orientation matters
        flipud=0.0,         # don't flip vertically (gravity reference)
        fliplr=0.5,
        degrees=15,
        scale=0.3,
        translate=0.1,
        hsv_h=0.02,
        hsv_s=0.5,
        hsv_v=0.3,
        # Verbose
        verbose=True,
        plots=True,
    )

    best = MODEL_DIR / "table_corners_pose" / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights: {best}")


if __name__ == "__main__":
    main()
