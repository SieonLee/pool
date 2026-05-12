"""
Train YOLO ball detector on pose-warped 900×450 images.

Input:  datasets/pose-warped-balls/   (warped images + YOLO labels)
Output: models/checkpoints/ball_yolo_candidate_<timestamp>.pt
        (never overwrites ball_yolo_active.pt — run accept_candidate.py to promote)

Classes:
  0: cue_ball
  1: object_ball

Run:
  python scripts/train_ball_yolo.py --augment
  python scripts/train_ball_yolo.py --augment --epochs 200 --seed 42
  python scripts/train_ball_yolo.py --augment --val-stems pool_real_022_warped pool_real_016_warped
  python scripts/train_ball_yolo.py --augment --exclude-stems pool_real_007_warped pool_real_028_warped pool_real_031_warped

After training, promote with:
  python scripts/accept_candidate.py --candidate models/checkpoints/ball_yolo_candidate_<ts>.pt
"""
import argparse
import random
import shutil
import numpy as np
import cv2
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).parent.parent
DATASET = BASE / "datasets" / "pose-warped-balls"
IMG_TRAIN = DATASET / "images" / "train"
LBL_TRAIN = DATASET / "labels" / "train"
IMG_VAL = DATASET / "images" / "val"
LBL_VAL = DATASET / "labels" / "val"
RUNS_DIR = BASE / "models" / "runs"
CKPT_DIR = BASE / "models" / "checkpoints"
YAML_PATH = DATASET / "dataset.yaml"

WARP_W, WARP_H = 900, 450
AUG_PREFIX = "aug_"

DEFAULT_VAL_STEMS = ["pool_real_022_warped", "pool_real_016_warped"]


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def check_labels(exclude=None):
    """Return list of labeled stems in train folder, skipping excluded and unlabeled."""
    exclude = set(exclude or [])
    labeled = []
    for img in sorted(IMG_TRAIN.glob("*.jpg")):
        if img.stem.startswith(AUG_PREFIX):
            continue
        stem = img.stem
        if stem in exclude:
            print(f"  EXCLUDED {img.name}")
            continue
        lbl = LBL_TRAIN / (stem + ".txt")
        if lbl.exists() and lbl.stat().st_size > 0:
            labeled.append(stem)
        else:
            print(f"  WARNING: {img.name} has no labels — skipped")
    return labeled


def augment_images(stems, n_per=8, seed=42):
    """Generate mild augmentations of labeled warped images."""
    set_seeds(seed)

    removed = sum(1 for p in IMG_TRAIN.glob(f"{AUG_PREFIX}*") if p.unlink() or True)
    removed += sum(1 for p in LBL_TRAIN.glob(f"{AUG_PREFIX}*") if p.unlink() or True)
    if removed:
        print(f"  Removed {removed} previous aug files")

    total = 0
    for stem in stems:
        img_path = IMG_TRAIN / f"{stem}.jpg"
        lbl_path = LBL_TRAIN / f"{stem}.txt"
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        lines = lbl_path.read_text().strip().splitlines()
        boxes = [list(map(float, l.split())) for l in lines if l.strip()]

        for i in range(n_per):
            aug_img = img.copy().astype(np.float32)

            alpha = random.uniform(0.75, 1.30)
            beta = random.uniform(-20, 20)
            aug_img = np.clip(aug_img * alpha + beta, 0, 255).astype(np.uint8)

            hsv = cv2.cvtColor(aug_img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.75, 1.25), 0, 255)
            hsv[:, :, 0] = np.clip(hsv[:, :, 0] + random.uniform(-8, 8), 0, 179)
            aug_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

            if random.random() < 0.4:
                k = random.choice([3, 5])
                aug_img = cv2.GaussianBlur(aug_img, (k, k), 0)

            angle = random.uniform(-5, 5)
            h, w = aug_img.shape[:2]
            M_rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            aug_img = cv2.warpAffine(aug_img, M_rot, (w, h),
                                     borderMode=cv2.BORDER_REFLECT_101)

            aug_boxes = []
            cos_a = np.cos(np.radians(angle))
            sin_a = np.sin(np.radians(angle))
            for b in boxes:
                cls = int(b[0])
                cx_px = b[1] * w
                cy_px = b[2] * h
                bw_px = b[3] * w
                bh_px = b[4] * h
                dx = cx_px - w / 2
                dy = cy_px - h / 2
                ncx = cos_a * dx - sin_a * dy + w / 2
                ncy = sin_a * dx + cos_a * dy + h / 2
                x1 = ncx - bw_px / 2
                y1 = ncy - bh_px / 2
                x2 = ncx + bw_px / 2
                y2 = ncy + bh_px / 2
                if x2 < 0 or y2 < 0 or x1 > w or y1 > h:
                    continue
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                ncx2 = (x1 + x2) / 2 / w
                ncy2 = (y1 + y2) / 2 / h
                nbw = (x2 - x1) / w
                nbh = (y2 - y1) / h
                if nbw > 0.005 and nbh > 0.005:
                    aug_boxes.append([cls, ncx2, ncy2, nbw, nbh])

            if not aug_boxes:
                continue

            name = f"{AUG_PREFIX}{stem}_{i:03d}"
            cv2.imwrite(str(IMG_TRAIN / f"{name}.jpg"), aug_img,
                        [cv2.IMWRITE_JPEG_QUALITY, 93])
            lbl_out = LBL_TRAIN / f"{name}.txt"
            lbl_out.write_text(
                "\n".join(f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}"
                           for b in aug_boxes)
            )
            total += 1

    print(f"  Generated {total} augmented images")
    return total


def build_val_split(stems, val_stems=None, seed=42):
    """Move specified stems (or a random sample) to the val split.

    If val_stems is given, those exact stems are used — fully deterministic.
    Otherwise falls back to random 20% sample with the given seed.
    """
    set_seeds(seed)
    IMG_VAL.mkdir(parents=True, exist_ok=True)
    LBL_VAL.mkdir(parents=True, exist_ok=True)

    # Move ALL current val files back to train before rebuilding the split.
    for img in list(IMG_VAL.glob("*.jpg")):
        shutil.move(str(img), IMG_TRAIN / img.name)
        lbl = LBL_VAL / (img.stem + ".txt")
        if lbl.exists():
            shutil.move(str(lbl), LBL_TRAIN / lbl.name)

    if val_stems is not None:
        # Deterministic: use exactly the requested stems (skip any not in train)
        chosen = [s for s in val_stems if (IMG_TRAIN / f"{s}.jpg").exists()]
        missing = set(val_stems) - set(chosen)
        if missing:
            print(f"  WARNING: val stems not found in train, skipping: {missing}")
    else:
        n_val = max(1, int(len(stems) * 0.2))
        chosen = random.sample(stems, n_val)

    for stem in chosen:
        for src_dir, dst_dir in [(IMG_TRAIN, IMG_VAL), (LBL_TRAIN, LBL_VAL)]:
            for ext in [".jpg", ".png", ".txt"]:
                src = src_dir / (stem + ext)
                if src.exists():
                    shutil.move(str(src), dst_dir / src.name)

    print(f"  Val split: {chosen}  ({len(chosen)}/{len(stems)})")
    return chosen


def write_yaml():
    yaml_content = f"""# YOLO detection dataset — billiards balls on pose-warped table
path: {DATASET}
train: images/train
val: images/val

nc: 2
names: ['cue_ball', 'object_ball']
"""
    YAML_PATH.write_text(yaml_content)
    print(f"  dataset.yaml → {YAML_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=896)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--augment", action="store_true",
                        help="Generate augmented training variants")
    parser.add_argument("--device", default="cpu",
                        help="Training device (cpu recommended; mps crashes intermittently)")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--val-stems", nargs="+", default=DEFAULT_VAL_STEMS,
                        metavar="STEM",
                        help="Exact stems to use as val split (default: pool_real_022_warped "
                             "pool_real_016_warped). Pass --val-stems random to use random split.")
    parser.add_argument("--exclude-stems", nargs="+", default=[],
                        metavar="STEM",
                        help="Image stems to exclude from training")
    args = parser.parse_args()

    set_seeds(args.seed)

    if not IMG_TRAIN.exists():
        print("No warped images found. Run export_warped_balls.py first.")
        return

    labeled = check_labels(exclude=args.exclude_stems)
    if not labeled:
        print("No labeled images found. Run annotate_balls.py first.")
        return

    print(f"\nLabeled images: {len(labeled)}")
    for s in labeled:
        lbl = LBL_TRAIN / f"{s}.txt"
        n = len([l for l in lbl.read_text().strip().splitlines() if l.strip()])
        print(f"  {s}: {n} boxes")

    if args.augment:
        print("\nGenerating augmentations...")
        augment_images(labeled, seed=args.seed)

    print("\nBuilding val split...")
    val_stems_arg = None if (args.val_stems == ["random"]) else args.val_stems
    val_stems = build_val_split(labeled, val_stems=val_stems_arg, seed=args.seed)

    print("\nWriting dataset.yaml...")
    write_yaml()

    n_train = len(list(IMG_TRAIN.glob("*.jpg")))
    n_val = len(list(IMG_VAL.glob("*.jpg")))
    print(f"  Train: {n_train}  Val: {n_val}")

    if n_train == 0:
        print("ERROR: no training images after split. Aborting.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"train_{timestamp}"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining YOLO ball detector...")
    print(f"  seed={args.seed}  epochs={args.epochs}  imgsz={args.imgsz}  "
          f"batch={args.batch}  device={args.device}")
    print(f"  run: {run_name}")

    from ultralytics import YOLO
    model = YOLO("yolo11n.pt")
    model.train(
        data=str(YAML_PATH),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        seed=args.seed,
        project=str(RUNS_DIR),
        name=run_name,
        exist_ok=False,
        box=7.5,
        cls=1.5,
        dfl=1.5,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        fliplr=0.5,
        flipud=0.0,
        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,
        verbose=True,
    )

    best = RUNS_DIR / run_name / "weights" / "best.pt"
    candidate = CKPT_DIR / f"ball_yolo_candidate_{timestamp}.pt"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    if best.exists():
        shutil.copy(best, candidate)
        print(f"\nCandidate saved → {candidate}")
        print(f"Run folder      → {RUNS_DIR / run_name}")
        print(f"\nTo evaluate and promote:")
        print(f"  python scripts/accept_candidate.py --candidate {candidate}")
    else:
        print(f"\nWARNING: best.pt not found at {best}")


if __name__ == "__main__":
    main()
