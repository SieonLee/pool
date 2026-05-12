"""
Convert GT JSON annotations → YOLO pose format dataset.

YOLO pose label format (one line per instance, all values normalized 0-1):
  <class> <cx> <cy> <bw> <bh> <kp0x> <kp0y> <v0> <kp1x> <kp1y> <v1> ...

Keypoint order: TL, TR, BR, BL  (class = 0 = "table")
Visibility: 2 = visible, 0 = not labeled

Run:
  python scripts/convert_to_yolo_pose.py [--val-ratio 0.2] [--seed 42]
"""
import json
import random
import shutil
import argparse
from pathlib import Path

BASE = Path(__file__).parent.parent
GT_DIR = BASE / "annotations" / "table_corners_gt"
PICTURE_DIR = BASE / "picture"
DATASET_DIR = BASE / "datasets" / "table-corners-pose"


def load_annotations():
    records = []
    for jf in sorted(GT_DIR.glob("*.json")):
        with open(jf) as f:
            data = json.load(f)
        img_stem = jf.stem
        # Find image file
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = PICTURE_DIR / (img_stem + ext)
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            print(f"  WARNING: image not found for {jf.name}, skipping")
            continue
        records.append((img_path, data))
    return records


def corners_to_yolo(data: dict) -> str:
    w, h = data["width"], data["height"]
    c = data["corners"]
    pts = [c["TL"], c["TR"], c["BR"], c["BL"]]

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    bx_min, bx_max = min(xs), max(xs)
    by_min, by_max = min(ys), max(ys)

    cx = (bx_min + bx_max) / 2 / w
    cy = (by_min + by_max) / 2 / h
    bw = (bx_max - bx_min) / w
    bh = (by_max - by_min) / h

    kps = []
    for px, py in pts:
        kps.append(f"{px/w:.6f} {py/h:.6f} 2")

    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {' '.join(kps)}"


def write_yaml(n_train: int, n_val: int):
    yaml_path = DATASET_DIR / "dataset.yaml"
    content = f"""# YOLO pose dataset — billiards table corners
path: {DATASET_DIR.resolve()}
train: images/train
val: images/val

nc: 1
names: ['table']

# 4 keypoints: TL, TR, BR, BL
kpt_shape: [4, 3]   # N keypoints, (x, y, visibility)

# Stats
# train: {n_train} images
# val:   {n_val} images
"""
    yaml_path.write_text(content)
    print(f"  Wrote {yaml_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = load_annotations()
    if not records:
        print("No annotations found. Run annotate.py first.")
        return

    random.seed(args.seed)
    random.shuffle(records)
    n_val = max(1, round(len(records) * args.val_ratio))
    val_records = records[:n_val]
    train_records = records[n_val:]

    print(f"Total: {len(records)}  train: {len(train_records)}  val: {len(val_records)}")

    for split, recs in [("train", train_records), ("val", val_records)]:
        img_out = DATASET_DIR / "images" / split
        lbl_out = DATASET_DIR / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path, data in recs:
            dst_img = img_out / img_path.name
            shutil.copy2(img_path, dst_img)

            label_line = corners_to_yolo(data)
            lbl_file = lbl_out / (img_path.stem + ".txt")
            lbl_file.write_text(label_line + "\n")

        print(f"  {split}: wrote {len(recs)} images + labels")

    write_yaml(len(train_records), len(val_records))
    print("\nDataset ready:", DATASET_DIR)


if __name__ == "__main__":
    main()
