"""
Offline augmentation: 12 annotated images → ~120 augmented images.
Saves augmented images + corner JSON alongside originals.

Augmentations applied:
  - Horizontal flip  (TL↔TR, BL↔BR)
  - Rotation         ±5, ±10, ±15 deg
  - Brightness/contrast
  - HSV jitter
  - Slight perspective warp
  - Combined: flip + rotate + color

Run:
  python scripts/augment.py [--per-image 9]
"""
import cv2
import json
import random
import argparse
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent.parent
GT_DIR = BASE / "annotations" / "table_corners_gt"
PICTURE_DIR = BASE / "picture"
AUG_IMG_DIR = PICTURE_DIR   # save alongside originals
AUG_GT_DIR = GT_DIR

PREFIX = "aug_"


def load_gt(json_path):
    with open(json_path) as f:
        data = json.load(f)
    c = data["corners"]
    pts = np.array([c["TL"], c["TR"], c["BR"], c["BL"]], dtype=np.float32)
    return data, pts


def save_aug(img, corners, stem, aug_id, orig_data):
    h, w = img.shape[:2]
    name = f"{PREFIX}{stem}_{aug_id:03d}"
    img_ext = ".jpg"
    img_path = AUG_IMG_DIR / (name + img_ext)
    cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])

    gt = {
        "image": name + img_ext,
        "width": w,
        "height": h,
        "corners": {
            "TL": corners[0].tolist(),
            "TR": corners[1].tolist(),
            "BR": corners[2].tolist(),
            "BL": corners[3].tolist(),
        },
        "order": ["TL", "TR", "BR", "BL"],
        "augmented_from": orig_data["image"],
    }
    gt_path = AUG_GT_DIR / (name + ".json")
    with open(gt_path, "w") as f:
        json.dump(gt, f, indent=2)
    return name


def transform_corners(M, corners):
    pts = corners.reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, M)
    return out.reshape(-1, 2)


def flip_h(img, corners):
    h, w = img.shape[:2]
    flipped = cv2.flip(img, 1)
    new_corners = corners.copy()
    new_corners[:, 0] = w - corners[:, 0]
    # TL↔TR (0↔1), BL↔BR (3↔2)
    new_corners = new_corners[[1, 0, 3, 2]]
    return flipped, new_corners


def rotate(img, corners, angle_deg):
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    # Compute bounding box of rotated image
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    rotated = cv2.warpAffine(img, M, (new_w, new_h),
                              borderMode=cv2.BORDER_REFLECT_101)
    # Transform corners
    ones = np.ones((4, 1))
    pts_h = np.hstack([corners, ones])
    new_corners = (M @ pts_h.T).T
    return rotated, new_corners.astype(np.float32)


def color_jitter(img, brightness=0.3, contrast=0.3, saturation=0.3):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    # Hue
    hsv[:, :, 0] += random.uniform(-10, 10)
    # Saturation
    hsv[:, :, 1] *= random.uniform(1 - saturation, 1 + saturation)
    # Value (brightness)
    hsv[:, :, 2] *= random.uniform(1 - brightness, 1 + brightness)
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def slight_perspective(img, corners, jitter_frac=0.03):
    h, w = img.shape[:2]
    src = corners.astype(np.float32)
    noise = (np.random.rand(4, 2) - 0.5) * 2 * jitter_frac * np.array([w, h])
    dst = (src + noise).astype(np.float32)
    # Keep dst inside image
    dst[:, 0] = np.clip(dst[:, 0], 0, w - 1)
    dst[:, 1] = np.clip(dst[:, 1], 0, h - 1)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (w, h),
                                  borderMode=cv2.BORDER_REFLECT_101)
    new_corners = transform_corners(M, corners)
    return warped, new_corners


def corners_in_bounds(corners, h, w, margin=10):
    return (np.all(corners[:, 0] >= margin) and
            np.all(corners[:, 0] <= w - margin) and
            np.all(corners[:, 1] >= margin) and
            np.all(corners[:, 1] <= h - margin))


def augment_one(img, corners, orig_data, stem, start_id):
    h, w = img.shape[:2]
    aug_id = start_id
    results = []

    def maybe_save(aug_img, aug_corners):
        nonlocal aug_id
        ah, aw = aug_img.shape[:2]
        if not corners_in_bounds(aug_corners, ah, aw):
            return
        name = save_aug(aug_img, aug_corners, stem, aug_id, orig_data)
        results.append(name)
        aug_id += 1

    # 1. Horizontal flip
    fi, fc = flip_h(img, corners)
    maybe_save(fi, fc)

    # 2. Rotations
    for angle in [-15, -10, -5, 5, 10, 15]:
        ri, rc = rotate(img, corners, angle)
        rc_color = color_jitter(ri)
        maybe_save(rc_color, rc)

    # 3. Color jitter only (3 variants)
    for _ in range(3):
        ci = color_jitter(img)
        maybe_save(ci, corners.copy())

    # 4. Flip + rotate combos
    for angle in [-10, 10]:
        fi2, fc2 = flip_h(img, corners)
        ri2, rc2 = rotate(fi2, fc2, angle)
        ri2 = color_jitter(ri2)
        maybe_save(ri2, rc2)

    # 5. Perspective jitter (3 variants)
    for _ in range(3):
        pi, pc = slight_perspective(img, corners, jitter_frac=0.04)
        pi = color_jitter(pi, brightness=0.2)
        maybe_save(pi, pc)

    return results, aug_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Remove previous augmented files
    removed = 0
    for jf in list(GT_DIR.glob(f"{PREFIX}*.json")):
        jf.unlink()
        removed += 1
    for img in list(PICTURE_DIR.glob(f"{PREFIX}*")):
        img.unlink()
        removed += 1
    if removed:
        print(f"Removed {removed} previous augmented files")

    gt_files = sorted(GT_DIR.glob("*.json"))
    if not gt_files:
        print("No annotations found. Run annotate.py first.")
        return

    total_aug = 0
    aug_id = 0
    for jf in gt_files:
        orig_data, corners = load_gt(jf)
        stem = jf.stem
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            c = PICTURE_DIR / (stem + ext)
            if c.exists():
                img_path = c
                break
        if img_path is None:
            print(f"  image not found: {stem}, skipping")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        names, aug_id = augment_one(img, corners, orig_data, stem, aug_id)
        print(f"  {stem}: +{len(names)} augmented")
        total_aug += len(names)

    print(f"\nTotal augmented: {total_aug}  (+ {len(gt_files)} originals = {total_aug + len(gt_files)} total)")
    print("Now run: python scripts/convert_to_yolo_pose.py")


if __name__ == "__main__":
    main()
