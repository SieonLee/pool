"""
Targeted augmentation for weak-pattern images.

For pool_real_002: extreme perspective angle + high contrast.
Strategy:
  - Heavy perspective jitter that mimics the extreme top-skew pattern
  - Brightness/contrast pushes matching pool_real_002 profile
  - Scale crops focusing on each corner region
  - Simulate similar extreme-angle views

Creates ~40 extra variants of pool_real_002.
Run:
  python scripts/augment_targeted.py
"""
import cv2
import json
import numpy as np
import random
from pathlib import Path

BASE = Path(__file__).parent.parent
GT_DIR = BASE / "annotations" / "table_corners_gt"
PICTURE_DIR = BASE / "picture"
PREFIX = "tgt_"
random.seed(7)
np.random.seed(7)


def load_gt(stem):
    jf = GT_DIR / (stem + ".json")
    with open(jf) as f:
        data = json.load(f)
    c = data["corners"]
    pts = np.array([c["TL"], c["TR"], c["BR"], c["BL"]], dtype=np.float32)
    return data, pts


def save_aug(img, corners, stem, aug_id, orig_data):
    h, w = img.shape[:2]
    corners = corners.copy()
    corners[:, 0] = np.clip(corners[:, 0], 0, w - 1)
    corners[:, 1] = np.clip(corners[:, 1], 0, h - 1)

    name = f"{PREFIX}{stem}_{aug_id:03d}"
    img_path = PICTURE_DIR / (name + ".jpg")
    cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 93])

    gt = {
        "image": name + ".jpg",
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
    gt_path = GT_DIR / (name + ".json")
    with open(gt_path, "w") as f:
        json.dump(gt, f, indent=2)
    return name


def transform_corners_affine(M, corners):
    ones = np.ones((4, 1))
    pts_h = np.hstack([corners, ones])
    return (M @ pts_h.T).T.astype(np.float32)


def transform_corners_perspective(M, corners):
    pts = corners.reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, M)
    return out.reshape(-1, 2).astype(np.float32)


def corners_valid(corners, h, w, margin=5):
    return (np.all(corners[:, 0] >= margin) and
            np.all(corners[:, 0] <= w - margin) and
            np.all(corners[:, 1] >= margin) and
            np.all(corners[:, 1] <= h - margin))


def color_jitter(img, b=0.4, c_range=0.35, s=0.4):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = np.clip(hsv[:, :, 0] + random.uniform(-12, 12), 0, 179)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(1 - s, 1 + s), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(1 - b, 1 + b), 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    # Contrast
    alpha = random.uniform(1 - c_range, 1 + c_range)
    beta = random.uniform(-20, 20)
    return np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def perspective_jitter(img, corners, jitter_frac=0.06):
    h, w = img.shape[:2]
    src = corners.astype(np.float32)
    # Asymmetric jitter — push TL/TR more (the hard corners)
    noise = np.random.randn(4, 2) * jitter_frac * np.array([w, h])
    dst = np.clip(src + noise, 0, [w - 1, h - 1]).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    new_corners = transform_corners_perspective(M, corners)
    return warped, new_corners


def rotate_aug(img, corners, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
    nw = int(h * sin_a + w * cos_a)
    nh = int(h * cos_a + w * sin_a)
    M[0, 2] += (nw - w) / 2
    M[1, 2] += (nh - h) / 2
    rotated = cv2.warpAffine(img, M, (nw, nh), borderMode=cv2.BORDER_REFLECT_101)
    new_corners = transform_corners_affine(M, corners)
    return rotated, new_corners


def scale_crop(img, corners, scale):
    """Zoom in/out around table centroid."""
    h, w = img.shape[:2]
    cx, cy = corners.mean(axis=0)
    M = np.float32([
        [scale, 0, cx * (1 - scale)],
        [0, scale, cy * (1 - scale)],
    ])
    scaled = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    new_corners = transform_corners_affine(M, corners)
    return scaled, new_corners


def simulate_high_contrast(img):
    """Push toward pool_real_002 style: bright, high-contrast."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] = np.clip(lab[:, :, 0] * random.uniform(1.1, 1.4) + random.uniform(5, 20), 0, 255)
    lab[:, :, 1] = np.clip(lab[:, :, 1] * random.uniform(0.8, 1.2), 0, 255)
    lab[:, :, 2] = np.clip(lab[:, :, 2] * random.uniform(0.8, 1.2), 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def generate_targeted(stem, n_target=40):
    # Remove previous targeted augments for this stem
    removed = 0
    for jf in list(GT_DIR.glob(f"{PREFIX}{stem}_*.json")):
        jf.unlink()
        removed += 1
    for img in list(PICTURE_DIR.glob(f"{PREFIX}{stem}_*")):
        img.unlink()
        removed += 1
    if removed:
        print(f"  Removed {removed} previous targeted files for {stem}")

    orig_data, corners = load_gt(stem)
    img_path = None
    for ext in [".jpg", ".jpeg", ".png"]:
        p = PICTURE_DIR / (stem + ext)
        if p.exists():
            img_path = p
            break
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    results = []
    aug_id = 0

    def try_save(aug_img, aug_corners):
        nonlocal aug_id
        ah, aw = aug_img.shape[:2]
        if not corners_valid(aug_corners, ah, aw):
            return
        save_aug(aug_img, aug_corners, stem, aug_id, orig_data)
        results.append(aug_id)
        aug_id += 1

    # Strategy: color/brightness ONLY — no aggressive perspective jitter
    # Pool_real_002 issue is brightness pattern, not geometry ambiguity

    # 1. Color jitter only — 8 variants (safe, no geometry change)
    for _ in range(8):
        ci = color_jitter(img, b=0.35, c_range=0.3, s=0.35)
        try_save(ci, corners.copy())

    # 2. High-contrast simulation — 6 variants
    for _ in range(6):
        hi = simulate_high_contrast(img)
        hi = color_jitter(hi, b=0.2, c_range=0.15)
        try_save(hi, corners.copy())

    # 3. Dark/dim simulation — 4 variants
    for _ in range(4):
        di = img.copy().astype(np.float32)
        factor = random.uniform(0.55, 0.80)
        di = np.clip(di * factor, 0, 255).astype(np.uint8)
        di = color_jitter(di, b=0.1, c_range=0.1)
        try_save(di, corners.copy())

    # 4. Small rotation + color — ±3, ±5, ±7 deg (6 variants, very gentle)
    for angle in [-7, -5, -3, 3, 5, 7]:
        ri, rc = rotate_aug(img, corners, angle)
        ri = color_jitter(ri, b=0.25)
        try_save(ri, rc)

    # 5. Scale crops only — no perspective (6 variants)
    for scale in [0.88, 0.92, 0.96, 1.04, 1.08, 1.12]:
        si, sc = scale_crop(img, corners, scale)
        si = color_jitter(si, b=0.2)
        try_save(si, sc)

    # 6. Very mild perspective jitter (max 0.02 frac) + color (6 variants)
    for _ in range(6):
        pi, pc = perspective_jitter(img, corners, jitter_frac=random.uniform(0.01, 0.02))
        pi = color_jitter(pi, b=0.25)
        try_save(pi, pc)

    print(f"  {stem}: generated {len(results)} targeted augments")
    return results


def main():
    # Target: pool_real_002 only (the weak pattern)
    stems = ["pool_real_002"]
    total = 0
    for stem in stems:
        r = generate_targeted(stem, n_target=40)
        total += len(r)
    print(f"\nTotal new targeted augments: {total}")
    print("Now run: python scripts/convert_to_yolo_pose.py && python scripts/train.py")


if __name__ == "__main__":
    main()
