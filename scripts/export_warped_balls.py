"""
Export pose-warped 900×450 images for YOLO ball labeling.

Uses table_corners_mvp_v1.pt to warp each source image and saves results to:
  datasets/pose-warped-balls/images/train/   ← warped images
  datasets/pose-warped-balls/labels/train/   ← empty .txt stubs (fill by annotate_balls.py)

By default exports ALL pool_real_* + original + images (skips contact sheets,
thumbnails, and images where pose detection fails).

Run:
  python scripts/export_warped_balls.py           # all available images
  python scripts/export_warped_balls.py --skip-existing   # skip already-exported
"""
import argparse
import cv2
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO

BASE = Path(__file__).parent.parent
CHECKPOINT = BASE / "models" / "checkpoints" / "table_corners_mvp_v1.pt"
PICTURE_DIR = BASE / "picture"
OUT_BASE = BASE / "datasets" / "pose-warped-balls"

WARP_W, WARP_H = 900, 450

# Skip these (contact sheets, thumbnails, known-bad)
SKIP_STEMS = {"new_uploads_contact_sheet", "search_contact_sheet", "thumb",
              "pexels-photo-10627132"}

# Explicitly excluded from training (low-res / overdetect)
LOW_RES_STEMS = {"pool_real_010", "pool_real_027"}

# Source diagonal threshold — used only for flagging metadata, NOT for routing
LOW_RES_DIAG = 600


def find_image(stem):
    for ext in [".jpg", ".jpeg", ".png"]:
        p = PICTURE_DIR / (stem + ext)
        if p.exists():
            return p
    return None


def get_corners(model, img):
    results = model(img, verbose=False)
    if not results or results[0].keypoints is None:
        return None
    kps = results[0].keypoints
    if len(kps) == 0:
        return None
    if kps.conf is not None:
        best = int(kps.conf.mean(dim=1).argmax())
    else:
        best = 0
    return kps.xy[best].cpu().numpy()


def warp(img, corners):
    dst = np.float32([[0, 0], [WARP_W, 0], [WARP_W, WARP_H], [0, WARP_H]])
    M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (WARP_W, WARP_H))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip stems that already have a warped image")
    args = parser.parse_args()

    for sub in ("images/train", "images/lowres", "labels/train"):
        (OUT_BASE / sub).mkdir(parents=True, exist_ok=True)

    # Collect all candidate images (exclude aug_, tgt_, contact sheets)
    exts = {".jpg", ".jpeg", ".png"}
    all_stems = []
    for p in sorted(PICTURE_DIR.iterdir()):
        if p.suffix.lower() not in exts:
            continue
        if p.stem.startswith("aug_") or p.stem.startswith("tgt_"):
            continue
        if p.stem in SKIP_STEMS:
            continue
        all_stems.append(p.stem)

    model = YOLO(str(CHECKPOINT))
    manifest = []

    # Load existing manifest to preserve label_status
    manifest_path = OUT_BASE / "manifest.json"
    existing = {}
    if manifest_path.exists():
        for entry in json.loads(manifest_path.read_text()):
            existing[entry["stem"]] = entry

    for stem in all_stems:
        src = find_image(stem)
        if src is None:
            continue

        is_low_res = stem in LOW_RES_STEMS
        subfolder = "lowres" if is_low_res else "train"
        out_img = OUT_BASE / "images" / subfolder / f"{stem}_warped.jpg"

        if args.skip_existing and out_img.exists():
            # Preserve existing entry
            if stem in existing:
                manifest.append(existing[stem])
            print(f"  {stem}: already exported — skipped")
            continue

        img = cv2.imread(str(src))
        if img is None:
            continue
        h, w = img.shape[:2]
        diag = (w**2 + h**2) ** 0.5

        corners = get_corners(model, img)
        if corners is None:
            print(f"  {stem}: no pose detection — skipped")
            continue

        warped = warp(img, corners)
        cv2.imwrite(str(out_img), warped, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Empty label stub (only for train split, don't overwrite existing labels)
        if not is_low_res:
            lbl = OUT_BASE / "labels" / "train" / f"{stem}_warped.txt"
            if not lbl.exists():
                lbl.write_text("")

        flag = " [EXCLUDED]" if is_low_res else ""
        print(f"  {stem}: {w}×{h} → {subfolder}/{out_img.name}{flag}")

        label_status = existing.get(stem, {}).get("label_status", "empty")
        manifest.append({
            "stem": stem,
            "source": str(src.name),
            "source_wh": [w, h],
            "source_diag": round(diag, 1),
            "source_low_res": diag < LOW_RES_DIAG,
            "excluded": is_low_res,
            "warped_path": str(out_img.relative_to(BASE)),
            "label_status": label_status,
        })

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    n_train = sum(1 for m in manifest if not m["excluded"])
    n_lr = sum(1 for m in manifest if m["excluded"])
    n_labeled = sum(1 for m in manifest if m["label_status"].startswith("labeled"))
    print(f"\n  {n_train} train images  ({n_labeled} labeled)  |  {n_lr} excluded")
    print(f"  Manifest → {manifest_path}")
    print(f"\nNext: python scripts/annotate_balls.py")


if __name__ == "__main__":
    main()
