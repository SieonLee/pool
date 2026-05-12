"""
Stage new warped images for labeling.

Copies warped images from stress-test output into the YOLO training
dataset folder (images/train/) with the correct naming convention,
so annotate_balls.py can pick them up.

Only copies images that don't already have a label file, or where
the label file is empty (i.e., truly unlabeled).

Run:
  python scripts/stage_new_labels.py        # dry-run, shows what would be copied
  python scripts/stage_new_labels.py --go   # actually copy
"""
import argparse
import shutil
from pathlib import Path

BASE     = Path(__file__).parent.parent
STRESS   = BASE / "review" / "stress_test"
IMG_TRAIN = BASE / "datasets" / "pose-warped-balls" / "images" / "train"
LBL_TRAIN = BASE / "datasets" / "pose-warped-balls" / "labels" / "train"

# Stems selected for labeling based on visual warp quality inspection.
# Criteria: balls distinguishable in warped image, table surface usable.
LABEL_STEMS = [
    "pool_real_011",   # blue table, clear rack + cue ball, ~10 balls
    "pool_real_012",   # green table, ~12 scattered balls, clear
    "pool_real_020",   # blue table, ~15 balls, excellent quality
    "pool_real_024",   # green table, white ball (cue) + ~10 obj balls visible
    "pool_real_001",   # green table, rack + cue visible, slightly blurry
    "pool_real_007",   # blue table, ~12 balls, blurry but labelable
    "pool_real_021",   # purple table, 5 balls, one white (cue)
    "pool_real_027",   # green snooker table, ~8 small colored balls
    "pool_real_028",   # teal table, ~5 balls at edges
]

SKIP_STEMS = {
    "pool_real_003": "extreme warp distortion — balls smeared",
    "pool_real_005": "extreme oblique angle — balls are thin streaks",
    "pool_real_006": "dark table + motion blur — unrecognizable",
    "pool_real_015": "only cue stick visible, no balls",
    "pool_real_025": "too blurry — balls indistinct",
    "pool_real_031": "diagonal warp + heavy blur — unusable",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--go", action="store_true",
                        help="Actually copy files (default: dry-run)")
    args = parser.parse_args()

    dry = not args.go
    if dry:
        print("DRY RUN — pass --go to actually copy\n")

    IMG_TRAIN.mkdir(parents=True, exist_ok=True)
    LBL_TRAIN.mkdir(parents=True, exist_ok=True)

    print("Images selected for labeling:")
    print("─" * 60)
    staged = []
    for stem in LABEL_STEMS:
        src = STRESS / stem / f"{stem}_warp.jpg"
        dst = IMG_TRAIN / f"{stem}_warped.jpg"
        lbl = LBL_TRAIN / f"{stem}_warped.txt"

        if not src.exists():
            print(f"  MISSING  {stem}  (no warp in stress_test/)")
            continue

        already_labeled = lbl.exists() and lbl.stat().st_size > 0
        if already_labeled:
            print(f"  SKIP     {stem}_warped.jpg  (already has labels)")
            continue

        status = "EXISTS" if dst.exists() else "NEW   "
        print(f"  {status}  {stem}_warped.jpg  →  images/train/")
        if not dry:
            shutil.copy(src, dst)
            # Create empty label file as placeholder
            if not lbl.exists():
                lbl.touch()
        staged.append(stem)

    print()
    print("Images skipped (warp too poor to label):")
    print("─" * 60)
    for stem, reason in SKIP_STEMS.items():
        print(f"  SKIP  {stem}  [{reason}]")

    print()
    if dry:
        print(f"Would stage {len(staged)} image(s). Run with --go to apply.")
    else:
        print(f"Staged {len(staged)} image(s) to datasets/pose-warped-balls/images/train/")
        print()
        print("Next step:")
        print("  python scripts/annotate_balls.py")
        print()
        print("Controls: drag box → C (cue_ball) or O (object_ball) → S (save)")

    print()
    n_train = len(list(IMG_TRAIN.glob("*.jpg")))
    lbl_with_content = [p for p in LBL_TRAIN.glob("*.txt") if p.stat().st_size > 0]
    print(f"Dataset status: {n_train} images in train/  |  {len(lbl_with_content)} labeled")


if __name__ == "__main__":
    main()
