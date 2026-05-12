"""
Quick sanity check: show each annotated image with its GT corners.
Useful after annotating to verify correctness before training.

Run:
  python scripts/check_annotations.py
  python scripts/check_annotations.py pool_real_005  # specific image

Controls: any key = next, Q = quit
"""
import cv2
import json
import sys
from pathlib import Path

BASE = Path(__file__).parent.parent
GT_DIR = BASE / "annotations" / "table_corners_gt"
PICTURE_DIR = BASE / "picture"

CORNER_COLORS = [(0, 255, 0), (0, 200, 255), (0, 0, 255), (255, 0, 255)]
CORNER_NAMES = ["TL", "TR", "BR", "BL"]
WARP_W, WARP_H = 900, 450
FONT = cv2.FONT_HERSHEY_SIMPLEX


def warp(img, corners):
    import numpy as np
    dst = [[0, 0], [WARP_W, 0], [WARP_W, WARP_H], [0, WARP_H]]
    M = cv2.getPerspectiveTransform(
        __import__("numpy").array(corners, dtype="float32"),
        __import__("numpy").array(dst, dtype="float32")
    )
    return cv2.warpPerspective(img, M, (WARP_W, WARP_H))


def main():
    filter_stem = sys.argv[1] if len(sys.argv) > 1 else None
    files = sorted(GT_DIR.glob("*.json"))
    if filter_stem:
        files = [f for f in files if filter_stem in f.stem]

    print(f"{len(files)} annotation(s) to review")

    for jf in files:
        with open(jf) as f:
            data = json.load(f)
        stem = jf.stem
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            c = PICTURE_DIR / (stem + ext)
            if c.exists():
                img_path = c
                break
        if img_path is None:
            print(f"  image not found: {stem}")
            continue

        img = cv2.imread(str(img_path))
        corners_dict = data["corners"]
        corners = [corners_dict["TL"], corners_dict["TR"], corners_dict["BR"], corners_dict["BL"]]

        h, w = img.shape[:2]
        scale = min(1.0, 1200 / max(h, w))
        disp = cv2.resize(img, (int(w * scale), int(h * scale)))

        for i, (px, py) in enumerate(corners):
            x, y = int(px * scale), int(py * scale)
            cv2.circle(disp, (x, y), 8, CORNER_COLORS[i], -1)
            cv2.putText(disp, CORNER_NAMES[i], (x + 10, y - 10),
                        FONT, 0.7, CORNER_COLORS[i], 2)

        import numpy as np
        pts = np.array([(int(x*scale), int(y*scale)) for x, y in corners], np.int32)
        cv2.polylines(disp, [pts], isClosed=True, color=(255,255,255), thickness=2)

        warped = warp(img, corners)
        warped_small = cv2.resize(warped, (int(WARP_W*0.4), int(WARP_H*0.4)))

        cv2.putText(disp, f"{stem}  (any key=next, Q=quit)",
                    (8, disp.shape[0]-10), FONT, 0.5, (220,220,220), 1)

        cv2.imshow("GT Check", disp)
        cv2.imshow("Warped GT", warped_small)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
