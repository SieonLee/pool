"""
Table corner annotation tool.

Click 4 corners in order: TL → TR → BR → BL
Saves JSON to annotations/table_corners_gt/<image_stem>.json

Controls:
  Left click   — place next corner
  Right click  — undo last corner
  S            — save and go to next image
  D            — skip image (no save)
  Q            — quit
"""
import cv2
import json
import sys
from pathlib import Path

PICTURE_DIR = Path(__file__).parent.parent / "picture"
GT_DIR = Path(__file__).parent.parent / "annotations" / "table_corners_gt"
GT_DIR.mkdir(parents=True, exist_ok=True)

CORNER_NAMES = ["TL", "TR", "BR", "BL"]
CORNER_COLORS = [
    (0, 255, 0),    # TL  green
    (0, 200, 255),  # TR  yellow
    (0, 0, 255),    # BR  red
    (255, 0, 255),  # BL  magenta
]
POINT_RADIUS = 6
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Skip contact sheets and thumbnails
SKIP_SUFFIXES = {"new_uploads_contact_sheet", "search_contact_sheet", "thumb"}


def load_image_list():
    exts = {".jpg", ".jpeg", ".png"}
    imgs = sorted(
        p for p in PICTURE_DIR.iterdir()
        if p.suffix.lower() in exts and p.stem not in SKIP_SUFFIXES
    )
    return imgs


def draw_state(canvas, corners, img_path, idx, total):
    h, w = canvas.shape[:2]
    # Draw placed corners
    for i, (x, y) in enumerate(corners):
        cv2.circle(canvas, (x, y), POINT_RADIUS, CORNER_COLORS[i], -1)
        cv2.putText(canvas, CORNER_NAMES[i], (x + 8, y - 8),
                    FONT, 0.7, CORNER_COLORS[i], 2)
    # Draw order lines
    if len(corners) >= 2:
        for i in range(len(corners) - 1):
            cv2.line(canvas, corners[i], corners[i + 1], (200, 200, 200), 1)
    if len(corners) == 4:
        cv2.line(canvas, corners[3], corners[0], (200, 200, 200), 1)

    # Status bar
    next_corner = CORNER_NAMES[len(corners)] if len(corners) < 4 else "DONE"
    status = f"[{idx+1}/{total}] {img_path.name}  |  Next: {next_corner}  |  S=save  D=skip  RClick=undo  Q=quit"
    cv2.rectangle(canvas, (0, h - 28), (w, h), (30, 30, 30), -1)
    cv2.putText(canvas, status, (8, h - 8), FONT, 0.5, (220, 220, 220), 1)

    if len(corners) == 4:
        cv2.putText(canvas, "All 4 corners placed — press S to save",
                    (8, 32), FONT, 0.7, (0, 255, 100), 2)


def annotate(start_from: str | None = None):
    images = load_image_list()
    if not images:
        print("No images found in", PICTURE_DIR)
        return

    # Resume: skip already-annotated unless user wants to re-label
    already_done = {p.stem for p in GT_DIR.glob("*.json")}

    start_idx = 0
    if start_from:
        for i, p in enumerate(images):
            if p.stem == start_from:
                start_idx = i
                break

    print(f"Found {len(images)} images, {len(already_done)} already annotated.")
    print("Skipping annotated images — to re-label one, pass its stem as argument.\n")

    idx = start_idx
    while idx < len(images):
        img_path = images[idx]
        if img_path.stem in already_done and start_from != img_path.stem:
            print(f"  skip (done): {img_path.name}")
            idx += 1
            continue

        orig = cv2.imread(str(img_path))
        if orig is None:
            print(f"  ERROR reading {img_path.name}, skipping")
            idx += 1
            continue

        # Resize for display if very large
        max_dim = 1200
        h, w = orig.shape[:2]
        scale = min(1.0, max_dim / max(h, w))
        disp_w, disp_h = int(w * scale), int(h * scale)
        display_orig = cv2.resize(orig, (disp_w, disp_h))

        corners: list[tuple[int, int]] = []

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(corners) < 4:
                    corners.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN:
                if corners:
                    corners.pop()

        cv2.namedWindow("annotate", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("annotate", on_mouse)

        action = None
        while action is None:
            canvas = display_orig.copy()
            draw_state(canvas, corners, img_path, idx, len(images))
            cv2.imshow("annotate", canvas)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('s') or key == ord('S'):
                if len(corners) == 4:
                    action = "save"
                else:
                    print(f"  Need 4 corners, only {len(corners)} placed.")
            elif key == ord('d') or key == ord('D'):
                action = "skip"
            elif key == ord('q') or key == ord('Q'):
                action = "quit"

        cv2.destroyAllWindows()

        if action == "quit":
            print("Quit.")
            break
        elif action == "skip":
            print(f"  Skipped: {img_path.name}")
            idx += 1
            continue
        elif action == "save":
            # Convert display coords back to original image coords
            orig_corners = [
                [round(x / scale), round(y / scale)]
                for x, y in corners
            ]
            gt = {
                "image": img_path.name,
                "width": w,
                "height": h,
                "corners": {
                    "TL": orig_corners[0],
                    "TR": orig_corners[1],
                    "BR": orig_corners[2],
                    "BL": orig_corners[3],
                },
                "order": ["TL", "TR", "BR", "BL"],
            }
            out_path = GT_DIR / f"{img_path.stem}.json"
            with open(out_path, "w") as f:
                json.dump(gt, f, indent=2)
            already_done.add(img_path.stem)
            print(f"  Saved: {out_path.name}")
            idx += 1

    print(f"\nDone. {len(already_done)} images annotated.")


if __name__ == "__main__":
    start = sys.argv[1] if len(sys.argv) > 1 else None
    annotate(start)
