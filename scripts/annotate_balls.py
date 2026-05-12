"""
Ball annotation tool for pose-warped 900×450 images.

Controls:
  Left-click drag    — draw bounding box
  C key              — tag last box as cue_ball  (class 0, shown in WHITE)
  O key              — tag last box as object_ball (class 1, shown in ORANGE)
  Z / Backspace      — undo last box
  S key              — save labels and advance to next image
  D key              — skip image (no changes saved)
  Q / Esc            — quit

Workflow:
  - Drag to draw a box around each ball
  - Press C or O to tag it (default: object_ball)
  - Cue ball box turns white; object balls turn orange
  - Press S when done with an image — labels saved to labels/train/<stem>.txt

Labels: YOLO format — class cx cy bw bh  (all normalized 0-1)

Run:
  python scripts/annotate_balls.py               # all unlabeled images
  python scripts/annotate_balls.py --redo        # re-label already-labeled images too
  python scripts/annotate_balls.py --image pool_real_002_warped.jpg
"""
import argparse
import cv2
import json
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent.parent
DATASET = BASE / "datasets" / "pose-warped-balls"
IMG_DIR = DATASET / "images" / "train"
LBL_DIR = DATASET / "labels" / "train"
MANIFEST = DATASET / "manifest.json"

WARP_W, WARP_H = 900, 450
CUE_COLOR = (255, 255, 255)
OBJ_COLOR = (0, 165, 255)
PENDING_COLOR = (180, 180, 50)
CLASS_NAMES = {0: "cue_ball", 1: "object_ball"}

FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_existing(lbl_path):
    boxes = []
    if lbl_path.exists() and lbl_path.stat().st_size > 0:
        for line in lbl_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) == 5:
                cls = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:])
                boxes.append([cls, cx, cy, bw, bh])
    return boxes


def boxes_to_yolo(boxes):
    lines = []
    for cls, cx, cy, bw, bh in boxes:
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return "\n".join(lines)


def yolo_to_pixel(cx, cy, bw, bh, W=WARP_W, H=WARP_H):
    x1 = int((cx - bw / 2) * W)
    y1 = int((cy - bh / 2) * H)
    x2 = int((cx + bw / 2) * W)
    y2 = int((cy + bh / 2) * H)
    return x1, y1, x2, y2


def pixel_to_yolo(x1, y1, x2, y2, W=WARP_W, H=WARP_H):
    cx = (x1 + x2) / 2 / W
    cy = (y1 + y2) / 2 / H
    bw = abs(x2 - x1) / W
    bh = abs(y2 - y1) / H
    return cx, cy, bw, bh


def draw_boxes(canvas, boxes, pending=None, drag=None):
    for cls, cx, cy, bw, bh in boxes:
        x1, y1, x2, y2 = yolo_to_pixel(cx, cy, bw, bh)
        color = CUE_COLOR if cls == 0 else OBJ_COLOR
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = "C" if cls == 0 else "O"
        cv2.putText(canvas, label, (x1 + 2, y1 + 14), FONT, 0.5, color, 1)

    if pending is not None:
        cls, cx, cy, bw, bh = pending
        x1, y1, x2, y2 = yolo_to_pixel(cx, cy, bw, bh)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), PENDING_COLOR, 2)
        cv2.putText(canvas, "?", (x1 + 2, y1 + 14), FONT, 0.5, PENDING_COLOR, 1)

    if drag is not None:
        cv2.rectangle(canvas, drag[0], drag[1], (100, 100, 200), 1)

    n_cue = sum(1 for b in boxes if b[0] == 0)
    n_obj = sum(1 for b in boxes if b[0] == 1)
    cv2.putText(canvas, f"Cue: {n_cue}  Obj: {n_obj}  Total: {len(boxes)}",
                (8, 24), FONT, 0.65, (200, 200, 200), 2)
    cv2.putText(canvas, "C=cue  O=obj  Z=undo  S=save  D=skip",
                (8, WARP_H - 10), FONT, 0.5, (150, 150, 150), 1)
    return canvas


def annotate_image(img_path, lbl_path):
    """Returns 'saved', 'skipped', or 'quit'."""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  Cannot read {img_path.name}")
        return "skipped"

    boxes = load_existing(lbl_path)
    pending = None  # last drawn box, awaiting C/O tag
    drag_start = None
    drag_cur = None
    drawing = False

    win = "annotate_balls"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    def on_mouse(event, x, y, flags, _):
        nonlocal drag_start, drag_cur, drawing, pending
        if event == cv2.EVENT_LBUTTONDOWN:
            drag_start = (x, y)
            drag_cur = (x, y)
            drawing = True
            pending = None
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            drag_cur = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and drawing:
            drawing = False
            x1, y1 = drag_start
            x2, y2 = x, y
            if abs(x2 - x1) > 6 and abs(y2 - y1) > 6:
                cx, cy, bw, bh = pixel_to_yolo(
                    min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
                )
                # Default to object_ball; user presses C to change
                pending = [1, cx, cy, bw, bh]
            drag_start = drag_cur = None

    cv2.setMouseCallback(win, on_mouse)

    print(f"  {img_path.name}  ({len(boxes)} existing labels)")

    while True:
        canvas = img.copy()
        drag_rect = (drag_start, drag_cur) if drawing and drag_start else None
        draw_boxes(canvas, boxes, pending, drag_rect)
        cv2.imshow(win, canvas)

        key = cv2.waitKey(20) & 0xFF

        if key == ord('c') or key == ord('C'):
            if pending is not None:
                pending[0] = 0
                boxes.append(pending)
                pending = None
            elif boxes:
                boxes[-1][0] = 0  # re-tag last saved

        elif key == ord('o') or key == ord('O'):
            if pending is not None:
                pending[0] = 1
                boxes.append(pending)
                pending = None
            elif boxes:
                boxes[-1][0] = 1

        elif key in (ord('z'), 8):  # Z or Backspace
            if pending is not None:
                pending = None
            elif boxes:
                boxes.pop()

        elif key == ord('s') or key == ord('S'):
            if pending is not None:
                pending[0] = 1
                boxes.append(pending)
                pending = None
            lbl_path.write_text(boxes_to_yolo(boxes))
            print(f"    Saved {len(boxes)} boxes → {lbl_path.name}")
            cv2.destroyWindow(win)
            return "saved"

        elif key == ord('d') or key == ord('D'):
            print(f"    Skipped")
            cv2.destroyWindow(win)
            return "skipped"

        elif key in (ord('q'), 27):  # Q or Esc
            cv2.destroyAllWindows()
            return "quit"

    cv2.destroyWindow(win)
    return "quit"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--redo", action="store_true",
                        help="Re-label already-labeled images")
    parser.add_argument("--image", default=None,
                        help="Annotate a specific warped image filename")
    args = parser.parse_args()

    if not IMG_DIR.exists():
        print("No warped images found. Run export_warped_balls.py first.")
        return

    if args.image:
        imgs = [IMG_DIR / args.image]
    else:
        imgs = sorted(IMG_DIR.glob("*.jpg")) + sorted(IMG_DIR.glob("*.png"))

    if not args.redo and not args.image:
        # Skip images that already have non-empty label files
        imgs = [p for p in imgs
                if not (LBL_DIR / (p.stem + ".txt")).exists()
                or (LBL_DIR / (p.stem + ".txt")).stat().st_size == 0]

    if not imgs:
        print("All images already labeled. Use --redo to re-label.")
        return

    print(f"Annotating {len(imgs)} image(s)...")
    saved = skipped = 0
    for img_path in imgs:
        lbl_path = LBL_DIR / (img_path.stem + ".txt")
        result = annotate_image(img_path, lbl_path)
        if result == "saved":
            saved += 1
        elif result == "skipped":
            skipped += 1
        elif result == "quit":
            break

    # Update manifest label_status
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text())
        for entry in manifest:
            stem = entry["stem"]
            lbl = LBL_DIR / f"{stem}_warped.txt"
            if lbl.exists() and lbl.stat().st_size > 0:
                n = len([l for l in lbl.read_text().strip().splitlines() if l.strip()])
                entry["label_status"] = f"labeled:{n}"
            elif lbl.exists():
                entry["label_status"] = "empty"
        MANIFEST.write_text(json.dumps(manifest, indent=2))

    print(f"\nSaved: {saved}  Skipped: {skipped}")
    print("Next: python scripts/train_ball_yolo.py")


if __name__ == "__main__":
    main()
