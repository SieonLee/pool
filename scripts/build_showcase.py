"""
Build showcase contact sheets from existing demo composite images.

Produces three sheets in review/showcase/:
  showcase_best.jpg      — top 5 high-confidence plan-ready results
  showcase_failures.jpg  — expected failure / no-plan cases
  showcase_confidence.jpg — confidence distribution strip (best→worst)

Each cell = one demo composite scaled to 300×490px (1/3 scale of 900×1470).

Usage:
  python scripts/build_showcase.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).parent.parent
DEMO_DIR  = BASE / "review" / "demo"
OUT_DIR   = BASE / "review" / "showcase"

# Cell size (scaled from 900×1470)
CELL_W = 300
CELL_H = 490
LABEL_H = 28

C_WHITE = (255, 255, 255)
C_BLACK = (0, 0, 0)
C_GOOD  = (80, 220, 100)
C_BAD   = (60,  80, 220)
C_MED   = (80, 200, 240)
FONT    = cv2.FONT_HERSHEY_SIMPLEX
FONT_B  = cv2.FONT_HERSHEY_DUPLEX


def load_result(stem: str) -> dict | None:
    p = DEMO_DIR / f"{stem}_result.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def load_demo_img(stem: str) -> np.ndarray | None:
    p = DEMO_DIR / f"{stem}_demo.jpg"
    if not p.exists():
        return None
    img = cv2.imread(str(p))
    if img is None:
        return None
    # Scale to cell size
    return cv2.resize(img, (CELL_W, CELL_H), interpolation=cv2.INTER_AREA)


def make_cell(stem: str) -> np.ndarray | None:
    """Load demo image and add a label strip at the top."""
    img = load_demo_img(stem)
    if img is None:
        return None
    result = load_result(stem)
    status = (result or {}).get("status", "unknown")
    plan = (result or {}).get("plan", {})
    conf = plan.get("confidence")

    # Label strip
    label = np.zeros((LABEL_H, CELL_W, 3), dtype=np.uint8)
    cv2.rectangle(label, (0, 0), (CELL_W, LABEL_H), (15, 15, 15), -1)

    # Status color
    if status == "plan_ready":
        col = C_GOOD if (conf or 0) >= 0.80 else C_MED if (conf or 0) >= 0.50 else C_BAD
        conf_s = f"  {conf:.0%}" if conf is not None else ""
        txt = f"{stem[-3:]}  READY{conf_s}"
    else:
        col = C_BAD
        txt = f"{stem[-3:]}  {status[:18].upper()}"

    cv2.putText(label, txt, (6, 19), FONT_B, 0.46, col, 1, cv2.LINE_AA)
    return np.vstack([label, img])


def build_sheet(stems: list[str], cols: int, title: str) -> np.ndarray:
    """Assemble cells into a grid."""
    cells = []
    for stem in stems:
        cell = make_cell(stem)
        if cell is None:
            # Placeholder
            cell = np.zeros((LABEL_H + CELL_H, CELL_W, 3), dtype=np.uint8)
            cv2.putText(cell, stem, (4, 20), FONT, 0.40, (80, 80, 80), 1)
        cells.append(cell)

    # Pad to full rows
    while len(cells) % cols:
        cells.append(np.zeros_like(cells[0]))

    rows = []
    for i in range(0, len(cells), cols):
        row = np.hstack(cells[i:i+cols])
        rows.append(row)
    grid = np.vstack(rows)

    # Title bar
    title_h = 40
    bar = np.full((title_h, grid.shape[1], 3), (18, 18, 18), dtype=np.uint8)
    cv2.rectangle(bar, (0, 0), (4, title_h), (80, 130, 220), -1)
    cv2.putText(bar, title, (12, 27), FONT_B, 0.68, (210, 210, 210), 1, cv2.LINE_AA)

    return np.vstack([bar, grid])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all results
    stems_all = sorted(p.stem.replace("_result", "")
                       for p in DEMO_DIR.glob("*_result.json"))

    ready = []
    failures = []
    for stem in stems_all:
        r = load_result(stem)
        if r is None:
            continue
        status = r.get("status", "unknown")
        plan   = r.get("plan", {})
        conf   = plan.get("confidence", 0.0) or 0.0
        if status == "plan_ready":
            ready.append((conf, stem))
        else:
            failures.append(stem)

    ready.sort(reverse=True)  # highest conf first

    # ── Sheet 1: Best results ──────────────────────────────────────────────────
    best_stems = [s for _, s in ready[:6]]
    if best_stems:
        sheet_best = build_sheet(best_stems, cols=3,
                                 title="SHOWCASE — Best Results (plan_ready, highest confidence)")
        cv2.imwrite(str(OUT_DIR / "showcase_best.jpg"), sheet_best,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"showcase_best.jpg    ({sheet_best.shape[1]}×{sheet_best.shape[0]}px)  — {len(best_stems)} images")

    # ── Sheet 2: Failure cases ─────────────────────────────────────────────────
    if failures:
        sheet_fail = build_sheet(failures[:6], cols=3,
                                 title="SHOWCASE — Expected Failures (no-plan cases)")
        cv2.imwrite(str(OUT_DIR / "showcase_failures.jpg"), sheet_fail,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"showcase_failures.jpg ({sheet_fail.shape[1]}×{sheet_fail.shape[0]}px)  — {len(failures)} images")

    # ── Sheet 3: Confidence distribution strip (all ready, sorted) ────────────
    conf_stems = [s for _, s in ready]
    if conf_stems:
        sheet_conf = build_sheet(conf_stems, cols=min(5, len(conf_stems)),
                                 title="SHOWCASE — All Plan-Ready Results (confidence: high → low)")
        cv2.imwrite(str(OUT_DIR / "showcase_confidence.jpg"), sheet_conf,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"showcase_confidence.jpg ({sheet_conf.shape[1]}×{sheet_conf.shape[0]}px)  — {len(conf_stems)} images")

    print(f"\nShowcase assets → {OUT_DIR}")


if __name__ == "__main__":
    main()
