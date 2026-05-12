"""
Table localization audit — visualizes the full localization pipeline for one image.

Produces review/table_audit/<stem>_audit.jpg with four panels:
  A. Original image + predicted corner polygon (yellow) + inner cloth bbox (green dashed)
  B. Warped image + cloth mask overlay (green tint) + cloth bbox (green dashed)
  C. Warped image + ALL detections: kept (normal) vs rejected-outside-cloth (red ⊗)
  D. Text summary: coverage ratio, dominant hue, balls kept/rejected, warnings

Usage:
  python scripts/audit_table_localization.py --image new_picture/maxresdefault.jpg
  python scripts/audit_table_localization.py --image picture/pool_real_013.jpg
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / "scripts"))

OUT_DIR = BASE / "review" / "table_audit"
DEFAULT_CORNER_CKPT = BASE / "models" / "checkpoints" / "table_corners_mvp_v1.pt"
DEFAULT_BALL_CKPT   = BASE / "models" / "checkpoints" / "ball_yolo_v7_below_baseline.pt"

FONT   = cv2.FONT_HERSHEY_SIMPLEX
FONT_B = cv2.FONT_HERSHEY_DUPLEX

# Panel layout
PANEL_W = 900
PANEL_H = 450
HEADER_H = 32

C_WHITE     = (255, 255, 255)
C_BLACK     = (0, 0, 0)
C_YELLOW    = (0, 220, 255)    # corner polygon — raw predicted
C_GREEN_HI  = (30, 220, 80)   # cloth bbox / kept balls
C_RED       = (50, 50, 230)   # rejected balls
C_ORANGE    = (30, 160, 255)  # object balls
C_BLUE_RING = (200, 120, 60)  # cue ring


def draw_corner_polygon_on_orig(img: np.ndarray, corners: np.ndarray,
                                 cloth_bounds_warp, M_inv) -> np.ndarray:
    """
    Draw predicted corners (yellow polygon) on the original image.
    If cloth_bounds_warp and M_inv provided, also back-project the cloth
    bounding box to original-image space (green dashed).
    """
    out = img.copy()
    h_orig, w_orig = img.shape[:2]

    # Scale for display (fit in PANEL_W x PANEL_H)
    scale = min(PANEL_W / w_orig, PANEL_H / h_orig)
    display_w = int(w_orig * scale)
    display_h = int(h_orig * scale)
    out = cv2.resize(out, (display_w, display_h))

    sc = scale  # corners are in original-image space

    # Draw predicted polygon (yellow)
    pts = (corners * sc).astype(np.int32)
    cv2.polylines(out, [pts.reshape(-1, 1, 2)], True, C_YELLOW, 2, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(out, tuple(pt), 6, C_YELLOW, -1)

    # Back-project cloth bounds from warp space to original space
    if cloth_bounds_warp is not None and M_inv is not None:
        x0, y0, x1, y1 = cloth_bounds_warp
        warp_corners = np.array([[x0, y0], [x1, y0],
                                  [x1, y1], [x0, y1]], dtype=np.float32)
        orig_corners = cv2.perspectiveTransform(
            warp_corners.reshape(-1, 1, 2), M_inv).reshape(-1, 2)
        orig_pts = (orig_corners * sc).astype(np.int32)
        cv2.polylines(out, [orig_pts.reshape(-1, 1, 2)], True, C_GREEN_HI, 2, cv2.LINE_AA)
        # Label
        ctr = orig_pts.mean(axis=0).astype(int)
        cv2.putText(out, "CLOTH BOUND", tuple(ctr - np.array([50, 10])),
                    FONT, 0.45, C_GREEN_HI, 1, cv2.LINE_AA)

    # Pad to PANEL_W x PANEL_H
    pad_w = PANEL_W - display_w
    pad_h = PANEL_H - display_h
    out = cv2.copyMakeBorder(out, 0, pad_h, 0, pad_w,
                              cv2.BORDER_CONSTANT, value=(20, 20, 20))
    return out


def draw_warp_cloth_panel(warped: np.ndarray, cloth_mask: np.ndarray,
                           bounds) -> np.ndarray:
    """Panel B: warped + cloth tint + bbox."""
    from table_refinement import draw_cloth_mask_overlay, draw_cloth_bounds_on_warp
    out = draw_cloth_mask_overlay(warped, cloth_mask, alpha=0.25)
    out = draw_cloth_bounds_on_warp(out, bounds, color=C_GREEN_HI, thickness=2)
    cv2.putText(out, "CLOTH MASK (green) + PLAYABLE BOUND (dashed)",
                (8, 22), FONT, 0.44, C_GREEN_HI, 1, cv2.LINE_AA)
    return out


def draw_detection_panel(warped: np.ndarray, kept: list[dict],
                          rejected: list[dict], bounds) -> np.ndarray:
    """Panel C: warped + kept balls (normal) + rejected balls (red ⊗)."""
    from table_refinement import draw_cloth_bounds_on_warp
    out = warped.copy()
    out = draw_cloth_bounds_on_warp(out, bounds, color=C_GREEN_HI, thickness=1)

    for b in kept:
        x, y = int(b["x"]), int(b["y"])
        r = max(8, min(28, int(b.get("r", 15))))
        is_cue = b["type"] == "cue_ball"
        fill = C_WHITE if is_cue else C_ORANGE
        ring = C_BLUE_RING if is_cue else (30, 80, 200)
        cv2.circle(out, (x, y), r, fill, -1)
        cv2.circle(out, (x, y), r + 2, ring, 2)
        label = "C" if is_cue else str(b["id"])
        (tw, _), _ = cv2.getTextSize(label, FONT_B, 0.40, 1)
        cv2.putText(out, label, (x - tw // 2, y + 5),
                    FONT_B, 0.40, C_BLACK if is_cue else C_WHITE, 1, cv2.LINE_AA)
        # Confidence tag
        tag = f"{b.get('confidence', 0):.2f}"
        cv2.putText(out, tag, (x - 12, y - r - 3), FONT, 0.30, (180, 180, 180), 1)

    for b in rejected:
        x, y = int(b["x"]), int(b["y"])
        r = max(8, min(28, int(b.get("r", 15))))
        # Draw ⊗ (circle + cross)
        cv2.circle(out, (x, y), r, C_RED, 2)
        d = int(r * 0.7)
        cv2.line(out, (x - d, y - d), (x + d, y + d), C_RED, 2, cv2.LINE_AA)
        cv2.line(out, (x + d, y - d), (x - d, y + d), C_RED, 2, cv2.LINE_AA)
        tag = f"id{b['id']} REJECTED"
        cv2.putText(out, tag, (x + r + 3, y + 4), FONT, 0.35, C_RED, 1)

    cv2.putText(out, f"KEPT: {len(kept)}  REJECTED: {len(rejected)}",
                (8, 22), FONT_B, 0.50, C_WHITE, 1, cv2.LINE_AA)
    return out


def draw_summary_panel(stats: dict, kept: list, rejected: list,
                        cloth_warnings: list, all_warnings: list,
                        stem: str) -> np.ndarray:
    """Panel D: text summary on dark background."""
    out = np.full((PANEL_H, PANEL_W, 3), (18, 18, 18), dtype=np.uint8)

    lines = [
        f"AUDIT: {stem}",
        "",
        f"  Cloth coverage : {stats.get('cloth_coverage', '?'):.1%}",
        f"  Dominant hue   : {stats.get('dominant_hue', '?')} (OpenCV 0–180)",
        f"  Cloth bounds   : {stats.get('bounds', 'none')}",
        "",
        f"  Balls kept     : {len(kept)}",
    ]
    for b in kept:
        lines.append(f"    id{b['id']} {b['type'][:3]}  "
                     f"warp=({b['x']:.0f},{b['y']:.0f})  "
                     f"conf={b.get('confidence', 0):.3f}")

    lines += ["", f"  Balls rejected : {len(rejected)}"]
    for b in rejected:
        lines.append(f"    id{b['id']} {b['type'][:3]}  "
                     f"warp=({b['x']:.0f},{b['y']:.0f})  "
                     f"conf={b.get('confidence', 0):.3f}  "
                     f"← {b.get('rejection_reason', '?')}")

    lines += ["", "  Cloth warnings:"]
    for w in (cloth_warnings or ["none"]):
        lines.append(f"    {w}")

    lines += ["", "  All perception warnings:"]
    for w in (all_warnings or ["none"]):
        lines.append(f"    {w}")

    y = 26
    for line in lines:
        if not line:
            y += 8
            continue
        scale = 0.62 if line.startswith("AUDIT") else 0.42
        col = C_WHITE if line.startswith("AUDIT") else (200, 200, 200)
        if "REJECTED" in line or "rejected" in line.lower():
            col = (80, 100, 220)
        elif "kept" in line.lower() or "coverage" in line.lower():
            col = (80, 220, 100)
        elif "warning" in line.lower():
            col = (80, 160, 255)
        cv2.putText(out, line, (12, y), FONT_B if line.startswith("AUDIT") else FONT,
                    scale, col, 1, cv2.LINE_AA)
        y += 20 if scale < 0.5 else 28
        if y > PANEL_H - 10:
            break

    return out


def panel_header(title: str, subtitle: str = "") -> np.ndarray:
    hdr = np.full((HEADER_H, PANEL_W, 3), (12, 12, 12), dtype=np.uint8)
    cv2.rectangle(hdr, (0, 0), (3, HEADER_H), (80, 130, 220), -1)
    cv2.putText(hdr, title, (10, 21), FONT_B, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    if subtitle:
        (sw, _), _ = cv2.getTextSize(subtitle, FONT, 0.36, 1)
        cv2.putText(hdr, subtitle, (PANEL_W - sw - 10, 21),
                    FONT, 0.36, (80, 80, 80), 1, cv2.LINE_AA)
    return hdr


def run_audit(img_path: Path, corner_ckpt: Path, ball_ckpt: Path,
              cue_conf: float = 0.25, obj_conf: float = 0.35) -> Path:
    from ultralytics import YOLO
    import stress_test as pipeline
    import table_refinement as tr

    pipeline.CUE_CONF_THRESH  = cue_conf
    pipeline.OBJ_CONF_THRESH  = obj_conf
    pipeline.BALL_CONF_THRESH = min(cue_conf, obj_conf)

    corner_model = YOLO(str(corner_ckpt))
    ball_model   = YOLO(str(ball_ckpt))

    perc, warped, timing = pipeline.perceive(corner_model, ball_model, img_path)
    if perc is None or warped is None:
        print(f"ERROR: perceive() failed on {img_path}", file=sys.stderr)
        sys.exit(1)

    balls_raw = perc.get("balls", [])
    corners   = perc.get("table", {}).get("corners")
    corner_conf = perc.get("table", {}).get("corner_confidence", 0)

    # Run cloth filter
    kept, extras = tr.apply_cloth_filter(warped, balls_raw)
    cloth_mask    = extras["cloth_mask"]
    bounds        = extras["cloth_bounds"]
    coverage      = extras["cloth_coverage"]
    rejected      = extras["rejected_balls"]
    cloth_warns   = extras["warnings"]
    dom_hue       = extras.get("dominant_hue")

    # Compute inverse warp matrix for back-projection
    M_inv = None
    if corners is not None:
        src_pts = np.array(corners, dtype=np.float32)
        dst_pts = np.float32([[0, 0], [900, 0], [900, 450], [0, 450]])
        M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)

    # ── Build audit panels ────────────────────────────────────────────────────
    orig = cv2.imread(str(img_path))

    # Panel A: original + corner polygon + cloth bounds back-projected
    if corners is not None and orig is not None:
        panel_a = draw_corner_polygon_on_orig(
            orig, np.array(corners, dtype=np.float32), bounds, M_inv)
    else:
        panel_a = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
        cv2.putText(panel_a, "No corners detected", (20, 220),
                    FONT_B, 0.7, C_RED, 2)

    # Panel B: warped + cloth mask tint + bbox
    panel_b = draw_warp_cloth_panel(warped, cloth_mask, bounds)

    # Panel C: warped + detections (kept normal, rejected ⊗)
    panel_c = draw_detection_panel(warped, kept, rejected, bounds)

    # Panel D: text summary
    stats_d = {
        "cloth_coverage": coverage,
        "dominant_hue": dom_hue,
        "bounds": f"({bounds[0]},{bounds[1]})–({bounds[2]},{bounds[3]})" if bounds else "none",
    }
    all_warns = perc.get("status", {}).get("warnings", [])
    panel_d = draw_summary_panel(stats_d, kept, rejected, cloth_warns, all_warns, img_path.stem)

    # ── Assemble 2×2 grid ────────────────────────────────────────────────────
    hdr_a = panel_header("A — PREDICTED CORNERS (yellow) + CLOTH BOUND (green)",
                          f"corner_conf={corner_conf:.3f}")
    hdr_b = panel_header("B — CLOTH MASK OVERLAY",
                          f"coverage={coverage:.1%}  hue≈{dom_hue}")
    hdr_c = panel_header("C — BALL DETECTIONS",
                          f"kept={len(kept)}  rejected={len(rejected)}")
    hdr_d = panel_header("D — AUDIT SUMMARY",
                          f"cloth_warns={len(cloth_warns)}")

    row1 = np.hstack([
        np.vstack([hdr_a, panel_a]),
        np.vstack([hdr_b, panel_b]),
    ])
    row2 = np.hstack([
        np.vstack([hdr_c, panel_c]),
        np.vstack([hdr_d, panel_d]),
    ])
    composite = np.vstack([row1, row2])

    # ── Save ──────────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = img_path.stem
    out_path = OUT_DIR / f"{stem}_audit.jpg"
    cv2.imwrite(str(out_path), composite, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Also save JSON summary
    summary = {
        "stem": stem,
        "corner_confidence": corner_conf,
        "cloth_coverage": coverage,
        "dominant_hue": dom_hue,
        "cloth_bounds": list(bounds) if bounds else None,
        "balls_raw": len(balls_raw),
        "balls_kept": len(kept),
        "balls_rejected": len(rejected),
        "kept": [{"id": b["id"], "type": b["type"],
                  "warp_xy": [b["x"], b["y"]], "conf": b.get("confidence")}
                 for b in kept],
        "rejected": [{"id": b["id"], "type": b["type"],
                      "warp_xy": [b["x"], b["y"]], "conf": b.get("confidence"),
                      "reason": b.get("rejection_reason")}
                     for b in rejected],
        "cloth_warnings": cloth_warns,
        "perception_warnings": all_warns,
        "timing_ms": timing,
    }
    (OUT_DIR / f"{stem}_audit.json").write_text(
        __import__("json").dumps(summary, indent=2))

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Table localization audit")
    parser.add_argument("--image", required=True)
    parser.add_argument("--corner-ckpt", default=str(DEFAULT_CORNER_CKPT))
    parser.add_argument("--ball-ckpt",   default=str(DEFAULT_BALL_CKPT))
    parser.add_argument("--cue-conf",  type=float, default=0.25)
    parser.add_argument("--obj-conf",  type=float, default=0.35)
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"ERROR: {img_path} not found", file=sys.stderr)
        sys.exit(1)

    out = run_audit(img_path, Path(args.corner_ckpt), Path(args.ball_ckpt),
                    args.cue_conf, args.obj_conf)
    print(f"Audit → {out}")
    print(f"JSON  → {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
