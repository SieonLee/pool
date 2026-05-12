"""
Table cloth region detection and ball filtering.

After a perspective warp, the 900×450 canvas includes the full rail area
(and potentially background content if the predicted polygon was loose).
This module segments the playing surface (cloth) by color and filters
ball detections to only those landing on the cloth.

Key insight: the cloth is the largest high-saturation uniform-colored region
in the warp. Rails are dark. Out-of-bounds warp pixels are black.
Balls are small isolated blobs removed by morphological opening.

Integration:
  from table_refinement import apply_cloth_filter
  balls_kept, perc_extra = apply_cloth_filter(warped, balls)
  # perc_extra: {cloth_mask, cloth_bounds, cloth_coverage, rejected_balls, warnings}
"""

import math
import cv2
import numpy as np

# ── Tuning constants ──────────────────────────────────────────────────────────

CLOTH_COVERAGE_MIN  = 0.30  # below this → "loose_table_polygon" (hard flag)
CLOTH_COVERAGE_WARN = 0.48  # below this → "low_cloth_coverage" (soft warn)

# HSV thresholds for "candidate cloth pixel"
CLOTH_MIN_V = 45    # exclude very dark (rails ≈ V<50, shadows)
CLOTH_MAX_V = 255   # no upper cap — bright lit cloth (V=240-255) must pass.
                    # White glare / chalk excluded by CLOTH_MIN_S instead.
CLOTH_MIN_S = 16    # exclude very desaturated (white chalk, gray floor)
                    # light-blue cloth (S≈30) passes; white cue ball (S≈5) does not

CLOTH_HUE_TOL = 24  # ± OpenCV hue units (0–180) around dominant hue

# Morphological constants
# Balls are ~30–50px diameter in the 900×450 warp space.
# Strategy: CLOSE with a kernel LARGER than the largest ball to fill ball-shaped holes
# in the cloth mask (ball pixels ≠ cloth hue, so they appear as holes).
# Then keep the largest connected component.
# We do NOT use OPEN — OPEN would erode the cloth AROUND the ball holes, rejecting
# pixels that are genuinely on the cloth.
_FILL_K_SIZE  = 55  # close kernel: fills ball-shaped holes (max ball diam ~50px)
_SMALL_K_SIZE = 9   # close kernel for minor noise/chalk marks before large close

# Inward margin applied to the raw cloth bounding box to get the playable bound
CLOTH_BBOX_MARGIN = 6

# A ball is "outside cloth" if its centre pixel is NOT on the cloth mask.
# We also check a small radius around the centre for robustness.
BALL_CLOTH_CHECK_R = 4  # px


# ── Core detection ────────────────────────────────────────────────────────────

def detect_cloth_mask(warped: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Segment the playing surface in a 900×450 warped table image.

    Returns:
        cloth_mask  — uint8 binary mask (255 = cloth, 0 = not cloth)
        stats       — dict with dominant_hue, coverage, error (if any)
    """
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    # Step 1: candidate cloth pixels (moderate brightness, non-trivial saturation)
    candidate = (
        (v_ch >= CLOTH_MIN_V) &
        (v_ch <= CLOTH_MAX_V) &
        (s_ch >= CLOTH_MIN_S)
    )

    h_vals = h_ch[candidate]
    if len(h_vals) < 500:
        # Too few coloured pixels — warp probably collapsed or image is black
        full = np.full(warped.shape[:2], 255, dtype=np.uint8)
        return full, {"error": "too_few_candidate_pixels", "candidate_px": int(len(h_vals))}

    # Step 2: histogram over 36 bins (5° each) → dominant hue
    hist, edges = np.histogram(h_vals, bins=36, range=(0, 180))
    dominant_bin = int(np.argmax(hist))
    dominant_hue = float(edges[dominant_bin])   # OpenCV hue (0–180)

    # Step 3: cloth hue mask (wrap-safe)
    lo = (dominant_hue - CLOTH_HUE_TOL) % 180
    hi = (dominant_hue + CLOTH_HUE_TOL) % 180
    if lo <= hi:
        hue_ok = (h_ch >= lo) & (h_ch <= hi)
    else:                           # wraps around 0 (e.g. red)
        hue_ok = (h_ch >= lo) | (h_ch <= hi)

    cloth_raw = (candidate & hue_ok).astype(np.uint8) * 255

    # Step 4: morphological cleanup
    #
    # Problem: balls sitting on the cloth appear as holes in cloth_raw because
    # ball pixels don't match the cloth hue.  We need to FILL those holes so
    # the cloth region is solid, then keep only the largest component.
    #
    # We use two-stage closing:
    #   (a) small close — fills chalk dots, shadow patches, rail sights
    #   (b) large close — fills ball-sized holes (~30–50px diameter)
    # We do NOT use morphological OPEN: OPEN would erode cloth around ball-holes
    # and incorrectly mark ball-centre pixels as "not cloth".
    sk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_SMALL_K_SIZE, _SMALL_K_SIZE))
    fk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_FILL_K_SIZE,  _FILL_K_SIZE))

    cloth = cv2.morphologyEx(cloth_raw, cv2.MORPH_CLOSE, sk)   # minor gaps
    cloth = cv2.morphologyEx(cloth,     cv2.MORPH_CLOSE, fk)   # ball-sized holes

    # Step 5: keep only the largest connected component
    n_labels, labels, stats_cc, _ = cv2.connectedComponentsWithStats(cloth, connectivity=8)
    if n_labels <= 1:
        # No component found — return permissive full mask
        full = np.full(warped.shape[:2], 255, dtype=np.uint8)
        return full, {"error": "no_cloth_component", "dominant_hue": round(dominant_hue, 1)}

    areas = stats_cc[1:, cv2.CC_STAT_AREA]   # skip background (label 0)
    best_label = int(np.argmax(areas)) + 1
    cloth_clean = (labels == best_label).astype(np.uint8) * 255

    coverage = float(np.sum(cloth_clean > 0)) / (warped.shape[0] * warped.shape[1])
    return cloth_clean, {
        "dominant_hue": round(dominant_hue, 1),
        "hue_tolerance": CLOTH_HUE_TOL,
        "cloth_coverage": round(coverage, 3),
        "candidate_px": int(len(h_vals)),
    }


def cloth_bounds(cloth_mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """
    Tight bounding box of the cloth region (with inward margin).
    Returns (x_min, y_min, x_max, y_max) or None if mask empty.
    """
    ys, xs = np.where(cloth_mask > 0)
    if len(xs) < 200:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    m = CLOTH_BBOX_MARGIN
    return (
        min(x0 + m, x1),
        min(y0 + m, y1),
        max(x1 - m, x0),
        max(y1 - m, y0),
    )


def ball_on_cloth(ball: dict, cloth_mask: np.ndarray) -> bool:
    """
    Return True if the ball's centre (± small radius) is within the cloth mask.
    """
    h, w = cloth_mask.shape[:2]
    bx, by = int(round(ball["x"])), int(round(ball["y"]))
    r = BALL_CLOTH_CHECK_R
    # Sample a small disc of points
    for dy in range(-r, r + 1, r):
        for dx in range(-r, r + 1, r):
            px = max(0, min(w - 1, bx + dx))
            py = max(0, min(h - 1, by + dy))
            if cloth_mask[py, px] > 0:
                return True
    return False


# ── Main entry point ──────────────────────────────────────────────────────────

def apply_cloth_filter(
    warped: np.ndarray,
    balls: list[dict],
) -> tuple[list[dict], dict]:
    """
    Run cloth detection on `warped` and split balls into kept / rejected.

    Returns:
        kept_balls   — balls whose centres land on the cloth
        extras       — dict with cloth_mask, cloth_bounds, cloth_coverage,
                       rejected_balls, warnings, dominant_hue
    """
    cloth_mask, det_stats = detect_cloth_mask(warped)
    coverage = det_stats.get("cloth_coverage", 1.0)
    bounds   = cloth_bounds(cloth_mask)
    warnings = []

    if "error" in det_stats:
        # Detection failed — pass all balls through, warn
        warnings.append(f"cloth_detection_failed:{det_stats['error']}")
        return balls, {
            "cloth_mask": cloth_mask,
            "cloth_bounds": bounds,
            "cloth_coverage": coverage,
            "rejected_balls": [],
            "warnings": warnings,
            "dominant_hue": det_stats.get("dominant_hue"),
        }

    # Coverage sanity checks
    if coverage < CLOTH_COVERAGE_MIN:
        warnings.append(f"loose_table_polygon(coverage={coverage:.2f})")
    elif coverage < CLOTH_COVERAGE_WARN:
        warnings.append(f"low_cloth_coverage(coverage={coverage:.2f})")

    # Per-ball cloth membership
    kept, rejected = [], []
    for b in balls:
        if ball_on_cloth(b, cloth_mask):
            kept.append(b)
        else:
            rej = dict(b)
            rej["rejection_reason"] = "outside_cloth"
            rejected.append(rej)

    if rejected:
        n_cue_rej  = sum(1 for b in rejected if b["type"] == "cue_ball")
        n_obj_rej  = sum(1 for b in rejected if b["type"] == "object_ball")
        parts = []
        if n_cue_rej:
            parts.append(f"{n_cue_rej}_cue")
        if n_obj_rej:
            parts.append(f"{n_obj_rej}_obj")
        warnings.append(f"outside_cloth_rejected:{'_'.join(parts)}")

    return kept, {
        "cloth_mask": cloth_mask,
        "cloth_bounds": bounds,
        "cloth_coverage": round(coverage, 3),
        "rejected_balls": rejected,
        "warnings": warnings,
        "dominant_hue": det_stats.get("dominant_hue"),
    }


# ── Visualization helpers (used by audit + demo overlay) ─────────────────────

def draw_cloth_bounds_on_warp(img: np.ndarray, bounds: tuple | None,
                               color=(0, 220, 100), thickness=2, dashed=True) -> np.ndarray:
    """Draw the playable-region rectangle on a warp image."""
    out = img.copy()
    if bounds is None:
        return out
    x0, y0, x1, y1 = bounds
    if dashed:
        _dashed_rect(out, x0, y0, x1, y1, color, thickness)
    else:
        cv2.rectangle(out, (x0, y0), (x1, y1), color, thickness)
    return out


def draw_cloth_mask_overlay(img: np.ndarray, cloth_mask: np.ndarray,
                             alpha: float = 0.18) -> np.ndarray:
    """Tint cloth pixels with a soft green to show detected region."""
    out = img.copy()
    tint = np.zeros_like(out)
    tint[cloth_mask > 0] = (30, 180, 60)   # BGR green
    cv2.addWeighted(tint, alpha, out, 1.0 - alpha, 0, out)
    return out


def _dashed_rect(img, x0, y0, x1, y1, color, thickness, dash=14, gap=7):
    """Draw a dashed rectangle."""
    pts = [(x0, y0, x1, y0), (x1, y0, x1, y1),
           (x1, y1, x0, y1), (x0, y1, x0, y0)]
    for ax, ay, bx, by in pts:
        length = math.hypot(bx - ax, by - ay)
        if length < 1:
            continue
        ux, uy = (bx - ax) / length, (by - ay) / length
        t = 0.0
        draw = True
        while t < length:
            seg = dash if draw else gap
            t2 = min(t + seg, length)
            if draw:
                p1 = (int(ax + ux * t), int(ay + uy * t))
                p2 = (int(ax + ux * t2), int(ay + uy * t2))
                cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)
            t = t2
            draw = not draw
