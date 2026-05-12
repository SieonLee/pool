"""
Demo overlay — three-panel composite for portfolio / demo use.

Produces a single vertical composite image with three labeled panels:
  Panel 1 — DETECTION:       all detected balls, no shot lines
  Panel 2 — TACTICAL:        ghost ball, top-3 candidate paths, all pockets
  Panel 3 — RECOMMENDED SHOT: final clean shot with confidence badge

Works with the data format from stress_test.perceive() and stress_test.plan()
(the single-image pipeline format used by run_full_pipeline.py).

Usage (standalone):
  python scripts/demo_overlay.py --result path/to/<stem>_result.json --warp path/to/<stem>_warp.jpg

Usage (as module):
  from demo_overlay import draw_demo_composite
  composite = draw_demo_composite(warped_img, perc, plan_r)
  cv2.imwrite("out.jpg", composite)
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

# Cloth bounds color / style
C_CLOTH_BOUND = (40, 200, 80)   # dashed green rectangle for playable region

BASE = Path(__file__).parent.parent

# ── Dimensions ────────────────────────────────────────────────────────────────
WARP_W, WARP_H = 900, 450   # native warp space
HEADER_H = 38               # height of the title strip above each panel
PANEL_GAP = 3               # black gap between panels

# ── Colors (BGR) ──────────────────────────────────────────────────────────────
C_WHITE      = (255, 255, 255)
C_BLACK      = (0,   0,   0)
C_CUE_RING   = (100, 200, 255)   # blue ring around cue ball
C_OB         = (60,  160, 255)   # object ball fill (warm blue)
C_OB_RING    = (30,   80, 220)   # object ball ring (darker)
C_GHOST      = (200, 200, 200)   # ghost ball dashed circle
C_PATH_CUE   = (255, 230, 100)   # yellow — cue→ghost path
C_PATH_OB    = (80,  200, 255)   # cyan  — OB→pocket path
C_POCKET_HI  = (50,  255, 150)   # selected pocket (bright green)
C_POCKET_DIM = (40,   80,  55)   # unselected pocket fill
C_POCKET_RIM = (60,  130,  80)   # unselected pocket rim
C_BADGE_BG   = (30,   30,  30)   # badge background
C_BADGE_WARN = (50,  130, 255)   # warning badge text (orange-ish)
C_BADGE_OK   = (100, 200, 100)   # ok / info badge
C_CONF_HIGH  = (80,  220, 100)   # conf ≥ 0.80
C_CONF_MED   = (80,  200, 240)   # conf 0.50–0.79
C_CONF_LOW   = (60,  100, 240)   # conf < 0.50
C_HEADER_BG  = (18,  18,  18)
C_HEADER_TXT = (210, 210, 210)
C_HEADER_SUB = (90,  90,  90)
C_CAND_PATH  = (80,  80,  40)    # dim yellow for non-selected candidates
C_CAND_OB    = (30,  80, 100)    # dim cyan for non-selected OB paths

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

POCKETS = {
    "TL": ( 22,  22), "TR": (878,  22),
    "ML": (  0, 225), "MR": (900, 225),
    "BL": ( 22, 428), "BR": (878, 428),
}

_WARN_LABELS = {
    "low_corner_confidence":    "LOW CORNER CONF",
    "low_cue_confidence":       "LOW CUE CONF",
    "high_ball_count":          "HIGH BALL COUNT",
    "extreme_thin_cut":         "EXTREME CUT",
    "long_shot":                "LONG SHOT",
    "cue_object_overlap":       "CUE / OB OVERLAP",
    "near_rail_cue_selected":   "NEAR RAIL",
    "near_rail_ghost_selected": "NEAR RAIL (ghost)",
}


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _dashed_circle(img, cx, cy, r, color, n=24, t=2):
    for i in range(0, n, 2):
        a0 = 2 * math.pi * i / n
        a1 = 2 * math.pi * (i + 1) / n
        cv2.line(img,
                 (int(cx + r * math.cos(a0)), int(cy + r * math.sin(a0))),
                 (int(cx + r * math.cos(a1)), int(cy + r * math.sin(a1))),
                 color, t, cv2.LINE_AA)


def _arrow(img, p1, p2, color, t=2):
    cv2.line(img, p1, p2, color, t, cv2.LINE_AA)
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    ln = math.hypot(dx, dy)
    if ln < 5:
        return
    ux, uy = dx / ln, dy / ln
    tip = min(ln * 0.08, 18)
    ang = math.radians(25)
    for s in (-1, 1):
        ax = p2[0] - tip * (ux * math.cos(ang) + s * uy * math.sin(ang))
        ay = p2[1] - tip * (uy * math.cos(ang) - s * ux * math.sin(ang))
        cv2.line(img, p2, (int(ax), int(ay)), color, t, cv2.LINE_AA)


def _badge(img, x, y, text, fg=C_BADGE_WARN, scale=0.44):
    (tw, th), bl = cv2.getTextSize(text, FONT, scale, 1)
    px, py = 7, 4
    x2 = x + tw + 2 * px
    y2 = y + th + 2 * py + bl
    cv2.rectangle(img, (x, y), (x2, y2), C_BADGE_BG, -1)
    cv2.rectangle(img, (x, y), (x2, y2), fg, 1)
    cv2.putText(img, text, (x + px, y2 - py - bl // 2), FONT, scale, fg, 1, cv2.LINE_AA)
    return y2 + 5


def _warn_label(w: str) -> str:
    for k, v in _WARN_LABELS.items():
        if w.startswith(k):
            return f"! {v}"
    if "cue_recovered_by_threshold" in w:
        return "! CUE RECOVERED (thresh)"
    if "cue_recovered_by_appearance" in w:
        return "! CUE RECOVERED (appear)"
    if "multiple_cue" in w:
        return "! MULTI-CUE"
    return f"! {w.upper()[:22]}"


def _conf_color(conf: float):
    if conf >= 0.80:
        return C_CONF_HIGH
    if conf >= 0.50:
        return C_CONF_MED
    return C_CONF_LOW


def _ball_radii(perc: dict, ob_id=None):
    """Return (cue_r, ob_r) from perception balls list."""
    cue_r = ob_r = 15
    for b in perc.get("balls", []):
        r = max(8, min(28, int(b.get("r", 15))))
        if b["type"] == "cue_ball":
            cue_r = r
        if ob_id is not None and b["id"] == ob_id:
            ob_r = r
    return cue_r, ob_r


def _draw_all_pockets(img, selected_pocket=None):
    """Draw all 6 pockets; highlight the selected one."""
    for name, (px, py) in POCKETS.items():
        if name == selected_pocket:
            cv2.circle(img, (px, py), 22, C_POCKET_HI, -1)
            cv2.circle(img, (px, py), 22, C_WHITE, 2)
            (tw, _), _ = cv2.getTextSize(name, FONT_BOLD, 0.50, 1)
            cv2.putText(img, name, (px - tw // 2, py + 5),
                        FONT_BOLD, 0.50, C_BLACK, 1, cv2.LINE_AA)
        else:
            cv2.circle(img, (px, py), 16, C_POCKET_DIM, -1)
            cv2.circle(img, (px, py), 16, C_POCKET_RIM, 1)
            (tw, _), _ = cv2.getTextSize(name, FONT, 0.36, 1)
            cv2.putText(img, name, (px - tw // 2, py + 5),
                        FONT, 0.36, (90, 160, 100), 1, cv2.LINE_AA)


def _draw_ball(img, x, y, r, is_cue: bool, label: str):
    if is_cue:
        cv2.circle(img, (x, y), r, C_WHITE, -1)
        cv2.circle(img, (x, y), r + 2, C_CUE_RING, 2)
    else:
        cv2.circle(img, (x, y), r, C_OB, -1)
        cv2.circle(img, (x, y), r + 2, C_OB_RING, 2)
    (tw, _), _ = cv2.getTextSize(label, FONT_BOLD, 0.40, 1)
    cv2.putText(img, label, (x - tw // 2, y + 5),
                FONT_BOLD, 0.40, (255, 255, 255) if not is_cue else (60, 60, 60),
                1, cv2.LINE_AA)


def _draw_cloth_bound(img: np.ndarray, perc: dict):
    """Draw the refined playable-region boundary (dashed green) from cloth_bounds in perc."""
    bounds = (perc.get("table") or {}).get("cloth_bounds")
    if not bounds:
        return
    try:
        from table_refinement import _dashed_rect
        x0, y0, x1, y1 = int(bounds[0]), int(bounds[1]), int(bounds[2]), int(bounds[3])
        _dashed_rect(img, x0, y0, x1, y1, C_CLOTH_BOUND, thickness=2)
    except Exception:
        pass  # table_refinement not available — skip silently


def _draw_rejected_balls(img: np.ndarray, perc: dict):
    """Draw ⊗ marker at positions of cloth-rejected balls (from perc['rejected_balls'])."""
    for b in perc.get("rejected_balls", []):
        x, y = int(b["x"]), int(b["y"])
        r = max(8, min(28, int(b.get("r", 15))))
        cv2.circle(img, (x, y), r, (50, 50, 200), 2)
        d = int(r * 0.7)
        cv2.line(img, (x - d, y - d), (x + d, y + d), (50, 50, 200), 2, cv2.LINE_AA)
        cv2.line(img, (x + d, y - d), (x - d, y + d), (50, 50, 200), 2, cv2.LINE_AA)


def _panel_header(title: str, subtitle: str) -> np.ndarray:
    hdr = np.full((HEADER_H, WARP_W, 3), C_HEADER_BG, dtype=np.uint8)
    # Left accent line
    cv2.rectangle(hdr, (0, 0), (3, HEADER_H), (80, 130, 220), -1)
    cv2.putText(hdr, title, (12, 26), FONT_BOLD, 0.62, C_HEADER_TXT, 1, cv2.LINE_AA)
    if subtitle:
        (sw, _), _ = cv2.getTextSize(subtitle, FONT, 0.38, 1)
        cv2.putText(hdr, subtitle, (WARP_W - sw - 12, 26), FONT, 0.38, C_HEADER_SUB, 1, cv2.LINE_AA)
    return hdr


def _dark_overlay_card(img, reason: str, detail: str = ""):
    """Darken img and draw a centered NOT-READY card in place."""
    dark = np.zeros_like(img)
    cv2.addWeighted(dark, 0.55, img, 0.45, 0, img)
    cw, ch = 420, 140
    cx = (WARP_W - cw) // 2
    cy = (WARP_H - ch) // 2
    card = img.copy()
    cv2.rectangle(card, (cx, cy), (cx + cw, cy + ch), (20, 20, 20), -1)
    cv2.addWeighted(card, 0.85, img, 0.15, 0, img)
    cv2.rectangle(img, (cx, cy), (cx + cw, cy + ch), (60, 60, 90), 2)

    hdr = "NO RECOMMENDATION"
    (hw, _), _ = cv2.getTextSize(hdr, FONT_BOLD, 0.65, 2)
    cv2.putText(img, hdr, (cx + (cw - hw) // 2, cy + 36),
                FONT_BOLD, 0.65, (80, 80, 240), 2, cv2.LINE_AA)
    cv2.line(img, (cx + 20, cy + 46), (cx + cw - 20, cy + 46), (70, 70, 100), 1)

    _REASON_TITLES = {
        "cue_missing":            "NO CUE BALL DETECTED",
        "low_quality_warp":       "LOW QUALITY IMAGE",
        "no_candidates":          "NO VALID SHOT FOUND",
        "no_table_detected":      "TABLE NOT FOUND",
        "not_ready_for_planner":  "NOT READY",
        "too_few_balls":          "TOO FEW BALLS",
    }
    title = _REASON_TITLES.get(reason, reason.upper().replace("_", " "))
    (tw, _), _ = cv2.getTextSize(title, FONT_BOLD, 0.58, 1)
    cv2.putText(img, title, (cx + (cw - tw) // 2, cy + 74),
                FONT_BOLD, 0.58, (200, 200, 255), 1, cv2.LINE_AA)

    if detail:
        for i, line in enumerate(detail.split("\n")):
            (lw, _), _ = cv2.getTextSize(line, FONT, 0.46, 1)
            cv2.putText(img, line, (cx + (cw - lw) // 2, cy + 102 + i * 22),
                        FONT, 0.46, (150, 150, 150), 1, cv2.LINE_AA)


# ── Panel 1: Detection ────────────────────────────────────────────────────────

def draw_detection_panel(warp: np.ndarray, perc: dict) -> np.ndarray:
    """
    Shows all detected balls labeled with their IDs.
    No shot geometry — pure perception output.
    """
    img = warp.copy()
    balls = perc.get("balls", [])
    n_balls = len(balls)
    cue_present = any(b["type"] == "cue_ball" for b in balls)
    corner_conf = (perc.get("table") or {}).get("corner_confidence")
    warns = perc.get("status", {}).get("warnings", [])

    # All 6 pockets (dim — they're context, not the focus here)
    for name, (px, py) in POCKETS.items():
        cv2.circle(img, (px, py), 12, C_POCKET_DIM, -1)
        cv2.circle(img, (px, py), 12, C_POCKET_RIM, 1)

    # Draw every detected ball
    for b in balls:
        x, y = int(b["x"]), int(b["y"])
        r = max(8, min(28, int(b.get("r", 15))))
        is_cue = b["type"] == "cue_ball"
        label = "C" if is_cue else str(b["id"])
        _draw_ball(img, x, y, r, is_cue, label)

        # Confidence tag for each ball (above ball)
        ball_conf = b.get("confidence", None)
        if ball_conf is not None:
            tag = f"{ball_conf:.2f}"
            (tw, _), _ = cv2.getTextSize(tag, FONT, 0.30, 1)
            cv2.putText(img, tag, (x - tw // 2, y - r - 3),
                        FONT, 0.30, (160, 160, 160), 1, cv2.LINE_AA)

    # Cloth boundary (dashed green) + rejected balls (⊗)
    _draw_cloth_bound(img, perc)
    _draw_rejected_balls(img, perc)

    # Top-left info panel
    n_rejected = len(perc.get("rejected_balls", []))
    panel_x, panel_y = 8, 8
    panel_w = 240 if n_rejected else 200
    panel_h = 60 if not n_rejected else 76
    ov = img.copy()
    cv2.rectangle(ov, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(ov, 0.78, img, 0.22, 0, img)

    cue_col = C_CONF_HIGH if cue_present else C_CONF_LOW
    cv2.putText(img, f"BALLS DETECTED: {n_balls}",
                (panel_x + 8, panel_y + 20), FONT_BOLD, 0.52, C_WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, f"CUE BALL: {'FOUND' if cue_present else 'MISSING'}",
                (panel_x + 8, panel_y + 42), FONT, 0.46, cue_col, 1, cv2.LINE_AA)
    if n_rejected:
        cv2.putText(img, f"OUTSIDE CLOTH (rejected): {n_rejected}",
                    (panel_x + 8, panel_y + 62), FONT, 0.42, (80, 80, 220), 1, cv2.LINE_AA)

    # Corner confidence badge (top-right)
    bx = WARP_W - 220
    by = 8
    if corner_conf is not None:
        cc_col = C_CONF_HIGH if corner_conf >= 0.80 else C_CONF_MED if corner_conf >= 0.60 else C_CONF_LOW
        by = _badge(img, bx, by, f"TABLE CONF  {corner_conf:.2f}", fg=cc_col)

    # Perception warning badges
    for w in warns:
        if any(k in w for k in ("low_quality_warp", "no_table", "low_corner")):
            by = _badge(img, bx, by, _warn_label(w))

    return img


# ── Panel 2: Tactical ─────────────────────────────────────────────────────────

def draw_tactical_panel(warp: np.ndarray, perc: dict, plan_r: dict) -> np.ndarray:
    """
    Shows all detected balls, all 6 pockets labeled,
    top-3 candidate paths (dim), and the selected shot prominently.
    """
    img = warp.copy()
    status = plan_r.get("status", "")

    if status != "plan_ready":
        # Resolve reason
        warns = perc.get("status", {}).get("warnings", [])
        if any("low_quality_warp" in w or "no_table" in w for w in warns):
            reason = "low_quality_warp"
        elif not perc.get("status", {}).get("cue_present", False):
            reason = "cue_missing"
        elif status == "no_candidates":
            reason, detail = "no_candidates", _format_rejections(plan_r)
            _dark_overlay_card(img, reason, detail)
            return img
        else:
            reason = "not_ready_for_planner"
        _dark_overlay_card(img, reason)
        return img

    sel = plan_r["selected"]
    top3 = plan_r.get("top3", [sel])
    n_cand = plan_r.get("n_candidates", 0)
    cue_r, ob_r = _ball_radii(perc, sel["ob_id"])

    # All 6 pockets
    _draw_all_pockets(img, selected_pocket=sel["pocket"])

    # Refined playable region boundary
    _draw_cloth_bound(img, perc)

    # Non-selected top-3 candidates (draw behind selected)
    for cand in reversed(top3[1:3]):
        c_xy  = tuple(int(v) for v in cand["cue_xy"])
        o_xy  = tuple(int(v) for v in cand["ob_xy"])
        gh    = tuple(int(v) for v in cand["ghost"])
        p_xy  = tuple(int(v) for v in cand["pocket_xy"])
        cv2.line(img, c_xy, gh, C_CAND_PATH, 1, cv2.LINE_AA)
        cv2.line(img, o_xy, p_xy, C_CAND_OB, 1, cv2.LINE_AA)
        _dashed_circle(img, gh[0], gh[1], ob_r, (70, 70, 70), n=16, t=1)

    # Selected shot — paths
    cue_xy = tuple(int(v) for v in sel["cue_xy"])
    ob_xy  = tuple(int(v) for v in sel["ob_xy"])
    ghost  = tuple(int(v) for v in sel["ghost"])
    cont   = tuple(int(v) for v in sel["contact"])
    pkt_xy = tuple(int(v) for v in sel["pocket_xy"])

    _arrow(img, cue_xy, ghost, C_PATH_CUE, t=2)
    _arrow(img, ob_xy, pkt_xy, C_PATH_OB, t=2)
    _dashed_circle(img, ghost[0], ghost[1], ob_r, C_GHOST, t=2)
    cv2.circle(img, cont, 5, C_PATH_OB, -1)
    cv2.circle(img, cont, 5, C_WHITE, 1)

    # All balls (draw after paths so they're on top)
    for b in perc.get("balls", []):
        x, y = int(b["x"]), int(b["y"])
        r = max(8, min(28, int(b.get("r", 15))))
        _draw_ball(img, x, y, r, b["type"] == "cue_ball",
                   "C" if b["type"] == "cue_ball" else str(b["id"]))

    # Info badges (top-left)
    bx, by = 8, 8
    by = _badge(img, bx, by, f"{n_cand} CANDIDATES", fg=C_BADGE_OK)
    if len(top3) > 1:
        by = _badge(img, bx, by,
                    f"BEST: ob{sel['ob_id']} → {sel['pocket']}  cut {sel['cut_deg']:.0f}°",
                    fg=C_BADGE_OK)

    # Quality + recovery warnings (top-right)
    flags = plan_r.get("quality_flags", [])
    warns = perc.get("status", {}).get("warnings", [])
    all_warns = list(flags) + [w for w in warns if "recovered" in w]
    wx, wy = WARP_W - 240, 8
    for w in all_warns:
        wy = _badge(img, wx, wy, _warn_label(w))

    return img


def _format_rejections(plan_r: dict) -> str:
    rj = plan_r.get("rejections", {})
    parts = [f"{v} {k.replace('_', ' ')}" for k, v in rj.items() if v]
    return "  ".join(parts) if parts else ""


# ── Panel 3: Recommended Shot ─────────────────────────────────────────────────

def draw_shot_panel(warp: np.ndarray, perc: dict, plan_r: dict) -> np.ndarray:
    """
    Clean final-output view: selected shot with larger visual elements,
    prominent confidence badge, and shot details in a bottom panel.
    """
    img = warp.copy()
    status = plan_r.get("status", "")

    if status != "plan_ready":
        warns = perc.get("status", {}).get("warnings", [])
        if any("low_quality_warp" in w or "no_table" in w for w in warns):
            reason = "low_quality_warp"
        elif not perc.get("status", {}).get("cue_present", False):
            reason = "cue_missing"
        elif status == "no_candidates":
            reason, detail = "no_candidates", _format_rejections(plan_r)
            _dark_overlay_card(img, reason, detail)
            return img
        else:
            reason = "not_ready_for_planner"
        _dark_overlay_card(img, reason)
        return img

    sel    = plan_r["selected"]
    conf   = plan_r.get("confidence", 1.0)
    flags  = plan_r.get("quality_flags", [])
    warns  = perc.get("status", {}).get("warnings", [])
    pocket = sel["pocket"]
    cut    = sel["cut_deg"]

    cue_r, ob_r = _ball_radii(perc, sel["ob_id"])
    cue_xy = tuple(int(v) for v in sel["cue_xy"])
    ob_xy  = tuple(int(v) for v in sel["ob_xy"])
    ghost  = tuple(int(v) for v in sel["ghost"])
    cont   = tuple(int(v) for v in sel["contact"])
    pkt_xy = tuple(int(v) for v in sel["pocket_xy"])

    # All 6 pockets (highlight selected)
    _draw_all_pockets(img, selected_pocket=pocket)

    # All other balls (dim — not in the shot)
    for b in perc.get("balls", []):
        if b["type"] == "object_ball" and b["id"] != sel["ob_id"]:
            x, y = int(b["x"]), int(b["y"])
            r = max(8, min(28, int(b.get("r", 15))))
            cv2.circle(img, (x, y), r, (40, 110, 180), -1)
            cv2.circle(img, (x, y), r + 1, (25, 60, 130), 1)

    # Shot paths — thicker for readability
    _arrow(img, cue_xy, ghost, C_PATH_CUE, t=3)
    _arrow(img, ob_xy, pkt_xy, C_PATH_OB, t=3)

    # Ghost ball
    _dashed_circle(img, ghost[0], ghost[1], ob_r, C_GHOST, t=2)
    cv2.circle(img, cont, 6, C_PATH_OB, -1)
    cv2.circle(img, cont, 6, C_WHITE, 1)

    # Target OB — highlighted
    cv2.circle(img, ob_xy, ob_r, C_OB, -1)
    cv2.circle(img, ob_xy, ob_r + 3, C_OB_RING, 2)
    (tw, _), _ = cv2.getTextSize(str(sel["ob_id"]), FONT_BOLD, 0.45, 1)
    cv2.putText(img, str(sel["ob_id"]), (ob_xy[0] - tw // 2, ob_xy[1] + 6),
                FONT_BOLD, 0.45, C_WHITE, 1, cv2.LINE_AA)

    # Cue ball
    cv2.circle(img, cue_xy, cue_r, C_WHITE, -1)
    cv2.circle(img, cue_xy, cue_r + 3, C_CUE_RING, 2)
    cv2.putText(img, "C", (cue_xy[0] - 5, cue_xy[1] + 6),
                FONT_BOLD, 0.45, (60, 60, 60), 1, cv2.LINE_AA)

    # ── Bottom info panel ──────────────────────────────────────────────────────
    p_x, p_y = 8, WARP_H - 78
    p_w, p_h = 310, 70
    ov = img.copy()
    cv2.rectangle(ov, (p_x, p_y), (p_x + p_w, p_y + p_h), (18, 18, 18), -1)
    cv2.addWeighted(ov, 0.82, img, 0.18, 0, img)
    cv2.rectangle(img, (p_x, p_y), (p_x + p_w, p_y + p_h), (55, 55, 75), 1)

    cc = _conf_color(conf)
    conf_pct = f"{conf:.0%}"
    cv2.putText(img, f"CONFIDENCE  {conf_pct}",
                (p_x + 10, p_y + 24), FONT_BOLD, 0.60, cc, 1, cv2.LINE_AA)
    cv2.putText(img, f"POCKET {pocket}   CUT {cut:.1f}deg",
                (p_x + 10, p_y + 46), FONT, 0.50, C_WHITE, 1, cv2.LINE_AA)
    dist = int(sel.get("cue_dist", 0) + sel.get("ob_dist", 0))
    cv2.putText(img, f"DISTANCE  {dist}px",
                (p_x + 10, p_y + 64), FONT, 0.44, (160, 160, 160), 1, cv2.LINE_AA)

    # ── Warning / quality badges (top-right) ───────────────────────────────────
    all_flags = list(flags) + [w for w in warns if "recovered" in w]
    wx, wy = WARP_W - 240, 8
    for w in all_flags:
        wy = _badge(img, wx, wy, _warn_label(w))

    return img


# ── Composite builder ─────────────────────────────────────────────────────────

def draw_demo_composite(
    warp: np.ndarray,
    perc: dict,
    plan_r: dict,
    stem: str = "",
) -> np.ndarray:
    """
    Build and return the 3-panel vertical composite image.

    Args:
        warp:   900×450 warped table image (from stress_test.perceive)
        perc:   perception dict (from stress_test.perceive)
        plan_r: plan dict (from stress_test.plan)
        stem:   image stem for the subtitle line (optional)

    Returns:
        numpy BGR image of shape (3*(HEADER_H+WARP_H)+gaps, WARP_W, 3)
    """
    if warp is None:
        warp = np.zeros((WARP_H, WARP_W, 3), dtype=np.uint8)

    panels_cfg = [
        ("01  DETECTION",
         f"all detected balls{' — ' + stem if stem else ''}",
         draw_detection_panel(warp, perc)),
        ("02  TACTICAL",
         "shot candidates & geometry",
         draw_tactical_panel(warp, perc, plan_r)),
        ("03  RECOMMENDED SHOT",
         "best shot selected by planner",
         draw_shot_panel(warp, perc, plan_r)),
    ]

    row_h = HEADER_H + WARP_H
    total_h = row_h * 3 + PANEL_GAP * 2
    canvas = np.zeros((total_h, WARP_W, 3), dtype=np.uint8)

    y = 0
    for i, (title, subtitle, panel_img) in enumerate(panels_cfg):
        hdr = _panel_header(title, subtitle)
        canvas[y: y + HEADER_H] = hdr
        y += HEADER_H
        h = min(panel_img.shape[0], WARP_H)
        w = min(panel_img.shape[1], WARP_W)
        canvas[y: y + h, :w] = panel_img[:h, :w]
        y += WARP_H
        if i < 2:
            y += PANEL_GAP  # gap between panels (stays black)

    return canvas


# ── Standalone entry point ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Billiards AI — demo composite overlay")
    parser.add_argument("--result", required=True,
                        help="Path to <stem>_result.json from run_full_pipeline.py")
    parser.add_argument("--warp", default=None,
                        help="Path to warped table image (optional; embedded in result.json dir)")
    parser.add_argument("--out", default=None,
                        help="Output path (default: <stem>_demo.jpg next to result)")
    args = parser.parse_args()

    result_path = Path(args.result)
    if not result_path.exists():
        print(f"ERROR: result JSON not found: {result_path}", file=sys.stderr)
        sys.exit(1)

    result = json.loads(result_path.read_text())
    perc   = result["perception"]
    plan_r = result["plan"]
    stem   = result.get("stem", result_path.stem.replace("_result", ""))

    # Try to load warp image
    warp_img = None
    if args.warp:
        warp_img = cv2.imread(args.warp)
    else:
        # Look for warp next to result
        for name in [f"{stem}_warp.jpg", f"{stem}_warp.png",
                     f"{stem}_warped.jpg", f"{stem}_warped.png"]:
            p = result_path.parent / name
            if p.exists():
                warp_img = cv2.imread(str(p))
                break
    if warp_img is None:
        print("WARNING: warp image not found — using black canvas", file=sys.stderr)

    composite = draw_demo_composite(warp_img, perc, plan_r, stem=stem)

    out_path = Path(args.out) if args.out else result_path.parent / f"{stem}_demo.jpg"
    cv2.imwrite(str(out_path), composite, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Demo composite → {out_path}  ({composite.shape[1]}×{composite.shape[0]}px)")


if __name__ == "__main__":
    main()
