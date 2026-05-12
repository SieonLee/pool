"""
Single-image MVP pipeline runner.

Runs the full stack on one photo and writes the overlay + JSON to --out.

Usage:
  python scripts/run_mvp.py --image picture/pool_real_023.jpg --out review/mvp_single/

Required checkpoints (auto-located from project root):
  models/checkpoints/table_corners_mvp_v1.pt
  models/checkpoints/ball_yolo_v1.pt

Output:
  <out>/<stem>_mvp.jpg   — annotated overlay image
  <out>/<stem>_mvp.json  — structured result (plan or reason)
  <out>/<stem>_warp.jpg  — warped table (debug)
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

CORNER_CKPT = BASE / "models" / "checkpoints" / "table_corners_mvp_v1.pt"
BALL_CKPT   = BASE / "models" / "checkpoints" / "ball_yolo_v1.pt"

WARP_W, WARP_H   = 900, 450
BALL_CONF_THRESH  = 0.25
CUE_FALLBACK_CONF = 0.10
MAX_BALLS         = 16
BALL_R            = 15
MAX_CUT_DEG       = 70

CUE_RECOVER_MIN_BRIGHT = 140
CUE_RECOVER_MAX_SAT    = 120
CUE_RECOVER_MIN_ASPECT = 0.32

POCKETS = {
    "TL": ( 22,  22), "TR": (878,  22),
    "ML": (  0, 225), "MR": (900, 225),
    "BL": ( 22, 428), "BR": (878, 428),
}

LOW_QUALITY_WARP_STEMS = {
    "pool_real_002": "source_resolution_too_low",
    "pool_real_010": "source_resolution_too_low",
}

# ── Visual constants (BGR) ─────────────────────────────────────────────────────
C_WHITE    = (255, 255, 255)
C_BLACK    = (0,   0,   0)
C_CUE_RING = (100, 200, 255)
C_OB       = (60,  160, 255)
C_OB_RING  = (30,   80, 220)
C_GHOST    = (200, 200, 200)
C_PATH_CUE = (255, 230, 100)
C_PATH_OB  = (80,  200, 255)
C_POCKET   = (50,  255, 150)
C_BADGE_BG = (30,   30,  30)
C_BADGE_W  = (80,  160, 255)
FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_B     = cv2.FONT_HERSHEY_DUPLEX


# ─────────────────────────────────────────────────────────────────────────────
# Perception helpers (inline — no dependency on perceive.py module)
# ─────────────────────────────────────────────────────────────────────────────

def get_corners(corner_model, img):
    results = corner_model(img, verbose=False)
    if not results or results[0].keypoints is None:
        return None, None
    kps = results[0].keypoints
    if len(kps) == 0:
        return None, None
    if kps.conf is not None:
        best = int(kps.conf.mean(dim=1).argmax())
        conf = float(kps.conf[best].mean().item())
    else:
        best, conf = 0, None
    return kps.xy[best].cpu().numpy(), conf


def warp_image(img, corners):
    dst = np.float32([[0, 0], [WARP_W, 0], [WARP_W, WARP_H], [0, WARP_H]])
    M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (WARP_W, WARP_H)), M


def run_yolo(model, warped, conf):
    results = model(warped, verbose=False, conf=conf)
    dets = []
    if results and results[0].boxes is not None:
        for i in range(len(results[0].boxes)):
            b = results[0].boxes
            dets.append((int(b.cls[i].item()), float(b.conf[i].item()),
                         *b.xyxy[i].cpu().numpy().tolist()))
    return dets


def patch_appearance(warped, x1, y1, x2, y2):
    px1, py1 = max(0, int(x1)), max(0, int(y1))
    px2, py2 = min(WARP_W, int(x2)), min(WARP_H, int(y2))
    patch = warped[py1:py2, px1:px2]
    if patch.size == 0:
        return 0.0, 255.0, 0.0
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    bright = float(hsv[:, :, 2].mean())
    sat    = float(hsv[:, :, 1].mean())
    w, h   = px2 - px1, py2 - py1
    aspect = min(w, h) / max(w, h) if max(w, h) > 0 else 0.0
    return bright, sat, aspect


def perceive(corner_model, ball_model, img_path: Path):
    """Full perception pass. Returns (perc_dict, warped_img)."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None, None

    stem = img_path.stem

    # Low-quality gate
    lq_reason = LOW_QUALITY_WARP_STEMS.get(stem)
    if lq_reason:
        corners, corner_conf = get_corners(corner_model, img)
        warped = None
        if corners is not None:
            warped, _ = warp_image(img, corners)
        return {
            "image": img_path.name,
            "status": {
                "cue_present": False, "ball_count": 0,
                "cue_ball_count": 0, "object_ball_count": 0,
                "ready_for_planner": False,
                "warp_quality": "low_quality_warp",
                "warnings": ["low_quality_warp", lq_reason],
            }
        }, warped

    corners, corner_conf = get_corners(corner_model, img)
    if corners is None:
        return {"image": img_path.name, "status": {
            "cue_present": False, "ball_count": 0,
            "ready_for_planner": False, "warnings": ["no_table_detected"]}}, None

    warped, _ = warp_image(img, corners)

    # Primary ball detection
    raw = run_yolo(ball_model, warped, BALL_CONF_THRESH)
    warnings = []

    cue_dets = [d for d in raw if d[0] == 0]
    obj_dets = [d for d in raw if d[0] == 1]

    if len(cue_dets) > 1:
        warnings.append(f"multiple_cue_detections:{len(cue_dets)}_kept_highest_conf")
        cue_dets = [max(cue_dets, key=lambda d: d[1])]

    obj_dets = sorted(obj_dets, key=lambda d: -d[1])
    combined = cue_dets + obj_dets
    if len(combined) > MAX_BALLS:
        warnings.append(f"capped_at_{MAX_BALLS}")
        combined = combined[:MAX_BALLS]

    # Pass 2: lower-threshold cue
    if not cue_dets:
        fb = run_yolo(ball_model, warped, CUE_FALLBACK_CONF)
        fb_cue = sorted([d for d in fb if d[0] == 0], key=lambda d: -d[1])
        if fb_cue:
            combined.insert(0, fb_cue[0])
            warnings.append(f"cue_recovered_by_threshold(conf={fb_cue[0][1]:.3f})")
            cue_dets = [fb_cue[0]]

    # Pass 3: appearance recovery
    if not cue_dets and obj_dets:
        candidates = []
        for det in obj_dets:
            _, cf, x1, y1, x2, y2 = det
            bright, sat, aspect = patch_appearance(warped, x1, y1, x2, y2)
            if (bright >= CUE_RECOVER_MIN_BRIGHT and
                    sat <= CUE_RECOVER_MAX_SAT and
                    aspect >= CUE_RECOVER_MIN_ASPECT):
                candidates.append((bright - sat, bright, sat, det))
        if candidates:
            _, bright, sat, best = max(candidates, key=lambda x: x[0])
            promoted_x = (best[2] + best[4]) / 2
            promoted_y = (best[3] + best[5]) / 2
            combined = [d for d in combined
                        if not (d[0] == 1
                                and abs((d[2]+d[4])/2 - promoted_x) < 5
                                and abs((d[3]+d[5])/2 - promoted_y) < 5)]
            combined.insert(0, (0, best[1], best[2], best[3], best[4], best[5]))
            cue_dets = [combined[0]]
            warnings.append(f"cue_recovered_by_appearance(bright={bright:.1f},sat={sat:.1f})")

    def det_to_ball(idx, cls, cf, x1, y1, x2, y2):
        cx, cy = (x1+x2)/2, (y1+y2)/2
        r = max(x2-x1, y2-y1) / 2
        return {"id": idx, "type": "cue_ball" if cls == 0 else "object_ball",
                "x": round(cx, 1), "y": round(cy, 1), "r": round(r, 1),
                "confidence": round(cf, 3)}

    balls = [det_to_ball(i, *d) for i, d in enumerate(combined)]
    cue_balls = [b for b in balls if b["type"] == "cue_ball"]
    obj_balls  = [b for b in balls if b["type"] == "object_ball"]
    cue_present = len(cue_balls) == 1

    if not cue_present:
        warnings.append("cue_missing")
    if not obj_balls:
        warnings.append("no_object_balls")

    ready = cue_present and len(balls) >= 2 and len(balls) <= MAX_BALLS

    return {
        "image": img_path.name,
        "table": {
            "corners": corners.tolist(),
            "warp_size": [WARP_W, WARP_H],
            "corner_confidence": round(corner_conf, 3) if corner_conf else None,
        },
        "balls": balls,
        "status": {
            "cue_present": cue_present,
            "ball_count": len(balls),
            "cue_ball_count": len(cue_balls),
            "object_ball_count": len(obj_balls),
            "ready_for_planner": ready,
            "warnings": warnings,
        },
    }, warped


# ─────────────────────────────────────────────────────────────────────────────
# Planner (inline — geometry only, no external dependency)
# ─────────────────────────────────────────────────────────────────────────────

def normalize(v):
    n = math.hypot(*v)
    return (v[0]/n, v[1]/n) if n > 1e-9 else (0.0, 0.0)


def path_blocked(seg_start, seg_end, balls, exclude_ids, ball_r=BALL_R):
    sx, sy = seg_start
    ex, ey = seg_end
    dx, dy = ex - sx, ey - sy
    seg_len = math.hypot(dx, dy)
    if seg_len < 1:
        return False
    t_min = ball_r / seg_len
    t_max = 1.0 - ball_r / seg_len
    for b in balls:
        if b["id"] in exclude_ids:
            continue
        bx, by = b["x"], b["y"]
        t = ((bx - sx)*dx + (by - sy)*dy) / (seg_len * seg_len)
        t = max(t_min, min(t_max, t))
        nx = sx + t*dx - bx
        ny = sy + t*dy - by
        if math.hypot(nx, ny) < ball_r * 2:
            return True
    return False


def plan(perc: dict) -> dict:
    st = perc.get("status", {})
    if not st.get("ready_for_planner"):
        return {"status": "not_ready_for_planner"}

    balls = perc["balls"]
    cue = next(b for b in balls if b["type"] == "cue_ball")
    obs = [b for b in balls if b["type"] == "object_ball"]
    cx, cy = cue["x"], cue["y"]

    candidates = []
    for ob in obs:
        ox, oy = ob["x"], ob["y"]
        for pname, (px, py) in POCKETS.items():
            # Ghost ball: cue-radius behind OB on pocket→OB line
            dpx, dpy = ox - px, oy - py
            dn = math.hypot(dpx, dpy)
            if dn < 1:
                continue
            ux, uy = dpx/dn, dpy/dn
            r_ob = max(ob.get("r", BALL_R), BALL_R)
            gx = ox + ux * r_ob * 2
            gy = oy + uy * r_ob * 2

            # Ghost in bounds
            m = 5
            if not (m <= gx <= WARP_W-m and m <= gy <= WARP_H-m):
                continue

            # Cut angle
            v_cue = normalize((gx-cx, gy-cy))
            v_ob  = normalize((px-ox, py-oy))
            dot   = max(-1.0, min(1.0, v_cue[0]*v_ob[0] + v_cue[1]*v_ob[1]))
            cut   = math.degrees(math.acos(dot))
            if cut > MAX_CUT_DEG:
                continue

            # Contact point
            ux2, uy2 = normalize((ox-gx, oy-gy))
            contact = (gx + ux2*r_ob, gy + uy2*r_ob)

            # Blocking checks
            if path_blocked((cx, cy), (gx, gy), balls,
                            exclude_ids={cue["id"]}):
                continue
            if path_blocked((ox, oy), (px, py), balls,
                            exclude_ids={cue["id"], ob["id"]}):
                continue

            cue_dist = math.hypot(gx-cx, gy-cy)
            ob_dist  = math.hypot(px-ox, py-oy)
            score    = cut + 0.02 * (cue_dist + ob_dist)

            candidates.append({
                "ob_id": ob["id"], "pocket": pname,
                "cut_deg": round(cut, 1),
                "cue_dist": round(cue_dist, 1),
                "ob_dist": round(ob_dist, 1),
                "total_dist": round(cue_dist + ob_dist, 1),
                "score": round(score, 2),
                "ghost": [round(gx, 1), round(gy, 1)],
                "contact": [round(contact[0], 1), round(contact[1], 1)],
                "pocket_xy": [px, py],
                "cue_xy": [round(cx, 1), round(cy, 1)],
                "ob_xy": [round(ox, 1), round(oy, 1)],
            })

    if not candidates:
        return {"status": "no_candidates"}

    candidates.sort(key=lambda c: c["score"])
    selected = candidates[0]
    top3 = candidates[:3]

    # Quality flags
    quality_flags = []
    corner_conf = perc.get("table", {}).get("corner_confidence") or 1.0
    cue_conf    = cue.get("confidence", 1.0)
    if corner_conf < 0.7:
        quality_flags.append("low_corner_confidence")
    if cue_conf < 0.40:
        quality_flags.append("low_cue_confidence")
    if len(balls) > 12:
        quality_flags.append("high_ball_count")
    if selected["cut_deg"] > 55:
        quality_flags.append("extreme_thin_cut")
    if selected["total_dist"] > 700:
        quality_flags.append("long_shot")
    sx, sy = selected["cue_xy"]
    gx, gy = selected["ghost"]
    rail = 40
    if min(sx, WARP_W-sx, sy, WARP_H-sy) < rail:
        quality_flags.append("near_rail_cue_selected")
    if min(gx, WARP_W-gx, gy, WARP_H-gy) < rail:
        quality_flags.append("near_rail_ghost_selected")
    dist_cue_to_ob = min(math.hypot(cue["x"]-b["x"], cue["y"]-b["y"])
                         for b in obs) if obs else 999
    if dist_cue_to_ob < 35:
        quality_flags.append("cue_object_overlap")

    # Confidence score (0–1, penalties applied)
    conf = 1.0
    penalties = {
        "low_corner_confidence": 0.15,
        "low_cue_confidence": 0.20,
        "extreme_thin_cut": 0.40,
        "long_shot": 0.10,
        "near_rail_cue_selected": 0.05,
        "near_rail_ghost_selected": 0.05,
        "cue_object_overlap": 0.20,
        "high_ball_count": 0.05,
    }
    for f in quality_flags:
        conf -= penalties.get(f, 0.0)
    for w in st.get("warnings", []):
        if "cue_recovered" in w:
            conf -= 0.15
    conf = round(max(0.0, min(1.0, conf)), 2)

    return {
        "status": "plan_ready",
        "selected": selected,
        "top3": top3,
        "quality_flags": quality_flags,
        "confidence": conf,
        "n_candidates": len(candidates),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Overlay drawing (inline)
# ─────────────────────────────────────────────────────────────────────────────

def _dashed_circle(img, cx, cy, r, color, n=24, t=2):
    for i in range(0, n, 2):
        a0 = 2*math.pi*i/n; a1 = 2*math.pi*(i+1)/n
        p0 = (int(cx+r*math.cos(a0)), int(cy+r*math.sin(a0)))
        p1 = (int(cx+r*math.cos(a1)), int(cy+r*math.sin(a1)))
        cv2.line(img, p0, p1, color, t)


def _arrow(img, p1, p2, color, t=2):
    cv2.line(img, p1, p2, color, t, cv2.LINE_AA)
    dx, dy = p2[0]-p1[0], p2[1]-p1[1]
    ln = math.hypot(dx, dy)
    if ln < 5:
        return
    ux, uy = dx/ln, dy/ln
    tip = min(ln*0.08, 18)
    ang = math.radians(25)
    for s in (-1, 1):
        ax = p2[0] - tip*(ux*math.cos(ang) + s*uy*math.sin(ang))
        ay = p2[1] - tip*(uy*math.cos(ang) - s*ux*math.sin(ang))
        cv2.line(img, p2, (int(ax), int(ay)), color, t, cv2.LINE_AA)


def _badge(img, x, y, text, fg=C_BADGE_W):
    (tw, th), bl = cv2.getTextSize(text, FONT, 0.42, 1)
    px, py = 6, 4
    cv2.rectangle(img, (x, y), (x+tw+2*px, y+th+2*py+bl), C_BADGE_BG, -1)
    cv2.rectangle(img, (x, y), (x+tw+2*px, y+th+2*py+bl), fg, 1)
    cv2.putText(img, text, (x+px, y+th+py), FONT, 0.42, fg, 1, cv2.LINE_AA)
    return y + th + 2*py + bl + 4


def _warning_label(w: str) -> str:
    MAP = {
        "low_corner_confidence":    "! LOW CORNER CONF",
        "low_cue_confidence":       "! LOW CUE CONF",
        "high_ball_count":          "! HIGH BALL COUNT",
        "extreme_thin_cut":         "! EXTREME CUT",
        "long_shot":                "! LONG SHOT",
        "cue_object_overlap":       "! CUE/OB OVERLAP",
        "near_rail_cue_selected":   "! NEAR RAIL",
        "near_rail_ghost_selected": "! NEAR RAIL (ghost)",
        "capped_at_16":             "! CAPPED >16 BALLS",
    }
    for key, label in MAP.items():
        if w.startswith(key):
            return label
    if "cue_recovered_by_threshold" in w:
        return "! CUE RECOVERED (thresh)"
    if "cue_recovered_by_appearance" in w:
        return "! CUE RECOVERED (appear)"
    if "multiple_cue" in w:
        return "! MULTI-CUE DETECT"
    return f"! {w.upper()[:22]}"


def draw_plan_overlay(warp: np.ndarray, perc: dict, plan_result: dict) -> np.ndarray:
    img = warp.copy()
    sel   = plan_result["selected"]
    conf  = plan_result.get("confidence", 1.0)
    flags = plan_result.get("quality_flags", [])
    warns = perc.get("status", {}).get("warnings", [])

    cue_xy  = tuple(int(v) for v in sel["cue_xy"])
    ob_xy   = tuple(int(v) for v in sel["ob_xy"])
    ghost   = tuple(int(v) for v in sel["ghost"])
    contact = tuple(int(v) for v in sel["contact"])
    pkt_xy  = tuple(int(v) for v in sel["pocket_xy"])
    cut_deg = sel["cut_deg"]
    pocket  = sel["pocket"]

    cue_r = ob_r = 15
    for b in perc.get("balls", []):
        r = max(8, min(28, int(b["r"])))
        if b["type"] == "cue_ball":
            cue_r = r
        if b["id"] == sel["ob_id"]:
            ob_r = r

    # Pocket
    px, py = pkt_xy
    cv2.circle(img, (px, py), 22, C_POCKET, -1)
    cv2.circle(img, (px, py), 22, C_WHITE, 1)
    cv2.putText(img, pocket, (px-10, py+5), FONT_B, 0.5, C_BLACK, 1, cv2.LINE_AA)

    # Paths
    _arrow(img, ob_xy, pkt_xy, C_PATH_OB)
    _arrow(img, cue_xy, ghost, C_PATH_CUE)

    # Ghost
    _dashed_circle(img, ghost[0], ghost[1], ob_r, C_GHOST)
    cv2.circle(img, contact, 4, C_PATH_OB, -1)
    cv2.circle(img, contact, 4, C_WHITE, 1)

    # OB
    cv2.circle(img, ob_xy, ob_r, C_OB, -1)
    cv2.circle(img, ob_xy, ob_r+2, C_OB_RING, 2)
    cv2.putText(img, str(sel["ob_id"]), (ob_xy[0]-5, ob_xy[1]+5),
                FONT_B, 0.4, C_WHITE, 1, cv2.LINE_AA)

    # Cue ball
    cv2.circle(img, cue_xy, cue_r, C_WHITE, -1)
    cv2.circle(img, cue_xy, cue_r+2, C_CUE_RING, 2)
    cv2.putText(img, "C", (cue_xy[0]-5, cue_xy[1]+5),
                FONT_B, 0.4, (60, 60, 60), 1, cv2.LINE_AA)

    # Info panel
    panel = img.copy()
    cv2.rectangle(panel, (8, 8), (218, 74), (30, 30, 30), -1)
    cv2.addWeighted(panel, 0.72, img, 0.28, 0, img)

    cc = (80, 220, 100) if conf >= 0.8 else (80, 200, 240) if conf >= 0.5 else (60, 100, 240)
    cv2.putText(img, f"CONF  {conf:.2f}", (16, 28), FONT_B, 0.55, cc, 1, cv2.LINE_AA)
    cv2.putText(img, f"CUT   {cut_deg:.1f}deg  PKT {pocket}",
                (16, 50), FONT, 0.48, C_WHITE, 1, cv2.LINE_AA)
    dist = int(sel.get("cue_dist", 0) + sel.get("ob_dist", 0))
    cv2.putText(img, f"DIST  {dist}px", (16, 68), FONT, 0.44, (180, 180, 180), 1, cv2.LINE_AA)

    # Badges
    badge_x, badge_y = WARP_W - 230, 8
    all_badges = list(flags) + [w for w in warns if "recovered" in w]
    for w in all_badges:
        badge_y = _badge(img, badge_x, badge_y, _warning_label(w))

    return img


def draw_no_plan_overlay(warp: np.ndarray, reason: str, detail: str = "") -> np.ndarray:
    img = warp.copy()
    dark = np.zeros_like(img)
    cv2.addWeighted(dark, 0.55, img, 0.45, 0, img)

    cw, ch = 420, 160
    cx = (WARP_W - cw) // 2
    cy = (WARP_H - ch) // 2
    card = img.copy()
    cv2.rectangle(card, (cx, cy), (cx+cw, cy+ch), (20, 20, 20), -1)
    cv2.addWeighted(card, 0.85, img, 0.15, 0, img)
    cv2.rectangle(img, (cx, cy), (cx+cw, cy+ch), (60, 60, 80), 2)

    header = "NO RECOMMENDATION"
    (hw, _), _ = cv2.getTextSize(header, FONT_B, 0.7, 2)
    cv2.putText(img, header, (cx+(cw-hw)//2, cy+38),
                FONT_B, 0.7, (80, 80, 240), 2, cv2.LINE_AA)
    cv2.line(img, (cx+20, cy+50), (cx+cw-20, cy+50), (80, 80, 100), 1)

    TITLES = {
        "low_quality_warp":        "LOW QUALITY WARP",
        "cue_missing":             "NO CUE BALL",
        "no_visible_cue":          "NO VISIBLE CUE",
        "cue_at_edge":             "CUE AT EDGE",
        "too_few_balls":           "TOO FEW BALLS",
        "no_candidates":           "NO VALID SHOT",
        "not_ready_for_planner":   "NOT READY",
        "no_table_detected":       "TABLE NOT FOUND",
    }
    title = TITLES.get(reason, reason.upper().replace("_", " "))
    (tw, _), _ = cv2.getTextSize(title, FONT_B, 0.6, 1)
    cv2.putText(img, title, (cx+(cw-tw)//2, cy+78),
                FONT_B, 0.6, (200, 200, 255), 1, cv2.LINE_AA)

    if detail:
        for i, line in enumerate(detail.split("\n")):
            (lw, _), _ = cv2.getTextSize(line, FONT, 0.48, 1)
            cv2.putText(img, line, (cx+(cw-lw)//2, cy+108+i*22),
                        FONT, 0.48, (160, 160, 160), 1, cv2.LINE_AA)

    return img


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def resolve_no_plan_reason(perc: dict) -> tuple[str, str]:
    warns = perc.get("status", {}).get("warnings", [])
    if any("low_quality_warp" in w or "source_resolution" in w for w in warns):
        return "low_quality_warp", "Image resolution too low\nfor reliable detection"
    if "no_table_detected" in warns:
        return "no_table_detected", "Could not find table corners\nin this image"
    if not perc.get("status", {}).get("cue_present", False):
        bc = perc.get("status", {}).get("ball_count", 0)
        if bc <= 1:
            return "too_few_balls", f"Only {bc} ball(s) detected"
        return "cue_missing", "Cue ball not detected\nin this image"
    return "not_ready_for_planner", ""


def run(image_path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    # Load models
    from ultralytics import YOLO
    corner_model = YOLO(str(CORNER_CKPT))
    ball_model   = YOLO(str(BALL_CKPT))

    # 1–3. Perceive
    print(f"  [1/3] Perception ...")
    perc, warped = perceive(corner_model, ball_model, image_path)
    if perc is None:
        print("  ERROR: could not read image")
        return {"error": "unreadable_image"}

    if warped is not None:
        cv2.imwrite(str(out_dir / f"{stem}_warp.jpg"), warped)

    # 4. Plan
    print(f"  [2/3] Planning ...")
    plan_result = plan(perc)

    # 5. Overlay
    print(f"  [3/3] Rendering overlay ...")
    base_img = warped if warped is not None else np.zeros((WARP_H, WARP_W, 3), np.uint8)

    if plan_result["status"] == "plan_ready":
        overlay = draw_plan_overlay(base_img, perc, plan_result)
        sel = plan_result["selected"]
        out_meta = {
            "stem": stem,
            "status": "plan_ready",
            "confidence": plan_result.get("confidence"),
            "selected_shot": {
                "ob_id":    sel["ob_id"],
                "pocket":   sel["pocket"],
                "cut_deg":  sel["cut_deg"],
                "cue_xy":   sel["cue_xy"],
                "ob_xy":    sel["ob_xy"],
                "ghost":    sel["ghost"],
                "contact":  sel["contact"],
                "pocket_xy": sel["pocket_xy"],
                "cue_dist": sel.get("cue_dist"),
                "ob_dist":  sel.get("ob_dist"),
                "score":    sel.get("score"),
            },
            "quality_flags": plan_result.get("quality_flags", []),
            "perception_warnings": perc.get("status", {}).get("warnings", []),
            "ball_count": perc.get("status", {}).get("ball_count"),
            "corner_confidence": perc.get("table", {}).get("corner_confidence"),
        }
    else:
        reason, detail = resolve_no_plan_reason(perc)
        if plan_result["status"] == "no_candidates":
            reason, detail = "no_candidates", "No valid shot found\nwithin cut/block constraints"
        overlay = draw_no_plan_overlay(base_img, reason, detail)
        out_meta = {
            "stem": stem,
            "status": "no_plan",
            "reason": reason,
            "plan_status": plan_result["status"],
            "perception_warnings": perc.get("status", {}).get("warnings", []),
            "ball_count": perc.get("status", {}).get("ball_count"),
        }

    out_jpg  = out_dir / f"{stem}_mvp.jpg"
    out_json = out_dir / f"{stem}_mvp.json"
    cv2.imwrite(str(out_jpg), overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])
    out_json.write_text(json.dumps(out_meta, indent=2))

    return out_meta


def main():
    parser = argparse.ArgumentParser(
        description="Run full MVP pipeline on a single image.")
    parser.add_argument("--image", required=True,
                        help="Path to input photo (JPG or PNG)")
    parser.add_argument("--out", default="review/mvp_single",
                        help="Output directory (default: review/mvp_single/)")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: image not found: {image_path}")
        sys.exit(1)

    out_dir = Path(args.out)
    stem = image_path.stem

    print(f"\nMVP Pipeline")
    print(f"  Input  : {image_path}")
    print(f"  Output : {out_dir}/")
    print(f"  Corner : {CORNER_CKPT.name}")
    print(f"  Balls  : {BALL_CKPT.name}")
    print()

    result = run(image_path, out_dir)

    print()
    if result.get("status") == "plan_ready":
        sel = result["selected_shot"]
        print(f"  RESULT : plan_ready")
        print(f"  Pocket : {sel['pocket']}  Cut: {sel['cut_deg']}deg  "
              f"Dist: {int((sel.get('cue_dist') or 0) + (sel.get('ob_dist') or 0))}px")
        print(f"  Conf   : {result.get('confidence'):.2f}")
        if result.get("quality_flags"):
            print(f"  Flags  : {', '.join(result['quality_flags'])}")
    else:
        print(f"  RESULT : no_plan  [{result.get('reason', result.get('plan_status'))}]")

    print()
    print(f"  Overlay → {out_dir / f'{stem}_mvp.jpg'}")
    print(f"  JSON    → {out_dir / f'{stem}_mvp.json'}")
    if (out_dir / f"{stem}_warp.jpg").exists():
        print(f"  Warp    → {out_dir / f'{stem}_warp.jpg'}")


if __name__ == "__main__":
    main()
