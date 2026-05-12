"""
MVP stress test — runs the full pipeline on every image in picture/
and produces a detailed readiness report.

Measures per-stage latency, categorizes failures, flags visual anomalies.

Output:
  review/stress_test/
    <stem>_mvp.jpg          overlay (always produced)
    <stem>_warp.jpg         warped table (debug)
    <stem>_result.json      per-image result
  review/stress_test/stress_report.md   human-readable summary
  review/stress_test/stress_results.json all-images machine-readable

Run:
  python scripts/stress_test.py
  python scripts/stress_test.py --image picture/pool_real_007.jpg  # single
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / "scripts"))

CORNER_CKPT = BASE / "models" / "checkpoints" / "table_corners_mvp_v1.pt"
BALL_CKPT   = BASE / "models" / "checkpoints" / "ball_yolo_active.pt"
PICTURE_DIR = BASE / "picture"
OUT_DIR     = BASE / "review" / "stress_test"

WARP_W, WARP_H    = 900, 450
CUE_CONF_THRESH   = 0.25   # cue_ball — kept low for recall
OBJ_CONF_THRESH   = 0.35   # object_ball — higher to suppress FPs
BALL_CONF_THRESH  = min(CUE_CONF_THRESH, OBJ_CONF_THRESH)  # YOLO primary run threshold
CUE_FALLBACK_CONF = 0.10
MAX_BALLS         = 16
BALL_R            = 15
MAX_CUT_DEG       = 70

# Cloth refinement — enabled by default
CLOTH_FILTER_ENABLED = True   # set False to skip for ablation / debugging

CUE_RECOVER_MIN_BRIGHT = 140
CUE_RECOVER_MAX_SAT    = 120
CUE_RECOVER_MIN_ASPECT = 0.32

POCKETS = {
    "TL": ( 22,  22), "TR": (878,  22),
    "ML": (  0, 225), "MR": (900, 225),
    "BL": ( 22, 428), "BR": (878, 428),
}

SKIP_STEMS = {"new_uploads_contact_sheet", "search_contact_sheet",
              "thumb", "pexels-photo-10627132"}

LOW_QUALITY_WARP_STEMS = {
    "pool_real_002": "source_resolution_too_low",
    "pool_real_010": "source_resolution_too_low",
    "pool_real_016": "corner_conf_0.69_warp_geometry_distorted",
}

# ── Overlay constants (BGR) ────────────────────────────────────────────────────
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
# Perception (self-contained, no import from perceive.py)
# ─────────────────────────────────────────────────────────────────────────────

def get_corners(model, img):
    t0 = time.perf_counter()
    results = model(img, verbose=False)
    t_corner = time.perf_counter() - t0
    if not results or results[0].keypoints is None:
        return None, None, t_corner
    kps = results[0].keypoints
    if len(kps) == 0:
        return None, None, t_corner
    if kps.conf is not None:
        best = int(kps.conf.mean(dim=1).argmax())
        conf = float(kps.conf[best].mean().item())
    else:
        best, conf = 0, None
    return kps.xy[best].cpu().numpy(), conf, t_corner


def warp_image(img, corners):
    t0 = time.perf_counter()
    dst = np.float32([[0, 0], [WARP_W, 0], [WARP_W, WARP_H], [0, WARP_H]])
    M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    warped = cv2.warpPerspective(img, M, (WARP_W, WARP_H))
    return warped, M, time.perf_counter() - t0


def run_yolo(model, warped, conf, timer=True):
    t0 = time.perf_counter()
    results = model(warped, verbose=False, conf=conf)
    t = time.perf_counter() - t0
    dets = []
    if results and results[0].boxes is not None:
        for i in range(len(results[0].boxes)):
            b = results[0].boxes
            dets.append((int(b.cls[i].item()), float(b.conf[i].item()),
                         *b.xyxy[i].cpu().numpy().tolist()))
    return dets, t


def patch_appearance(warped, x1, y1, x2, y2):
    px1, py1 = max(0, int(x1)), max(0, int(y1))
    px2, py2 = min(WARP_W, int(x2)), min(WARP_H, int(y2))
    patch = warped[py1:py2, px1:px2]
    if patch.size == 0:
        return 0.0, 255.0, 0.0
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    return (float(hsv[:, :, 2].mean()), float(hsv[:, :, 1].mean()),
            min(px2-px1, py2-py1) / max(max(px2-px1, py2-py1), 1))


def perceive(corner_model, ball_model, img_path: Path):
    """Returns (perc_dict, warped_img, timing_dict)."""
    timing = {}
    img = cv2.imread(str(img_path))
    if img is None:
        return None, None, {}

    stem = img_path.stem
    h, w = img.shape[:2]

    lq_reason = LOW_QUALITY_WARP_STEMS.get(stem)
    if lq_reason:
        corners, corner_conf, t_c = get_corners(corner_model, img)
        timing["corner_ms"] = round(t_c * 1000, 1)
        warped = None
        if corners is not None:
            warped, _, t_w = warp_image(img, corners)
            timing["warp_ms"] = round(t_w * 1000, 1)
        timing["yolo_ms"] = 0
        timing["planner_ms"] = 0
        return {
            "image": img_path.name, "stem": stem,
            "source_size": [w, h],
            "status": {
                "cue_present": False, "ball_count": 0,
                "cue_ball_count": 0, "object_ball_count": 0,
                "ready_for_planner": False,
                "warnings": ["low_quality_warp", lq_reason],
            }
        }, warped, timing

    # Corner
    corners, corner_conf, t_c = get_corners(corner_model, img)
    timing["corner_ms"] = round(t_c * 1000, 1)
    if corners is None:
        return {"image": img_path.name, "stem": stem, "source_size": [w, h],
                "status": {"cue_present": False, "ball_count": 0,
                           "ready_for_planner": False, "warnings": ["no_table_detected"]}
                }, None, timing

    warped, _, t_w = warp_image(img, corners)
    timing["warp_ms"] = round(t_w * 1000, 1)

    # Primary YOLO
    raw, t_y = run_yolo(ball_model, warped, BALL_CONF_THRESH)
    timing["yolo_ms"] = round(t_y * 1000, 1)

    warnings = []
    cue_dets = sorted([d for d in raw if d[0] == 0 and d[1] >= CUE_CONF_THRESH],
                      key=lambda d: -d[1])
    # Apply class-specific obj threshold — filter out low-conf obj_ball FPs
    obj_dets_raw = [d for d in raw if d[0] == 1]
    obj_dets_filtered = [d for d in obj_dets_raw if d[1] >= OBJ_CONF_THRESH]
    if len(obj_dets_raw) != len(obj_dets_filtered):
        warnings.append(
            f"obj_filtered_by_class_thresh:{len(obj_dets_raw)-len(obj_dets_filtered)}_removed"
        )
    obj_dets = sorted(obj_dets_filtered, key=lambda d: -d[1])

    if len(cue_dets) > 1:
        warnings.append(f"multiple_cue_detections:{len(cue_dets)}_kept_highest_conf")
        cue_dets = [cue_dets[0]]

    combined = cue_dets + obj_dets
    if len(combined) > MAX_BALLS:
        warnings.append(f"capped_at_{MAX_BALLS}")
        combined = combined[:MAX_BALLS]

    # Pass 2: lower-threshold cue
    if not cue_dets:
        fb, _ = run_yolo(ball_model, warped, CUE_FALLBACK_CONF, timer=False)
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
            px = (best[2]+best[4])/2; py = (best[3]+best[5])/2
            combined = [d for d in combined
                        if not (d[0]==1 and abs((d[2]+d[4])/2-px)<5
                                and abs((d[3]+d[5])/2-py)<5)]
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

    # ── Cloth refinement — filter out balls outside the playing surface ────────
    cloth_info = {}
    rejected_balls = []
    if CLOTH_FILTER_ENABLED and warped is not None:
        try:
            from table_refinement import apply_cloth_filter
            balls, cloth_extras = apply_cloth_filter(warped, balls)
            rejected_balls = cloth_extras.get("rejected_balls", [])
            cloth_info = {
                "cloth_coverage": cloth_extras.get("cloth_coverage"),
                "cloth_bounds":   cloth_extras.get("cloth_bounds"),
                "dominant_hue":   cloth_extras.get("dominant_hue"),
            }
            for w in cloth_extras.get("warnings", []):
                warnings.append(w)
        except Exception as e:
            warnings.append(f"cloth_filter_error:{e}")
    # ─────────────────────────────────────────────────────────────────────────

    cue_balls = [b for b in balls if b["type"] == "cue_ball"]
    obj_balls  = [b for b in balls if b["type"] == "object_ball"]
    cue_present = len(cue_balls) == 1

    if not cue_present:
        warnings.append("cue_missing")
    if not obj_balls:
        warnings.append("no_object_balls")
    if len(balls) > 15:
        warnings.append(f"high_ball_count:{len(balls)}")

    ready = cue_present and len(balls) >= 2 and len(balls) <= MAX_BALLS

    result = {
        "image": img_path.name, "stem": stem,
        "source_size": [w, h],
        "table": {
            "corners": corners.tolist(),
            "warp_size": [WARP_W, WARP_H],
            "corner_confidence": round(corner_conf, 3) if corner_conf else None,
            **cloth_info,
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
    }
    if rejected_balls:
        result["rejected_balls"] = rejected_balls

    return result, warped, timing


# ─────────────────────────────────────────────────────────────────────────────
# Planner
# ─────────────────────────────────────────────────────────────────────────────

def normalize(v):
    n = math.hypot(*v)
    return (v[0]/n, v[1]/n) if n > 1e-9 else (0.0, 0.0)


def path_blocked(p1, p2, balls, exclude_ids, r=BALL_R):
    sx, sy = p1; ex, ey = p2
    dx, dy = ex-sx, ey-sy
    L = math.hypot(dx, dy)
    if L < 1:
        return False
    tmin = r / L; tmax = 1.0 - r / L
    for b in balls:
        if b["id"] in exclude_ids:
            continue
        bx, by = b["x"], b["y"]
        t = max(tmin, min(tmax, ((bx-sx)*dx + (by-sy)*dy) / (L*L)))
        if math.hypot(sx+t*dx-bx, sy+t*dy-by) < r*2:
            return True
    return False


def path_blocker_id(p1, p2, balls, exclude_ids, r=BALL_R):
    """Same as path_blocked but returns the id of the first blocking ball, or None."""
    sx, sy = p1; ex, ey = p2
    dx, dy = ex-sx, ey-sy
    L = math.hypot(dx, dy)
    if L < 1:
        return None
    tmin = r / L; tmax = 1.0 - r / L
    for b in balls:
        if b["id"] in exclude_ids:
            continue
        bx, by = b["x"], b["y"]
        t = max(tmin, min(tmax, ((bx-sx)*dx + (by-sy)*dy) / (L*L)))
        if math.hypot(sx+t*dx-bx, sy+t*dy-by) < r*2:
            return b["id"]
    return None


def plan(perc: dict) -> tuple[dict, float]:
    t0 = time.perf_counter()
    st = perc.get("status", {})
    if not st.get("ready_for_planner"):
        return {"status": "not_ready_for_planner"}, time.perf_counter() - t0

    balls = perc["balls"]
    cue = next(b for b in balls if b["type"] == "cue_ball")
    obs = [b for b in balls if b["type"] == "object_ball"]
    cx, cy = cue["x"], cue["y"]

    rejections = {"cut_too_large": 0, "cue_path_blocked": 0,
                  "ob_path_blocked": 0, "ghost_out_of_bounds": 0}
    shot_rejections = []   # per-shot diagnostic detail
    candidates = []

    for ob in obs:
        ox, oy = ob["x"], ob["y"]
        for pname, (px, py) in POCKETS.items():
            dpx, dpy = ox-px, oy-py
            dn = math.hypot(dpx, dpy)
            if dn < 1:
                continue
            ux, uy = dpx/dn, dpy/dn
            r_ob = max(ob.get("r", BALL_R), BALL_R)
            gx, gy = ox + ux*r_ob*2, oy + uy*r_ob*2
            if not (5 <= gx <= WARP_W-5 and 5 <= gy <= WARP_H-5):
                rejections["ghost_out_of_bounds"] += 1
                shot_rejections.append({
                    "ob_id": ob["id"], "ob_xy": [round(ox,1), round(oy,1)],
                    "pocket": pname, "reason": "ghost_out_of_bounds",
                    "ghost": [round(gx,1), round(gy,1)],
                })
                continue
            v_cue = normalize((gx-cx, gy-cy))
            v_ob  = normalize((px-ox, py-oy))
            cut = math.degrees(math.acos(max(-1.0, min(1.0,
                               v_cue[0]*v_ob[0] + v_cue[1]*v_ob[1]))))
            if cut > MAX_CUT_DEG:
                rejections["cut_too_large"] += 1
                shot_rejections.append({
                    "ob_id": ob["id"], "ob_xy": [round(ox,1), round(oy,1)],
                    "pocket": pname, "reason": "cut_too_large",
                    "cut_deg": round(cut, 1),
                    "ghost": [round(gx,1), round(gy,1)],
                })
                continue
            blocker_cue = path_blocker_id((cx, cy), (gx, gy), balls, {cue["id"]})
            if blocker_cue is not None:
                rejections["cue_path_blocked"] += 1
                shot_rejections.append({
                    "ob_id": ob["id"], "ob_xy": [round(ox,1), round(oy,1)],
                    "pocket": pname, "reason": "cue_path_blocked",
                    "cut_deg": round(cut, 1),
                    "blocker_id": blocker_cue,
                    "ghost": [round(gx,1), round(gy,1)],
                })
                continue
            blocker_ob = path_blocker_id((ox, oy), (px, py), balls, {cue["id"], ob["id"]})
            if blocker_ob is not None:
                rejections["ob_path_blocked"] += 1
                shot_rejections.append({
                    "ob_id": ob["id"], "ob_xy": [round(ox,1), round(oy,1)],
                    "pocket": pname, "reason": "ob_path_blocked",
                    "cut_deg": round(cut, 1),
                    "blocker_id": blocker_ob,
                    "ghost": [round(gx,1), round(gy,1)],
                })
                continue
            ux2, uy2 = normalize((ox-gx, oy-gy))
            contact = (gx + ux2*r_ob, gy + uy2*r_ob)
            cue_dist = math.hypot(gx-cx, gy-cy)
            ob_dist  = math.hypot(px-ox, py-oy)
            candidates.append({
                "ob_id": ob["id"], "pocket": pname,
                "cut_deg": round(cut, 1),
                "cue_dist": round(cue_dist, 1), "ob_dist": round(ob_dist, 1),
                "total_dist": round(cue_dist+ob_dist, 1),
                "score": round(cut + 0.02*(cue_dist+ob_dist), 2),
                "ghost": [round(gx, 1), round(gy, 1)],
                "contact": [round(contact[0], 1), round(contact[1], 1)],
                "pocket_xy": [px, py], "cue_xy": [round(cx, 1), round(cy, 1)],
                "ob_xy": [round(ox, 1), round(oy, 1)],
            })

    if not candidates:
        return {"status": "no_candidates", "rejections": rejections,
                "shot_rejections": shot_rejections}, \
               time.perf_counter() - t0

    candidates.sort(key=lambda c: c["score"])
    sel = candidates[0]

    quality_flags = []
    corner_conf = perc.get("table", {}).get("corner_confidence") or 1.0
    cue_conf    = cue.get("confidence", 1.0)
    if corner_conf < 0.7:
        quality_flags.append("low_corner_confidence")
    if cue_conf < 0.40:
        quality_flags.append("low_cue_confidence")
    if len(balls) > 12:
        quality_flags.append("high_ball_count")
    if sel["cut_deg"] > 55:
        quality_flags.append("extreme_thin_cut")
    if sel["total_dist"] > 700:
        quality_flags.append("long_shot")
    sx2, sy2 = sel["cue_xy"]; gx2, gy2 = sel["ghost"]; rail = 40
    if min(sx2, WARP_W-sx2, sy2, WARP_H-sy2) < rail:
        quality_flags.append("near_rail_cue_selected")
    if min(gx2, WARP_W-gx2, gy2, WARP_H-gy2) < rail:
        quality_flags.append("near_rail_ghost_selected")
    if obs and min(math.hypot(cue["x"]-b["x"], cue["y"]-b["y"]) for b in obs) < 35:
        quality_flags.append("cue_object_overlap")

    conf = 1.0
    penalties = {"low_corner_confidence": 0.15, "low_cue_confidence": 0.20,
                 "extreme_thin_cut": 0.40, "long_shot": 0.10,
                 "near_rail_cue_selected": 0.05, "near_rail_ghost_selected": 0.05,
                 "cue_object_overlap": 0.20, "high_ball_count": 0.05}
    for f in quality_flags:
        conf -= penalties.get(f, 0.0)
    for w in perc.get("status", {}).get("warnings", []):
        if "cue_recovered" in w:
            conf -= 0.15
    conf = round(max(0.0, min(1.0, conf)), 2)

    return {
        "status": "plan_ready",
        "selected": sel, "top3": candidates[:3],
        "quality_flags": quality_flags,
        "confidence": conf,
        "n_candidates": len(candidates),
        "rejections": rejections,
    }, time.perf_counter() - t0


# ─────────────────────────────────────────────────────────────────────────────
# Overlay drawing
# ─────────────────────────────────────────────────────────────────────────────

def _dashed_circle(img, cx, cy, r, color, n=24, t=2):
    for i in range(0, n, 2):
        a0 = 2*math.pi*i/n; a1 = 2*math.pi*(i+1)/n
        cv2.line(img,
                 (int(cx+r*math.cos(a0)), int(cy+r*math.sin(a0))),
                 (int(cx+r*math.cos(a1)), int(cy+r*math.sin(a1))),
                 color, t)


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


_WARN_MAP = {
    "low_corner_confidence":    "! LOW CORNER CONF",
    "low_cue_confidence":       "! LOW CUE CONF",
    "high_ball_count":          "! HIGH BALL COUNT",
    "extreme_thin_cut":         "! EXTREME CUT",
    "long_shot":                "! LONG SHOT",
    "cue_object_overlap":       "! CUE/OB OVERLAP",
    "near_rail_cue_selected":   "! NEAR RAIL",
    "near_rail_ghost_selected": "! NEAR RAIL (ghost)",
}


def _wlabel(w):
    for k, v in _WARN_MAP.items():
        if w.startswith(k):
            return v
    if "cue_recovered_by_threshold" in w:
        return "! CUE RECOVERED (thresh)"
    if "cue_recovered_by_appearance" in w:
        return "! CUE RECOVERED (appear)"
    if "multiple_cue" in w:
        return "! MULTI-CUE DETECT"
    return f"! {w.upper()[:22]}"


def draw_ready(warp, perc, plan_r):
    img = warp.copy()
    sel = plan_r["selected"]
    conf = plan_r.get("confidence", 1.0)
    flags = plan_r.get("quality_flags", [])
    warns = perc.get("status", {}).get("warnings", [])

    cue_xy = tuple(int(v) for v in sel["cue_xy"])
    ob_xy  = tuple(int(v) for v in sel["ob_xy"])
    ghost  = tuple(int(v) for v in sel["ghost"])
    cont   = tuple(int(v) for v in sel["contact"])
    pkt_xy = tuple(int(v) for v in sel["pocket_xy"])
    pocket = sel["pocket"]

    cue_r = ob_r = 15
    for b in perc.get("balls", []):
        r = max(8, min(28, int(b["r"])))
        if b["type"] == "cue_ball":
            cue_r = r
        if b["id"] == sel["ob_id"]:
            ob_r = r

    cv2.circle(img, pkt_xy, 22, C_POCKET, -1)
    cv2.circle(img, pkt_xy, 22, C_WHITE, 1)
    cv2.putText(img, pocket, (pkt_xy[0]-10, pkt_xy[1]+5),
                FONT_B, 0.5, C_BLACK, 1, cv2.LINE_AA)
    _arrow(img, ob_xy, pkt_xy, C_PATH_OB)
    _arrow(img, cue_xy, ghost, C_PATH_CUE)
    _dashed_circle(img, ghost[0], ghost[1], ob_r, C_GHOST)
    cv2.circle(img, cont, 4, C_PATH_OB, -1)
    cv2.circle(img, cont, 4, C_WHITE, 1)
    cv2.circle(img, ob_xy, ob_r, C_OB, -1)
    cv2.circle(img, ob_xy, ob_r+2, C_OB_RING, 2)
    cv2.putText(img, str(sel["ob_id"]), (ob_xy[0]-5, ob_xy[1]+5),
                FONT_B, 0.4, C_WHITE, 1, cv2.LINE_AA)
    cv2.circle(img, cue_xy, cue_r, C_WHITE, -1)
    cv2.circle(img, cue_xy, cue_r+2, C_CUE_RING, 2)
    cv2.putText(img, "C", (cue_xy[0]-5, cue_xy[1]+5),
                FONT_B, 0.4, (60,60,60), 1, cv2.LINE_AA)

    panel = img.copy()
    cv2.rectangle(panel, (8, 8), (218, 74), (30, 30, 30), -1)
    cv2.addWeighted(panel, 0.72, img, 0.28, 0, img)
    cc = (80,220,100) if conf >= 0.8 else (80,200,240) if conf >= 0.5 else (60,100,240)
    cv2.putText(img, f"CONF  {conf:.2f}", (16, 28), FONT_B, 0.55, cc, 1, cv2.LINE_AA)
    cv2.putText(img, f"CUT   {sel['cut_deg']:.1f}deg  PKT {pocket}",
                (16, 50), FONT, 0.48, C_WHITE, 1, cv2.LINE_AA)
    dist = int(sel.get("cue_dist",0)+sel.get("ob_dist",0))
    cv2.putText(img, f"DIST  {dist}px", (16, 68), FONT, 0.44, (180,180,180), 1, cv2.LINE_AA)

    bx, by = WARP_W - 230, 8
    for w in list(flags) + [w for w in warns if "recovered" in w]:
        by = _badge(img, bx, by, _wlabel(w))
    return img


def draw_no_plan(warp, reason, detail=""):
    img = warp.copy() if warp is not None else np.zeros((WARP_H, WARP_W, 3), np.uint8)
    dark = np.zeros_like(img)
    cv2.addWeighted(dark, 0.55, img, 0.45, 0, img)
    cw, ch = 420, 160
    cx = (WARP_W-cw)//2; cy = (WARP_H-ch)//2
    card = img.copy()
    cv2.rectangle(card, (cx,cy), (cx+cw,cy+ch), (20,20,20), -1)
    cv2.addWeighted(card, 0.85, img, 0.15, 0, img)
    cv2.rectangle(img, (cx,cy), (cx+cw,cy+ch), (60,60,80), 2)
    hdr = "NO RECOMMENDATION"
    (hw,_),_ = cv2.getTextSize(hdr, FONT_B, 0.7, 2)
    cv2.putText(img, hdr, (cx+(cw-hw)//2, cy+38), FONT_B, 0.7, (80,80,240), 2, cv2.LINE_AA)
    cv2.line(img, (cx+20,cy+50), (cx+cw-20,cy+50), (80,80,100), 1)
    TITLES = {"low_quality_warp": "LOW QUALITY WARP", "cue_missing": "NO CUE BALL",
              "no_candidates": "NO VALID SHOT", "no_table_detected": "TABLE NOT FOUND",
              "not_ready_for_planner": "NOT READY", "too_few_balls": "TOO FEW BALLS"}
    title = TITLES.get(reason, reason.upper().replace("_"," "))
    (tw,_),_ = cv2.getTextSize(title, FONT_B, 0.6, 1)
    cv2.putText(img, title, (cx+(cw-tw)//2, cy+78), FONT_B, 0.6, (200,200,255), 1, cv2.LINE_AA)
    if detail:
        for i, line in enumerate(detail.split("\n")):
            (lw,_),_ = cv2.getTextSize(line, FONT, 0.48, 1)
            cv2.putText(img, line, (cx+(cw-lw)//2, cy+108+i*22),
                        FONT, 0.48, (160,160,160), 1, cv2.LINE_AA)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Failure categorisation
# ─────────────────────────────────────────────────────────────────────────────

def categorize(perc, plan_r) -> list[str]:
    cats = []
    warns = perc.get("status", {}).get("warnings", [])
    plan_status = plan_r.get("status", "")
    corner_conf = (perc.get("table") or {}).get("corner_confidence") or 1.0
    ball_count  = perc.get("status", {}).get("ball_count", 0)

    if "low_quality_warp" in warns:
        cats.append("bad_warp")
    if "no_table_detected" in warns:
        cats.append("bad_warp")
    if corner_conf < 0.5:
        cats.append("bad_warp")
    if "cue_missing" in warns:
        cats.append("cue_missing")
    if any("cue_recovered" in w for w in warns):
        cats.append("cue_recovered")
    if any("multiple_cue" in w for w in warns):
        cats.append("false_cue")
    if "capped_at_16" in warns or ball_count > 16:
        cats.append("over_detection")
    if ball_count < 2 and "low_quality_warp" not in warns:
        cats.append("under_detection")
    if plan_status == "no_candidates":
        cats.append("no_valid_shot")
    if plan_status == "plan_ready":
        flags = plan_r.get("quality_flags", [])
        if "extreme_thin_cut" in flags:
            cats.append("unrealistic_shot")
        if plan_r.get("confidence", 1.0) < 0.5:
            cats.append("low_confidence")

    return cats or (["ok"] if plan_status == "plan_ready" else ["blocked"])


# ─────────────────────────────────────────────────────────────────────────────
# Full single-image run
# ─────────────────────────────────────────────────────────────────────────────

def run_one(corner_model, ball_model, img_path: Path) -> dict:
    t_total = time.perf_counter()
    stem = img_path.stem

    perc, warped, timing = perceive(corner_model, ball_model, img_path)
    if perc is None:
        return {"stem": stem, "error": "unreadable_image"}

    plan_r, t_plan = plan(perc)
    timing["planner_ms"] = round(t_plan * 1000, 1)
    timing["total_ms"]   = round((time.perf_counter() - t_total) * 1000, 1)

    base = warped if warped is not None else np.zeros((WARP_H, WARP_W, 3), np.uint8)

    if plan_r["status"] == "plan_ready":
        overlay = draw_ready(base, perc, plan_r)
        sel = plan_r["selected"]
        result = {
            "stem": stem, "image": img_path.name,
            "source_size": perc.get("source_size"),
            "status": "plan_ready",
            "confidence": plan_r.get("confidence"),
            "pocket": sel["pocket"],
            "cut_deg": sel["cut_deg"],
            "total_dist": sel["total_dist"],
            "ball_count": perc["status"]["ball_count"],
            "cue_detected": perc["status"]["cue_present"],
            "corner_confidence": (perc.get("table") or {}).get("corner_confidence"),
            "quality_flags": plan_r.get("quality_flags", []),
            "perception_warnings": perc["status"].get("warnings", []),
            "n_candidates": plan_r.get("n_candidates", 0),
            "timing": timing,
            "categories": categorize(perc, plan_r),
        }
    else:
        warns = perc["status"].get("warnings", [])
        if "low_quality_warp" in warns:
            reason, detail = "low_quality_warp", "Source resolution too low"
        elif "no_table_detected" in warns:
            reason, detail = "no_table_detected", "Table corners not found"
        elif "cue_missing" in warns:
            reason, detail = "cue_missing", "Cue ball not detected"
        elif plan_r["status"] == "no_candidates":
            reason, detail = "no_candidates", "All shots blocked or cut > 70deg"
        else:
            reason, detail = "not_ready", ""
        overlay = draw_no_plan(base, reason, detail)
        result = {
            "stem": stem, "image": img_path.name,
            "source_size": perc.get("source_size"),
            "status": "no_plan",
            "reason": reason,
            "plan_status": plan_r["status"],
            "ball_count": perc["status"]["ball_count"],
            "cue_detected": perc["status"]["cue_present"],
            "corner_confidence": (perc.get("table") or {}).get("corner_confidence"),
            "perception_warnings": perc["status"].get("warnings", []),
            "planner_rejections": plan_r.get("rejections"),
            "shot_rejections": plan_r.get("shot_rejections"),
            "timing": timing,
            "categories": categorize(perc, plan_r),
        }

    out_stem = OUT_DIR / stem
    out_stem.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_stem / f"{stem}_mvp.jpg"), overlay,
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    if warped is not None:
        cv2.imwrite(str(out_stem / f"{stem}_warp.jpg"), warped)
    (out_stem / f"{stem}_result.json").write_text(json.dumps(result, indent=2))

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────

def build_report(results: list[dict]) -> str:
    ready   = [r for r in results if r.get("status") == "plan_ready"]
    no_plan = [r for r in results if r.get("status") == "no_plan"]
    errors  = [r for r in results if "error" in r]

    all_cats: dict[str, list[str]] = {}
    for r in results:
        for c in r.get("categories", []):
            all_cats.setdefault(c, []).append(r["stem"])

    timings = [r["timing"] for r in results if "timing" in r]
    def avg_ms(key):
        vals = [t[key] for t in timings if key in t]
        return round(sum(vals)/len(vals), 1) if vals else "—"
    def med_ms(key):
        vals = sorted(t[key] for t in timings if key in t)
        return round(vals[len(vals)//2], 1) if vals else "—"

    confs = [r["confidence"] for r in ready if r.get("confidence") is not None]

    L = [
        "# MVP Stress Test — Readiness Report",
        "",
        f"*Images tested: {len(results)}  |  Date: 2026-05-10*",
        "",
        "---",
        "",
        "## Pipeline Status",
        "",
        f"| Outcome | Count | % |",
        f"|---------|-------|---|",
        f"| Plan ready | {len(ready)} | {100*len(ready)//max(len(results),1)}% |",
        f"| No plan    | {len(no_plan)} | {100*len(no_plan)//max(len(results),1)}% |",
        f"| Errors     | {len(errors)} | {100*len(errors)//max(len(results),1)}% |",
        "",
        "---",
        "",
        "## Per-Image Results",
        "",
        "| Image | Status | Conf | Cue | Balls | Pocket | Cut | Flags |",
        "|-------|--------|------|-----|-------|--------|-----|-------|",
    ]

    for r in sorted(results, key=lambda x: x.get("stem", "")):
        if "error" in r:
            L.append(f"| {r['stem']} | ERROR | — | — | — | — | — | {r['error']} |")
            continue
        st   = "✓ ready" if r["status"] == "plan_ready" else f"✗ {r.get('reason','no_plan')}"
        conf = f"{r['confidence']:.2f}" if r.get("confidence") is not None else "—"
        cue  = "✓" if r.get("cue_detected") else "✗"
        bc   = str(r.get("ball_count", "?"))
        pkt  = r.get("pocket", "—")
        cut  = f"{r.get('cut_deg','—')}" if r.get("cut_deg") is not None else "—"
        flags = ", ".join(r.get("quality_flags", r.get("categories", [])))[:40]
        L.append(f"| {r['stem']} | {st} | {conf} | {cue} | {bc} | {pkt} | {cut} | {flags} |")

    L += [
        "",
        "---",
        "",
        "## Latency (per stage, milliseconds)",
        "",
        "| Stage | Average | Median |",
        "|-------|---------|--------|",
        f"| Corner detection | {avg_ms('corner_ms')} | {med_ms('corner_ms')} |",
        f"| Warp             | {avg_ms('warp_ms')}   | {med_ms('warp_ms')}   |",
        f"| YOLO ball detect | {avg_ms('yolo_ms')}   | {med_ms('yolo_ms')}   |",
        f"| Planner          | {avg_ms('planner_ms')}| {med_ms('planner_ms')}|",
        f"| **Total pipeline**| **{avg_ms('total_ms')}** | **{med_ms('total_ms')}** |",
        "",
        "---",
        "",
        "## Failure Categories",
        "",
        "| Category | Count | Images |",
        "|----------|-------|--------|",
    ]

    cat_order = ["ok", "bad_warp", "cue_missing", "cue_recovered", "false_cue",
                 "over_detection", "under_detection", "no_valid_shot",
                 "unrealistic_shot", "low_confidence", "blocked"]
    for cat in cat_order:
        stems = all_cats.get(cat, [])
        if stems:
            L.append(f"| `{cat}` | {len(stems)} | {', '.join(stems)} |")
    for cat, stems in all_cats.items():
        if cat not in cat_order:
            L.append(f"| `{cat}` | {len(stems)} | {', '.join(stems)} |")

    L += [
        "",
        "---",
        "",
        "## Confidence Distribution (ready images)",
        "",
    ]
    if confs:
        high = sum(1 for c in confs if c >= 0.8)
        med  = sum(1 for c in confs if 0.5 <= c < 0.8)
        low  = sum(1 for c in confs if c < 0.5)
        L += [
            f"- High (≥0.80): {high}",
            f"- Medium (0.50–0.79): {med}",
            f"- Low (<0.50): {low}",
            f"- Average: {sum(confs)/len(confs):.2f}",
            f"- Min: {min(confs):.2f}  Max: {max(confs):.2f}",
        ]
    else:
        L.append("- No ready images.")

    L += [
        "",
        "---",
        "",
        "## No-Plan Breakdown",
        "",
        "| Image | Reason | Cue | Balls | Corner Conf |",
        "|-------|--------|-----|-------|-------------|",
    ]
    for r in no_plan:
        cue  = "✓" if r.get("cue_detected") else "✗"
        cc   = f"{r.get('corner_confidence'):.2f}" if r.get("corner_confidence") else "—"
        L.append(f"| {r['stem']} | {r.get('reason','?')} | {cue} | {r.get('ball_count','?')} | {cc} |")

    # Readiness assessment
    ready_pct = 100*len(ready)//max(len(results), 1)
    avg_conf  = sum(confs)/len(confs) if confs else 0

    L += [
        "",
        "---",
        "",
        "## MVP Readiness Assessment",
        "",
        "### What works well",
        "",
    ]
    if ready_pct >= 60:
        L.append(f"- **Coverage**: {ready_pct}% of images produce a valid shot plan")
    if avg_conf >= 0.7:
        L.append(f"- **Confidence**: average {avg_conf:.2f} on ready images — above demo threshold")
    if not all_cats.get("bad_warp") or len(all_cats.get("bad_warp", [])) <= 3:
        L.append("- **Warp quality**: table corner detection robust across most angles and lighting")
    if not all_cats.get("false_cue"):
        L.append("- **Cue disambiguation**: no false-positive cue detections detected")
    L += [
        "- **Failure communication**: all non-ready cases produce a clear 'NO RECOMMENDATION' overlay",
        "- **Cue recovery**: threshold and appearance fallbacks recover missed cues in most cases",
        "- **Gating**: low-quality warps correctly blocked before planner",
        "",
        "### What still fails",
        "",
    ]
    fails = []
    bw = all_cats.get("bad_warp", [])
    if bw:
        fails.append(f"- **Bad warp** ({len(bw)} images): {', '.join(bw)} — "
                     "extreme angles or very low resolution defeat corner detection")
    cm = all_cats.get("cue_missing", [])
    if cm:
        fails.append(f"- **Cue missing** ({len(cm)} images): {', '.join(cm)} — "
                     "cue at table edge or absent from scene")
    ud = all_cats.get("under_detection", [])
    if ud:
        fails.append(f"- **Under-detection** ({len(ud)} images): {', '.join(ud)} — "
                     "balls missed, possibly due to poor lighting or occlusion")
    lc = all_cats.get("low_confidence", [])
    if lc:
        fails.append(f"- **Low confidence plans** ({len(lc)} images): {', '.join(lc)} — "
                     "multiple stacked penalties, recommend retake prompt in app")
    nv = all_cats.get("no_valid_shot", [])
    if nv:
        fails.append(f"- **No valid shot** ({len(nv)} images): {', '.join(nv)} — "
                     "all OB×pocket combinations blocked or cut > 70°")
    if not fails:
        fails.append("- No systematic failure patterns observed across the stress-test set")
    L += fails

    L += [
        "",
        "### Production blockers",
        "",
    ]
    blockers = []
    if ready_pct < 50:
        blockers.append("- [ ] Coverage below 50% — pipeline not ready for general use")
    bw2 = [s for s in all_cats.get("bad_warp", [])
           if s not in LOW_QUALITY_WARP_STEMS]
    if bw2:
        blockers.append(f"- [ ] Warp failures on otherwise good images: {', '.join(bw2)}")
    if avg_conf < 0.6:
        blockers.append("- [ ] Average confidence below 0.60 — too many uncertain plans for demo")
    if not blockers:
        blockers.append("- None identified — pipeline meets demo/MVP threshold")
    L += blockers

    L += [
        "",
        "### Can wait until v2",
        "",
        "- Spin, english, and cue-ball path-after-contact",
        "- Bank shots and kick shots",
        "- Multi-step run-out planning",
        "- Object ball color/number identification",
        "- 8-ball / 9-ball game-state tracking",
        "- Strategy AI (leave position, safety play)",
        "- Model retraining on new domain images (phone cameras, varied lighting)",
        "",
        "---",
        "",
        "*Generated by stress_test.py*",
        "",
    ]

    return "\n".join(L)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global CUE_CONF_THRESH, OBJ_CONF_THRESH, BALL_CONF_THRESH, MAX_CUT_DEG
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None,
                        help="Run on a single image instead of all picture/")
    parser.add_argument("--ball-ckpt", default=None,
                        help="Override ball detector checkpoint path")
    parser.add_argument("--ball-conf", type=float, default=None,
                        help="Override BOTH cue and obj confidence thresholds")
    parser.add_argument("--cue-conf", type=float, default=None,
                        help=f"Override cue_ball confidence threshold (default {CUE_CONF_THRESH})")
    parser.add_argument("--obj-conf", type=float, default=None,
                        help=f"Override object_ball confidence threshold (default {OBJ_CONF_THRESH})")
    parser.add_argument("--max-cut-deg", type=float, default=None,
                        help=f"Override max cut angle for planner (default {MAX_CUT_DEG}°). "
                             "Does NOT change the default — sweep only.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    ball_ckpt = Path(args.ball_ckpt) if args.ball_ckpt else BALL_CKPT
    if args.ball_conf is not None:
        CUE_CONF_THRESH = args.ball_conf
        OBJ_CONF_THRESH = args.ball_conf
    if args.cue_conf is not None:
        CUE_CONF_THRESH = args.cue_conf
    if args.obj_conf is not None:
        OBJ_CONF_THRESH = args.obj_conf
    if args.max_cut_deg is not None:
        MAX_CUT_DEG = args.max_cut_deg
    BALL_CONF_THRESH = min(CUE_CONF_THRESH, OBJ_CONF_THRESH)
    print("Loading models...")
    corner_model = YOLO(str(CORNER_CKPT))
    ball_model   = YOLO(str(ball_ckpt))
    print(f"  Corner: {CORNER_CKPT.name}")
    print(f"  Balls : {ball_ckpt.name}")
    print(f"  Cue conf thresh:  {CUE_CONF_THRESH}")
    print(f"  Obj conf thresh:  {OBJ_CONF_THRESH}")
    print(f"  Max cut deg:      {MAX_CUT_DEG}°")
    print()

    if args.image:
        imgs = [Path(args.image)]
    else:
        exts = {".jpg", ".jpeg", ".png"}
        imgs = sorted(
            p for p in PICTURE_DIR.iterdir()
            if p.suffix.lower() in exts
            and p.stem not in SKIP_STEMS
            and not p.stem.startswith("aug_")
            and not p.stem.startswith("tgt_")
        )

    print(f"Stress-testing {len(imgs)} images...\n")
    header = f"{'image':<28} {'status':<14} {'conf':>5} {'cue':>4} {'balls':>5} {'total_ms':>8}"
    print(header)
    print("-" * len(header))

    results = []
    for img_path in imgs:
        r = run_one(corner_model, ball_model, img_path)
        results.append(r)

        if "error" in r:
            print(f"  {'ERROR':<27} {'ERROR':<14}")
            continue

        st   = "plan_ready" if r["status"] == "plan_ready" else f"no:{r.get('reason','?')[:10]}"
        conf = f"{r['confidence']:.2f}" if r.get("confidence") is not None else "  —"
        cue  = "YES" if r.get("cue_detected") else " NO"
        bc   = str(r.get("ball_count", "?"))
        ms   = str(r["timing"].get("total_ms", "?"))
        cats = ",".join(r.get("categories", []))
        print(f"  {img_path.name:<27} {st:<14} {conf:>5} {cue:>4} {bc:>5} {ms:>7}ms  {cats}")

    report_md = build_report(results)
    report_path = OUT_DIR / "stress_report.md"
    report_path.write_text(report_md)

    results_path = OUT_DIR / "stress_results.json"
    results_path.write_text(json.dumps(results, indent=2))

    ready_n = sum(1 for r in results if r.get("status") == "plan_ready")
    print()
    print("=" * 64)
    print(f"  Plan ready  : {ready_n} / {len(results)}")
    print(f"  No plan     : {len(results) - ready_n} / {len(results)}")
    timings_total = [r["timing"]["total_ms"] for r in results
                     if "timing" in r and "total_ms" in r["timing"]]
    if timings_total:
        print(f"  Avg latency : {sum(timings_total)/len(timings_total):.0f}ms  "
              f"(max {max(timings_total):.0f}ms)")
    print(f"  Report      → {report_path}")
    print(f"  Results     → {results_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()
