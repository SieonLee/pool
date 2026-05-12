"""
Shot planner — gated debug mode + quality evaluation.

Gate: ready_for_planner=true AND cue_present=true AND ball_count >= 2
Input: review/perception/<stem>/<stem>_perception.json
Output per image → review/planner/<stem>/
  <stem>_planner_selected.jpg   selected shot (ghost ball + contact point)
  <stem>_planner_top3.jpg       top-3 candidates overlay
  <stem>_planner_rejected.jpg   sample of blocked/rejected paths
  <stem>_plan.json              full planner result with quality flags

Summary → review/planner/summary.json
Report  → review/planner/report.txt

Run:
  python scripts/plan.py                           # all perception outputs
  python scripts/plan.py --stem pool_real_019      # single stem
  python scripts/plan.py --image picture/foo.jpg   # by source image name
"""
import argparse
import cv2
import json
import math
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent.parent
PERCEPTION_DIR = BASE / "review" / "perception"
OUT_DIR        = BASE / "review" / "planner"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WARP_W, WARP_H = 900, 450
BALL_R = 15          # collision radius for path-blocking (px)
MAX_CUT_DEG = 70     # hard cut-angle rejection threshold
GHOST_MARGIN = 5     # px — ghost-ball table-edge clearance

# Thresholds for quality flags
THIN_CUT_DEG     = 45   # shot flag: difficult cut
EXTREME_CUT_DEG  = 55   # plan flag: very difficult
LONG_SHOT_PX     = 700  # shot flag: cue_dist + ob_dist sum
SUSPLONG_PX      = 900  # plan flag: alarmingly long
NEAR_RAIL_PX     = 40   # flag when cue/ghost within this of any rail
OVERLAP_DIST_PX  = 35   # flag when cue ball is this close to another ball
LOW_CORNER_CONF  = 0.7
LOW_CUE_CONF     = 0.40
HIGH_BALL_COUNT  = 12

# ── pocket positions in 900×450 warped space ─────────────────────────────────
POCKETS = {
    "TL": ( 22,  22),
    "TR": (878,  22),
    "ML": (  0, 225),
    "MR": (900, 225),
    "BL": ( 22, 428),
    "BR": (878, 428),
}
POCKET_LIST = list(POCKETS.items())

# ── visual ────────────────────────────────────────────────────────────────────
FONT      = cv2.FONT_HERSHEY_SIMPLEX
COL_CUE   = (255, 255, 255)
COL_OBJ   = (0, 165, 255)
COL_PKT   = (0, 220, 220)
COL_SEL   = (0, 255, 80)
COL_2ND   = (200, 200, 0)
COL_3RD   = (100, 180, 0)
COL_GHOST = (160, 160, 255)
COL_CONT  = (255, 80, 80)
COL_REJ   = (60, 60, 200)
COL_WARN  = (0, 90, 255)


# ── geometry helpers ──────────────────────────────────────────────────────────

def normalize(v):
    d = math.hypot(v[0], v[1])
    return (v[0] / d, v[1] / d) if d > 1e-9 else (0.0, 0.0)


def dot2(a, b):
    return a[0] * b[0] + a[1] * b[1]


def dist_point_to_line(px, py, ax, ay, bx, by):
    """Signed-t distance from point P to infinite line A→B."""
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    nx, ny = ax + t * dx, ay + t * dy
    return math.hypot(px - nx, py - ny), t


def ghost_ball_pos(ob, pocket, ball_r=BALL_R):
    dx, dy = ob[0] - pocket[0], ob[1] - pocket[1]
    d = math.hypot(dx, dy)
    if d < 1e-9:
        return ob
    return (ob[0] + dx / d * 2 * ball_r, ob[1] + dy / d * 2 * ball_r)


def contact_point(ghost, ob, ball_r=BALL_R):
    """Midpoint of ghost→OB at distance ball_r from each center = collision surface."""
    u = normalize((ob[0] - ghost[0], ob[1] - ghost[1]))
    return (ghost[0] + u[0] * ball_r, ghost[1] + u[1] * ball_r)


def cut_angle_deg(cue, ob, pocket):
    v1 = normalize((ob[0] - cue[0], ob[1] - cue[1]))
    v2 = normalize((pocket[0] - ob[0], pocket[1] - ob[1]))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot2(v1, v2)))))


def path_blocked(a, b, balls_xy, exclude_ids, ball_r=BALL_R):
    """True if any non-excluded ball blocks segment a→b (endpoint regions excluded)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    seg_len = math.hypot(dx, dy)
    if seg_len < 1e-9:
        return False
    t_min = ball_r / seg_len
    t_max = 1.0 - ball_r / seg_len
    if t_min >= t_max:
        return False
    for bid, bx, by in balls_xy:
        if bid in exclude_ids:
            continue
        t = ((bx - a[0]) * dx + (by - a[1]) * dy) / (seg_len * seg_len)
        if t < t_min or t > t_max:
            continue
        nx, ny = a[0] + t * dx, a[1] + t * dy
        if math.hypot(bx - nx, by - ny) < 2 * ball_r:
            return True
    return False


def ghost_in_bounds(g, margin=GHOST_MARGIN):
    return margin <= g[0] <= WARP_W - margin and margin <= g[1] <= WARP_H - margin


def is_near_rail(pos, margin=NEAR_RAIL_PX):
    x, y = pos
    return x < margin or x > WARP_W - margin or y < margin or y > WARP_H - margin


def score_shot(cut_deg, cue_dist, ob_dist):
    return cut_deg + 0.02 * (cue_dist + ob_dist)


# ── quality flags ─────────────────────────────────────────────────────────────

def shot_flags(c):
    """Per-candidate quality flags."""
    flags = []
    total_dist = c["cue_dist"] + c["ob_dist"]
    if c["cut_deg"] > THIN_CUT_DEG:
        flags.append("thin_cut")
    if total_dist > LONG_SHOT_PX:
        flags.append("long_shot")
    if is_near_rail(c["cue_xy"]):
        flags.append("near_rail_cue")
    if is_near_rail(c["ghost"]):
        flags.append("near_rail_ghost")
    return flags


def plan_flags(plan, perc):
    """Plan-level and perception-level quality flags."""
    flags = []
    status = perc.get("status", {})
    table  = perc.get("table", {})
    balls  = perc.get("balls", [])

    # --- perception flags ---
    cc = table.get("corner_confidence")
    if cc is not None and cc < LOW_CORNER_CONF:
        flags.append("low_corner_confidence")

    warns = " ".join(status.get("warnings", []))
    if "multiple_cue_detections" in warns:
        flags.append("cue_overdetection")
    if "capped_at_" in warns:
        flags.append("ball_cap_applied")
    if status.get("ball_count", 0) > HIGH_BALL_COUNT:
        flags.append("high_ball_count")

    cue_balls = [b for b in balls if b["type"] == "cue_ball"]
    if cue_balls and cue_balls[0]["confidence"] < LOW_CUE_CONF:
        flags.append("low_cue_confidence")

    # cue/object overlap ambiguity
    if cue_balls:
        cx, cy = cue_balls[0]["x"], cue_balls[0]["y"]
        for b in balls:
            if b["type"] == "object_ball":
                if math.hypot(b["x"] - cx, b["y"] - cy) < OVERLAP_DIST_PX:
                    flags.append("cue_object_overlap")
                    break

    if plan["status"] != "plan_ready":
        return flags

    # --- plan flags ---
    sel = plan["selected"]
    if sel["cut_deg"] > EXTREME_CUT_DEG:
        flags.append("extreme_thin_cut")
    if sel["cue_dist"] + sel["ob_dist"] > SUSPLONG_PX:
        flags.append("suspiciously_long_shot")
    if plan["n_candidates"] == 1:
        flags.append("only_one_candidate")
    if plan["n_candidates"] == 0:
        flags.append("no_candidates")

    # Pocket too blocked: ob_path_blocked rejections high relative to object balls
    n_obj = status.get("object_ball_count", 0)
    ob_blocked = plan.get("rejections", {}).get("ob_path_blocked", 0)
    if n_obj > 0 and ob_blocked > n_obj * 1.5:
        flags.append("pocket_overcrowded")

    # Flag if selected shot has near-rail geometry
    if is_near_rail(sel["cue_xy"]):
        flags.append("near_rail_cue_selected")
    if is_near_rail(sel["ghost"]):
        flags.append("near_rail_ghost_selected")

    return flags


def planner_confidence(plan, perc, flags):
    """
    Confidence estimate [0.0, 1.0] that the selected shot is a real, playable shot.
    Starts at 1.0, penalised by quality flags.
    """
    score = 1.0
    penalties = {
        "extreme_thin_cut":          0.40,
        "thin_cut":                  0.20,
        "suspiciously_long_shot":    0.15,
        "long_shot":                 0.10,
        "near_rail_cue_selected":    0.10,
        "near_rail_ghost_selected":  0.10,
        "cue_object_overlap":        0.20,
        "low_corner_confidence":     0.10,
        "low_cue_confidence":        0.20,
        "high_ball_count":           0.08,
        "ball_cap_applied":          0.08,
        "cue_overdetection":         0.05,
        "only_one_candidate":        0.10,
        "pocket_overcrowded":        0.08,
    }
    # include shot flags from selected
    if plan["status"] == "plan_ready":
        flags = flags + shot_flags(plan["selected"])

    for f in flags:
        score -= penalties.get(f, 0.0)

    return round(max(0.0, min(1.0, score)), 3)


# ── planner core ──────────────────────────────────────────────────────────────

def plan_image(perc: dict) -> dict:
    status = perc.get("status", {})

    if not status.get("ready_for_planner", False):
        return {"status": "no_plan", "reason": "not_ready_for_planner",
                "candidates": [], "rejections": {}}
    if not status.get("cue_present", False):
        return {"status": "no_plan", "reason": "no_cue_ball",
                "candidates": [], "rejections": {}}
    if status.get("ball_count", 0) < 2:
        return {"status": "no_plan", "reason": "too_few_balls",
                "candidates": [], "rejections": {}}

    balls     = perc["balls"]
    cue_balls = [b for b in balls if b["type"] == "cue_ball"]
    obj_balls = [b for b in balls if b["type"] == "object_ball"]

    if not cue_balls or not obj_balls:
        return {"status": "no_plan", "reason": "missing_cue_or_objects",
                "candidates": [], "rejections": {}}

    cue    = (cue_balls[0]["x"], cue_balls[0]["y"])
    cue_id = cue_balls[0]["id"]
    balls_xy = [(b["id"], b["x"], b["y"]) for b in balls]

    candidates = []
    rejections = {"cut_too_large": 0, "cue_path_blocked": 0,
                  "ob_path_blocked": 0, "ghost_out_of_bounds": 0}
    # Track a few rejected samples for the overlay
    rejected_samples = []

    for ob in obj_balls:
        ob_pos = (ob["x"], ob["y"])
        ob_id  = ob["id"]

        for pkt_name, pkt_pos in POCKET_LIST:
            cut = cut_angle_deg(cue, ob_pos, pkt_pos)
            ghost = ghost_ball_pos(ob_pos, pkt_pos)
            reject_reason = None

            if cut > MAX_CUT_DEG:
                rejections["cut_too_large"] += 1
                reject_reason = "cut_too_large"
            elif not ghost_in_bounds(ghost):
                rejections["ghost_out_of_bounds"] += 1
                reject_reason = "ghost_out_of_bounds"
            elif path_blocked(cue, ghost, balls_xy, {cue_id, ob_id}):
                rejections["cue_path_blocked"] += 1
                reject_reason = "cue_path_blocked"
            elif path_blocked(ob_pos, pkt_pos, balls_xy, {cue_id, ob_id}):
                rejections["ob_path_blocked"] += 1
                reject_reason = "ob_path_blocked"

            if reject_reason:
                if len(rejected_samples) < 8 and reject_reason in (
                        "cue_path_blocked", "ob_path_blocked"):
                    rejected_samples.append({
                        "ob_id": ob_id, "pocket": pkt_name, "reason": reject_reason,
                        "cut_deg": round(cut, 1), "ghost": ghost,
                        "pocket_xy": pkt_pos, "cue_xy": cue, "ob_xy": ob_pos,
                    })
                continue

            cue_dist = math.hypot(ghost[0] - cue[0], ghost[1] - cue[1])
            ob_dist  = math.hypot(pkt_pos[0] - ob_pos[0], pkt_pos[1] - ob_pos[1])
            s = score_shot(cut, cue_dist, ob_dist)
            cont = contact_point(ghost, ob_pos)

            candidates.append({
                "ob_id":       ob_id,
                "pocket":      pkt_name,
                "cut_deg":     round(cut, 1),
                "cue_dist":    round(cue_dist, 1),
                "ob_dist":     round(ob_dist, 1),
                "total_dist":  round(cue_dist + ob_dist, 1),
                "score":       round(s, 2),
                "ghost":       (round(ghost[0], 1), round(ghost[1], 1)),
                "contact":     (round(cont[0], 1), round(cont[1], 1)),
                "pocket_xy":   pkt_pos,
                "cue_xy":      cue,
                "ob_xy":       ob_pos,
                "shot_flags":  shot_flags({
                    "cut_deg": cut, "cue_dist": cue_dist, "ob_dist": ob_dist,
                    "cue_xy": cue, "ghost": ghost,
                }),
            })

    if not candidates:
        return {"status": "no_plan", "reason": "all_shots_rejected",
                "rejections": rejections, "candidates": [],
                "rejected_samples": rejected_samples}

    candidates.sort(key=lambda c: c["score"])
    top3 = candidates[:3]

    return {
        "status":           "plan_ready",
        "selected":         top3[0],
        "top3":             top3,
        "n_candidates":     len(candidates),
        "rejections":       rejections,
        "rejected_samples": rejected_samples,
    }


# ── draw helpers ──────────────────────────────────────────────────────────────

def draw_dashed_line(canvas, p1, p2, color, thickness=1, dash=10, gap=6):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    dist = math.hypot(dx, dy)
    if dist < 1:
        return
    ux, uy = dx / dist, dy / dist
    t, on = 0.0, True
    while t < dist:
        t2 = min(t + (dash if on else gap), dist)
        if on:
            x1 = int(p1[0] + t * ux);  y1 = int(p1[1] + t * uy)
            x2 = int(p1[0] + t2 * ux); y2 = int(p1[1] + t2 * uy)
            cv2.line(canvas, (x1, y1), (x2, y2), color, thickness)
        t, on = t2, not on


def draw_pockets(canvas):
    for name, (px, py) in POCKET_LIST:
        cv2.circle(canvas, (int(px), int(py)), 13, COL_PKT, -1)
        cv2.circle(canvas, (int(px), int(py)), 13, (0, 0, 0), 1)
        cv2.putText(canvas, name, (int(px) - 14, int(py) + 20),
                    FONT, 0.36, COL_PKT, 1)


def draw_balls_simple(canvas, balls):
    for b in balls:
        cx, cy = int(b["x"]), int(b["y"])
        r = max(int(b["r"]), 8)
        if b["type"] == "cue_ball":
            cv2.circle(canvas, (cx, cy), r, COL_CUE, 2)
            cv2.circle(canvas, (cx, cy), 3, COL_CUE, -1)
            cv2.putText(canvas, "C", (cx - 5, cy + 5), FONT, 0.42, COL_CUE, 1)
        else:
            cv2.circle(canvas, (cx, cy), r, COL_OBJ, 2)
            cv2.putText(canvas, str(b["id"]), (cx - 6, cy + 5), FONT, 0.36, COL_OBJ, 1)


def draw_shot(canvas, c, color, show_ghost=True, show_contact=True,
              arrow_tip=0.04, line_thick=2):
    cue  = (int(c["cue_xy"][0]),    int(c["cue_xy"][1]))
    ob   = (int(c["ob_xy"][0]),     int(c["ob_xy"][1]))
    g    = (int(c["ghost"][0]),     int(c["ghost"][1]))
    pkt  = (int(c["pocket_xy"][0]), int(c["pocket_xy"][1]))

    # OB → pocket
    cv2.arrowedLine(canvas, ob, pkt, color, line_thick, tipLength=arrow_tip)
    # Cue → ghost
    cv2.arrowedLine(canvas, cue, g, color, line_thick, tipLength=arrow_tip)

    if show_ghost:
        # Ghost ball: dashed circle
        draw_dashed_circle(canvas, g, BALL_R, COL_GHOST, 1)

    if show_contact and "contact" in c:
        cont = (int(c["contact"][0]), int(c["contact"][1]))
        cv2.circle(canvas, cont, 4, COL_CONT, -1)
        cv2.circle(canvas, cont, 4, (0, 0, 0), 1)


def draw_dashed_circle(canvas, center, radius, color, thickness=1, n_dashes=16):
    for i in range(n_dashes):
        start_a = 360 * i / n_dashes
        end_a   = start_a + 360 / n_dashes * 0.6
        cv2.ellipse(canvas, center, (radius, radius), 0,
                    start_a, end_a, color, thickness)


def draw_shot_label(canvas, c, rank, color):
    ob   = (int(c["ob_xy"][0]),     int(c["ob_xy"][1]))
    pkt  = (int(c["pocket_xy"][0]), int(c["pocket_xy"][1]))
    mx   = (ob[0] + pkt[0]) // 2
    my   = max(18, (ob[1] + pkt[1]) // 2 - 16)
    flags_str = ",".join(c.get("shot_flags", [])) or ""
    flag_tag  = f" [{flags_str}]" if flags_str else ""
    label = f"#{rank} OB{c['ob_id']}→{c['pocket']}  cut={c['cut_deg']}°{flag_tag}"
    cv2.putText(canvas, label, (mx - 60, my), FONT, 0.34, color, 1)


def draw_flags_footer(canvas, flags, confidence):
    flag_str = "  ".join(flags) if flags else "—"
    conf_col = COL_SEL if confidence >= 0.7 else (COL_2ND if confidence >= 0.45 else (0, 60, 255))
    cv2.putText(canvas, f"flags: {flag_str}", (8, WARP_H - 24),
                FONT, 0.36, COL_WARN, 1)
    cv2.putText(canvas, f"confidence: {confidence:.2f}", (8, WARP_H - 8),
                FONT, 0.45, conf_col, 1)


# ── overlay builders ──────────────────────────────────────────────────────────

def overlay_selected(warp_path, perc, plan, flags, confidence):
    img = cv2.imread(str(warp_path))
    if img is None:
        return None
    canvas = img.copy()
    draw_pockets(canvas)
    draw_balls_simple(canvas, perc.get("balls", []))

    if plan["status"] != "plan_ready":
        reason = plan.get("reason", "no_plan")
        cv2.putText(canvas, f"NO PLAN: {reason}", (12, 30), FONT, 0.7, (0, 0, 255), 2)
        draw_flags_footer(canvas, flags, confidence)
        return canvas

    sel = plan["selected"]
    draw_shot(canvas, sel, COL_SEL, show_ghost=True, show_contact=True)
    draw_shot_label(canvas, sel, 1, COL_SEL)

    info = (f"SELECTED: OB{sel['ob_id']}→{sel['pocket']}  "
            f"cut={sel['cut_deg']}°  cue_d={sel['cue_dist']}  ob_d={sel['ob_dist']}")
    cv2.putText(canvas, info, (8, 20), FONT, 0.42, COL_SEL, 1)
    draw_flags_footer(canvas, flags, confidence)
    return canvas


def overlay_top3(warp_path, perc, plan, flags, confidence):
    img = cv2.imread(str(warp_path))
    if img is None:
        return None
    canvas = img.copy()
    draw_pockets(canvas)
    draw_balls_simple(canvas, perc.get("balls", []))

    if plan["status"] != "plan_ready":
        reason = plan.get("reason", "no_plan")
        cv2.putText(canvas, f"NO PLAN: {reason}", (12, 30), FONT, 0.7, (0, 0, 255), 2)
        draw_flags_footer(canvas, flags, confidence)
        return canvas

    colors = [COL_SEL, COL_2ND, COL_3RD]
    top3   = plan["top3"]

    for i in range(min(3, len(top3)) - 1, -1, -1):
        draw_shot(canvas, top3[i], colors[i],
                  show_ghost=(i == 0), show_contact=(i == 0),
                  line_thick=max(1, 3 - i))
        draw_shot_label(canvas, top3[i], i + 1, colors[i])

    sel   = plan["selected"]
    n_rej = sum(plan["rejections"].values())
    info  = (f"TOP-3 of {plan['n_candidates']} candidates  "
             f"({n_rej} rejected)  "
             f"conf={confidence:.2f}")
    cv2.putText(canvas, info, (8, 20), FONT, 0.42, COL_SEL, 1)
    draw_flags_footer(canvas, flags, confidence)
    return canvas


def overlay_rejected(warp_path, perc, plan):
    img = cv2.imread(str(warp_path))
    if img is None:
        return None
    canvas = img.copy()
    draw_pockets(canvas)
    draw_balls_simple(canvas, perc.get("balls", []))

    samples = plan.get("rejected_samples", [])
    if not samples:
        cv2.putText(canvas, "no blocked paths to show", (12, 30),
                    FONT, 0.6, (120, 120, 120), 1)
        return canvas

    for s in samples:
        color = (80, 80, 180) if s["reason"] == "cue_path_blocked" else (40, 40, 140)
        cue = (int(s["cue_xy"][0]),    int(s["cue_xy"][1]))
        ob  = (int(s["ob_xy"][0]),     int(s["ob_xy"][1]))
        g   = (int(s["ghost"][0]),     int(s["ghost"][1]))
        pkt = (int(s["pocket_xy"][0]), int(s["pocket_xy"][1]))
        draw_dashed_line(canvas, cue, g, color, 1)
        draw_dashed_line(canvas, ob, pkt, color, 1)
        draw_dashed_circle(canvas, g, BALL_R, color, 1)
        tag = f"OB{s['ob_id']}→{s['pocket']} [{s['reason'][:3]}]"
        cv2.putText(canvas, tag, (ob[0] + 6, ob[1] - 8), FONT, 0.30, color, 1)

    n_rej = sum(plan.get("rejections", {}).values())
    cv2.putText(canvas, f"showing {len(samples)} of {n_rej} rejected shots",
                (8, 20), FONT, 0.42, COL_REJ, 1)
    return canvas


# ── process one stem ──────────────────────────────────────────────────────────

def process_stem(stem):
    p = PERCEPTION_DIR / stem / f"{stem}_perception.json"
    if not p.exists():
        return None
    perc = json.loads(p.read_text())

    plan       = plan_image(perc)
    flags      = plan_flags(plan, perc)
    confidence = planner_confidence(plan, perc, flags)

    out_dir = OUT_DIR / stem
    out_dir.mkdir(exist_ok=True)
    warp_path = PERCEPTION_DIR / stem / f"{stem}_warp.jpg"

    cv2.imwrite(str(out_dir / f"{stem}_planner_selected.jpg"),
                overlay_selected(warp_path, perc, plan, flags, confidence))
    cv2.imwrite(str(out_dir / f"{stem}_planner_top3.jpg"),
                overlay_top3(warp_path, perc, plan, flags, confidence))
    cv2.imwrite(str(out_dir / f"{stem}_planner_rejected.jpg"),
                overlay_rejected(warp_path, perc, plan))

    result = {
        "image":             perc["image"],
        "stem":              stem,
        "perception_status": perc["status"],
        "perception_table":  perc.get("table", {}),
        "plan":              plan,
        "quality_flags":     flags,
        "confidence":        confidence,
    }
    with open(out_dir / f"{stem}_plan.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


# ── report ────────────────────────────────────────────────────────────────────

def build_report(results):
    lines = []
    W = 80

    def hr(c="═"): lines.append(c * W)
    def sec(t):    lines.append(f"\n{t}"); lines.append("─" * W)

    hr()
    lines.append("SHOT PLANNER EVALUATION REPORT")
    hr()

    # Per-image detail
    sec("PER-IMAGE DETAIL")
    for r in results:
        if r is None:
            continue
        stem  = r["stem"]
        plan  = r["plan"]
        flags = r["quality_flags"]
        conf  = r["confidence"]
        pstat = r["perception_status"]

        if plan["status"] == "plan_ready":
            sel = plan["selected"]
            lines.append(f"\n  ✓ {stem}")
            lines.append(f"    selected : OB{sel['ob_id']}→{sel['pocket']}  "
                         f"cut={sel['cut_deg']}°  "
                         f"cue_d={sel['cue_dist']}px  "
                         f"ob_d={sel['ob_dist']}px  "
                         f"total={sel['total_dist']}px")
            lines.append(f"    score    : {sel['score']}  "
                         f"conf={conf:.2f}  "
                         f"n_cand={plan['n_candidates']}")

            # Top-3 table
            lines.append(f"    top-3    :")
            for i, c in enumerate(plan["top3"]):
                sf = ",".join(c.get("shot_flags", [])) or "—"
                lines.append(f"      #{i+1}  OB{c['ob_id']}→{c['pocket']:2s}  "
                             f"cut={c['cut_deg']:5.1f}°  "
                             f"c_d={c['cue_dist']:6.1f}  "
                             f"o_d={c['ob_dist']:6.1f}  "
                             f"score={c['score']:6.2f}  flags=[{sf}]")

            rej = plan["rejections"]
            lines.append(f"    rejected : cut_large={rej['cut_too_large']}  "
                         f"cue_blk={rej['cue_path_blocked']}  "
                         f"ob_blk={rej['ob_path_blocked']}  "
                         f"ghost_oob={rej['ghost_out_of_bounds']}")
            lines.append(f"    flags    : {', '.join(flags) if flags else '—'}")
        else:
            reason = plan.get("reason", "?")
            lines.append(f"\n  ✗ {stem}")
            lines.append(f"    NO PLAN  : {reason}  "
                         f"[perc_ready={pstat.get('ready_for_planner')}]")
            lines.append(f"    flags    : {', '.join(flags) if flags else '—'}")
            if plan.get("rejections"):
                rej = plan["rejections"]
                lines.append(f"    rejected : cut_large={rej['cut_too_large']}  "
                             f"cue_blk={rej['cue_path_blocked']}  "
                             f"ob_blk={rej['ob_path_blocked']}")

    # Summary stats
    sec("SUMMARY")
    valid   = [r for r in results if r]
    n_ready = sum(1 for r in valid if r["plan"]["status"] == "plan_ready")
    n_no    = len(valid) - n_ready
    confs   = [r["confidence"] for r in valid if r["plan"]["status"] == "plan_ready"]
    avg_c   = sum(confs) / len(confs) if confs else 0.0

    lines.append(f"  Total processed  : {len(valid)}")
    lines.append(f"  Plan ready       : {n_ready}/{len(valid)}")
    lines.append(f"  No plan          : {n_no}/{len(valid)}")
    lines.append(f"  Avg confidence   : {avg_c:.2f}  "
                 f"(min={min(confs, default=0):.2f}  max={max(confs, default=0):.2f})")

    # Confidence buckets
    hi = sum(1 for c in confs if c >= 0.70)
    md = sum(1 for c in confs if 0.45 <= c < 0.70)
    lo = sum(1 for c in confs if c < 0.45)
    lines.append(f"  Confidence bands : high(≥0.70)={hi}  mid=[0.45,0.70)={md}  low(<0.45)={lo}")

    # Suspicious images
    sec("SUSPICIOUS / WEAK IMAGES")
    suspicious = []
    for r in valid:
        flags = r["quality_flags"]
        conf  = r["confidence"]
        issues = [f for f in flags if f in (
            "extreme_thin_cut", "cue_object_overlap", "low_corner_confidence",
            "low_cue_confidence", "ball_cap_applied", "only_one_candidate",
            "suspiciously_long_shot", "cue_overdetection",
        )]
        if issues or conf < 0.55:
            suspicious.append((r["stem"], conf, issues))

    if suspicious:
        for stem, conf, issues in sorted(suspicious, key=lambda x: x[1]):
            lines.append(f"  {stem:30s}  conf={conf:.2f}  [{', '.join(issues)}]")
    else:
        lines.append("  None detected.")

    # Perception weakness summary
    sec("PERCEPTION QUALITY")
    for r in valid:
        pstat = r["perception_status"]
        ptbl  = r.get("perception_table", {})
        cc    = ptbl.get("corner_confidence")
        bc    = pstat.get("ball_count", 0)
        warns = pstat.get("warnings", [])
        tag   = "⚠" if cc and cc < LOW_CORNER_CONF else " "
        lines.append(f"  {tag} {r['stem']:30s}  "
                     f"corner_conf={cc or '?':>5}  "
                     f"balls={bc:2d}  "
                     f"warns=[{', '.join(warns) or '—'}]")

    # Rejection totals
    sec("REJECTION BREAKDOWN (ALL IMAGES)")
    agg = {"cut_too_large": 0, "cue_path_blocked": 0,
           "ob_path_blocked": 0, "ghost_out_of_bounds": 0}
    for r in valid:
        for k in agg:
            agg[k] += r["plan"].get("rejections", {}).get(k, 0)
    for k, v in agg.items():
        lines.append(f"  {k:28s}: {v}")

    hr()
    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None)
    parser.add_argument("--stem",  default=None)
    args = parser.parse_args()

    if args.stem:
        stems = [args.stem]
    elif args.image:
        stems = [Path(args.image).stem]
    else:
        stems = sorted(
            p.parent.name
            for p in PERCEPTION_DIR.rglob("*_perception.json")
        )

    if not stems:
        print("No perception outputs found. Run scripts/perceive.py first.")
        return

    print(f"Running shot planner on {len(stems)} image(s)...")

    results = []
    for stem in stems:
        print(f"  {stem}...", end=" ", flush=True)
        r = process_stem(stem)
        results.append(r)
        if r:
            plan = r["plan"]
            conf = r["confidence"]
            if plan["status"] == "plan_ready":
                sel = plan["selected"]
                flags_str = f"  [{','.join(r['quality_flags'])}]" if r["quality_flags"] else ""
                print(f"OB{sel['ob_id']}→{sel['pocket']} cut={sel['cut_deg']}° "
                      f"conf={conf:.2f}{flags_str}")
            else:
                print(f"no_plan: {plan.get('reason','?')}  conf={conf:.2f}")
        else:
            print("FAILED (no perception JSON)")

    report_text = build_report(results)
    print("\n" + report_text)

    report_path = OUT_DIR / "report.txt"
    report_path.write_text(report_text)

    summary = {
        "n_images":     len(results),
        "n_plan_ready": sum(1 for r in results if r and r["plan"]["status"] == "plan_ready"),
        "n_no_plan":    sum(1 for r in results if r and r["plan"]["status"] != "plan_ready"),
        "avg_confidence": round(
            sum(r["confidence"] for r in results if r and r["plan"]["status"] == "plan_ready") /
            max(1, sum(1 for r in results if r and r["plan"]["status"] == "plan_ready")), 3),
        "results": [r for r in results if r],
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nOutputs  → {OUT_DIR}")
    print(f"Report   → {report_path}")
    print(f"Summary  → {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
