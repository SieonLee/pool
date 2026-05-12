"""
Generate clean MVP output overlays.

For planner-ready images:
  - Selected shot: cue path, OB path, ghost ball, pocket highlight
  - Cut angle, confidence, warning badges

For non-ready images:
  - Dark overlay with reason text

Output:
  review/mvp_overlay/<stem>_mvp.jpg
  review/mvp_overlay/<stem>_mvp.json
  review/mvp_overlay/mvp_summary_report.md
"""
import json
import math
import textwrap
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).parent.parent
PERC_DIR = BASE / "review" / "perception"
PLAN_DIR = BASE / "review" / "planner"
OUT_DIR = BASE / "review" / "mvp_overlay"

# ── Visual constants ──────────────────────────────────────────────────────────
WARP_W, WARP_H = 900, 450
BALL_R = 15

# Colors (BGR)
C_WHITE      = (255, 255, 255)
C_BLACK      = (0,   0,   0)
C_CUE        = (255, 255, 255)   # white ball
C_CUE_RING   = (100, 200, 255)   # blue ring around cue
C_OB         = (60,  160, 255)   # orange-ish OB
C_OB_RING    = (30,   80, 220)   # darker ring
C_GHOST      = (200, 200, 200)   # ghost ball (dashed)
C_PATH_CUE   = (255, 230, 100)   # cue→ghost path
C_PATH_OB    = (80,  200, 255)   # OB→pocket path
C_POCKET     = (50,  255, 150)   # pocket highlight
C_BADGE_BG   = (30,   30,  30)   # badge background
C_BADGE_WARN = (30,  130, 255)   # warning badge text (orange)
C_BADGE_OK   = (100, 200, 100)   # ok badge text
C_CONF_HIGH  = (80,  220, 100)   # conf ≥ 0.8
C_CONF_MED   = (80,  200, 240)   # conf 0.5–0.8
C_CONF_LOW   = (60,  100, 240)   # conf < 0.5

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD  = cv2.FONT_HERSHEY_DUPLEX

POCKET_NAMES = {
    "TL": (22,  22),  "TR": (878,  22),
    "ML": (0,  225),  "MR": (900, 225),
    "BL": (22, 428),  "BR": (878, 428),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_warp(stem: str) -> np.ndarray | None:
    """Load the warped table image from perception output."""
    perc_stem_dir = PERC_DIR / stem
    for name in [f"{stem}_warp.jpg", f"{stem}_warp.png",
                 f"{stem}_warped.jpg", f"{stem}_warped.png"]:
        p = perc_stem_dir / name
        if p.exists():
            return cv2.imread(str(p))
    # Fallback: search dataset
    for ext in [".jpg", ".png", ".jpeg"]:
        p = BASE / "datasets" / "pose-warped-balls" / "images" / "train" / f"{stem}_warped{ext}"
        if p.exists():
            return cv2.imread(str(p))
    return None


def dashed_circle(img, center, radius, color, thickness=2, n_segments=24):
    """Draw a dashed circle."""
    for i in range(0, n_segments, 2):
        a0 = 2 * math.pi * i / n_segments
        a1 = 2 * math.pi * (i + 1) / n_segments
        p0 = (int(center[0] + radius * math.cos(a0)),
              int(center[1] + radius * math.sin(a0)))
        p1 = (int(center[0] + radius * math.cos(a1)),
              int(center[1] + radius * math.sin(a1)))
        cv2.line(img, p0, p1, color, thickness)


def arrow_line(img, p1, p2, color, thickness=2, tip_frac=0.08):
    """Draw a line with an arrowhead at p2."""
    cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)
    # arrowhead
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = math.hypot(dx, dy)
    if length < 5:
        return
    ux, uy = dx / length, dy / length
    tip_len = min(length * tip_frac, 18)
    angle = math.radians(25)
    for sign in (-1, 1):
        ax = p2[0] - tip_len * (ux * math.cos(angle) + sign * uy * math.sin(angle))
        ay = p2[1] - tip_len * (uy * math.cos(angle) - sign * ux * math.sin(angle))
        cv2.line(img, p2, (int(ax), int(ay)), color, thickness, cv2.LINE_AA)


def draw_badge(img, x, y, text, bg=C_BADGE_BG, fg=C_BADGE_WARN, scale=0.42):
    """Draw a small rounded badge at (x, y). Returns new y for stacking."""
    (tw, th), bl = cv2.getTextSize(text, FONT, scale, 1)
    pad_x, pad_y = 6, 4
    rx1, ry1 = x, y
    rx2, ry2 = x + tw + 2 * pad_x, y + th + 2 * pad_y + bl
    # background
    cv2.rectangle(img, (rx1, ry1), (rx2, ry2), bg, -1)
    cv2.rectangle(img, (rx1, ry1), (rx2, ry2), fg, 1)
    # text
    cv2.putText(img, text, (rx1 + pad_x, ry2 - pad_y - bl // 2),
                FONT, scale, fg, 1, cv2.LINE_AA)
    return ry2 + 4


def conf_color(conf: float):
    if conf >= 0.8:
        return C_CONF_HIGH
    if conf >= 0.5:
        return C_CONF_MED
    return C_CONF_LOW


def warning_label(w: str) -> str:
    """Convert raw warning string to short display label (ASCII only for OpenCV)."""
    if w.startswith("cue_recovered_by_threshold"):
        return "! CUE RECOVERED (thresh)"
    if w.startswith("cue_recovered_by_appearance"):
        return "! CUE RECOVERED (appear)"
    if w == "low_corner_confidence":
        return "! LOW CORNER CONF"
    if w == "low_cue_confidence":
        return "! LOW CUE CONF"
    if w == "capped_at_16":
        return "! CAPPED >16 BALLS"
    if w == "long_shot":
        return "! LONG SHOT"
    if w == "near_rail":
        return "! NEAR RAIL"
    if w == "extreme_thin_cut":
        return "! EXTREME CUT"
    if w == "thin_cut":
        return "! THIN CUT"
    if w == "cue_object_overlap":
        return "! CUE/OB OVERLAP"
    if w.startswith("near_rail"):
        return "! NEAR RAIL"
    if w.startswith("multiple_cue"):
        return "! MULTI-CUE DETECT"
    if w == "high_ball_count":
        return "! HIGH BALL COUNT"
    if w == "cue_overdetection":
        return "! CUE OVERDETECT"
    # Generic fallback — strip underscores, cap
    label = w.upper().replace("_", " ")
    return f"! {label[:22]}"


# ── Ready overlay ─────────────────────────────────────────────────────────────

def draw_ready_overlay(warp: np.ndarray, plan_data: dict, perc_data: dict) -> np.ndarray:
    img = warp.copy()
    sel = plan_data["plan"]["selected"]
    conf = plan_data.get("confidence", 1.0)
    q_flags = plan_data.get("quality_flags", [])
    perc_warns = perc_data.get("status", {}).get("warnings", [])

    cue_xy  = tuple(int(v) for v in sel["cue_xy"])
    ob_xy   = tuple(int(v) for v in sel["ob_xy"])
    ghost   = tuple(int(v) for v in sel["ghost"])
    contact = tuple(int(v) for v in sel["contact"])
    pkt_xy  = tuple(int(v) for v in sel["pocket_xy"])
    cut_deg = sel["cut_deg"]
    pocket  = sel["pocket"]
    shot_flags = sel.get("shot_flags", [])

    # Estimate ball radius from perception
    cue_r = BALL_R
    ob_r  = BALL_R
    for b in perc_data.get("balls", []):
        if b["type"] == "cue_ball":
            cue_r = max(8, min(28, int(b["r"])))
        if b["id"] == sel["ob_id"]:
            ob_r = max(8, min(28, int(b["r"])))

    # ── Pocket highlight ──
    px, py = pkt_xy
    cv2.circle(img, (px, py), 22, C_POCKET, -1)
    cv2.circle(img, (px, py), 22, C_WHITE, 1)
    cv2.putText(img, pocket, (px - 10, py + 5), FONT_BOLD, 0.5, C_BLACK, 1, cv2.LINE_AA)

    # ── OB → pocket path ──
    arrow_line(img, ob_xy, pkt_xy, C_PATH_OB, thickness=2)

    # ── Cue → ghost path ──
    arrow_line(img, cue_xy, ghost, C_PATH_CUE, thickness=2)

    # ── Ghost ball ──
    dashed_circle(img, ghost, ob_r, C_GHOST, thickness=2)
    # contact point
    cv2.circle(img, contact, 4, C_PATH_OB, -1)
    cv2.circle(img, contact, 4, C_WHITE, 1)

    # ── Target OB ──
    cv2.circle(img, ob_xy, ob_r, C_OB, -1)
    cv2.circle(img, ob_xy, ob_r + 2, C_OB_RING, 2)
    cv2.putText(img, str(sel["ob_id"]),
                (ob_xy[0] - 5, ob_xy[1] + 5), FONT_BOLD, 0.4, C_WHITE, 1, cv2.LINE_AA)

    # ── Cue ball ──
    cv2.circle(img, cue_xy, cue_r, C_CUE, -1)
    cv2.circle(img, cue_xy, cue_r + 2, C_CUE_RING, 2)
    cv2.putText(img, "C", (cue_xy[0] - 5, cue_xy[1] + 5),
                FONT_BOLD, 0.4, (60, 60, 60), 1, cv2.LINE_AA)

    # ── Shot info panel (top-left) ──
    panel_x, panel_y = 8, 8
    panel_w, panel_h = 210, 66
    overlay = img.copy()
    cv2.rectangle(overlay, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h), C_BADGE_BG, -1)
    cv2.addWeighted(overlay, 0.72, img, 0.28, 0, img)

    cc = conf_color(conf)
    cv2.putText(img, f"CONF  {conf:.2f}",
                (panel_x + 8, panel_y + 20), FONT_BOLD, 0.55, cc, 1, cv2.LINE_AA)
    cv2.putText(img, f"CUT   {cut_deg:.1f}deg  PKT {pocket}",
                (panel_x + 8, panel_y + 40), FONT, 0.48, C_WHITE, 1, cv2.LINE_AA)
    dist_total = int(sel.get("cue_dist", 0) + sel.get("ob_dist", 0))
    cv2.putText(img, f"DIST  {dist_total}px",
                (panel_x + 8, panel_y + 58), FONT, 0.44, (180, 180, 180), 1, cv2.LINE_AA)

    # ── Warning badges (top-right column) ──
    all_warns = list(q_flags) + [w for w in perc_warns if "recovered" in w]
    all_warns += [f for f in shot_flags]
    badge_x = WARP_W - 230
    badge_y = 8
    for w in all_warns:
        label = warning_label(w)
        badge_y = draw_badge(img, badge_x, badge_y, label)

    return img


# ── Not-ready overlay ─────────────────────────────────────────────────────────

NOT_READY_REASONS = {
    "low_quality_warp":   ("LOW QUALITY WARP",   "Image resolution too low\nfor reliable detection"),
    "source_resolution_too_low": ("LOW QUALITY WARP", "Image resolution too low\nfor reliable detection"),
    "cue_missing":        ("NO CUE BALL",        "Cue ball not detected\nin this image"),
    "no_visible_cue":     ("NO VISIBLE CUE",     "Cue ball not visible\nfrom this angle"),
    "cue_at_edge":        ("CUE AT EDGE",        "Cue ball at table edge —\ndetection unreliable"),
    "too_few_balls":      ("TOO FEW BALLS",      "Not enough balls detected\nto plan a shot"),
    "not_ready_for_planner": ("NOT READY",       "Perception quality\nnot sufficient for planning"),
}


def draw_not_ready_overlay(warp: np.ndarray, reason: str, detail: str = "") -> np.ndarray:
    img = warp.copy()

    # Darken background
    dark = np.zeros_like(img)
    cv2.addWeighted(dark, 0.55, img, 0.45, 0, img)

    # Central card
    cw, ch = 420, 160
    cx = (WARP_W - cw) // 2
    cy = (WARP_H - ch) // 2
    card = img.copy()
    cv2.rectangle(card, (cx, cy), (cx + cw, cy + ch), (20, 20, 20), -1)
    cv2.addWeighted(card, 0.85, img, 0.15, 0, img)
    cv2.rectangle(img, (cx, cy), (cx + cw, cy + ch), (60, 60, 80), 2)

    # "NO RECOMMENDATION" header
    header = "NO RECOMMENDATION"
    (hw, hh), _ = cv2.getTextSize(header, FONT_BOLD, 0.7, 2)
    cv2.putText(img, header,
                (cx + (cw - hw) // 2, cy + 38),
                FONT_BOLD, 0.7, (80, 80, 240), 2, cv2.LINE_AA)

    # Separator line
    cv2.line(img, (cx + 20, cy + 50), (cx + cw - 20, cy + 50), (80, 80, 100), 1)

    # Reason title
    r_title, r_body = NOT_READY_REASONS.get(reason, (reason.upper(), detail or ""))
    (tw, _), _ = cv2.getTextSize(r_title, FONT_BOLD, 0.6, 1)
    cv2.putText(img, r_title,
                (cx + (cw - tw) // 2, cy + 78),
                FONT_BOLD, 0.6, (200, 200, 255), 1, cv2.LINE_AA)

    # Body text (may have \n)
    body = detail if detail else r_body
    lines = body.split("\n")
    for i, line in enumerate(lines):
        (lw, lh), _ = cv2.getTextSize(line, FONT, 0.48, 1)
        cv2.putText(img, line,
                    (cx + (cw - lw) // 2, cy + 108 + i * 22),
                    FONT, 0.48, (160, 160, 160), 1, cv2.LINE_AA)

    return img


# ── Reason resolution ─────────────────────────────────────────────────────────

def resolve_not_ready_reason(perc_data: dict, plan_data: dict) -> tuple[str, str]:
    warns = perc_data.get("status", {}).get("warnings", [])
    if any("low_quality_warp" in w or "source_resolution" in w for w in warns):
        return "low_quality_warp", ""
    if not perc_data.get("status", {}).get("cue_present", False):
        # Check if cue was at edge
        for b in perc_data.get("balls", []):
            if b["type"] == "cue_ball" and b["x"] < 30:
                return "cue_at_edge", "Cue at x<30px — near edge\nof warped table"
        ball_count = perc_data.get("status", {}).get("ball_count", 0)
        if ball_count <= 1:
            return "too_few_balls", f"Only {ball_count} ball(s) detected"
        return "cue_missing", ""
    plan_status = plan_data.get("plan", {}).get("status", "")
    if plan_status == "no_cue":
        return "no_visible_cue", ""
    if plan_status == "too_few_balls":
        bc = perc_data.get("status", {}).get("ball_count", 0)
        return "too_few_balls", f"Only {bc} ball(s) detected"
    return "not_ready_for_planner", plan_status


# ── Main ──────────────────────────────────────────────────────────────────────

def process_stem(stem: str) -> dict:
    perc_json = PERC_DIR / stem / f"{stem}_perception.json"
    plan_json = PLAN_DIR / stem / f"{stem}_plan.json"

    if not perc_json.exists():
        return {"stem": stem, "status": "missing_perception"}
    if not plan_json.exists():
        return {"stem": stem, "status": "missing_plan"}

    perc_data = json.loads(perc_json.read_text())
    plan_data = json.loads(plan_json.read_text())

    warp = load_warp(stem)
    if warp is None:
        return {"stem": stem, "status": "missing_warp"}

    plan_status = plan_data.get("plan", {}).get("status", "")
    is_ready = plan_status == "plan_ready"

    if is_ready:
        overlay = draw_ready_overlay(warp, plan_data, perc_data)
        sel = plan_data["plan"]["selected"]
        out_meta = {
            "stem": stem,
            "status": "plan_ready",
            "confidence": plan_data.get("confidence"),
            "selected_shot": {
                "ob_id":    sel["ob_id"],
                "pocket":   sel["pocket"],
                "cut_deg":  sel["cut_deg"],
                "cue_dist": sel.get("cue_dist"),
                "ob_dist":  sel.get("ob_dist"),
                "score":    sel.get("score"),
            },
            "quality_flags": plan_data.get("quality_flags", []),
            "perception_warnings": perc_data.get("status", {}).get("warnings", []),
        }
    else:
        nr_reason, nr_detail = resolve_not_ready_reason(perc_data, plan_data)
        overlay = draw_not_ready_overlay(warp, nr_reason, nr_detail)
        out_meta = {
            "stem": stem,
            "status": "no_plan",
            "reason": nr_reason,
            "confidence": plan_data.get("confidence"),
            "perception_warnings": perc_data.get("status", {}).get("warnings", []),
        }

    out_img  = OUT_DIR / f"{stem}_mvp.jpg"
    out_json = OUT_DIR / f"{stem}_mvp.json"
    cv2.imwrite(str(out_img), overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])
    out_json.write_text(json.dumps(out_meta, indent=2))

    return out_meta


def build_report(results: list[dict]) -> str:
    ready   = [r for r in results if r.get("status") == "plan_ready"]
    no_plan = [r for r in results if r.get("status") == "no_plan"]
    errors  = [r for r in results if r.get("status") not in ("plan_ready", "no_plan")]

    lines = [
        "# MVP Overlay — Summary Report",
        "",
        "## Pipeline Status",
        f"- **Planner-ready**: {len(ready)} / {len(results)}",
        f"- **No-plan**: {len(no_plan)} / {len(results)}",
        "",
        "## Selected Shots",
        "",
        "| Image | Pocket | Cut° | Dist (px) | Conf | Flags |",
        "|-------|--------|------|-----------|------|-------|",
    ]
    for r in sorted(ready, key=lambda x: x.get("confidence", 0), reverse=True):
        ss = r.get("selected_shot", {})
        flags = ", ".join(r.get("quality_flags", []) +
                          [w for w in r.get("perception_warnings", []) if "recover" in w])
        lines.append(
            f"| {r['stem']} | {ss.get('pocket','?')} | {ss.get('cut_deg','?'):.1f} "
            f"| {int((ss.get('cue_dist') or 0) + (ss.get('ob_dist') or 0))} "
            f"| {r.get('confidence','?'):.2f} | {flags or '—'} |"
        )

    lines += [
        "",
        "## Confidence Distribution",
        "",
    ]
    confs = [r.get("confidence", 0) for r in ready if r.get("confidence") is not None]
    if confs:
        high = sum(1 for c in confs if c >= 0.8)
        med  = sum(1 for c in confs if 0.5 <= c < 0.8)
        low  = sum(1 for c in confs if c < 0.5)
        avg  = sum(confs) / len(confs)
        lines += [
            f"- High (≥0.80): {high}",
            f"- Medium (0.50–0.79): {med}",
            f"- Low (<0.50): {low}",
            f"- Average: {avg:.2f}",
        ]

    lines += [
        "",
        "## No-Plan Cases",
        "",
        "| Image | Reason |",
        "|-------|--------|",
    ]
    for r in no_plan:
        lines.append(f"| {r['stem']} | {r.get('reason', '?')} |")

    lines += [
        "",
        "## Warning Summary",
        "",
    ]
    all_flags: dict[str, list[str]] = {}
    for r in ready:
        for f in r.get("quality_flags", []) + r.get("perception_warnings", []):
            all_flags.setdefault(f, []).append(r["stem"])
    if all_flags:
        for flag, stems in sorted(all_flags.items()):
            lines.append(f"- **{flag}**: {', '.join(stems)}")
    else:
        lines.append("- None")

    lines += [
        "",
        "## Remaining Blockers Before App Integration",
        "",
        "| Blocker | Count | Images |",
        "|---------|-------|--------|",
    ]
    blockers = []
    lqw = [r for r in no_plan if r.get("reason") == "low_quality_warp"]
    if lqw:
        blockers.append(("low_quality_warp — unusable images",
                         len(lqw), ", ".join(r["stem"] for r in lqw)))
    cue_miss = [r for r in no_plan if r.get("reason") in ("cue_missing", "cue_at_edge", "no_visible_cue")]
    if cue_miss:
        blockers.append(("cue not detected — more labels or augmentation needed",
                         len(cue_miss), ", ".join(r["stem"] for r in cue_miss)))
    low_c = [r for r in ready if (r.get("confidence") or 1.0) < 0.5]
    if low_c:
        blockers.append((f"low-confidence plans (conf<0.50) — review perception quality",
                         len(low_c), ", ".join(r["stem"] for r in low_c)))
    if not blockers:
        blockers.append(("None — pipeline is MVP-ready", 0, "—"))
    for b, n, imgs in blockers:
        lines.append(f"| {b} | {n} | {imgs} |")

    if errors:
        lines += ["", "## Processing Errors", ""]
        for r in errors:
            lines.append(f"- {r['stem']}: {r.get('status')}")

    lines += ["", f"*Generated by mvp_overlay.py*", ""]
    return "\n".join(lines)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stems = sorted(
        d.name for d in PERC_DIR.iterdir()
        if d.is_dir() and (d / f"{d.name}_perception.json").exists()
    )

    results = []
    for stem in stems:
        r = process_stem(stem)
        status_icon = "✓" if r.get("status") == "plan_ready" else "✗"
        conf = r.get("confidence")
        conf_str = f"  conf={conf:.2f}" if conf is not None else ""
        reason = f"  [{r.get('reason', r.get('status'))}]" if r.get("status") != "plan_ready" else ""
        print(f"  {status_icon} {stem}{conf_str}{reason}")
        results.append(r)

    ready_count   = sum(1 for r in results if r.get("status") == "plan_ready")
    no_plan_count = sum(1 for r in results if r.get("status") == "no_plan")

    report_md = build_report(results)
    report_path = OUT_DIR / "mvp_summary_report.md"
    report_path.write_text(report_md)

    print()
    print("=" * 60)
    print(f"  Plan ready : {ready_count} / {len(results)}")
    print(f"  No plan    : {no_plan_count} / {len(results)}")
    print(f"  Overlays   → {OUT_DIR}")
    print(f"  Report     → {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
