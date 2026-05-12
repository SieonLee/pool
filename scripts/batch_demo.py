"""
Batch demo composite generator + performance audit.

Runs the full pipeline on every picture/pool_real_*.jpg and produces:
  review/demo/
    <stem>_demo.jpg          — three-panel composite
    <stem>_result.json       — full result (same as run_full_pipeline)
  review/demo/performance_report.md — latency audit
  review/demo/batch_summary.md      — plan-ready rate, failure breakdown

Usage:
  python scripts/batch_demo.py
  python scripts/batch_demo.py --stems pool_real_013 pool_real_022
  python scripts/batch_demo.py --cue-conf 0.25 --obj-conf 0.35
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / "scripts"))

PICTURE_DIR = BASE / "picture"
OUT_DIR     = BASE / "review" / "demo"

DEFAULT_BALL_CKPT   = BASE / "models" / "checkpoints" / "ball_yolo_v7_below_baseline.pt"
DEFAULT_CORNER_CKPT = BASE / "models" / "checkpoints" / "table_corners_mvp_v1.pt"

SKIP_STEMS = {
    "pool_real_002", "pool_real_010", "pool_real_016",  # low_quality_warp
    "pool_real_031",                                     # harmful empty label
}


def run_batch(stems, corner_model, ball_model, cue_conf, obj_conf):
    from ultralytics import YOLO
    import stress_test as pipeline
    import demo_overlay as demo

    pipeline.CUE_CONF_THRESH  = cue_conf
    pipeline.OBJ_CONF_THRESH  = obj_conf
    pipeline.BALL_CONF_THRESH = min(cue_conf, obj_conf)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for stem in stems:
        img_path = PICTURE_DIR / f"{stem}.jpg"
        if not img_path.exists():
            print(f"  SKIP  {stem}  (file not found)")
            continue

        t_wall = time.perf_counter()

        perc, warped, timing = pipeline.perceive(corner_model, ball_model, img_path)
        if perc is None:
            print(f"  ERROR {stem}  (could not read image)")
            continue

        plan_r, t_plan = pipeline.plan(perc)
        timing["planner_ms"] = round(t_plan * 1000, 1)
        timing["total_ms"]   = round((time.perf_counter() - t_wall) * 1000, 1)

        # Overlay rendering time
        t_render = time.perf_counter()
        if warped is not None:
            composite = demo.draw_demo_composite(warped, perc, plan_r, stem=stem)
            demo_path = OUT_DIR / f"{stem}_demo.jpg"
            cv2.imwrite(str(demo_path), composite, [cv2.IMWRITE_JPEG_QUALITY, 95])
        timing["render_ms"] = round((time.perf_counter() - t_render) * 1000, 1)

        status  = plan_r.get("status", "unknown")
        n_balls = perc.get("status", {}).get("ball_count", 0)
        cue_ok  = perc.get("status", {}).get("cue_present", False)
        conf    = plan_r.get("confidence")
        warns   = perc.get("status", {}).get("warnings", [])
        flags   = plan_r.get("quality_flags", [])

        row = {
            "stem": stem,
            "status": status,
            "n_balls": n_balls,
            "cue_present": cue_ok,
            "confidence": conf,
            "quality_flags": flags,
            "warnings": warns,
            "timing": timing,
        }
        if status == "plan_ready":
            sel = plan_r["selected"]
            row["selected"] = {
                "ob_id": sel["ob_id"],
                "pocket": sel["pocket"],
                "cut_deg": sel["cut_deg"],
                "total_dist": sel.get("total_dist"),
            }

        results.append(row)

        # Write per-image result JSON
        full_result = {
            "stem": stem, "status": status,
            "perception": perc, "plan": plan_r, "timing": timing,
        }
        (OUT_DIR / f"{stem}_result.json").write_text(json.dumps(full_result, indent=2))

        icon = "✓" if status == "plan_ready" else "✗"
        conf_s = f"  conf={conf:.2f}" if conf is not None else ""
        print(f"  {icon} {stem:<28}  {status:<26}{conf_s}  {timing['total_ms']:.0f}ms")

    return results


def build_performance_report(results: list[dict]) -> str:
    timings = [r["timing"] for r in results if "timing" in r]
    if not timings:
        return "# Performance Report\n\nNo timing data.\n"

    def stat(vals):
        if not vals:
            return "—"
        return f"{sum(vals)/len(vals):.1f}ms  (min {min(vals):.1f}  max {max(vals):.1f})"

    keys = ["corner_ms", "warp_ms", "yolo_ms", "planner_ms", "render_ms", "total_ms"]
    labels = {
        "corner_ms":  "Corner detection  (YOLO pose)",
        "warp_ms":    "Perspective warp",
        "yolo_ms":    "Ball detector     (YOLO11n)",
        "planner_ms": "Shot planner",
        "render_ms":  "Demo render       (3-panel)",
        "total_ms":   "End-to-end total",
    }

    lines = [
        "# Performance Audit",
        "",
        f"**Images**: {len(results)}  |  **Device**: CPU  |  **YOLO**: YOLO11n",
        "",
        "## Stage Latency",
        "",
        "| Stage | Mean | Min | Max |",
        "|-------|------|-----|-----|",
    ]
    for k in keys:
        vals = [t[k] for t in timings if k in t]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        mn   = min(vals)
        mx   = max(vals)
        lines.append(f"| {labels[k]} | {mean:.1f}ms | {mn:.1f}ms | {mx:.1f}ms |")

    all_total = [t["total_ms"] for t in timings]
    if all_total:
        p50 = sorted(all_total)[len(all_total)//2]
        p90 = sorted(all_total)[int(len(all_total)*0.9)]
        lines += [
            "",
            "## Latency Distribution (end-to-end)",
            "",
            f"- p50: {p50:.0f}ms",
            f"- p90: {p90:.0f}ms",
            f"- Fastest: {min(all_total):.0f}ms",
            f"- Slowest: {max(all_total):.0f}ms",
            "",
            "## Per-Image Breakdown",
            "",
            "| Image | Corner | Warp | YOLO | Planner | Render | Total |",
            "|-------|--------|------|------|---------|--------|-------|",
        ]
        def _ms(t, k, fmt=".0f"):
            v = t.get(k)
            return f"{v:{fmt}}ms" if v is not None else "—"

        for r in sorted(results, key=lambda x: x["timing"].get("total_ms", 0)):
            t = r["timing"]
            lines.append(
                f"| {r['stem']} "
                f"| {_ms(t,'corner_ms')} "
                f"| {_ms(t,'warp_ms')} "
                f"| {_ms(t,'yolo_ms')} "
                f"| {_ms(t,'planner_ms','.1f')} "
                f"| {_ms(t,'render_ms')} "
                f"| {_ms(t,'total_ms')} |"
            )

    lines += ["", "*Generated by batch_demo.py*", ""]
    return "\n".join(lines)


def build_batch_summary(results: list[dict]) -> str:
    ready    = [r for r in results if r["status"] == "plan_ready"]
    no_cands = [r for r in results if r["status"] == "no_candidates"]
    not_rdy  = [r for r in results if r["status"] == "not_ready_for_planner"]
    other    = [r for r in results if r["status"] not in
                ("plan_ready", "no_candidates", "not_ready_for_planner")]

    lines = [
        "# Batch Demo — Summary",
        "",
        "## Pipeline Status",
        "",
        f"| Status | Count |",
        f"|--------|-------|",
        f"| Plan ready | {len(ready)} / {len(results)} |",
        f"| No candidates | {len(no_cands)} |",
        f"| Not ready (perception) | {len(not_rdy)} |",
        f"| Other | {len(other)} |",
        "",
        "## Plan-Ready Results",
        "",
        "| Image | OB→Pocket | Cut° | Dist | Conf | Flags |",
        "|-------|-----------|------|------|------|-------|",
    ]
    for r in sorted(ready, key=lambda x: -(x.get("confidence") or 0)):
        s = r.get("selected", {})
        flags = ", ".join(r.get("quality_flags", []))
        lines.append(
            f"| {r['stem']} "
            f"| ob{s.get('ob_id','?')} → {s.get('pocket','?')} "
            f"| {s.get('cut_deg','?'):.1f} "
            f"| {int(s.get('total_dist') or 0)}px "
            f"| {r.get('confidence', 0):.2f} "
            f"| {flags or '—'} |"
        )

    lines += [
        "",
        "## Not-Ready Cases",
        "",
        "| Image | Balls | Cue | Warnings |",
        "|-------|-------|-----|----------|",
    ]
    for r in not_rdy + no_cands:
        warns = ", ".join(r.get("warnings", []))
        lines.append(
            f"| {r['stem']} | {r['n_balls']} "
            f"| {'✓' if r['cue_present'] else '✗'} "
            f"| {warns or '—'} |"
        )

    lines += [
        "",
        "## Confidence Distribution",
        "",
    ]
    confs = [r["confidence"] for r in ready if r.get("confidence") is not None]
    if confs:
        high = sum(1 for c in confs if c >= 0.80)
        med  = sum(1 for c in confs if 0.50 <= c < 0.80)
        low  = sum(1 for c in confs if c < 0.50)
        lines += [
            f"- High (≥0.80): {high}  ({100*high/len(confs):.0f}%)",
            f"- Med  (0.50–0.79): {med}  ({100*med/len(confs):.0f}%)",
            f"- Low  (<0.50): {low}  ({100*low/len(confs):.0f}%)",
            f"- Average: {sum(confs)/len(confs):.2f}",
        ]

    lines += ["", "*Generated by batch_demo.py*", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Batch demo composite generator + performance audit")
    parser.add_argument("--stems", nargs="+", default=None,
                        help="Specific stems to process (default: all pool_real_*)")
    parser.add_argument("--ball-ckpt", default=str(DEFAULT_BALL_CKPT))
    parser.add_argument("--corner-ckpt", default=str(DEFAULT_CORNER_CKPT))
    parser.add_argument("--cue-conf", type=float, default=0.25)
    parser.add_argument("--obj-conf", type=float, default=0.35)
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    corner_model = YOLO(str(args.corner_ckpt))
    ball_model   = YOLO(str(args.ball_ckpt))

    if args.stems:
        stems = args.stems
    else:
        all_imgs = sorted(PICTURE_DIR.glob("pool_real_*.jpg"))
        stems = [p.stem for p in all_imgs if p.stem not in SKIP_STEMS]

    print(f"\n  Processing {len(stems)} images → {OUT_DIR}\n")
    results = run_batch(stems, corner_model, ball_model, args.cue_conf, args.obj_conf)

    # Write reports
    perf_report  = build_performance_report(results)
    batch_report = build_batch_summary(results)

    perf_path  = OUT_DIR / "performance_report.md"
    batch_path = OUT_DIR / "batch_summary.md"
    perf_path.write_text(perf_report)
    batch_path.write_text(batch_report)

    ready = sum(1 for r in results if r["status"] == "plan_ready")
    total = len(results)
    print(f"\n  Plan ready : {ready} / {total}  ({100*ready/total:.1f}%)")
    print(f"  Demo images → {OUT_DIR}")
    print(f"  Performance → {perf_path}")
    print(f"  Summary     → {batch_path}\n")


if __name__ == "__main__":
    main()
