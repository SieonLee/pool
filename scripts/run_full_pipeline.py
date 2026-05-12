"""
Single-image pipeline runner for the Billiards AI MVP.

Usage:
  python scripts/run_full_pipeline.py --image path/to/photo.jpg
  python scripts/run_full_pipeline.py --image photo.jpg --out-dir results/
  python scripts/run_full_pipeline.py --image photo.jpg --ball-ckpt models/checkpoints/ball_yolo_v7_below_baseline.pt

Output (in --out-dir, default: same directory as input image):
  <stem>_overlay.jpg   — single-panel warp overlay with shot visualization
  <stem>_demo.jpg      — three-panel demo composite (detection / tactical / shot)
  <stem>_result.json   — full machine-readable result

Prints a concise status summary to stdout.
"""
import argparse
import json
import sys
import time
from pathlib import Path

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / "scripts"))

DEFAULT_CORNER_CKPT = BASE / "models" / "checkpoints" / "table_corners_mvp_v1.pt"
DEFAULT_BALL_CKPT   = BASE / "models" / "checkpoints" / "ball_yolo_v7_below_baseline.pt"


def main():
    parser = argparse.ArgumentParser(description="Billiards AI — full pipeline on a single image")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: same as input image)")
    parser.add_argument("--ball-ckpt", default=str(DEFAULT_BALL_CKPT),
                        help="Ball detector checkpoint (default: v7)")
    parser.add_argument("--corner-ckpt", default=str(DEFAULT_CORNER_CKPT),
                        help="Corner detector checkpoint")
    parser.add_argument("--cue-conf", type=float, default=0.25,
                        help="Cue-ball confidence threshold (default: 0.25)")
    parser.add_argument("--obj-conf", type=float, default=0.35,
                        help="Object-ball confidence threshold (default: 0.35)")
    parser.add_argument("--no-overlay", action="store_true",
                        help="Skip overlay rendering (faster)")
    parser.add_argument("--no-demo", action="store_true",
                        help="Skip three-panel demo composite rendering")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"ERROR: image not found: {img_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else img_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO
        import cv2
    except ImportError as e:
        print(f"ERROR: missing dependency — {e}", file=sys.stderr)
        print("Run: pip install ultralytics opencv-python", file=sys.stderr)
        sys.exit(1)

    import stress_test as pipeline
    import demo_overlay as demo

    # Override module-level thresholds
    pipeline.CUE_CONF_THRESH  = args.cue_conf
    pipeline.OBJ_CONF_THRESH  = args.obj_conf
    pipeline.BALL_CONF_THRESH = min(args.cue_conf, args.obj_conf)

    t_start = time.perf_counter()

    corner_model = YOLO(str(args.corner_ckpt))
    ball_model   = YOLO(str(args.ball_ckpt))

    perc, warped, timing = pipeline.perceive(corner_model, ball_model, img_path)
    if perc is None:
        print(f"ERROR: could not read image: {img_path}", file=sys.stderr)
        sys.exit(1)

    plan_r, t_plan = pipeline.plan(perc)
    timing["planner_ms"] = round(t_plan * 1000, 1)

    total_ms = round((time.perf_counter() - t_start) * 1000, 1)

    # Build result document
    stem = img_path.stem
    status = plan_r.get("status", "unknown")
    warnings = perc.get("status", {}).get("warnings", [])

    result = {
        "image": img_path.name,
        "stem": stem,
        "status": status,
        "ball_ckpt": Path(args.ball_ckpt).name,
        "thresholds": {"cue_conf": args.cue_conf, "obj_conf": args.obj_conf},
        "perception": perc,
        "plan": plan_r,
        "timing": {**timing, "total_ms": total_ms},
    }

    # Write JSON
    json_path = out_dir / f"{stem}_result.json"
    json_path.write_text(json.dumps(result, indent=2))

    # Resolve failure reason (used by both overlay renderers)
    def _no_plan_reason():
        w = perc.get("status", {}).get("warnings", [])
        if any("low_quality_warp" in x or "no_table" in x for x in w):
            return "low_quality_warp", ""
        if not perc.get("status", {}).get("cue_present", False):
            return "cue_missing", ""
        if status == "no_candidates":
            rj = plan_r.get("rejections", {})
            detail = "  ".join(f"{v} {k}" for k, v in rj.items() if v)
            return "no_candidates", detail
        return "not_ready_for_planner", ""

    # Write single-panel overlay — always produced (both renderers handle warped=None)
    overlay_path = None
    if not args.no_overlay:
        if status == "plan_ready":
            # draw_ready requires warped; plan_ready guarantees table was found
            overlay = pipeline.draw_ready(warped, perc, plan_r)
        else:
            reason, detail = _no_plan_reason()
            overlay = pipeline.draw_no_plan(warped, reason, detail)
        overlay_path = out_dir / f"{stem}_overlay.jpg"
        cv2.imwrite(str(overlay_path), overlay)

    # Write three-panel demo composite — always produced (handles warped=None internally)
    demo_path = None
    if not args.no_demo:
        composite = demo.draw_demo_composite(warped, perc, plan_r, stem=stem)
        demo_path = out_dir / f"{stem}_demo.jpg"
        cv2.imwrite(str(demo_path), composite, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Print summary
    n_balls   = perc.get("status", {}).get("ball_count", 0)
    cue_ok    = perc.get("status", {}).get("cue_present", False)
    corner_cf = (perc.get("table") or {}).get("corner_confidence")

    print()
    print(f"━━━  Billiards AI  —  {img_path.name}  ━━━")
    print(f"  Status       : {status.upper()}")
    print(f"  Balls found  : {n_balls}  (cue={'✓' if cue_ok else '✗'})")
    if corner_cf is not None:
        print(f"  Corner conf  : {corner_cf:.3f}")

    if status == "plan_ready":
        sel  = plan_r["selected"]
        conf = plan_r.get("confidence", "?")
        nc   = plan_r.get("n_candidates", "?")
        flags = plan_r.get("quality_flags", [])
        print(f"  Best shot    : ob={sel['ob_id']} → {sel['pocket']}  cut={sel['cut_deg']}°")
        print(f"  Candidates   : {nc}")
        print(f"  Shot conf    : {conf}")
        if flags:
            print(f"  Quality flags: {', '.join(flags)}")
    elif status == "no_candidates":
        rj = plan_r.get("rejections", {})
        total_rj = sum(rj.values())
        print(f"  Rejections   : {total_rj} total — " +
              ", ".join(f"{k}={v}" for k, v in rj.items() if v))
    elif status == "not_ready_for_planner":
        print(f"  Reason       : {', '.join(warnings) or 'unknown'}")

    if warnings:
        print(f"  Warnings     : {', '.join(warnings)}")

    print(f"  Latency      : {total_ms:.0f}ms")
    print(f"  JSON         : {json_path}")
    print(f"  Overlay      : {overlay_path or '(skipped)'}")
    print(f"  Demo         : {demo_path or '(skipped)'}")
    print()


if __name__ == "__main__":
    main()
