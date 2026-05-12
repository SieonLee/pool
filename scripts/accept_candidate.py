"""
Evaluate a candidate ball detector checkpoint against acceptance criteria.

Runs the full eval pipeline (eval_ball_yolo + stress_test), reads JSON outputs,
and prints ACCEPT or REJECT. With --promote, copies the candidate to
models/checkpoints/ball_yolo_active.pt only on ACCEPT.

Acceptance criteria (all must pass):
  Rate-based (scale with stress-set size):
    - plan_ready_rate >= 63.6%  (= 21/33 original baseline)
    - cue_missing_rate <= 27.3% (= 9/33 original baseline)
  Hard fails (absolute, regardless of set size):
    - cue_ball recall > 0  (from eval_ball_yolo summary)
    - pool_real_015 must NOT be plan_ready (LQ-warp gating intact)

Run:
  python scripts/accept_candidate.py --candidate models/checkpoints/ball_yolo_candidate_<ts>.pt
  python scripts/accept_candidate.py --candidate ... --promote
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).parent.parent
CKPT_DIR = BASE / "models" / "checkpoints"
ACTIVE_CKPT = CKPT_DIR / "ball_yolo_active.pt"
EVAL_SUMMARY = BASE / "review" / "pose_warped_ball_yolo" / "summary.json"
STRESS_RESULTS = BASE / "review" / "stress_test" / "stress_results.json"

# Rate-based thresholds derived from the original 21/33 baseline
PLAN_READY_RATE_MIN = 21 / 33          # 63.6%
CUE_MISSING_RATE_MAX = 9 / 33         # 27.3%

# Hard-fail: this image must always be blocked (LQ-warp gating check)
LQ_WARP_SENTINEL = "pool_real_015"

SCRIPTS = Path(__file__).parent


def run(cmd: list[str], label: str) -> int:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(BASE))
    return result.returncode


def compute_cue_recall(eval_results: list[dict]) -> float:
    tp = sum(r.get("tp_cue", 0) for r in eval_results)
    fn = sum(r.get("fn_cue", 0) for r in eval_results)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True,
                        help="Path to candidate checkpoint (.pt)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="eval_ball_yolo confidence threshold (default 0.25)")
    parser.add_argument("--cue-conf", type=float, default=None,
                        help="stress_test cue_ball conf threshold (default: stress_test default)")
    parser.add_argument("--obj-conf", type=float, default=None,
                        help="stress_test object_ball conf threshold (default: stress_test default)")
    parser.add_argument("--promote", action="store_true",
                        help="Copy candidate to ball_yolo_active.pt if ACCEPT")
    args = parser.parse_args()

    candidate = Path(args.candidate)
    if not candidate.exists():
        print(f"ERROR: candidate not found: {candidate}")
        sys.exit(1)

    print(f"\nCandidate: {candidate.name}")
    print(f"Active:    {ACTIVE_CKPT.name}  ({'exists' if ACTIVE_CKPT.exists() else 'MISSING'})")

    # 1. Run eval_ball_yolo
    rc = run(
        [sys.executable, str(SCRIPTS / "eval_ball_yolo.py"),
         "--checkpoint", str(candidate),
         "--conf", str(args.conf)],
        "Step 1/2 — eval_ball_yolo.py",
    )
    if rc != 0:
        print(f"\nERROR: eval_ball_yolo.py exited with code {rc}")
        sys.exit(1)

    # 2. Run stress_test
    stress_cmd = [sys.executable, str(SCRIPTS / "stress_test.py"),
                  "--ball-ckpt", str(candidate)]
    if args.cue_conf is not None:
        stress_cmd += ["--cue-conf", str(args.cue_conf)]
    if args.obj_conf is not None:
        stress_cmd += ["--obj-conf", str(args.obj_conf)]
    rc = run(stress_cmd, "Step 2/2 — stress_test.py")
    if rc != 0:
        print(f"\nERROR: stress_test.py exited with code {rc}")
        sys.exit(1)

    # 3. Read results
    if not EVAL_SUMMARY.exists():
        print(f"\nERROR: eval summary not found: {EVAL_SUMMARY}")
        sys.exit(1)
    if not STRESS_RESULTS.exists():
        print(f"\nERROR: stress results not found: {STRESS_RESULTS}")
        sys.exit(1)

    eval_summary = json.loads(EVAL_SUMMARY.read_text())
    stress_results = json.loads(STRESS_RESULTS.read_text())

    eval_image_results = eval_summary.get("results", [])
    cue_recall = compute_cue_recall(eval_image_results)

    total = len(stress_results)
    plan_ready = sum(1 for r in stress_results if r.get("status") == "plan_ready")
    cue_missing = sum(
        1 for r in stress_results if "cue_missing" in r.get("categories", [])
    )
    plan_ready_rate = plan_ready / total if total > 0 else 0.0
    cue_missing_rate = cue_missing / total if total > 0 else 0.0

    # LQ-warp sentinel: pool_real_015 must NOT be plan_ready
    sentinel_ok = not any(
        r.get("stem") == LQ_WARP_SENTINEL and r.get("status") == "plan_ready"
        for r in stress_results
    )

    # Thresholds expressed as counts for this run's set size
    plan_ready_needed = PLAN_READY_RATE_MIN * total
    cue_missing_allowed = CUE_MISSING_RATE_MAX * total

    # 4. Evaluate criteria
    checks = [
        (
            f"plan_ready_rate >= {PLAN_READY_RATE_MIN*100:.1f}%"
            f"  (need ≥{plan_ready_needed:.1f}/{total})",
            plan_ready_rate >= PLAN_READY_RATE_MIN,
            f"{plan_ready}/{total}  ({plan_ready_rate*100:.1f}%)",
        ),
        (
            f"cue_missing_rate <= {CUE_MISSING_RATE_MAX*100:.1f}%"
            f"  (allow ≤{cue_missing_allowed:.1f}/{total})",
            cue_missing_rate <= CUE_MISSING_RATE_MAX,
            f"{cue_missing}/{total}  ({cue_missing_rate*100:.1f}%)",
        ),
        (
            "cue_recall > 0  [hard fail]",
            cue_recall > 0,
            f"{cue_recall:.3f}",
        ),
        (
            f"{LQ_WARP_SENTINEL} blocked (LQ-warp gate intact)  [hard fail]",
            sentinel_ok,
            "OK" if sentinel_ok else "BROKEN — sentinel is plan_ready",
        ),
    ]

    print(f"\n{'='*60}")
    print(f"  ACCEPTANCE CHECK — {candidate.name}")
    print(f"  Stress set: {total} images")
    print(f"{'='*60}")
    all_pass = True
    for label, passed, value in checks:
        icon = "✓" if passed else "✗"
        print(f"  {icon}  {label}")
        print(f"       got: {value}")
        if not passed:
            all_pass = False

    print(f"\n{'='*60}")
    if all_pass:
        print(f"  RESULT: ACCEPT")
        if args.promote:
            CKPT_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(candidate), str(ACTIVE_CKPT))
            print(f"  Promoted → {ACTIVE_CKPT}")
        else:
            print(f"  (run with --promote to copy to ball_yolo_active.pt)")
    else:
        print(f"  RESULT: REJECT")
        if args.promote:
            print(f"  (not promoted — criteria not met)")
    print(f"{'='*60}\n")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
