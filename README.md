# Billiards AI — Pool Shot Advisor

An end-to-end computer vision pipeline that analyzes a photo of a pool table and recommends the optimal shot. Given a static image, the system detects the table corners, normalizes the perspective, detects all balls, and uses geometric ghost-ball planning to identify the highest-probability pot.

---

## Quick Start

```bash
# Single image — produces _demo.jpg + _overlay.jpg + _result.json
python scripts/run_full_pipeline.py --image picture/pool_real_021.jpg

# Batch all stress-test images + performance report
python scripts/batch_demo.py
```

**Output — three-panel demo composite** (`<stem>_demo.jpg`, 900×1470px):

| Panel | Content |
|-------|---------|
| **01 DETECTION** | All detected balls with IDs and per-ball YOLO confidence; corner-confidence and ball-count badges |
| **02 TACTICAL** | Ghost ball position, top-3 candidate shot paths (dim), selected path highlighted, all 6 pockets labeled |
| **03 RECOMMENDED SHOT** | Clean selected shot — thick cue/OB paths, color-coded confidence badge, shot details panel |

**Console output** on a successful image:

```
━━━  Billiards AI  —  pool_real_021.jpg  ━━━
  Status       : PLAN_READY
  Balls found  : 6  (cue=✓)
  Corner conf  : 0.973
  Best shot    : ob=4 → TR  cut=8.2°
  Candidates   : 3
  Shot conf    : 1.0
  Latency      : 64ms
  JSON         : pool_real_021_result.json
  Overlay      : pool_real_021_overlay.jpg
  Demo         : pool_real_021_demo.jpg
```

---

## How It Works

```
Input photo
  │
  ├─ Corner detection  ← YOLO keypoint model (4 corners)         ~28ms
  ├─ Perspective warp  ← OpenCV → canonical 900×450px space       ~1ms
  ├─ Ball detection    ← YOLO11n (cls=0 cue / cls=1 object)       ~33ms
  │    cue_conf=0.25 (high recall)  obj_conf=0.35 (suppress FPs)
  ├─ Cue identity gate ← MobileNetV3-Small classifier             <1ms
  │    suppresses false-cue detections; threshold=0.30
  ├─ Shot planning     ← ghost-ball geometry, path blocking        ~0.1ms
  └─ Demo render       ← three-panel composite                    ~4ms
```

**Corner detection** finds the four table rails with a YOLO keypoint model and applies a perspective transform to a canonical 900×450 top-down image. All downstream geometry lives in this fixed space — pocket positions are hard-coded, distances are physically interpretable.

**Ball detection** runs YOLO11n with class-specific thresholds: `cue_conf=0.25` (low — maximizes recall) and `obj_conf=0.35` (higher — suppresses false positives). A whiteness-fallback pass promotes the brightest low-saturation ball to cue when the primary pass misses it.

**Cue identity gate** filters each YOLO cue prediction through a MobileNetV3-Small binary classifier (`cue_id_v3.pt`, threshold=0.30). This suppresses false-cue detections from bright object balls near the rail. Trained on 1740 crops with hard-negative mining; production metrics: precision=0.909, recall=1.000.

**Shot planning** uses ghost-ball geometry: for each object ball × pocket pair, it computes the ghost position, cut angle (rejected if >70°), and sweeps both paths for blocking balls. Scores by `cut_angle + 0.02×(cue_dist + ob_dist)` — favoring straight, short shots.

---

## Performance

Measured on 27 real images, CPU only (Apple M-series):

| Stage | Mean | p50 | p90 |
|-------|------|-----|-----|
| Corner detection | 28ms | — | — |
| Perspective warp | 1ms | — | — |
| Ball YOLO | 33ms | — | — |
| Shot planner | 0.1ms | — | — |
| Demo render (3-panel) | 4ms | — | — |
| **End-to-end total** | **76ms** | **65ms** | **97ms** |

| Metric | Value |
|--------|-------|
| Plan-ready rate (stress set, 27 images) | **19 / 27 (70.4%)** |
| Cue-ball recall | **0.913** |
| Cue identity precision / recall | **0.909 / 1.000** |
| High-confidence shots (conf ≥ 0.80) | **58% of plan-ready** |
| Supported input resolution | ≥ 480×360 (720p+ recommended) |

---

## Production Configuration

| Parameter | Value | Reason |
|-----------|-------|--------|
| Ball detector | `ball_yolo_v7_below_baseline.pt` | Best available, cue_recall=0.913 |
| Cue classifier | `cue_id_v3.pt` | prec=0.909, rec=1.000 at threshold=0.30 |
| `cue_conf` | 0.25 | Low for recall; classifier gate handles FP suppression |
| `obj_conf` | 0.35 | Gained +1 plan_ready over 0.30 with no regressions |
| `cue_id_threshold` | 0.30 | Operating point from PR-curve analysis |
| `MAX_CUT_DEG` | 70° | Shots beyond 70° are unreliable for most players |
| Device | CPU | MPS caused a YOLO crash during training (v4 failure) |
| Selector | Off | Default geometric scorer only; advanced selector not enabled |
| Playable-region filter | Active | Rejects ghost positions within 5px of table edge |

---

## What It Handles

- Mid-game and post-break table layouts (3–15 balls)
- Standard pool cloth: green ✓  blue ✓  purple ✓ (limited)
- Overhead and steep-angle photography (≥45° from table surface)
- Missing or low-confidence cue ball — fallback recovery + identity gate
- Transparent quality flags when output is uncertain (`low_corner_confidence`, `extreme_thin_cut`, etc.)

## Known Limitations

- **Controlled-photo MVP** — not a general-purpose deployment; optimized for the 27-image stress set and similar overhead photos
- Gray or dark table cloth → corner detection fails
- Motion blur / in-progress shots → unreliable detection
- Dense packed rack → may find no valid shot geometry (correct behavior — reported as `no_candidates`)
- Single best shot only — no multi-ball run-out or safety-shot planning

---

## Project Structure

```
billiards/
├── README.md                               this file
├── DEMO_READY.md                           demo guide (best inputs, failure types, status codes)
├── review/
│   ├── technical_summary.md                full engineering deep-dive
│   ├── demo/                               27 demo composites + performance_report.md
│   ├── showcase/                           showcase_best.jpg, showcase_failures.jpg
│   └── stress_test/stress_report.md        model version history and rejection notes
├── picture/                                27-image stress test set
├── models/checkpoints/
│   ├── table_corners_mvp_v1.pt             frozen corner detector
│   ├── ball_yolo_v7_below_baseline.pt      active ball detector
│   └── cue_id_v3.pt                        active cue identity classifier
└── scripts/
    ├── run_full_pipeline.py                ← demo entry point (single image)
    ├── batch_demo.py                       ← batch runner + performance audit
    ├── demo_overlay.py                     three-panel composite renderer
    ├── stress_test.py                      core perception + planner + stress evaluation
    ├── perceive.py                         standalone perception with cue_id integration
    ├── train_ball_yolo.py                  deterministic training (seed=42, pinned val)
    ├── train_cue_classifier.py             cue identity classifier training
    └── accept_candidate.py                 checkpoint acceptance gate
```

---

## Technical Stack

| Component | Technology |
|-----------|-----------|
| Table corner detection | Ultralytics YOLO keypoint model |
| Ball detection | Ultralytics YOLO11n (2-class) |
| Cue identity gate | MobileNetV3-Small, BCEWithLogitsLoss |
| Perspective warp | OpenCV `getPerspectiveTransform` |
| Shot planning | Pure Python ghost-ball geometry |
| Inference | CPU — no GPU required |

---

## Reproducing the Stress Test

```bash
python scripts/stress_test.py \
  --ball-ckpt models/checkpoints/ball_yolo_v7_below_baseline.pt \
  --cue-conf 0.25 --obj-conf 0.35
# Output: review/stress_test/  (overlays + per-image JSON + stress_report.md)
```

Acceptance criteria (from `baseline_metrics.json`): plan_ready_rate ≥ 63.6%, cue_missing_rate ≤ 27.3%, cue_recall > 0, sentinel image (pool_real_015) always blocked.
