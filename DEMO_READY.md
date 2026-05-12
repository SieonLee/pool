# Billiards AI — Demo Guide
**Frozen: 2026-05-11 | Controlled-photo MVP — not general-purpose deployment**
**Perception patch: 2026-05-11 — cloth brightness threshold corrected (CLOTH_MAX_V 238→255)**

---

## Quick Start

```bash
# Single image — recommended for demo
python scripts/run_full_pipeline.py --image picture/pool_real_021.jpg

# Batch all images + performance report
python scripts/batch_demo.py

# All outputs land in the same directory as the input image (or --out-dir):
#   <stem>_demo.jpg      ← recommended demo output (three-panel composite)
#   <stem>_overlay.jpg   ← single-panel shot overlay
#   <stem>_result.json   ← machine-readable full result
```

All three output files are **always produced**, even on failure cases.

---

## Recommended Demo Output

The primary demo artifact is the **three-panel composite** (`<stem>_demo.jpg`, 900×1470px):

```
┌──────────────────────────────────────┐
│  01  DETECTION                       │  ← all detected balls, IDs, corner conf
│  (450px panel)                       │
├──────────────────────────────────────┤
│  02  TACTICAL                        │  ← top-3 candidates, ghost ball, all pockets
│  (450px panel)                       │
├──────────────────────────────────────┤
│  03  RECOMMENDED SHOT                │  ← clean selected shot + confidence badge
│  (450px panel)                       │
└──────────────────────────────────────┘
```

On failure, panels 2 and 3 show a dark overlay with a specific reason card
(`NO CUE BALL DETECTED`, `TABLE NOT FOUND`, `NO VALID SHOT FOUND`).
Panel 1 always shows what the detector found.

---

## Best Showcase Images

These images produce the cleanest demos. Use for portfolio screenshots and live demo.

| Image | Balls | Confidence | Shot | Notes |
|-------|-------|-----------|------|-------|
| `pool_real_021` | 6 | **1.00** | ob4 → TR, cut=8.2° | Clean overhead, no flags |
| `pool_real_033` | 9 | **1.00** | ob5 → ML, cut=2.7° | Near-straight mid pocket |
| `pool_real_034` | 8 | **1.00** | ob2 → BR, cut=0.0° | Perfectly straight shot |
| `pool_real_023` | 16 | 0.95 | ob12 → MR, cut=4.5° | Full rack, still finds shot |
| `pool_real_024` | 13 | 0.90 | ob2 → TL, cut=21.3° | Dense layout, planner succeeds |
| `pool_real_036` | varies | 0.90 | ob1 → TR, cut=12.6° | Clean geometry |
| `maxresdefault` | 6 | 0.80 | ob5 → ML, cut=10.0° | Bar/oblique angle — real-world test image |

All demo composites pre-generated: `review/demo/<stem>_demo.jpg`  
Showcase contact sheets: `review/showcase/showcase_best.jpg`, `showcase_confidence.jpg`

---

## Expected Failure Types

These failures are **by design** — the system correctly identifies that it cannot make a recommendation and says so clearly.

| Failure type | Example images | System response | Why correct |
|-------------|---------------|-----------------|-------------|
| No cue ball in scene | `pool_real_009`, `pool_real_020`, `pool_real_025` | `cue_missing` — dark card panel 2+3 | Cue genuinely not visible |
| Table not detectable | `pool_real_032` (gray cloth) | `no_table_detected` — black canvas | Corner model not trained on gray felt |
| All shot paths blocked | `pool_real_007` (16 balls, full rack) | `no_candidates` + rejection counts | Ghost-ball geometry: no clear path exists |
| Low-resolution source | `pool_real_002`, `pool_real_010` | `low_quality_warp` | Source too small for reliable warp |

Showcase of failure cases: `review/showcase/showcase_failures.jpg`

---

## Input Photo Requirements

| Requirement | Target |
|-------------|--------|
| Camera angle | Overhead or steep downward (≥45° from table surface) |
| Cloth color | Green or blue — gray/dark fails corner detection |
| Resolution | 720p or higher (1080p ideal) |
| Ball count | 3–15 (8–12 best for interesting shots) |
| Motion | Static — no blur, no action |
| Cue ball | Clearly visible, not fully hidden in cluster |
| Lighting | Works from dim indoor to bright bar overhead lighting. Heavy shadows degrade warp quality. Pure backlighting (silhouette) will fail. |

---

## Performance

All measurements on CPU (Apple M-series), 27 real images:

| Stage | Mean latency |
|-------|-------------|
| Corner detection (YOLO keypoint) | 28ms |
| Perspective warp | 1ms |
| Ball detector (YOLO11n) | 33ms |
| Shot planner | 0.1ms |
| Demo render (3-panel) | 4ms |
| **End-to-end p50** | **65ms** |
| **End-to-end p90** | **97ms** |

---

## Production Configuration (Frozen)

```
Ball detector  : models/checkpoints/ball_yolo_v7_below_baseline.pt
Cue classifier : models/checkpoints/cue_id_v3.pt   threshold=0.30
cue_conf       : 0.25   (low for recall; classifier handles FP suppression)
obj_conf       : 0.35
MAX_CUT_DEG    : 70°
Selector       : OFF    (geometric scorer only)
Device         : CPU
```

Do not change any of these before the demo — see **Do Not Touch** section below.

---

## Status Codes

| Status | Meaning |
|--------|---------|
| `plan_ready` | Full pipeline success — shot selected and rendered |
| `no_candidates` | Balls detected, cue found, but no shot geometry clears all checks |
| `not_ready_for_planner` | Perception failed — cue missing or table not found |

## Quality Flags (appear on plan_ready results)

| Flag | Meaning | Confidence penalty |
|------|---------|-------------------|
| `low_corner_confidence` | Table warp may be slightly off | −15% |
| `low_cue_confidence` | Cue ball detection uncertain | −20% |
| `extreme_thin_cut` | Cut angle > 55° — hard shot | −40% |
| `long_shot` | Combined path > 700px | −10% |
| `near_rail_cue_selected` | Cue ball near rail | −5% |
| `near_rail_ghost_selected` | Ghost position near rail | −5% |
| `cue_object_overlap` | Cue within 35px of object ball | −20% |
| `high_ball_count` | ≥13 balls (crowded) | −5% |

---

## Portfolio Blurb

### Non-Technical

I built an end-to-end AI system that looks at a photo of a pool table and tells you which shot to take. You hand it a picture, and within 65–97 milliseconds it figures out where all the balls are, which ones you could reasonably pocket, and draws you a clean diagram showing exactly how to line up the cue. It handles messy real-world photos — different lighting, different cloth colors, crowded layouts — and when it genuinely can't make a recommendation, it tells you why rather than guessing.

### Technical

The pipeline chains four learned/geometric stages: a YOLO keypoint model finds the four table rails and warps the image to a canonical 900×450 top-down space; YOLO11n detects balls in two classes with class-specific confidence thresholds; a MobileNetV3-Small binary classifier gates each cue-ball prediction to suppress false detections (precision=0.909, recall=1.000 on the held-out eval set); and a pure-Python ghost-ball planner evaluates every object-ball × pocket pair for geometric feasibility (cut angle, path occlusion, ghost position bounds) and selects the minimum-score shot. End-to-end: p50=65ms, p90=97ms on CPU.

### Engineering Highlights

- **Cue hallucination debugging** — Diagnosed and fixed a false-cue problem where the detector would hallucinate 2–5 cue balls per image. Root cause: YOLO fires on any bright low-saturation region near the rail, which visually overlaps with the cue. Solution: a hard-negative mining pipeline that extracts YOLO false-positive crops (including from empty-label images) and trains a binary identity classifier as a post-hoc gate.
- **Hard-negative mining failure** — An initial attempt at whiteness hard-negatives (mine the fallback-path false positives as negatives) caused a catastrophic regression (cue score 0.869→0.001 on held-out images). Root cause: the whiteness candidates and real cue balls share the same HSV feature space. Permanently excluded; documented so no one repeats it.
- **Deterministic, reproducible training** — Fixed random seed, pinned validation split (by stem, not by index), timestamped checkpoint naming. Any training run can be exactly reproduced months later.
- **Geometry-aware acceptance gate** — Checkpoints are accepted only by running the full stress-test pipeline and comparing plan_ready_rate and cue_missing_rate against rate-based thresholds (not absolute counts), so the gate scales as the image set grows.
- **Three-panel demo overlay** — Each output shows the AI's full reasoning: what it detected (panel 1), which shots it considered (panel 2), and the final recommendation (panel 3). Failure panels replace panels 2+3 with a specific reason card.

---

## ⛔ Do Not Touch Before Demo

The following are **frozen**. Do not change any of these until after the demo has been recorded/presented.

| Item | Status | Why frozen |
|------|--------|-----------|
| Ball detector checkpoint | `ball_yolo_v7_below_baseline.pt` | Best cue_recall (0.913) after 7 training runs |
| Cue classifier checkpoint | `cue_id_v3.pt` | prec=0.909, rec=1.000 — best to date |
| `cue_conf` threshold | 0.25 | Tuned: lower drops plan_ready, higher drops cue recall |
| `obj_conf` threshold | 0.35 | +1 plan_ready vs 0.30, 0 regressions |
| `cue_id_threshold` | 0.30 | PR-curve operating point — tightening drops recall to 0 |
| `MAX_CUT_DEG` | 70° | Calibrated for recreational play |
| Selector | OFF | Selector-on mode not validated on stress set |
| Planner scoring function | `cut + 0.02*dist` | Balanced after empirical review |
| Geometry validation bounds | `ghost ± 5px`, `path_radius = ball_r` | Changing breaks existing plan_ready images |
| Training dataset | 27 images + exclusion list | Any retrain risks regression on stress set |

**Do not retrain.** If a new training run is needed after the demo, start a new experiment branch, run the full acceptance gate (`accept_candidate.py`), and compare against `baseline_metrics.json`.

---

## Cloth / Playable-Region Filter

Added in perception pipeline after ball detection. Segments the playing surface by dominant HSV hue in the warped image and rejects ball detections whose centres land off the cloth.

### How it works

1. **Candidate pixels** — V ≥ 45 (exclude dark rails) and S ≥ 16 (exclude white chalk/cue ball). No upper-V cap — bright lit cloth (V=240-255) is valid and must pass.
2. **Dominant hue histogram** — 36-bin histogram over candidate pixels; largest bin → cloth hue.
3. **Hue mask** — ±24 hue units around dominant hue (wrap-safe for red cloth).
4. **Morphological closing** — two-pass close: small (9px) for noise/chalk marks, large (55px) to fill ball-sized holes in the mask (balls ≠ cloth hue → appear as holes).
5. **Largest connected component** — keeps only the main cloth region.
6. **Ball membership** — centre ± 4px sampled; any cloth pixel → ball is kept.

### Coverage sanity checks

| Coverage | Action |
|----------|--------|
| < 30% | `loose_table_polygon` hard flag — cloth detection failed |
| 30–48% | `low_cloth_coverage` soft warning |
| ≥ 48% | Normal |

### Known remaining limitation

If the corner model produces a **loose polygon** (e.g. corner confidence < 0.75), the warp includes background floor/chairs/rails. The cloth filter will still detect the dominant hue correctly — but if background objects happen to share the cloth hue, they may not be rejected. The cloth filter cannot compensate for a fundamentally wrong polygon. Fix: improve corner localization or add polygon tightening heuristics (post-demo work).

### Bug fixed: 2026-05-11

`CLOTH_MAX_V` was originally set to 238, which incorrectly excluded cloth pixels on bright bar-lit tables (where cloth V ≈ 240-255 under overhead lighting). This caused:
- Coverage to collapse to ~19% on bright-lit images (only shadowed areas passed)
- All detected balls rejected as "outside cloth"
- `pool_real_021`, `pool_real_034` and other correctly-detected images to regress from `plan_ready` to `not_ready_for_planner`

**Fix**: Raised `CLOTH_MAX_V` to 255 (no upper brightness cap). White glare and chalk are excluded by `CLOTH_MIN_S=16` (saturation), not by brightness. After fix: 17/27 plan-ready (63%), up from 13/27 with the broken threshold. All previously plan-ready images restored; two additional images newly plan-ready.

---

## Architecture (Quick Reference)

```
Input photo
  → Corner YOLO (keypoints)      table_corners_mvp_v1.pt
  → Perspective warp (900×450)
  → Ball YOLO11n (2 classes)     ball_yolo_v7_below_baseline.pt
  → Cue identity gate            cue_id_v3.pt  threshold=0.30
  → Ghost-ball planner           geometry only, pure Python
  → Demo overlay                 demo_overlay.draw_demo_composite()
  → <stem>_demo.jpg + _overlay.jpg + _result.json
```

Full engineering detail: [`review/technical_summary.md`](review/technical_summary.md)
