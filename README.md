# Pool

This repository contains a billiards perception and shot recommendation MVP.

The project takes a single table photo, estimates the table area, warps it into a top-down view, detects balls, identifies the cue ball, and produces a simple geometric shot recommendation. The current public version is a cleaned export of the working project, so large sample photos, generated review artifacts, and model weight files are intentionally excluded.

## What is in this repo

- `scripts/`
  Core pipeline, evaluation, training, and audit scripts
- `annotations/`
  Reviewed table and ball annotation data in JSON form
- `datasets/`
  Exported labels and dataset metadata that do not include the original image files
- `models/`
  Lightweight experiment metadata such as configs and result CSVs
- `DEMO_READY.md`
  Notes about the frozen demo configuration and expected behavior

## Pipeline summary

The MVP pipeline is organized as:

1. table localization
2. perspective warp
3. ball detection
4. cue-ball identification
5. geometric shot planning
6. overlay rendering

The more recent work in this project focused on reliability audits. In particular:

- table-constrained filtering was added so background hallucinations do not leak into planner input
- geometry-aware validation was added to catch bad warps and invalid shot paths
- selector-off was frozen as the stable demo path
- cue-ball confusion was audited separately from planner logic

## Running the project

Some scripts expect local image assets and model checkpoints that are not included in this public export. That means the code and dataset structure are here, but a few entry points will not run end to end unless you restore the private assets on your machine.

Representative commands:

```bash
python scripts/run_full_pipeline.py
python scripts/perceive.py
python scripts/plan.py
python scripts/stress_test.py
```

A representative local run on `pool_real_021.jpg` produced:

```text
Status      : PLAN_READY
Balls found : 6 (cue present)
Best shot   : ob=4 -> TR
Confidence  : 1.0
Latency     : 144ms
```

The pipeline writes three main outputs for a successful run:

- `<stem>_result.json`
- `<stem>_overlay.jpg`
- `<stem>_demo.jpg`

If you are trying to understand the project, the best starting points are:

- `scripts/run_full_pipeline.py`
- `scripts/perceive.py`
- `scripts/stress_test.py`
- `DEMO_READY.md`

## Current state

This is still an MVP, not a production system.

What is reasonably solid in the frozen version:

- table-surface enforcement prevents obvious off-table hallucinations
- selector-off is the documented demo path
- geometry-aware validation is kept on
- audits and failure analysis scripts are included

What is still weak:

- table localization can still be the main bottleneck on hard views
- ball recall drops on rail-adjacent, dark, or partially visible balls
- cue-ball precision is still fragile because bright object-ball regions can look cue-like

## Notes on missing files

This repository does not include:

- sample and real photo assets
- generated review images
- local debug outputs
- model checkpoint binaries

Those were left out on purpose so the public repository only contains code, annotations, labels, and documentation.
