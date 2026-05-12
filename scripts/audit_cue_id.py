"""
Before/after cue identity audit.

Runs the full perception pipeline on all 38 stress-test images in two modes:
  BEFORE: existing pipeline (YOLO only + whiteness fallback)
  AFTER:  YOLO + cue identity classifier gate

Metrics reported:
  - Cue precision (TP / (TP + FP))
  - Cue recall    (TP / (TP + FN))
  - False cue rate per image
  - High-confidence false cues (YOLO conf ≥ 0.7 and ≥ 0.9)
  - Plan-ready count change
  - Per-image cue status (correct / false_cue / missed_cue / cue_missing)

GT cue locations come from the YOLO warp-space label files.
Images not in the training label set are flagged as "no_gt".

Run:
  python scripts/audit_cue_id.py --ckpt models/checkpoints/cue_id_v1.pt --threshold 0.6
  python scripts/audit_cue_id.py --before-only   # just re-measure current pipeline
  python scripts/audit_cue_id.py --ckpt ... --threshold ... --save-crops  # dump FP crops
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

CORNER_CKPT    = BASE / "models" / "checkpoints" / "table_corners_mvp_v1.pt"
BALL_CKPT_V7   = BASE / "models" / "checkpoints" / "ball_yolo_v7_below_baseline.pt"
PICTURE_DIR    = BASE / "picture"
LABEL_DIR      = BASE / "datasets" / "pose-warped-balls" / "labels" / "train"
REPORT_DIR     = BASE / "review" / "cue_id"

WARP_W, WARP_H = 900, 450
CUE_CONF       = 0.25
OBJ_CONF       = 0.35
FALLBACK_CONF  = 0.10
CUE_MIN_BRIGHT = 140
CUE_MAX_SAT    = 120
CUE_MIN_ASPECT = 0.32
MAX_BALLS      = 16
CROP_SIZE      = 64
CROP_CONTEXT   = 1.6

SKIP_STEMS = {"new_uploads_contact_sheet", "search_contact_sheet",
              "thumb", "pexels-photo-10627132"}
LOW_QUALITY_WARP_STEMS = {"pool_real_002", "pool_real_010", "pool_real_016"}

# Images that genuinely have no cue in scene — cue_missing is correct
NO_CUE_STEMS = {"pool_real_020", "pool_real_025", "pool_real_028"}


# ─────────────────────────────────────────────────────────────────────────────
# Re-use stress_test perception helpers (inline to avoid import side-effects)
# ─────────────────────────────────────────────────────────────────────────────

def get_corners(model, img):
    r = model(img, verbose=False)
    if not r or r[0].keypoints is None or len(r[0].keypoints) == 0:
        return None, None
    kps = r[0].keypoints
    best = int(kps.conf.mean(dim=1).argmax()) if kps.conf is not None else 0
    conf = float(kps.conf[best].mean().item()) if kps.conf is not None else None
    return kps.xy[best].cpu().numpy(), conf


def warp_image(img, corners):
    dst = np.float32([[0,0],[WARP_W,0],[WARP_W,WARP_H],[0,WARP_H]])
    M   = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (WARP_W, WARP_H))


def run_yolo(model, warped, conf):
    r = model(warped, verbose=False, conf=conf)
    dets = []
    if r and r[0].boxes is not None:
        for b in r[0].boxes:
            dets.append((int(b.cls.item()), float(b.conf.item()),
                         *b.xyxy[0].cpu().numpy().tolist()))
    return dets


def patch_appearance(warped, x1, y1, x2, y2):
    p = warped[max(0,int(y1)):min(WARP_H,int(y2)),
               max(0,int(x1)):min(WARP_W,int(x2))]
    if p.size == 0:
        return 0.0, 255.0, 0.0
    hsv = cv2.cvtColor(p, cv2.COLOR_BGR2HSV)
    w, h = int(x2)-int(x1), int(y2)-int(y1)
    return float(hsv[:,:,2].mean()), float(hsv[:,:,1].mean()), \
           min(w,h)/max(max(w,h),1)


def detect_balls(ball_model, warped):
    """
    Run YOLO with class-specific thresholds; apply whiteness fallback if no cue.
    Returns (balls, warnings, raw_cue_dets).
    raw_cue_dets = all raw cls=0 predictions with conf >= FALLBACK_CONF (for audit).
    """
    raw  = run_yolo(ball_model, warped, FALLBACK_CONF)
    warns = []

    all_cue_raw = [d for d in raw if d[0] == 0]   # for audit (FP counting)

    cue_dets = sorted([d for d in raw if d[0] == 0 and d[1] >= CUE_CONF],
                      key=lambda d: -d[1])
    obj_dets_raw = [d for d in raw if d[0] == 1]
    obj_dets = sorted([d for d in obj_dets_raw if d[1] >= OBJ_CONF],
                      key=lambda d: -d[1])

    if len(cue_dets) > 1:
        warns.append(f"multi_cue:{len(cue_dets)}")
        cue_dets = [cue_dets[0]]

    combined = cue_dets + obj_dets
    if len(combined) > MAX_BALLS:
        combined = combined[:MAX_BALLS]

    # Whiteness fallback
    recovered = False
    if not cue_dets and obj_dets:
        candidates = []
        for det in obj_dets:
            _, cf, x1, y1, x2, y2 = det
            bright, sat, asp = patch_appearance(warped, x1, y1, x2, y2)
            if bright >= CUE_MIN_BRIGHT and sat <= CUE_MAX_SAT and asp >= CUE_MIN_ASPECT:
                candidates.append((bright - sat, bright, sat, det))
        if candidates:
            _, bright, sat, best = max(candidates, key=lambda x: x[0])
            px = (best[2]+best[4])/2; py = (best[3]+best[5])/2
            combined = [d for d in combined
                        if not (d[0]==1 and abs((d[2]+d[4])/2-px)<5
                                and abs((d[3]+d[5])/2-py)<5)]
            combined.insert(0, (0, best[1], best[2], best[3], best[4], best[5]))
            cue_dets = [combined[0]]
            warns.append(f"cue_recovered_appearance")
            recovered = True

    balls = []
    for i, (cls, cf, x1, y1, x2, y2) in enumerate(combined):
        cx = (x1+x2)/2; cy = (y1+y2)/2; r = max(x2-x1,y2-y1)/2
        balls.append({"id": i, "type": "cue_ball" if cls==0 else "object_ball",
                      "x": round(cx,1), "y": round(cy,1), "r": round(r,1),
                      "confidence": round(cf,3)})

    return balls, warns, all_cue_raw, recovered


# ─────────────────────────────────────────────────────────────────────────────
# Cue identity classifier
# ─────────────────────────────────────────────────────────────────────────────

def load_cue_classifier(ckpt_path):
    import torch
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    arch = ckpt.get("arch", "unknown")

    if "mobilenet" in arch:
        try:
            import torchvision.models as tvm
            import torch.nn as nn
            model = tvm.mobilenet_v3_small(weights=None)
            model.classifier = nn.Sequential(
                nn.Linear(576, 64), nn.Hardswish(), nn.Dropout(0.4), nn.Linear(64,1))
        except ImportError:
            raise RuntimeError("torchvision needed for mobilenet checkpoint")
    else:
        import torch.nn as nn
        class SmallCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3,32,3,padding=1),  nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
                )
                self.head = nn.Sequential(
                    nn.Flatten(), nn.Dropout(0.5), nn.Linear(128*16,64), nn.ReLU(), nn.Linear(64,1))
            def forward(self, x): return self.head(self.features(x))
        model = SmallCNN()

    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt.get("best_row", {})


def crop_ball(warped, cx, cy, r, size=CROP_SIZE, context=CROP_CONTEXT):
    cr = int(math.ceil(r * context))
    x1 = max(0, int(cx-cr)); y1 = max(0, int(cy-cr))
    x2 = min(WARP_W, x1+cr*2); y2 = min(WARP_H, y1+cr*2)
    patch = warped[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    return cv2.resize(patch, (size, size), interpolation=cv2.INTER_LINEAR)


def classifier_score(model, warped, cx, cy, r):
    import torch
    crop = crop_ball(warped, cx, cy, r)
    if crop is None:
        return 0.0
    img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.tensor(img_rgb.transpose(2,0,1)).unsqueeze(0)
    with torch.no_grad():
        logit = model(x).item()
    import torch.nn.functional as F
    return float(torch.sigmoid(torch.tensor(logit)).item())


# ─────────────────────────────────────────────────────────────────────────────
# GT loader
# ─────────────────────────────────────────────────────────────────────────────

def load_gt_cue(stem):
    """
    Returns GT cue centre (cx_px, cy_px) from warped label file, or None.
    Returns 'no_gt' string if no label file exists for this image.
    """
    label_path = LABEL_DIR / f"{stem}_warped.txt"
    if not label_path.exists():
        return "no_gt"
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5 or int(parts[0]) != 0:
            continue
        cx = float(parts[1]) * WARP_W
        cy = float(parts[2]) * WARP_H
        return (cx, cy)
    return None   # label file exists but no cue label


def cue_match(pred_ball, gt_cue_xy, dist_thresh=30):
    """True if predicted cue centre is within dist_thresh px of GT cue."""
    if gt_cue_xy is None or gt_cue_xy == "no_gt":
        return None   # can't evaluate
    px, py = pred_ball["x"], pred_ball["y"]
    gx, gy = gt_cue_xy
    return math.hypot(px - gx, py - gy) <= dist_thresh


# ─────────────────────────────────────────────────────────────────────────────
# Per-image processing
# ─────────────────────────────────────────────────────────────────────────────

def process_image(img_path, corner_model, ball_model, cue_classifier, threshold,
                  save_crops_dir=None):
    stem   = img_path.stem
    img    = cv2.imread(str(img_path))
    if img is None:
        return None

    if stem in LOW_QUALITY_WARP_STEMS:
        return {"stem": stem, "status": "lq_warp", "mode": "skip"}

    corners, corner_conf = get_corners(corner_model, img)
    if corners is None:
        return {"stem": stem, "status": "no_table", "mode": "skip"}

    warped = warp_image(img, corners)
    gt_cue = load_gt_cue(stem)

    balls_before, warns_before, raw_cue_before, recovered_before = \
        detect_balls(ball_model, warped)

    # ── BEFORE: existing pipeline result ──────────────────────────────────────
    pred_cue_before = next((b for b in balls_before if b["type"] == "cue_ball"), None)

    before_status = _cue_status(pred_cue_before, gt_cue, stem)
    before_result = {
        "stem": stem,
        "gt_cue": "no_gt" if gt_cue == "no_gt" else
                  (None if gt_cue is None else list(gt_cue)),
        "pred_cue_conf": pred_cue_before["confidence"] if pred_cue_before else None,
        "cue_status": before_status,
        "raw_cue_count": len(raw_cue_before),
        "raw_cue_high_conf": sum(1 for d in raw_cue_before if d[1] >= 0.7),
        "n_balls": len(balls_before),
        "plan_ready": bool(pred_cue_before and 2 <= len(balls_before) <= MAX_BALLS),
        "warns": warns_before,
        "recovered": recovered_before,
    }

    if cue_classifier is None:
        return {"stem": stem, "before": before_result, "after": None}

    # ── AFTER: with classifier gate ───────────────────────────────────────────
    balls_after  = []
    verified_cue = None
    fp_crops_saved = []

    # Re-examine every YOLO cue prediction (including at fallback conf)
    cue_candidates = sorted(
        [d for d in raw_cue_before if d[1] >= FALLBACK_CONF],
        key=lambda d: -d[1]
    )

    for det in cue_candidates:
        cls, cf, x1, y1, x2, y2 = det
        cx = (x1+x2)/2; cy = (y1+y2)/2; r = max(x2-x1, y2-y1)/2
        score = classifier_score(cue_classifier, warped, cx, cy, r)

        if score >= threshold:
            # Classifier confirms this as cue_ball
            verified_cue = {"id": 0, "type": "cue_ball",
                            "x": round(cx,1), "y": round(cy,1),
                            "r": round(r,1), "confidence": round(cf,3),
                            "classifier_score": round(score,3)}
            break
        else:
            # Classifier rejects this cue prediction
            if save_crops_dir and gt_cue != "no_gt":
                crop = crop_ball(warped, cx, cy, r)
                if crop is not None:
                    fname = f"rejected_{stem}_{len(fp_crops_saved):02d}.jpg"
                    cv2.imwrite(str(save_crops_dir / fname), crop)
                    fp_crops_saved.append(fname)

    # Re-run whiteness fallback only if no YOLO cue was verified
    if verified_cue is None:
        _, _, _, _ = detect_balls(ball_model, warped)
        # Rebuild obj_dets from raw (above threshold)
        raw_all  = run_yolo(ball_model, warped, FALLBACK_CONF)
        obj_dets = sorted([d for d in raw_all if d[0] == 1 and d[1] >= OBJ_CONF],
                          key=lambda d: -d[1])
        # Whiteness fallback — same heuristic as before
        recovered_after = False
        for det in obj_dets:
            _, cf, x1, y1, x2, y2 = det
            bright, sat, asp = patch_appearance(warped, x1, y1, x2, y2)
            if bright >= CUE_MIN_BRIGHT and sat <= CUE_MAX_SAT and asp >= CUE_MIN_ASPECT:
                cx = (x1+x2)/2; cy = (y1+y2)/2; r = max(x2-x1,y2-y1)/2
                score = classifier_score(cue_classifier, warped, cx, cy, r)
                if score >= threshold:
                    verified_cue = {"id": 0, "type": "cue_ball",
                                    "x": round(cx,1), "y": round(cy,1),
                                    "r": round(r,1), "confidence": round(cf,3),
                                    "classifier_score": round(score,3),
                                    "recovered": True}
                    recovered_after = True
                    break

    after_status = _cue_status(verified_cue, gt_cue, stem)
    n_obj_after  = sum(1 for b in balls_before if b["type"] == "object_ball")

    after_result = {
        "stem": stem,
        "gt_cue": before_result["gt_cue"],
        "pred_cue_conf": verified_cue["confidence"] if verified_cue else None,
        "classifier_score": verified_cue.get("classifier_score") if verified_cue else None,
        "cue_status": after_status,
        "n_balls": (1 + n_obj_after) if verified_cue else n_obj_after,
        "plan_ready": bool(verified_cue and (1 + n_obj_after) >= 2
                           and (1 + n_obj_after) <= MAX_BALLS),
        "fp_crops_rejected": len(fp_crops_saved),
    }

    return {"stem": stem, "before": before_result, "after": after_result}


def _cue_status(pred_cue, gt_cue, stem):
    """Classify the cue prediction relative to GT."""
    if stem in NO_CUE_STEMS:
        if pred_cue is None:
            return "correct_no_cue"
        return "false_cue"

    if gt_cue == "no_gt":
        if pred_cue is None:
            return "no_gt_no_pred"
        return "no_gt_has_pred"

    if gt_cue is None:
        # Label file exists but no cue label → scene has no cue
        if pred_cue is None:
            return "correct_no_cue"
        return "false_cue"

    # GT cue exists
    if pred_cue is None:
        return "missed_cue"

    matched = cue_match(pred_cue, gt_cue)
    return "correct_cue" if matched else "false_cue"


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(results, mode="before"):
    key = mode   # "before" or "after"
    records = [r[key] for r in results if r and r.get(key) and r[key] != "skip"]

    tp = sum(1 for r in records if r["cue_status"] == "correct_cue")
    fp = sum(1 for r in records if r["cue_status"] == "false_cue")
    fn = sum(1 for r in records if r["cue_status"] == "missed_cue")
    plan_ready = sum(1 for r in records if r.get("plan_ready"))
    n_imgs = len(records)

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    fpr       = fp / max(n_imgs, 1)

    high_conf_fp = sum(r.get("raw_cue_high_conf", 0)
                       for r in records if r["cue_status"] == "false_cue")

    return {
        "mode": mode, "n_images": n_imgs,
        "TP": tp, "FP": fp, "FN": fn,
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "false_cue_rate_per_image": round(fpr, 4),
        "high_conf_fp": high_conf_fp,
        "plan_ready": plan_ready,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def print_metrics(m, label):
    print(f"\n── {label} ─────────────────────────────────────────────")
    print(f"  Images evaluated : {m['n_images']}")
    print(f"  TP / FP / FN     : {m['TP']} / {m['FP']} / {m['FN']}")
    print(f"  Precision        : {m['precision']:.4f}")
    print(f"  Recall           : {m['recall']:.4f}")
    print(f"  False-cue / img  : {m['false_cue_rate_per_image']:.4f}")
    print(f"  High-conf FP (≥0.7): {m['high_conf_fp']}")
    print(f"  Plan-ready       : {m['plan_ready']}")


def write_report(results, metrics_before, metrics_after, threshold, report_dir):
    report_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Cue Identity Audit — Before / After",
        f"**Threshold**: {threshold}  (None = before-only mode)",
        "",
        "## Summary Metrics",
        "",
        "| metric | BEFORE | AFTER |",
        "|--------|--------|-------|",
    ]

    def _fmt(m, key):
        v = m.get(key, "—")
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    for key in ["precision", "recall", "false_cue_rate_per_image",
                "high_conf_fp", "plan_ready", "TP", "FP", "FN"]:
        a_val = _fmt(metrics_after, key) if metrics_after else "—"
        lines.append(f"| {key} | {_fmt(metrics_before, key)} | {a_val} |")

    lines += ["", "## Per-Image Detail", "",
              "| stem | GT cue | before_status | after_status | "
              "before_plan | after_plan | note |",
              "|------|--------|--------------|-------------|"
              "------------|-----------|------|"]

    for r in results:
        if not r:
            continue
        b = r.get("before", {})
        a = r.get("after")
        stem = r["stem"]
        gt   = str(b.get("gt_cue", "?"))[:12]
        bs   = b.get("cue_status", "?")
        as_  = a.get("cue_status", "?") if a else "—"
        bp   = "✓" if b.get("plan_ready") else "✗"
        ap   = ("✓" if a.get("plan_ready") else "✗") if a else "—"
        note = ""
        if a and bs != as_:
            if bs == "false_cue" and as_ in ("missed_cue","correct_no_cue"):
                note = "FP suppressed ✓"
            elif bs == "correct_cue" and as_ == "missed_cue":
                note = "⚠ true cue dropped"
            elif bs == "missed_cue" and as_ == "missed_cue":
                note = ""
        lines.append(f"| {stem} | {gt} | {bs} | {as_} | {bp} | {ap} | {note} |")

    (report_dir / "audit_report.md").write_text("\n".join(lines))

    full = {"threshold": threshold,
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
            "per_image": results}
    (report_dir / "audit_results.json").write_text(json.dumps(full, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",       default=None,
                        help="Cue identity classifier checkpoint")
    parser.add_argument("--threshold",  type=float, default=0.6,
                        help="Classifier score threshold (default 0.6)")
    parser.add_argument("--before-only", action="store_true")
    parser.add_argument("--ball-ckpt",  default=str(BALL_CKPT_V7))
    parser.add_argument("--save-crops", action="store_true",
                        help="Save rejected FP cue crops to review/cue_id/rejected/")
    args = parser.parse_args()

    from ultralytics import YOLO
    corner_model = YOLO(str(CORNER_CKPT))
    ball_model   = YOLO(args.ball_ckpt)

    cue_classifier = None
    threshold      = None
    if not args.before_only and args.ckpt:
        print(f"Loading cue classifier: {args.ckpt}")
        cue_classifier, best_row = load_cue_classifier(args.ckpt)
        threshold = args.threshold
        print(f"  Threshold: {threshold}  (best_row from training: {best_row})")

    save_crops_dir = None
    if args.save_crops:
        save_crops_dir = REPORT_DIR / "rejected_crops"
        save_crops_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted(
        p for p in PICTURE_DIR.iterdir()
        if p.suffix.lower() in {".jpg",".jpeg",".png"}
        and p.stem not in SKIP_STEMS
        and not p.stem.startswith("aug_")
    )
    print(f"\nAuditing {len(imgs)} images...")

    results = []
    for img_path in imgs:
        print(f"  {img_path.stem:40s}", end=" ", flush=True)
        r = process_image(img_path, corner_model, ball_model,
                          cue_classifier, threshold, save_crops_dir)
        if r is None:
            print("SKIP")
            continue
        b = (r.get("before") or {})
        a = (r.get("after")  or {})
        bs = b.get("cue_status", "skip")
        as_ = a.get("cue_status", "—") if a else "—"
        print(f"before={bs:18s}  after={as_}")
        results.append(r)

    metrics_before = compute_metrics(results, "before")
    metrics_after  = compute_metrics(results, "after") if cue_classifier else None

    print_metrics(metrics_before, "BEFORE (existing pipeline)")
    if metrics_after:
        print_metrics(metrics_after, f"AFTER  (classifier gate @ {threshold})")

        delta_pr    = metrics_after["plan_ready"]  - metrics_before["plan_ready"]
        delta_prec  = metrics_after["precision"]   - metrics_before["precision"]
        delta_rec   = metrics_after["recall"]      - metrics_before["recall"]
        delta_fpr   = metrics_after["false_cue_rate_per_image"] \
                    - metrics_before["false_cue_rate_per_image"]

        print(f"\n── Delta ────────────────────────────────────────────────────")
        print(f"  Δ Precision      : {delta_prec:+.4f}")
        print(f"  Δ Recall         : {delta_rec:+.4f}")
        print(f"  Δ False-cue/img  : {delta_fpr:+.4f}")
        print(f"  Δ Plan-ready     : {delta_pr:+d}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_report(results, metrics_before, metrics_after, threshold, REPORT_DIR)
    print(f"\nReport  → {REPORT_DIR / 'audit_report.md'}")
    print(f"Results → {REPORT_DIR / 'audit_results.json'}")


if __name__ == "__main__":
    main()
