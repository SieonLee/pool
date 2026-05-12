"""
Cue identity classifier analysis.

Produces:
  1. Calibration on val-set crops
       - Reliability diagram (PNG)
       - Score histogram: positives vs negatives (PNG)
       - Precision-Recall curve (PNG)
       - FP score distribution (PNG)
  2. Pipeline FP gallery  — crops that pass the classifier gate but are not cues
  3. Pipeline FN gallery  — true cue crops that the classifier rejects
  4. Contact sheets grouped by failure type
  5. Confusion matrices (val set + pipeline)
  6. calibration_report.md

Run:
  python scripts/analyze_cue_classifier.py \\
      --ckpt models/checkpoints/cue_id_v1.pt \\
      --threshold 0.3 \\
      --ball-ckpt models/checkpoints/ball_yolo_v7_below_baseline.pt
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / "scripts"))

CROP_DIR    = BASE / "datasets" / "cue_crops"
PICTURE_DIR = BASE / "picture"
LABEL_DIR   = BASE / "datasets" / "pose-warped-balls" / "labels" / "train"
CORNER_CKPT = BASE / "models" / "checkpoints" / "table_corners_mvp_v1.pt"
REPORT_DIR  = BASE / "review" / "cue_id"
GALLERY_DIR = REPORT_DIR / "galleries"

WARP_W, WARP_H = 900, 450
CROP_SIZE      = 64
CROP_CONTEXT   = 1.6
SEED           = 42
VAL_FRAC       = 0.20

# ── appearance constants (mirror audit_cue_id.py) ────────────────────────────
CUE_CONF      = 0.25
OBJ_CONF      = 0.35
FALLBACK_CONF = 0.10
NO_CUE_STEMS  = {"pool_real_020", "pool_real_025", "pool_real_028"}
SKIP_STEMS    = {"new_uploads_contact_sheet", "search_contact_sheet",
                 "thumb", "pexels-photo-10627132"}
LQ_STEMS      = {"pool_real_002", "pool_real_010", "pool_real_016"}
GALLERY_CELL  = 128   # px per crop in gallery


# ─────────────────────────────────────────────────────────────────────────────
# Model loading (mirrors audit_cue_id.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_path):
    import torch
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    arch = ckpt.get("arch", "unknown")
    if "mobilenet" in arch:
        import torchvision.models as tvm
        import torch.nn as nn
        model = tvm.mobilenet_v3_small(weights=None)
        model.classifier = nn.Sequential(
            nn.Linear(576, 64), nn.Hardswish(), nn.Dropout(0.4), nn.Linear(64, 1))
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
    return model, arch, ckpt


def score_crop(model, img_bgr):
    """Run classifier on a BGR uint8 64×64 crop → sigmoid probability."""
    import torch
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.tensor(img_rgb.transpose(2, 0, 1)).unsqueeze(0)
    with torch.no_grad():
        logit = model(x).item()
    import torch.nn.functional as F
    return float(torch.sigmoid(torch.tensor(logit)).item())


# ─────────────────────────────────────────────────────────────────────────────
# Calibration: run inference on val set crops
# ─────────────────────────────────────────────────────────────────────────────

def run_val_inference(model):
    """
    Reproduces the val split (same seed/frac as train_cue_classifier.py).
    Returns (probs, labels, files).
    """
    manifest_path = CROP_DIR / "manifest.json"
    if not manifest_path.exists():
        print("  manifest.json not found — skipping calibration")
        return [], [], []

    mf   = json.loads(manifest_path.read_text())
    recs = mf.get("crops", [])

    # Build parallel lists matching the load_crops order
    cue_recs = [r for r in recs if r["label"] == "cue_ball"]
    obj_recs = [r for r in recs if r["label"] == "not_cue"]

    rng = random.Random(SEED)
    pos_idx = list(range(len(cue_recs)))
    neg_idx = list(range(len(obj_recs)))
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    n_val_pos = max(1, int(len(pos_idx) * VAL_FRAC))
    n_val_neg = max(1, int(len(neg_idx) * VAL_FRAC))
    val_pos   = pos_idx[:n_val_pos]
    val_neg   = neg_idx[:n_val_neg]

    probs, labels, files = [], [], []
    for i in val_pos:
        rec  = cue_recs[i]
        path = CROP_DIR / "cue_ball" / rec["file"]
        img  = cv2.imread(str(path))
        if img is None:
            continue
        probs.append(score_crop(model, img))
        labels.append(1)
        files.append(rec["file"])

    for i in val_neg:
        rec  = obj_recs[i]
        path = CROP_DIR / "not_cue" / rec["file"]
        img  = cv2.imread(str(path))
        if img is None:
            continue
        probs.append(score_crop(model, img))
        labels.append(0)
        files.append(rec["file"])

    print(f"  Val inference: {sum(l==1 for l in labels)} pos, "
          f"{sum(l==0 for l in labels)} neg, total={len(labels)}")
    return probs, labels, files


def compute_calibration(probs, labels, n_bins=10):
    """Return reliability diagram data: (mean_conf, frac_pos, counts) per bin."""
    bins = np.linspace(0, 1, n_bins + 1)
    mean_conf, frac_pos, counts = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = [(lo <= p < hi) for p in probs]
        if not any(mask):
            continue
        bin_probs  = [p for p, m in zip(probs, mask) if m]
        bin_labels = [l for l, m in zip(labels, mask) if m]
        mean_conf.append(np.mean(bin_probs))
        frac_pos.append(np.mean(bin_labels))
        counts.append(len(bin_probs))
    return mean_conf, frac_pos, counts


def pr_curve(probs, labels, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    labs = np.array(labels)
    ps, rs = [], []
    for t in thresholds:
        preds = (np.array(probs) >= t).astype(int)
        tp = int(((preds==1) & (labs==1)).sum())
        fp = int(((preds==1) & (labs==0)).sum())
        fn = int(((preds==0) & (labs==1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        ps.append(prec)
        rs.append(rec)
    return ps, rs, list(thresholds)


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib plots
# ─────────────────────────────────────────────────────────────────────────────

def make_plots(probs, labels, threshold, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plots")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    labs  = np.array(labels)
    probs_arr = np.array(probs)
    pos_p = probs_arr[labs == 1]
    neg_p = probs_arr[labs == 0]

    # ── 1. Score histogram ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 26)
    ax.hist(neg_p, bins=bins, alpha=0.7, color="#e74c3c", label=f"not_cue  (n={len(neg_p)})")
    ax.hist(pos_p, bins=bins, alpha=0.7, color="#2ecc71", label=f"cue_ball (n={len(pos_p)})")
    ax.axvline(threshold, color="k", linestyle="--", linewidth=1.5,
               label=f"threshold={threshold}")
    ax.set_xlabel("Classifier score (sigmoid)")
    ax.set_ylabel("Count")
    ax.set_title("Score distributions — val set")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(str(out_dir / "score_histogram.png"), dpi=120)
    plt.close(fig)

    # ── 2. Reliability diagram ────────────────────────────────────────────────
    mc, fp_frac, cnts = compute_calibration(probs, labels)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    sc = ax.scatter(mc, fp_frac, c=cnts, cmap="Blues", s=60, zorder=3)
    plt.colorbar(sc, ax=ax, label="Count")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Reliability diagram")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(str(out_dir / "reliability_diagram.png"), dpi=120)
    plt.close(fig)

    # ── 3. Precision-Recall curve ─────────────────────────────────────────────
    ps, rs, ts = pr_curve(probs, labels)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rs, ps, "b-", lw=2)
    # Mark operating threshold
    idx = int(np.argmin(np.abs(np.array(ts) - threshold)))
    ax.scatter([rs[idx]], [ps[idx]], color="red", zorder=5, s=80,
               label=f"threshold={threshold}  P={ps[idx]:.3f} R={rs[idx]:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve — val set")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_dir / "pr_curve.png"), dpi=120)
    plt.close(fig)

    # ── 4. FP score distribution (zoomed on negatives) ────────────────────────
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(neg_p, bins=25, color="#e74c3c", alpha=0.85, label="not_cue crops (val)")
    ax.axvline(threshold, color="k", linestyle="--", lw=1.5,
               label=f"gate threshold={threshold}")
    ax.set_xlabel("Classifier score")
    ax.set_ylabel("Count")
    ax.set_title("False-positive score distribution — negatives only")
    ax.legend(fontsize=9)
    fp_passing = int((neg_p >= threshold).sum())
    ax.text(0.97, 0.95, f"FP passing gate: {fp_passing}/{len(neg_p)}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            color="darkred")
    fig.tight_layout()
    fig.savefig(str(out_dir / "fp_score_dist.png"), dpi=120)
    plt.close(fig)

    print(f"  Plots saved → {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Failure type heuristics
# ─────────────────────────────────────────────────────────────────────────────

def classify_failure(crop_bgr, cx, cy, warp_w=WARP_W, warp_h=WARP_H):
    """
    Heuristically label a crop with a failure type.
    Returns a string tag.
    """
    if crop_bgr is None:
        return "unknown"

    lab   = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    hsv   = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    gray  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    L_mean = float(lab[:, :, 0].mean())       # 0-255 (CIELAB L* scaled)
    S_mean = float(hsv[:, :, 1].mean())       # saturation
    V_mean = float(hsv[:, :, 2].mean())       # brightness

    # Lap var: sharpness proxy
    # Colour variance: stripe proxy
    row_var = float(np.var([gray[i, :].mean() for i in range(gray.shape[0])]))

    # 1. Near image border → rail / cushion
    margin = 60
    if cx < margin or cx > (warp_w - margin) or cy < margin or cy > (warp_h - margin):
        return "rail_edge"

    # 2. Very blurry
    if lap_var < 5.0:
        return "blurry"

    # 3. Bright specular highlight (very high brightness, low saturation)
    if V_mean > 200 and S_mean < 30:
        return "specular_highlight"

    # 4. Striped — high row-wise variance (bright and dark rows alternating)
    if row_var > 600 and S_mean > 40:
        return "striped_ball"

    # 5. Bright white-ish but moderately saturated → solid coloured near-white
    if V_mean > 160 and S_mean < 60 and L_mean > 160:
        return "bright_solid"

    # 6. Partially occluded / clipped → low brightness, high variance at edges
    edge = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    if float(edge.mean()) < 30:
        return "occluded_clipped"

    return "other"


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline image analysis — extract crops for FP/FN images
# ─────────────────────────────────────────────────────────────────────────────

def warp_image(img, corners):
    dst = np.float32([[0,0],[WARP_W,0],[WARP_W,WARP_H],[0,WARP_H]])
    M   = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (WARP_W, WARP_H))


def extract_crop(warped, cx, cy, r, size=CROP_SIZE, context=CROP_CONTEXT):
    cr = int(math.ceil(r * context))
    x1 = max(0, int(cx - cr)); y1 = max(0, int(cy - cr))
    x2 = min(WARP_W, x1 + cr * 2); y2 = min(WARP_H, y1 + cr * 2)
    patch = warped[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    return cv2.resize(patch, (size, size), interpolation=cv2.INTER_LINEAR)


def get_pipeline_crops(img_path, corner_model, ball_model, cue_classifier, threshold):
    """
    Process one image through the full pipeline.
    Returns dict with: warped, gt_cue, cue_dets, all_cue_scores, status
    """
    from ultralytics import YOLO

    img = cv2.imread(str(img_path))
    if img is None:
        return None

    r = corner_model(img, verbose=False)
    if not r or r[0].keypoints is None or len(r[0].keypoints) == 0:
        return None
    kps  = r[0].keypoints
    best = int(kps.conf.mean(dim=1).argmax()) if kps.conf is not None else 0
    corners = kps.xy[best].cpu().numpy()
    warped  = warp_image(img, corners)

    raw = ball_model(warped, verbose=False, conf=FALLBACK_CONF)
    dets = []
    if raw and raw[0].boxes is not None:
        for b in raw[0].boxes:
            dets.append((int(b.cls.item()), float(b.conf.item()),
                         *b.xyxy[0].cpu().numpy().tolist()))

    cue_dets = sorted([d for d in dets if d[0] == 0], key=lambda d: -d[1])

    cue_analysis = []
    for det in cue_dets:
        cls, cf, x1, y1, x2, y2 = det
        cx = (x1+x2)/2; cy = (y1+y2)/2; r_ball = max(x2-x1, y2-y1)/2
        crop  = extract_crop(warped, cx, cy, r_ball)
        score = score_crop(cue_classifier, crop) if crop is not None else 0.0
        ftype = classify_failure(crop, cx, cy)
        cue_analysis.append({
            "cx": cx, "cy": cy, "r": r_ball, "yolo_conf": cf,
            "classifier_score": score, "passes_gate": score >= threshold,
            "failure_type": ftype,
            "crop": crop,
        })

    return {"stem": img_path.stem, "warped": warped, "cue_analysis": cue_analysis}


def load_gt_cue(stem):
    label_path = LABEL_DIR / f"{stem}_warped.txt"
    if not label_path.exists():
        return "no_gt"
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) >= 5 and int(parts[0]) == 0:
            return (float(parts[1]) * WARP_W, float(parts[2]) * WARP_H)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Contact sheet builder
# ─────────────────────────────────────────────────────────────────────────────

BORDER = {
    "TP":    (0, 200, 0),    # green
    "FP":    (0, 0, 220),    # red  (BGR)
    "FN":    (0, 140, 255),  # orange
    "TN":    (160, 160, 160),# grey
    "PASS":  (0, 200, 0),
    "FAIL":  (0, 0, 220),
}

def annotate_crop(crop_bgr, label, score, tag, cell=GALLERY_CELL):
    """
    Resize crop to cell×cell, add coloured border + text overlay.
    tag: TP / FP / FN / TN
    """
    if crop_bgr is None:
        tile = np.zeros((cell, cell, 3), dtype=np.uint8)
        cv2.putText(tile, "MISSING", (4, cell//2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (200, 200, 200), 1)
        return tile

    tile = cv2.resize(crop_bgr, (cell, cell), interpolation=cv2.INTER_LINEAR)

    # Border
    color = BORDER.get(tag, (200, 200, 200))
    cv2.rectangle(tile, (0, 0), (cell-1, cell-1), color, 3)

    # Score text
    sc_txt = f"{score:.2f}" if score is not None else "n/a"
    cv2.putText(tile, f"{tag} {sc_txt}", (3, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1,
                cv2.LINE_AA)
    # Label (failure type)
    short = label[:12] if label else ""
    cv2.putText(tile, short, (3, cell - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (200, 200, 200), 1,
                cv2.LINE_AA)
    return tile


def make_contact_sheet(tiles, cols, title=""):
    """tiles: list of (annotated_crop, stem_text)"""
    rows = math.ceil(len(tiles) / cols)
    cell = GALLERY_CELL
    header_h = 28 if title else 0
    canvas = np.zeros((rows * cell + header_h, cols * cell, 3), dtype=np.uint8)

    if title:
        cv2.putText(canvas, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (220, 220, 220), 1, cv2.LINE_AA)

    for i, (tile, stem) in enumerate(tiles):
        r = i // cols; c = i % cols
        y0 = r * cell + header_h; y1 = y0 + cell
        x0 = c * cell;            x1 = x0 + cell
        canvas[y0:y1, x0:x1] = tile
        # stem text below (within cell)
        cv2.putText(canvas, stem[:14], (x0+2, y1-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.22, (160, 160, 160), 1)
    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# Val set confusion contact sheet
# ─────────────────────────────────────────────────────────────────────────────

def make_val_confusion_sheet(probs, labels, files, threshold):
    """
    Contact sheet for val-set crops grouped by confusion type.
    Shows TP, FP, FN, TN quadrants.
    """
    groups = {"FP": [], "FN": [], "TP": [], "TN": []}
    for prob, lab, fname in zip(probs, labels, files):
        pred = int(prob >= threshold)
        if   lab == 1 and pred == 1: tag = "TP"
        elif lab == 0 and pred == 1: tag = "FP"
        elif lab == 1 and pred == 0: tag = "FN"
        else:                        tag = "TN"

        subdir = "cue_ball" if lab == 1 else "not_cue"
        path   = CROP_DIR / subdir / fname
        img    = cv2.imread(str(path))
        tile   = annotate_crop(img, tag, prob, tag)
        groups[tag].append((tile, fname[:12]))

    # Build one sheet per group
    sheets = {}
    for tag, items in groups.items():
        if not items:
            continue
        cols = min(8, len(items))
        sheet = make_contact_sheet(items, cols,
                                   title=f"Val set — {tag}  ({len(items)} crops)")
        sheets[tag] = (sheet, len(items))

    return groups, sheets


# ─────────────────────────────────────────────────────────────────────────────
# Report writer
# ─────────────────────────────────────────────────────────────────────────────

def write_report(probs, labels, threshold, val_groups, pipeline_cases, out_dir):
    """Write calibration_report.md"""
    labs = np.array(labels)
    p    = np.array(probs)
    pred = (p >= threshold).astype(int)

    tp = int(((pred==1) & (labs==1)).sum())
    fp = int(((pred==1) & (labs==0)).sum())
    fn = int(((pred==0) & (labs==1)).sum())
    tn = int(((pred==0) & (labs==0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

    mc, fp_frac, cnts = compute_calibration(probs, labels)
    ece = sum(abs(mc_i - fp_i) * cnt for mc_i, fp_i, cnt in
              zip(mc, fp_frac, cnts)) / max(sum(cnts), 1)

    lines = [
        "# Cue Identity Classifier — Calibration & Error Analysis",
        "",
        f"**Checkpoint**: `cue_id_v1.pt`  |  **Threshold**: {threshold}",
        f"**Val set**: {len(probs)} crops  "
        f"({sum(labs==1)} cue_ball, {sum(labs==0)} not_cue)",
        "",
        "## Val-Set Confusion Matrix",
        "",
        "| | Pred: cue_ball | Pred: not_cue |",
        "|---|---|---|",
        f"| **GT: cue_ball** | TP = {tp} | FN = {fn} |",
        f"| **GT: not_cue**  | FP = {fp} | TN = {tn} |",
        "",
        f"- **Precision**: {prec:.4f}",
        f"- **Recall**: {rec:.4f}",
        f"- **ECE** (expected calibration error): {ece:.4f}",
        "",
        "## Calibration Summary",
        "",
        "| Confidence bin | Fraction positive | Count |",
        "|----------------|------------------|-------|",
    ]
    for mc_i, fp_i, cnt in zip(mc, fp_frac, cnts):
        gap = mc_i - fp_i
        flag = " ← overconfident" if gap > 0.1 else (" ← underconfident" if gap < -0.1 else "")
        lines.append(f"| {mc_i:.2f} | {fp_i:.2f} | {cnt}{flag} |")

    pos_scores = [p for p, l in zip(probs, labels) if l == 1]
    neg_scores = [p for p, l in zip(probs, labels) if l == 0]
    lines += [
        "",
        "## Score Distribution Summary",
        "",
        f"- cue_ball scores: mean={np.mean(pos_scores):.3f}  "
        f"std={np.std(pos_scores):.3f}  "
        f"min={np.min(pos_scores):.3f}  max={np.max(pos_scores):.3f}",
        f"- not_cue  scores: mean={np.mean(neg_scores):.3f}  "
        f"std={np.std(neg_scores):.3f}  "
        f"min={np.min(neg_scores):.3f}  max={np.max(neg_scores):.3f}",
        f"- Negatives passing gate (≥{threshold}): "
        f"{int((np.array(neg_scores)>=threshold).sum())}/{len(neg_scores)}",
        f"- Positives blocked by gate (<{threshold}): "
        f"{int((np.array(pos_scores)<threshold).sum())}/{len(pos_scores)}",
        "",
        "## Pipeline FP / FN Analysis",
        "",
    ]

    for case in pipeline_cases:
        lines.append(f"### `{case['stem']}` — {case['pipeline_status']}")
        for ca in case["cue_analysis"]:
            lines.append(
                f"  - YOLO conf={ca['yolo_conf']:.3f}  "
                f"classifier_score={ca['classifier_score']:.3f}  "
                f"passes_gate={ca['passes_gate']}  "
                f"failure_type=`{ca['failure_type']}`  "
                f"cx={ca['cx']:.0f}  cy={ca['cy']:.0f}"
            )
        if not case["cue_analysis"]:
            lines.append("  *(no YOLO cue detections at fallback conf)*")
        lines.append("")

    lines += [
        "## Gallery Files",
        "",
        "| File | Contents |",
        "|------|----------|",
        "| `galleries/val_FP.jpg` | Val-set false positives |",
        "| `galleries/val_FN.jpg` | Val-set false negatives |",
        "| `galleries/val_TP.jpg` | Val-set true positives (sample) |",
        "| `galleries/pipeline_fp.jpg` | Pipeline-level FPs (full image crops) |",
        "| `galleries/pipeline_fn.jpg` | Pipeline-level FNs (full image crops) |",
        "| `calibration/score_histogram.png` | Score distribution by class |",
        "| `calibration/reliability_diagram.png` | Calibration curve |",
        "| `calibration/pr_curve.png` | Precision-Recall curve |",
        "| `calibration/fp_score_dist.png` | FP score distribution |",
        "",
        "## Recommendations for Classifier v2",
        "",
        "Based on this analysis:",
        "",
    ]

    # Auto-recommendations based on data
    fp_ftypes = [ca["failure_type"] for case in pipeline_cases
                 if case["pipeline_status"] == "FP"
                 for ca in case["cue_analysis"] if ca["passes_gate"]]
    fn_scores = [ca["classifier_score"] for case in pipeline_cases
                 if case["pipeline_status"] == "FN"
                 for ca in case["cue_analysis"]]

    if any(ft in ("bright_solid", "specular_highlight") for ft in fp_ftypes):
        lines.append("1. **Add more bright-solid hard negatives** — "
                     "FPs are bright white-ish objects (solid-coloured balls, rails, highlights).")
    if any(ft == "striped_ball" for ft in fp_ftypes):
        lines.append("2. **Add striped-ball negatives** — classifier confuses stripes with cue ball.")
    if any(ft == "rail_edge" for ft in fp_ftypes):
        lines.append("3. **Add rail/edge negatives** — false cues near table boundary.")
    if fn_scores and max(fn_scores or [1]) < 0.5:
        lines.append("4. **Augment cue ball crops** — FN cue had very low score; "
                     "likely unusual lighting or partial occlusion. "
                     "Add rotate/flip/brightness augmentation for cue class.")
    if ece > 0.10:
        lines.append(f"5. **Apply temperature scaling** — ECE={ece:.3f} indicates "
                     "poor calibration (overconfident). Post-hoc calibration recommended.")

    lines.append("")
    (out_dir / "calibration_report.md").write_text("\n".join(lines))
    print(f"  Report → {out_dir / 'calibration_report.md'}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",       required=True)
    parser.add_argument("--threshold",  type=float, default=0.3)
    parser.add_argument("--ball-ckpt",  default=str(
        BASE / "models" / "checkpoints" / "ball_yolo_v7_below_baseline.pt"))
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO

    GALLERY_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading classifier: {args.ckpt}")
    model, arch, ckpt = load_model(args.ckpt)
    threshold = args.threshold
    print(f"  arch={arch}  threshold={threshold}")

    corner_model = YOLO(str(CORNER_CKPT))
    ball_model   = YOLO(args.ball_ckpt)

    # ── 1. Val set calibration ────────────────────────────────────────────────
    print("\nRunning val-set inference...")
    probs, labels, files = run_val_inference(model)

    if probs:
        print("\nGenerating calibration plots...")
        make_plots(probs, labels, threshold, REPORT_DIR / "calibration")

        print("\nGenerating val-set confusion contact sheets...")
        val_groups, sheets = make_val_confusion_sheet(probs, labels, files, threshold)
        for tag, (sheet, count) in sheets.items():
            path = GALLERY_DIR / f"val_{tag}.jpg"
            cv2.imwrite(str(path), sheet)
            print(f"  val_{tag}.jpg  ({count} crops)")

        # Print confusion matrix
        tp = len(val_groups.get("TP", []))
        fp = len(val_groups.get("FP", []))
        fn = len(val_groups.get("FN", []))
        tn = len(val_groups.get("TN", []))
        prec = tp/(tp+fp) if (tp+fp)>0 else 0
        rec  = tp/(tp+fn) if (tp+fn)>0 else 0
        print(f"\n  Val confusion @ threshold {threshold}:")
        print(f"    TP={tp}  FP={fp}  FN={fn}  TN={tn}")
        print(f"    Precision={prec:.4f}  Recall={rec:.4f}")
    else:
        val_groups = {}
        sheets = {}

    # ── 2. Pipeline FP/FN analysis ────────────────────────────────────────────
    print("\nAnalysing pipeline FP/FN images...")

    # Load audit results to find which images are FP/FN
    audit_path = REPORT_DIR / "audit_results.json"
    pipeline_cases = []
    fp_gallery_items = []
    fn_gallery_items = []

    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        for record in audit.get("per_image", []):
            if not record:
                continue
            b = record.get("before", {})
            a = record.get("after", {})
            before_status = b.get("cue_status", "")
            after_status  = a.get("cue_status", "")

            # Categorise: is this an FP or FN in the AFTER pipeline?
            is_fp = (after_status == "false_cue")
            is_fn = (after_status == "missed_cue")

            if not (is_fp or is_fn):
                continue

            stem = record["stem"]
            img_path = PICTURE_DIR / f"{stem}.jpg"
            if not img_path.exists():
                img_path = next(PICTURE_DIR.glob(f"{stem}.*"), None)
            if not img_path:
                continue

            print(f"  Analysing {stem} ({'FP' if is_fp else 'FN'})...", flush=True)
            result = get_pipeline_crops(img_path, corner_model, ball_model,
                                        model, threshold)
            if result is None:
                continue

            pipeline_status = "FP" if is_fp else "FN"
            result["pipeline_status"] = pipeline_status
            pipeline_cases.append(result)

            # Build gallery tiles
            for ca in result["cue_analysis"]:
                crop = ca["crop"]
                if crop is None:
                    continue
                tag = pipeline_status
                tile = annotate_crop(crop, ca["failure_type"],
                                     ca["classifier_score"], tag)
                label_txt = f"{stem[:10]} s={ca['classifier_score']:.2f}"
                if is_fp:
                    fp_gallery_items.append((tile, label_txt))
                else:
                    fn_gallery_items.append((tile, label_txt))

    # For FN images: also extract crop using GT position to show what was missed
    print("\nExtracting GT-position crops for FN images...")
    for case in pipeline_cases:
        if case["pipeline_status"] != "FN":
            continue
        stem = case["stem"]
        gt   = load_gt_cue(stem)
        if gt is None or gt == "no_gt":
            continue
        gt_cx, gt_cy = gt
        warped = case["warped"]

        # Estimate ball radius from YOLO detections, or use default
        r_est = 18.0
        for ca in case["cue_analysis"]:
            r_est = ca["r"]; break

        crop  = extract_crop(warped, gt_cx, gt_cy, r_est)
        if crop is None:
            continue
        score = score_crop(model, crop)
        ftype = classify_failure(crop, gt_cx, gt_cy)
        print(f"    {stem}: GT crop score={score:.3f}  type={ftype}")
        tile  = annotate_crop(crop, f"GT:{ftype}", score, "FN")
        fn_gallery_items.append((tile, f"GT {stem[:10]}"))

    # Save pipeline FP/FN galleries
    if fp_gallery_items:
        cols = min(6, len(fp_gallery_items))
        sheet = make_contact_sheet(fp_gallery_items, cols,
                                   title=f"Pipeline FPs (threshold {threshold})")
        cv2.imwrite(str(GALLERY_DIR / "pipeline_fp.jpg"), sheet)
        print(f"\n  pipeline_fp.jpg  ({len(fp_gallery_items)} crops)")

    if fn_gallery_items:
        cols = min(6, len(fn_gallery_items))
        sheet = make_contact_sheet(fn_gallery_items, cols,
                                   title=f"Pipeline FNs (threshold {threshold})")
        cv2.imwrite(str(GALLERY_DIR / "pipeline_fn.jpg"), sheet)
        print(f"  pipeline_fn.jpg  ({len(fn_gallery_items)} crops)")

    # ── 3. Write report ───────────────────────────────────────────────────────
    print("\nWriting calibration report...")
    write_report(probs, labels, threshold, val_groups, pipeline_cases, REPORT_DIR)

    print(f"\nDone. All outputs → {REPORT_DIR}")
    print(f"  Plots     → {REPORT_DIR / 'calibration'}/")
    print(f"  Galleries → {GALLERY_DIR}/")
    print(f"  Report    → {REPORT_DIR / 'calibration_report.md'}")


if __name__ == "__main__":
    main()
