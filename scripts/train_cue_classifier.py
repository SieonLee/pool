"""
Train a small cue-ball identity classifier.

Input:  datasets/cue_crops/{cue_ball,not_cue}/  (built by extract_cue_crops.py)
Output: models/checkpoints/cue_id_v1.pt
        review/cue_id/training_report.md

Architecture:
  MobileNetV3-Small (pretrained) with fine-tuned classifier head, if torchvision
  is available.  Falls back to a lightweight 3-conv custom CNN otherwise.

Design choices:
  - No oversampling of cue_ball class.  Natural imbalance is preserved and
    handled via pos_weight in BCEWithLogitsLoss.
  - Optimised for PRECISION over recall: threshold selected at the point that
    maximises F0.5 (penalises FP twice as much as FN).
  - Train-time augmentation: horizontal/vertical flip, colour jitter, rotation.

Run:
  python scripts/train_cue_classifier.py
  python scripts/train_cue_classifier.py --epochs 60 --lr 1e-3
  python scripts/train_cue_classifier.py --no-pretrain   # custom CNN only
"""
import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / "scripts"))

CROP_DIR   = BASE / "datasets" / "cue_crops"
CKPT_DIR   = BASE / "models" / "checkpoints"
REPORT_DIR = BASE / "review" / "cue_id"

CROP_SIZE  = 64
SEED       = 42
VAL_FRAC   = 0.20

# Stems whose cue_ball crops are forced into the train split regardless of the
# random draw.  Add any stem where the sole crop ended up in val and caused an
# FN in the production pipeline.
FORCE_TRAIN_STEMS = {
    "pool_real_011_warped",  # 1 cue crop; random split put it in val → FN score=0.000
}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def load_crops(crop_dir: Path):
    """
    Returns (images_list, labels_list) where each image is a uint8 HWC BGR
    numpy array and label is 1 (cue_ball) or 0 (not_cue).
    """
    imgs, labels = [], []

    for p in sorted((crop_dir / "cue_ball").glob("*.jpg")):
        img = cv2.imread(str(p))
        if img is not None:
            imgs.append(img)
            labels.append(1)

    for p in sorted((crop_dir / "not_cue").glob("*.jpg")):
        img = cv2.imread(str(p))
        if img is not None:
            imgs.append(img)
            labels.append(0)

    return imgs, labels


def stratified_split(imgs, labels, val_frac=VAL_FRAC, seed=SEED,
                     force_train_stems=None):
    """
    Stratified 80/20 split by class.
    force_train_stems: set of stem strings — any cue_ball crop whose manifest
    stem matches will be moved from val to train, regardless of the random draw.
    """
    # Load manifest once to get per-crop stems (matches manifest order)
    stem_map = {}   # global crop index → stem string
    manifest_path = CROP_DIR / "manifest.json"
    if force_train_stems and manifest_path.exists():
        import json
        mf   = json.loads(manifest_path.read_text())
        recs = mf.get("crops", [])
        # Manifest order matches load_crops order (cue_ball first, then not_cue)
        cue_recs = [r for r in recs if r["label"] == "cue_ball"]
        obj_recs = [r for r in recs if r["label"] == "not_cue"]
        for i, r in enumerate(cue_recs):
            stem_map[("pos", i)] = r.get("stem", "")
        for i, r in enumerate(obj_recs):
            stem_map[("neg", i)] = r.get("stem", "")

    rng = random.Random(seed)
    pos_idx = [i for i, l in enumerate(labels) if l == 1]
    neg_idx = [i for i, l in enumerate(labels) if l == 0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    n_val_pos = max(1, int(len(pos_idx) * val_frac))
    n_val_neg = max(1, int(len(neg_idx) * val_frac))

    val_pos_raw   = pos_idx[:n_val_pos]
    train_pos_raw = pos_idx[n_val_pos:]

    # Apply force_train_stems: move matching val positives → train
    if force_train_stems and stem_map:
        forced_to_train = [i for i in val_pos_raw
                           if stem_map.get(("pos", i), "") in force_train_stems]
        val_pos_final   = [i for i in val_pos_raw if i not in set(forced_to_train)]
        train_pos_final = train_pos_raw + forced_to_train
        if forced_to_train:
            forced_stems = [stem_map[("pos", i)] for i in forced_to_train]
            print(f"  [force_train] moved {len(forced_to_train)} val→train: {forced_stems}")
    else:
        val_pos_final   = val_pos_raw
        train_pos_final = train_pos_raw

    val_idx   = val_pos_final   + neg_idx[:n_val_neg]
    train_idx = train_pos_final + neg_idx[n_val_neg:]

    return (
        [imgs[i] for i in train_idx],   [labels[i] for i in train_idx],
        [imgs[i] for i in val_idx],     [labels[i] for i in val_idx],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation
# ─────────────────────────────────────────────────────────────────────────────

def augment(img):
    """Light augmentation for a single crop (BGR numpy array).

    Kept identical to v1 — saturation jitter was tried in v2 but caused
    regressions (model learned to reject high-saturation crops, which also
    rejects true cue balls in tight coloured-ball clusters).
    """
    # Horizontal flip
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
    # Vertical flip
    if random.random() < 0.3:
        img = cv2.flip(img, 0)
    # Rotation ±30°
    angle = random.uniform(-30, 30)
    M = cv2.getRotationMatrix2D((CROP_SIZE / 2, CROP_SIZE / 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (CROP_SIZE, CROP_SIZE),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    # Brightness/contrast jitter
    alpha = random.uniform(0.8, 1.2)   # contrast
    beta  = random.uniform(-20, 20)    # brightness
    img   = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return img


def to_tensor(img_bgr):
    """BGR uint8 HWC → float32 CHW, normalised to [0,1]."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb.astype(np.float32).transpose(2, 0, 1) / 255.0


# ─────────────────────────────────────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────────────────────────────────────

def build_mobilenet_v3(pretrained=True):
    import torchvision.models as tvm
    import torch.nn as nn
    weights = tvm.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model   = tvm.mobilenet_v3_small(weights=weights)
    # Replace final classifier — 576 is MobileNetV3-Small's last feature dim
    model.classifier = nn.Sequential(
        nn.Linear(576, 64),
        nn.Hardswish(),
        nn.Dropout(0.4),
        nn.Linear(64, 1),   # raw logit
    )
    return model, "mobilenet_v3_small"


def build_simple_cnn():
    import torch.nn as nn

    class SmallCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),  nn.BatchNorm2d(32),  nn.ReLU(),
                nn.MaxPool2d(2),                                          # 32×32
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
                nn.MaxPool2d(2),                                          # 16×16
                nn.Conv2d(64, 128, 3, padding=1),nn.BatchNorm2d(128), nn.ReLU(),
                nn.AdaptiveAvgPool2d(4),                                  # 4×4
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.5),
                nn.Linear(128 * 16, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            return self.head(self.features(x))

    return SmallCNN(), "simple_cnn"


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model, imgs, labels, optimizer, pos_weight, augment_fn=None,
              batch_size=16, training=True):
    import torch
    import torch.nn.functional as F

    model.train(training)
    rng = random.Random()
    order = list(range(len(imgs)))
    if training:
        rng.shuffle(order)

    total_loss = 0.0
    all_logits, all_labels = [], []

    for start in range(0, len(order), batch_size):
        batch_idx = order[start:start + batch_size]
        batch_imgs = []
        batch_lbl  = []
        for i in batch_idx:
            img = imgs[i].copy()
            if training and augment_fn:
                img = augment_fn(img)
            batch_imgs.append(to_tensor(img))
            batch_lbl.append(float(labels[i]))

        x = torch.tensor(np.stack(batch_imgs), dtype=torch.float32)
        y = torch.tensor(batch_lbl, dtype=torch.float32).unsqueeze(1)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(x)
            pw = torch.tensor([[pos_weight]], dtype=torch.float32)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)

        if training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * len(batch_idx)
        all_logits.extend(logits.detach().squeeze(1).tolist())
        all_labels.extend(batch_lbl)

    return total_loss / max(len(imgs), 1), all_logits, all_labels


def precision_recall_at_thresholds(logits, labels, thresholds=None):
    import torch
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    import torch.nn.functional as F
    probs  = torch.sigmoid(torch.tensor(logits)).numpy()
    labels = np.array(labels)
    results = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())
        prec  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f05   = (1 + 0.5**2) * prec * rec / (0.5**2 * prec + rec + 1e-9)
        f1    = 2 * prec * rec / (prec + rec + 1e-9)
        results.append({"threshold": t, "precision": round(prec, 4),
                        "recall": round(rec, 4), "f05": round(f05, 4),
                        "f1": round(f1, 4),
                        "tp": tp, "fp": fp, "fn": fn, "tn": tn})
    return results


def best_precision_threshold(pr_rows, min_recall=0.60):
    """
    Return the threshold that maximises F0.5 subject to recall >= min_recall.
    Falls back to the row with maximum F0.5 if no row meets min_recall.
    """
    candidates = [r for r in pr_rows if r["recall"] >= min_recall]
    pool = candidates if candidates else pr_rows
    return max(pool, key=lambda r: r["f05"])


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=75)
    parser.add_argument("--lr",         type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--no-pretrain", action="store_true",
                        help="Use simple CNN instead of MobileNetV3")
    parser.add_argument("--min-recall", type=float, default=0.60,
                        help="Minimum recall required when selecting threshold")
    args = parser.parse_args()

    import torch
    import torch.optim as optim

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────────────────
    print("Loading crops...")
    imgs, labels = load_crops(CROP_DIR)
    if not imgs:
        print("ERROR: no crops found. Run extract_cue_crops.py first.")
        sys.exit(1)

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    print(f"  cue_ball: {n_pos}   not_cue: {n_neg}   total: {len(imgs)}")

    if n_pos == 0 or n_neg == 0:
        print("ERROR: need both cue_ball and not_cue crops.")
        sys.exit(1)

    tr_imgs, tr_labels, va_imgs, va_labels = stratified_split(
        imgs, labels, force_train_stems=FORCE_TRAIN_STEMS)
    print(f"  train: {len(tr_imgs)} ({sum(tr_labels)} pos)  "
          f"val: {len(va_imgs)} ({sum(va_labels)} pos)")

    # pos_weight: penalise FP more — weight NOT oversampling.
    # For the classifier we want high precision, so we set pos_weight slightly
    # below the natural ratio to push the model toward caution on cue calls.
    nat_ratio = n_neg / max(n_pos, 1)
    pos_weight = max(1.0, nat_ratio * 0.5)   # half-natural penalty
    print(f"  natural ratio neg/pos = {nat_ratio:.1f} → pos_weight = {pos_weight:.2f}")

    # ── Build model ───────────────────────────────────────────────────────────
    if not args.no_pretrain:
        try:
            model, arch = build_mobilenet_v3(pretrained=True)
            print(f"Model: {arch} (pretrained)")
        except ImportError:
            print("torchvision not available — falling back to simple CNN")
            model, arch = build_simple_cnn()
    else:
        model, arch = build_simple_cnn()
        print(f"Model: {arch}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Train ─────────────────────────────────────────────────────────────────
    best_f05     = -1.0
    best_state   = None
    best_row     = None
    history      = []

    print(f"\nTraining {arch} for {args.epochs} epochs...")
    print(f"{'Ep':>4}  {'tr_loss':>8}  {'va_loss':>8}  "
          f"{'prec@0.5':>9}  {'rec@0.5':>8}  {'f05@0.5':>8}")
    print("─" * 60)

    for ep in range(1, args.epochs + 1):
        tr_loss, _, _ = run_epoch(model, tr_imgs, tr_labels, optimizer,
                                  pos_weight, augment_fn=augment,
                                  batch_size=args.batch_size, training=True)
        va_loss, va_logits, va_labs = run_epoch(model, va_imgs, va_labels,
                                                optimizer, pos_weight,
                                                training=False)
        scheduler.step()

        pr = precision_recall_at_thresholds(va_logits, va_labs,
                                            thresholds=[0.3, 0.4, 0.5, 0.6, 0.7])
        row_05  = next(r for r in pr if r["threshold"] == 0.5)
        best_pr = best_precision_threshold(pr, min_recall=args.min_recall)

        if best_pr["f05"] > best_f05:
            best_f05   = best_pr["f05"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_row   = {"epoch": ep, **best_pr}

        history.append({"epoch": ep, "tr_loss": round(tr_loss, 4),
                        "va_loss": round(va_loss, 4), "pr": pr})

        print(f"{ep:>4}  {tr_loss:>8.4f}  {va_loss:>8.4f}  "
              f"{row_05['precision']:>9.4f}  {row_05['recall']:>8.4f}  "
              f"{row_05['f05']:>8.4f}"
              + ("  ← best" if ep == best_row.get("epoch") else ""))

    # ── Save checkpoint ───────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    ckpt_path = CKPT_DIR / "cue_id_v3.pt"
    torch.save({
        "arch":         arch,
        "crop_size":    CROP_SIZE,
        "state_dict":   best_state,
        "best_row":     best_row,
        "train_stats":  {"n_pos": n_pos, "n_neg": n_neg,
                         "n_train": len(tr_imgs), "n_val": len(va_imgs)},
        "pos_weight":   pos_weight,
    }, str(ckpt_path))
    print(f"\nSaved checkpoint → {ckpt_path}")

    # ── Full val evaluation with best threshold ────────────────────────────────
    _, va_logits, va_labs = run_epoch(model, va_imgs, va_labels, optimizer,
                                     pos_weight, training=False)
    pr_all = precision_recall_at_thresholds(
        va_logits, va_labs,
        thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )
    selected = best_precision_threshold(pr_all, min_recall=args.min_recall)

    # ── Confusion examples (val set, at selected threshold) ───────────────────
    import torch
    import torch.nn.functional as F
    probs  = torch.sigmoid(torch.tensor(va_logits)).numpy()
    thresh = selected["threshold"]
    preds  = (probs >= thresh).astype(int)
    va_labs_arr = np.array(va_labs)

    # Build confusion records
    # Load manifest to get filenames for val crops
    manifest_path = CROP_DIR / "manifest.json"
    confusion_records = []
    if manifest_path.exists():
        mf       = json.loads(manifest_path.read_text())
        all_recs = mf.get("crops", [])
        # Replicate the stratified split (including FORCE_TRAIN_STEMS) to
        # recover val file names — must match exactly what main() did.
        all_imgs_check, all_labels_check = load_crops(CROP_DIR)
        rng_check = random.Random(SEED)
        pos_idx = [i for i, l in enumerate(all_labels_check) if l == 1]
        neg_idx = [i for i, l in enumerate(all_labels_check) if l == 0]
        rng_check.shuffle(pos_idx); rng_check.shuffle(neg_idx)
        n_val_pos = max(1, int(len(pos_idx) * VAL_FRAC))
        n_val_neg = max(1, int(len(neg_idx) * VAL_FRAC))

        cue_recs_check = [r for r in all_recs if r["label"] == "cue_ball"]
        val_pos_raw = pos_idx[:n_val_pos]
        # Apply force_train filter to mirror the actual split
        forced = {i for i in val_pos_raw
                  if cue_recs_check[i]["stem"] in FORCE_TRAIN_STEMS} \
                 if cue_recs_check else set()
        val_pos_final = [i for i in val_pos_raw if i not in forced]
        val_src_idx = val_pos_final + neg_idx[:n_val_neg]

        for local_i, src_i in enumerate(val_src_idx):
            if local_i >= len(va_labs):
                break   # safety: val_src_idx must not exceed actual val size
            if src_i < len(all_recs):
                rec = all_recs[src_i]
                gt  = va_labs[local_i]
                pr  = int(preds[local_i])
                pb  = float(probs[local_i])
                if gt == 1 and pr == 0:
                    confusion_records.append({"type": "FN", "file": rec["file"],
                                              "prob": round(pb, 3)})
                elif gt == 0 and pr == 1:
                    confusion_records.append({"type": "FP", "file": rec["file"],
                                              "prob": round(pb, 3),
                                              "source": rec.get("source","?")})

    # ── Write training report ─────────────────────────────────────────────────
    report_lines = [
        "# Cue Identity Classifier — Training Report",
        f"**Architecture**: {arch}",
        f"**Dataset**: {n_pos} cue_ball  /  {n_neg} not_cue  (total {n_pos + n_neg})",
        f"**Train**: {len(tr_imgs)}  |  **Val**: {len(va_imgs)}",
        f"**Epochs**: {args.epochs}  |  **LR**: {args.lr}  |  **pos_weight**: {pos_weight:.2f}",
        f"**Best checkpoint**: epoch {best_row.get('epoch')}  "
        f"(F0.5={best_row.get('f05','?')} @ threshold {best_row.get('threshold','?')})",
        "",
        "## Precision / Recall at Multiple Thresholds (val set)",
        "",
        "| threshold | precision | recall | F0.5 | F1 | TP | FP | FN | TN |",
        "|-----------|-----------|--------|------|----|----|----|----|-----|",
    ]
    for r in pr_all:
        marker = "  ← selected" if r["threshold"] == selected["threshold"] else ""
        report_lines.append(
            f"| {r['threshold']:.1f} | {r['precision']:.4f} | {r['recall']:.4f} | "
            f"{r['f05']:.4f} | {r['f1']:.4f} | {r['tp']} | {r['fp']} | "
            f"{r['fn']} | {r['tn']} |{marker}"
        )

    report_lines += [
        "",
        f"## Selected Operating Threshold: **{selected['threshold']}**",
        f"- Precision : {selected['precision']:.4f}",
        f"- Recall    : {selected['recall']:.4f}",
        f"- F0.5      : {selected['f05']:.4f}",
        f"- F1        : {selected['f1']:.4f}",
        "",
        "## Confusion Examples (val set at selected threshold)",
        "",
    ]

    fn_list = [c for c in confusion_records if c["type"] == "FN"]
    fp_list = [c for c in confusion_records if c["type"] == "FP"]

    if fn_list:
        report_lines.append("### False Negatives (missed cue balls — recall failures)")
        for c in sorted(fn_list, key=lambda x: x["prob"]):
            report_lines.append(f"  - `{c['file']}` (prob={c['prob']})")
    else:
        report_lines.append("### False Negatives: none on val set")

    report_lines.append("")

    if fp_list:
        report_lines.append("### False Positives (hallucinated cue balls — precision failures)")
        for c in sorted(fp_list, key=lambda x: -x["prob"]):
            src = c.get("source", "?")
            report_lines.append(f"  - `{c['file']}` (prob={c['prob']}, source={src})")
    else:
        report_lines.append("### False Positives: none on val set")

    report_path = REPORT_DIR / "training_report.md"
    report_path.write_text("\n".join(report_lines))

    # ── Save full results JSON ────────────────────────────────────────────────
    results_path = REPORT_DIR / "training_results.json"
    results_path.write_text(json.dumps({
        "arch": arch, "epochs": args.epochs,
        "dataset": {"n_pos": n_pos, "n_neg": n_neg},
        "best_row": best_row,
        "selected_threshold": selected,
        "pr_table": pr_all,
        "confusion": confusion_records,
    }, indent=2))

    print(f"\n{'─'*60}")
    print(f"Selected threshold : {selected['threshold']}")
    print(f"  Precision        : {selected['precision']:.4f}")
    print(f"  Recall           : {selected['recall']:.4f}")
    print(f"  F0.5             : {selected['f05']:.4f}")
    print(f"  FP on val        : {selected['fp']}")
    print(f"  FN on val        : {selected['fn']}")
    print(f"\nReport   → {report_path}")
    print(f"Results  → {results_path}")
    print(f"Checkpoint → {ckpt_path}")
    print(f"\nNext: python scripts/audit_cue_id.py "
          f"--ckpt {ckpt_path} "
          f"--threshold 0.3 "
          f"--ball-ckpt models/checkpoints/ball_yolo_v7_below_baseline.pt")


if __name__ == "__main__":
    main()
