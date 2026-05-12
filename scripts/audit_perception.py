"""
Perception failure audit.

For each weak/failing image, generates:
  review/audit/<stem>/<stem>_raw_005.jpg   YOLO raw dets @ conf=0.05
  review/audit/<stem>/<stem>_raw_015.jpg   YOLO raw dets @ conf=0.15
  review/audit/<stem>/<stem>_raw_025.jpg   YOLO raw dets @ conf=0.25 (pipeline default)
  review/audit/<stem>/<stem>_labels.jpg    ground-truth labels from train set
  review/audit/<stem>/<stem>_corners.jpg   original image with pose corners
  review/audit/<stem>/<stem>_audit.json    per-image audit summary

Also writes:
  review/audit/audit_report.txt   human-readable report
  review/audit/audit_summary.json machine-readable

Run:
  python scripts/audit_perception.py                  # default weak set
  python scripts/audit_perception.py --all            # all perception stems
  python scripts/audit_perception.py --stem pool_real_002
"""
import argparse
import cv2
import json
import math
import numpy as np
from pathlib import Path
from ultralytics import YOLO

BASE = Path(__file__).parent.parent
CORNER_CKPT  = BASE / "models/checkpoints/table_corners_mvp_v1.pt"
BALL_CKPT    = BASE / "models/checkpoints/ball_yolo_v1.pt"
PICTURE_DIR  = BASE / "picture"
PERCEPTION_DIR = BASE / "review/perception"
DATASET_DIR  = BASE / "datasets/pose-warped-balls"
TRAIN_IMG    = DATASET_DIR / "images/train"
LOWRES_IMG   = DATASET_DIR / "images/lowres"
VAL_IMG      = DATASET_DIR / "images/val"
TRAIN_LBL    = DATASET_DIR / "labels/train"
VAL_LBL      = DATASET_DIR / "labels/val"
OUT_DIR      = BASE / "review/audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WARP_W, WARP_H = 900, 450
FONT = cv2.FONT_HERSHEY_SIMPLEX

WEAK_STEMS = [
    "pool_real_002", "pool_real_010", "pool_real_015",
    "pool_real_018", "pool_real_014", "pool_real_022",
    "pool_real_013", "pool_real_004", "pool_real_024",
]

CONF_LEVELS = [("005", 0.05), ("015", 0.15), ("025", 0.25)]

CLS_NAMES  = {0: "cue_ball", 1: "object_ball"}
CLS_COLORS = {0: (255, 255, 255), 1: (0, 165, 255)}


# ── helpers ───────────────────────────────────────────────────────────────────

def find_source(stem):
    for ext in [".jpg", ".jpeg", ".png"]:
        p = PICTURE_DIR / (stem + ext)
        if p.exists():
            return p
    return None


def find_warped(stem):
    for d in [TRAIN_IMG, LOWRES_IMG, VAL_IMG]:
        p = d / f"{stem}_warped.jpg"
        if p.exists():
            return p, d.name
    return None, None


def load_label_boxes(stem):
    for lbl_dir in [TRAIN_LBL, VAL_LBL]:
        lbl = lbl_dir / f"{stem}_warped.txt"
        if lbl.exists():
            break
    else:
        return None, "label_file_missing"
    text = lbl.read_text().strip()
    if not text:
        return [], "label_empty"
    boxes = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 5:
            cls = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])
            boxes.append({"cls": cls, "cx": cx, "cy": cy, "bw": bw, "bh": bh})
    return boxes, f"labeled:{len(boxes)}"


def draw_raw_dets(warped_img, boxes_raw, conf_thresh, label_boxes):
    canvas = warped_img.copy()

    for b in boxes_raw:
        cls, cf, x1, y1, x2, y2 = b
        col = CLS_COLORS[cls]
        cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
        tag = f"{'C' if cls==0 else 'O'} {cf:.2f}"
        cv2.putText(canvas, tag, (int(x1), int(y1) - 4), FONT, 0.38, col, 1)

    # Header
    n_cue = sum(1 for b in boxes_raw if b[0] == 0)
    n_obj = sum(1 for b in boxes_raw if b[0] == 1)
    cv2.putText(canvas,
                f"conf>{conf_thresh:.2f}: cue={n_cue} obj={n_obj} total={len(boxes_raw)}",
                (6, 18), FONT, 0.50, (220, 220, 220), 1)

    # Draw GT boxes in green (dashed look via alpha blend)
    if label_boxes:
        for b in label_boxes:
            cx_px = int(b["cx"] * WARP_W)
            cy_px = int(b["cy"] * WARP_H)
            bw_px = int(b["bw"] * WARP_W)
            bh_px = int(b["bh"] * WARP_H)
            x1g = cx_px - bw_px // 2
            y1g = cy_px - bh_px // 2
            x2g = cx_px + bw_px // 2
            y2g = cy_px + bh_px // 2
            cv2.rectangle(canvas, (x1g, y1g), (x2g, y2g), (0, 255, 0), 1)
        cv2.putText(canvas, f"GT:{len(label_boxes)} boxes (green)",
                    (6, WARP_H - 6), FONT, 0.38, (0, 200, 0), 1)

    return canvas


def draw_label_vis(warped_img, label_boxes, status):
    canvas = warped_img.copy()
    if label_boxes:
        for b in label_boxes:
            cx_px = int(b["cx"] * WARP_W)
            cy_px = int(b["cy"] * WARP_H)
            bw_px = int(b["bw"] * WARP_W)
            bh_px = int(b["bh"] * WARP_H)
            x1 = cx_px - bw_px // 2;  y1 = cy_px - bh_px // 2
            x2 = cx_px + bw_px // 2;  y2 = cy_px + bh_px // 2
            col = CLS_COLORS[b["cls"]]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 2)
            cv2.circle(canvas, (cx_px, cy_px), 3, col, -1)
            tag = "CUE" if b["cls"] == 0 else "OBJ"
            cv2.putText(canvas, tag, (x1, y1 - 4), FONT, 0.38, col, 1)

    cv2.putText(canvas, f"GT labels: {status}", (6, 18), FONT, 0.50, (220, 220, 220), 1)
    return canvas


def draw_corners_vis(src_img, corner_model, stem):
    results = corner_model(src_img, verbose=False)
    corners = None
    conf = None
    if results and results[0].keypoints is not None:
        kps = results[0].keypoints
        if len(kps) > 0:
            best = int(kps.conf.mean(dim=1).argmax()) if kps.conf is not None else 0
            conf = float(kps.conf[best].mean()) if kps.conf is not None else None
            corners = kps.xy[best].cpu().numpy()

    canvas = src_img.copy()
    if corners is not None:
        names = ["TL", "TR", "BR", "BL"]
        cols  = [(0,255,0),(0,200,255),(0,0,255),(255,0,255)]
        for i, (x, y) in enumerate(corners):
            x, y = int(round(x)), int(round(y))
            cv2.circle(canvas, (x, y), 8, cols[i], -1)
            cv2.putText(canvas, f"{names[i]} {conf:.2f}" if conf else names[i],
                        (x+8, y-8), FONT, 0.55, cols[i], 2)
        pts = np.array([(int(x), int(y)) for x, y in corners], np.int32)
        cv2.polylines(canvas, [pts], True, (255,255,255), 2)
    else:
        cv2.putText(canvas, "NO CORNERS DETECTED", (20, 40), FONT, 1.0, (0,0,255), 2)
    return canvas, corners, conf


# ── audit one stem ────────────────────────────────────────────────────────────

def audit_stem(stem, corner_model, ball_model):
    out = OUT_DIR / stem
    out.mkdir(exist_ok=True)

    audit = {"stem": stem, "issues": [], "actions": []}

    # 1. Source image
    src_path = find_source(stem)
    if src_path is None:
        audit["issues"].append("source_image_missing")
        return audit
    src_img = cv2.imread(str(src_path))
    h_src, w_src = src_img.shape[:2]
    diag = (w_src**2 + h_src**2) ** 0.5
    audit["source_size"] = [w_src, h_src]
    audit["source_diag"] = round(diag, 1)
    if diag < 600:
        audit["issues"].append(f"low_res_source(diag={diag:.0f}px)")

    # 2. Corner detection on source
    corners_img, corners, corner_conf = draw_corners_vis(src_img, corner_model, stem)
    cv2.imwrite(str(out / f"{stem}_corners.jpg"), corners_img)
    audit["corner_conf"] = round(corner_conf, 3) if corner_conf else None
    if corners is None:
        audit["issues"].append("corner_detection_failed")
        audit["actions"].append("SKIP: cannot warp without corners")
        return audit
    if corner_conf and corner_conf < 0.70:
        audit["issues"].append(f"low_corner_conf({corner_conf:.3f})")

    # 3. Warped image
    warp_path, warp_split = find_warped(stem)
    if warp_path is None:
        audit["issues"].append("warped_image_missing")
        audit["actions"].append("run: python scripts/export_warped_balls.py")
        return audit
    warped = cv2.imread(str(warp_path))
    audit["warp_split"] = warp_split
    if warp_split == "lowres":
        audit["issues"].append("warped_in_lowres_split(excluded_from_training)")
        audit["actions"].append(f"MOVE: {stem}_warped.jpg from lowres/ to train/ and create label stub")

    # 4. Label status
    label_boxes, label_status = load_label_boxes(stem)
    audit["label_status"] = label_status
    labels_img = draw_label_vis(warped, label_boxes or [], label_status)
    cv2.imwrite(str(out / f"{stem}_labels.jpg"), labels_img)

    if "empty" in label_status or "missing" in label_status:
        audit["issues"].append(f"label_status:{label_status}")
        audit["actions"].append(f"LABEL: python scripts/annotate_balls.py --image {stem}_warped.jpg")
    else:
        n_labels = len(label_boxes) if label_boxes else 0
        n_cue_gt = sum(1 for b in (label_boxes or []) if b["cls"] == 0)
        n_obj_gt = sum(1 for b in (label_boxes or []) if b["cls"] == 1)
        audit["gt_cue_count"] = n_cue_gt
        audit["gt_obj_count"] = n_obj_gt
        if n_cue_gt == 0:
            audit["issues"].append("no_cue_ball_in_labels")
            audit["actions"].append(f"RE-LABEL (add cue): python scripts/annotate_balls.py --image {stem}_warped.jpg --redo")

    # 5. Raw YOLO detections at multiple thresholds
    raw_by_conf = {}
    for tag, thresh in CONF_LEVELS:
        res = ball_model(warped, verbose=False, conf=thresh)
        boxes_raw = []
        if res and res[0].boxes is not None:
            bxs = res[0].boxes
            for i in range(len(bxs)):
                cls = int(bxs.cls[i])
                cf  = float(bxs.conf[i])
                x1, y1, x2, y2 = bxs.xyxy[i].cpu().numpy()
                boxes_raw.append((cls, round(cf, 3), x1, y1, x2, y2))

        raw_by_conf[thresh] = {
            "n_total": len(boxes_raw),
            "n_cue":   sum(1 for b in boxes_raw if b[0] == 0),
            "n_obj":   sum(1 for b in boxes_raw if b[0] == 1),
            "max_cue_conf": max((b[1] for b in boxes_raw if b[0] == 0), default=0.0),
            "max_obj_conf": max((b[1] for b in boxes_raw if b[0] == 1), default=0.0),
        }

        det_img = draw_raw_dets(warped, boxes_raw, thresh, label_boxes)
        cv2.imwrite(str(out / f"{stem}_raw_{tag}.jpg"), det_img)

    audit["detections"] = raw_by_conf

    # 6. Diagnose
    d025 = raw_by_conf[0.25]
    d005 = raw_by_conf[0.05]

    if d025["n_total"] == 0 and d005["n_total"] == 0:
        audit["issues"].append("zero_detections_even_at_005")
        audit["diagnosis"] = "model_blind_to_this_image"
        audit["actions"].append("LABEL + RETRAIN: model has no concept of balls in this scene")
    elif d025["n_total"] == 0 and d005["n_total"] > 0:
        max_conf = max(d005["max_cue_conf"], d005["max_obj_conf"])
        audit["issues"].append(f"detections_only_below_025(max_conf={max_conf:.3f})")
        audit["diagnosis"] = f"model_uncertain(max_conf={max_conf:.3f}) — labeling + retraining needed"
        audit["actions"].append("LABEL + RETRAIN: detections exist but below confidence threshold")
    elif d025["n_cue"] == 0 and d005["n_cue"] == 0:
        audit["issues"].append("no_cue_ball_detected_at_any_threshold")
        if label_boxes and sum(1 for b in label_boxes if b["cls"] == 0) == 0:
            audit["diagnosis"] = "cue_absent_from_scene(confirmed_by_labels)"
            # Remove RE-LABEL action — cue is genuinely absent, not a labeling error
            audit["actions"] = [a for a in audit["actions"] if "RE-LABEL" not in a]
            audit["actions"].append("OK: no cue in scene — not_ready_for_planner is correct")
        else:
            audit["diagnosis"] = "cue_present_but_not_detected — needs more cue diversity in training"
            audit["actions"].append("LABEL + RETRAIN: cue ball not detected")
    elif d025["n_cue"] > 1:
        audit["issues"].append(f"multiple_cue_detections:{d025['n_cue']}")
        audit["diagnosis"] = "over_detection — model confusing object balls for cue"
        audit["actions"].append("CHECK LABELS: ensure cue/obj distinction is correct, retrain")

    if not audit.get("actions"):
        audit["actions"].append("OK: no labeling action needed — monitor after retrain")

    return audit


# ── report builder ────────────────────────────────────────────────────────────

def build_report(audits):
    lines = []
    W = 80
    lines.append("=" * W)
    lines.append("PERCEPTION FAILURE AUDIT REPORT")
    lines.append("=" * W)

    n_zero = n_label_needed = n_ok = 0

    for a in audits:
        stem = a["stem"]
        lines.append(f"\n{'─'*W}")
        lines.append(f"  {stem}")
        lines.append(f"{'─'*W}")

        lines.append(f"  source  : {a.get('source_size','?')}  diag={a.get('source_diag','?')}px  "
                     f"warp_split={a.get('warp_split','?')}")
        lines.append(f"  corners : conf={a.get('corner_conf','?')}")
        lines.append(f"  labels  : {a.get('label_status','?')}  "
                     f"GT_cue={a.get('gt_cue_count','?')}  GT_obj={a.get('gt_obj_count','?')}")

        det = a.get("detections", {})
        for thresh in [0.05, 0.15, 0.25]:
            d = det.get(thresh, {})
            nt = d.get('n_total', '?')
            lines.append(f"  @{thresh:.2f}     : total={nt!s:>3}  "
                         f"cue={d.get('n_cue','?')}(max={d.get('max_cue_conf',0):.3f})  "
                         f"obj={d.get('n_obj','?')}(max={d.get('max_obj_conf',0):.3f})")

        lines.append(f"  issues  : {', '.join(a.get('issues', ['—']))}")
        lines.append(f"  diag    : {a.get('diagnosis', '—')}")
        lines.append(f"  actions :")
        for act in a.get("actions", []):
            lines.append(f"    → {act}")

        if "zero_detections" in " ".join(a.get("issues", [])):
            n_zero += 1
        if any("LABEL" in act for act in a.get("actions", [])):
            n_label_needed += 1
        else:
            n_ok += 1

    lines.append(f"\n{'='*W}")
    lines.append("SUMMARY")
    lines.append(f"{'─'*W}")
    lines.append(f"  Audited          : {len(audits)}")
    lines.append(f"  Need labeling    : {n_label_needed}")
    lines.append(f"  Zero detections  : {n_zero}")
    lines.append(f"  No action needed : {n_ok}")
    lines.append(f"\n  Images to label (run in order):")
    seen = set()
    for a in audits:
        for act in a.get("actions", []):
            if ("LABEL" in act or "MOVE" in act) and "OK:" not in act:
                key = (a["stem"], act.split(":")[0])
                if key not in seen:
                    seen.add(key)
                    lines.append(f"    {a['stem']:25s}  {act}")
    lines.append("=" * W)
    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stem", default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    print("Loading models...")
    corner_model = YOLO(str(CORNER_CKPT))
    ball_model   = YOLO(str(BALL_CKPT))

    if args.stem:
        stems = [args.stem]
    elif args.all:
        stems = sorted(
            p.parent.name
            for p in PERCEPTION_DIR.rglob("*_perception.json")
        )
    else:
        stems = WEAK_STEMS

    print(f"Auditing {len(stems)} image(s)...\n")
    audits = []
    for stem in stems:
        print(f"  {stem}...", end=" ", flush=True)
        a = audit_stem(stem, corner_model, ball_model)
        audits.append(a)
        issues = a.get("issues", [])
        print(f"issues={len(issues)}  {', '.join(issues[:2]) or '—'}")

    report = build_report(audits)
    print("\n" + report)

    (OUT_DIR / "audit_report.txt").write_text(report)
    with open(OUT_DIR / "audit_summary.json", "w") as f:
        json.dump({"stems": stems, "audits": audits}, f, indent=2)

    print(f"\nOutputs → {OUT_DIR}")


if __name__ == "__main__":
    main()
