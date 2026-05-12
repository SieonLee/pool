"""
Unified perception pipeline: image → planner-ready JSON.

Pipeline:
  1. Pose corner model → predict 4 table corners
  2. Warp to 900×450 top-view
  3. YOLO ball detector → cue_ball / object_ball detections
  4. Postprocess: NMS, cue resolution, max-ball cap
  5. [Optional] Cue identity classifier gate — rejects false-positive cue calls
  6. Output JSON + debug overlays

Output per image → review/perception/<stem>/
  <stem>_corners.jpg      original + predicted corners
  <stem>_warp.jpg         clean warped table
  <stem>_balls.jpg        warped + YOLO detections
  <stem>_perception.json  planner-ready JSON

Summary → review/perception/summary.json

Run:
  python scripts/perceive.py                          # all labeled eval images
  python scripts/perceive.py --image picture/foo.jpg  # single image
  python scripts/perceive.py --all                    # all images in picture/
  python scripts/perceive.py --cue-id-ckpt models/checkpoints/cue_id_v3.pt \\
      --cue-id-threshold 0.3     # enable classifier gate
"""
import argparse
import cv2
import json
import math
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent.parent
CORNER_CKPT = BASE / "models" / "checkpoints" / "table_corners_mvp_v1.pt"
BALL_CKPT   = BASE / "models" / "checkpoints" / "ball_yolo_active.pt"
PICTURE_DIR = BASE / "picture"
OUT_DIR     = BASE / "review" / "perception"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WARP_W, WARP_H = 900, 450
MAX_BALLS = 16
BALL_CONF_THRESH    = 0.25
CUE_FALLBACK_CONF   = 0.10  # second-pass threshold when no cue found at 0.25
CORNER_NAMES = ["TL", "TR", "BR", "BL"]
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Appearance thresholds for cue recovery from object_ball candidates.
# Based on measured brightness/saturation of confirmed cue balls across dataset:
#   images(219/130), 014(220/94), 018(201/94), 022(148/75)
CUE_RECOVER_MIN_BRIGHT  = 140   # HSV V channel mean over bbox patch
CUE_RECOVER_MAX_SAT     = 120   # HSV S channel mean over bbox patch
CUE_RECOVER_MIN_ASPECT  = 0.32  # min(w,h)/max(w,h) — balls can be squished by warp

# Cue identity classifier — loaded at runtime when --cue-id-ckpt is provided.
# When active, every YOLO cue_ball prediction is confirmed by the classifier
# before being accepted.  Classifier rejects go to not_cue; if nothing passes
# the classifier, cue is left as missing rather than hallucinating one.
CUE_ID_CKPT      = None    # Path or None
CUE_ID_THRESHOLD = 0.30    # classifier sigmoid score threshold (v3 tuned: prec=0.909, rec=1.000, plan_ready 24→23)
CUE_ID_CROP_SIZE = 64
CUE_ID_CONTEXT   = 1.6     # crop radius = ball_r * context
_CUE_ID_MODEL    = None    # loaded once on first use

CUE_COLOR = (255, 255, 255)
OBJ_COLOR = (0, 165, 255)
CORNER_COLORS = [(0, 255, 0), (0, 200, 255), (0, 0, 255), (255, 0, 255)]

# Images used for current eval baseline
EVAL_STEMS = [
    "images", "pool_real_004", "pool_real_013", "pool_real_014",
    "pool_real_015", "pool_real_016", "pool_real_017", "pool_real_018",
    "pool_real_019", "pool_real_023", "pool_real_024", "pool_real_026",
    "original", "pool_real_022", "pool_real_002", "pool_real_010",
]
SKIP_STEMS = {"new_uploads_contact_sheet", "search_contact_sheet", "thumb",
              "pexels-photo-10627132"}

# Images whose warped result is unrecognizable due to low source resolution.
# These are hard-gated: skip YOLO entirely and return a fixed status.
# Reason: labeling attempts confirmed no identifiable balls; bad labels are
# worse than no labels. Do not force planner on these images.
LOW_QUALITY_WARP_STEMS = {
    "pool_real_002": "source_resolution_too_low",   # 1000x802 but warp unrecognizable
    "pool_real_010": "source_resolution_too_low",   # 340x255 source, upscaled to 900x450
}


# ── helpers ───────────────────────────────────────────────────────────────────

def find_image(stem):
    for ext in [".jpg", ".jpeg", ".png"]:
        p = PICTURE_DIR / (stem + ext)
        if p.exists():
            return p
    return None


def get_corners(corner_model, img):
    results = corner_model(img, verbose=False)
    if not results or results[0].keypoints is None:
        return None, None
    kps = results[0].keypoints
    if len(kps) == 0:
        return None, None
    if kps.conf is not None:
        best = int(kps.conf.mean(dim=1).argmax())
        conf = float(kps.conf[best].mean().item())
    else:
        best, conf = 0, None
    return kps.xy[best].cpu().numpy(), conf


def warp_image(img, corners):
    dst = np.float32([[0, 0], [WARP_W, 0], [WARP_W, WARP_H], [0, WARP_H]])
    M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (WARP_W, WARP_H)), M


def run_ball_detector(ball_model, warped, conf_thresh=BALL_CONF_THRESH):
    results = ball_model(warped, verbose=False, conf=conf_thresh)
    dets = []
    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        for i in range(len(boxes)):
            cls  = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            dets.append((cls, conf, float(x1), float(y1), float(x2), float(y2)))
    return dets


def try_cue_at_lower_thresh(ball_model, warped, conf=CUE_FALLBACK_CONF):
    """
    Re-run YOLO at a lower threshold and return only the highest-confidence
    cue_ball detection, or None if still nothing found.
    Used when the primary run at 0.25 misses the cue.
    """
    results = ball_model(warped, verbose=False, conf=conf)
    best = None
    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        for i in range(len(boxes)):
            if int(boxes.cls[i].item()) != 0:
                continue
            cf = float(boxes.conf[i].item())
            if best is None or cf > best[1]:
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                best = (0, cf, float(x1), float(y1), float(x2), float(y2))
    return best


def _patch_appearance(warped, x1, y1, x2, y2):
    """Return (brightness, saturation, aspect_ratio) of a bbox patch."""
    px1, py1 = max(0, int(x1)), max(0, int(y1))
    px2, py2 = min(WARP_W, int(x2)), min(WARP_H, int(y2))
    patch = warped[py1:py2, px1:px2]
    if patch.size == 0:
        return 0.0, 255.0, 0.0
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    bright = float(hsv[:, :, 2].mean())
    sat    = float(hsv[:, :, 1].mean())
    w, h   = px2 - px1, py2 - py1
    aspect = min(w, h) / max(w, h) if max(w, h) > 0 else 0.0
    return bright, sat, aspect


def cue_recovery_by_appearance(warped, obj_dets):
    """
    Conservative fallback: if no cue_ball was detected, promote the
    object_ball candidate that looks most like a white/bright cue ball.

    Criteria (all must pass):
      - brightness  > CUE_RECOVER_MIN_BRIGHT  (not a dark ball)
      - saturation  < CUE_RECOVER_MAX_SAT     (not a colored ball)
      - aspect      > CUE_RECOVER_MIN_ASPECT  (roughly round, not a line)

    Returns the best (cls=0) det tuple with modified class, or None.
    Adds 'cue_recovered_by_appearance' warning upstream.
    """
    candidates = []
    for det in obj_dets:
        _, cf, x1, y1, x2, y2 = det
        bright, sat, aspect = _patch_appearance(warped, x1, y1, x2, y2)
        if (bright  >= CUE_RECOVER_MIN_BRIGHT and
                sat <= CUE_RECOVER_MAX_SAT     and
                aspect >= CUE_RECOVER_MIN_ASPECT):
            # Score: reward brightness, penalise saturation
            score = bright - sat
            candidates.append((score, bright, sat, det))

    if not candidates:
        return None

    best_score, best_bright, best_sat, best_det = max(candidates, key=lambda x: x[0])
    cls, cf, x1, y1, x2, y2 = best_det
    return (0, cf, x1, y1, x2, y2), round(best_bright, 1), round(best_sat, 1)


# ── Cue identity classifier helpers ──────────────────────────────────────────

def _load_cue_id_model():
    """Load the cue identity classifier once; cache in _CUE_ID_MODEL."""
    global _CUE_ID_MODEL
    if _CUE_ID_MODEL is not None:
        return _CUE_ID_MODEL
    if CUE_ID_CKPT is None:
        return None
    try:
        import torch
        import torch.nn as nn
        ckpt = torch.load(str(CUE_ID_CKPT), map_location="cpu", weights_only=False)
        arch = ckpt.get("arch", "simple_cnn")

        if "mobilenet" in arch:
            import torchvision.models as tvm
            model = tvm.mobilenet_v3_small(weights=None)
            model.classifier = nn.Sequential(
                nn.Linear(576, 64), nn.Hardswish(), nn.Dropout(0.4), nn.Linear(64, 1))
        else:
            class SmallCNN(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.features = nn.Sequential(
                        nn.Conv2d(3,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                        nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                        nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
                    )
                    self.head = nn.Sequential(
                        nn.Flatten(), nn.Dropout(0.5), nn.Linear(128*16, 64), nn.ReLU(), nn.Linear(64,1))
                def forward(self, x): return self.head(self.features(x))
            model = SmallCNN()

        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        _CUE_ID_MODEL = model
        return model
    except Exception as e:
        print(f"  [cue_id] WARNING: could not load classifier: {e}")
        return None


def _cue_id_score(warped, x1, y1, x2, y2):
    """
    Run the cue identity classifier on a ball bbox patch.
    Returns sigmoid probability [0,1] of being the cue ball.
    Returns None if classifier not loaded.
    """
    model = _load_cue_id_model()
    if model is None:
        return None
    try:
        import torch
        cx   = (x1 + x2) / 2;  cy = (y1 + y2) / 2
        r    = max(x2 - x1, y2 - y1) / 2 * CUE_ID_CONTEXT
        cr   = int(math.ceil(r))
        px1  = max(0, int(cx - cr)); py1 = max(0, int(cy - cr))
        px2  = min(WARP_W, px1 + cr * 2); py2 = min(WARP_H, py1 + cr * 2)
        patch = warped[py1:py2, px1:px2]
        if patch.size == 0:
            return None
        patch = cv2.resize(patch, (CUE_ID_CROP_SIZE, CUE_ID_CROP_SIZE))
        rgb   = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x     = torch.tensor(rgb.transpose(2, 0, 1)).unsqueeze(0)
        with torch.no_grad():
            logit = model(x).item()
        return float(torch.sigmoid(torch.tensor(logit)).item())
    except Exception:
        return None


def _cue_id_verify(warped, det):
    """
    Return True if the cue identity classifier confirms this detection is a
    cue ball, or if no classifier is loaded (pass-through).
    Returns (verified: bool, score: float | None).
    """
    if CUE_ID_CKPT is None:
        return True, None
    _, _, x1, y1, x2, y2 = det
    score = _cue_id_score(warped, x1, y1, x2, y2)
    if score is None:
        return True, None   # classifier unavailable → pass-through
    return score >= CUE_ID_THRESHOLD, score


def postprocess(dets):
    """
    - Keep conf >= BALL_CONF_THRESH (already filtered)
    - Resolve cue_ball: keep highest-conf only
    - Cap total balls at MAX_BALLS (by confidence)
    - Return sorted list: cue_ball first, then object_balls by confidence
    """
    warnings = []

    cue_dets = [(c, conf, x1, y1, x2, y2)
                for c, conf, x1, y1, x2, y2 in dets if c == 0]
    obj_dets = [(c, conf, x1, y1, x2, y2)
                for c, conf, x1, y1, x2, y2 in dets if c == 1]

    # Resolve multiple cue detections
    if len(cue_dets) > 1:
        warnings.append(f"multiple_cue_detections:{len(cue_dets)}_kept_highest_conf")
        cue_dets = [max(cue_dets, key=lambda d: d[1])]

    # Sort object balls by confidence desc
    obj_dets = sorted(obj_dets, key=lambda d: -d[1])

    combined = cue_dets + obj_dets

    # Cap at MAX_BALLS
    if len(combined) > MAX_BALLS:
        warnings.append(f"capped_at_{MAX_BALLS}_from_{len(combined)}")
        combined = combined[:MAX_BALLS]

    return combined, warnings


def det_to_ball(idx, cls, conf, x1, y1, x2, y2):
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    r  = max(x2 - x1, y2 - y1) / 2
    return {
        "id": idx,
        "type": "cue_ball" if cls == 0 else "object_ball",
        "x": round(cx, 1),
        "y": round(cy, 1),
        "r": round(r, 1),
        "confidence": round(conf, 3),
    }


# ── debug visuals ──────────────────────────────────────────────────────────────

def draw_corners(img, corners, conf=None):
    canvas = img.copy()
    for i, (x, y) in enumerate(corners):
        x, y = int(round(x)), int(round(y))
        cv2.circle(canvas, (x, y), 8, CORNER_COLORS[i], -1)
        label = CORNER_NAMES[i] + (f" {conf:.2f}" if conf else "")
        cv2.putText(canvas, label, (x + 10, y - 10), FONT, 0.65, CORNER_COLORS[i], 2)
    pts = np.array([(int(round(x)), int(round(y))) for x, y in corners], np.int32)
    cv2.polylines(canvas, [pts], True, (255, 255, 255), 2)
    return canvas


def draw_balls(warped, balls, warnings):
    canvas = warped.copy()
    for b in balls:
        cx, cy, r = int(b["x"]), int(b["y"]), int(b["r"])
        color = CUE_COLOR if b["type"] == "cue_ball" else OBJ_COLOR
        cv2.circle(canvas, (cx, cy), max(r, 8), color, 2)
        cv2.circle(canvas, (cx, cy), 2, color, -1)
        label = f"{'C' if b['type']=='cue_ball' else 'O'}{b['id']} {b['confidence']:.2f}"
        cv2.putText(canvas, label, (cx - 14, cy + int(r) + 14), FONT, 0.38, color, 1)

    n_cue = sum(1 for b in balls if b["type"] == "cue_ball")
    n_obj = sum(1 for b in balls if b["type"] == "object_ball")
    cv2.putText(canvas, f"cue={n_cue}  obj={n_obj}  total={len(balls)}",
                (8, 24), FONT, 0.65, (200, 200, 200), 2)
    if warnings:
        cv2.putText(canvas, " | ".join(warnings), (8, WARP_H - 10),
                    FONT, 0.42, (0, 100, 255), 1)
    return canvas


# ── main pipeline ─────────────────────────────────────────────────────────────

def process_image(corner_model, ball_model, img_path: Path, conf_thresh=BALL_CONF_THRESH):
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    stem = img_path.stem
    img_out = OUT_DIR / stem
    img_out.mkdir(exist_ok=True)

    # Hard gate: low-quality warp — do not run YOLO, return fixed failure status.
    # Labeling confirmed: balls unrecognizable after pose-warp due to source quality.
    lq_reason = LOW_QUALITY_WARP_STEMS.get(stem)
    if lq_reason:
        result = {
            "image": img_path.name,
            "status": {
                "cue_present": False,
                "ball_count": 0,
                "cue_ball_count": 0,
                "object_ball_count": 0,
                "ready_for_planner": False,
                "warp_quality": "low_quality_warp",
                "warnings": ["low_quality_warp", lq_reason],
            },
        }
        with open(img_out / f"{stem}_perception.json", "w") as f:
            json.dump(result, f, indent=2)
        # Still save warp image for visual inspection
        corners, corner_conf = get_corners(corner_model, img)
        if corners is not None:
            warped, _ = warp_image(img, corners)
            cv2.imwrite(str(img_out / f"{stem}_warp.jpg"), warped)
            corners_vis = draw_corners(img, corners, corner_conf)
            cv2.imwrite(str(img_out / f"{stem}_corners.jpg"), corners_vis)
        return result

    # 1. Corner detection
    corners, corner_conf = get_corners(corner_model, img)
    if corners is None:
        return {
            "image": img_path.name,
            "status": {
                "cue_present": False,
                "ball_count": 0,
                "ready_for_planner": False,
                "warnings": ["no_table_detected"],
            }
        }

    # Save corners overlay
    corners_vis = draw_corners(img, corners, corner_conf)
    cv2.imwrite(str(img_out / f"{stem}_corners.jpg"), corners_vis)

    # 2. Warp
    warped, M = warp_image(img, corners)
    cv2.imwrite(str(img_out / f"{stem}_warp.jpg"), warped)

    # 3. YOLO ball detection (primary pass at conf_thresh)
    raw_dets = run_ball_detector(ball_model, warped, conf_thresh)

    # 4. Postprocess
    final_dets, pp_warnings = postprocess(raw_dets)

    # Build ball list
    balls = [det_to_ball(i, *d) for i, d in enumerate(final_dets)]

    # 5a. Cue identity classifier gate (if loaded) ─────────────────────────────
    # For every YOLO cue_ball detection, run the identity classifier.
    # If classifier rejects it, demote to object_ball.
    # Motivation: YOLO v7 makes high-confidence semantic errors on striped balls,
    # partial whites, and specular highlights; whiteness alone is not sufficient.
    recovery_warnings = []
    cue_id_rejections = 0

    if CUE_ID_CKPT is not None:
        verified_balls = []
        for b in balls:
            if b["type"] != "cue_ball":
                verified_balls.append(b)
                continue
            x1 = b["x"] - b["r"]; y1 = b["y"] - b["r"]
            x2 = b["x"] + b["r"]; y2 = b["y"] + b["r"]
            ok, score = _cue_id_verify(warped, (0, b["confidence"], x1, y1, x2, y2))
            if ok:
                if score is not None:
                    b = dict(b, cue_id_score=round(score, 3))
                verified_balls.append(b)
            else:
                # Demote: reclassify as object_ball
                b = dict(b, type="object_ball",
                         cue_id_score=round(score, 3) if score is not None else None,
                         cue_id_rejected=True)
                verified_balls.append(b)
                cue_id_rejections += 1
        balls = verified_balls
        if cue_id_rejections:
            recovery_warnings.append(
                f"cue_id_rejected:{cue_id_rejections}"
            )

    # 5b. Cue recovery — only if no cue remains after classifier gate
    cue_balls = [b for b in balls if b["type"] == "cue_ball"]
    obj_balls = [b for b in balls if b["type"] == "object_ball"]

    if not cue_balls:
        # Pass 2: re-run at lower threshold (catches conf 0.10–0.24)
        fallback_det = try_cue_at_lower_thresh(ball_model, warped)
        if fallback_det is not None:
            # Also verify the fallback cue with classifier
            ok, score = _cue_id_verify(warped, fallback_det)
            if ok:
                cue_ball_entry = det_to_ball(len(balls), *fallback_det)
                if score is not None:
                    cue_ball_entry = dict(cue_ball_entry, cue_id_score=round(score, 3))
                balls.insert(0, cue_ball_entry)
                cue_balls = [cue_ball_entry]
                recovery_warnings.append(
                    f"cue_recovered_by_threshold(conf={fallback_det[1]:.3f})"
                )
            else:
                recovery_warnings.append(
                    f"cue_threshold_fallback_rejected_by_classifier"
                    f"(score={score:.3f})" if score is not None else
                    "cue_threshold_fallback_rejected_by_classifier"
                )

    if not cue_balls and obj_balls:
        # Pass 3: appearance fallback — promote most cue-like object_ball,
        # but only if it also passes the identity classifier.
        obj_raw = [(1, b["confidence"], b["x"] - b["r"], b["y"] - b["r"],
                    b["x"] + b["r"], b["y"] + b["r"]) for b in obj_balls]
        result_ap = cue_recovery_by_appearance(warped, obj_raw)
        if result_ap is not None:
            recovered_det, bright, sat = result_ap
            ok, score = _cue_id_verify(warped, recovered_det)
            if ok:
                # Remove the promoted ball from obj_balls list
                promoted_x = (recovered_det[2] + recovered_det[4]) / 2
                promoted_y = (recovered_det[3] + recovered_det[5]) / 2
                balls = [b for b in balls
                         if not (b["type"] == "object_ball"
                                 and abs(b["x"] - promoted_x) < 5
                                 and abs(b["y"] - promoted_y) < 5)]
                cue_ball_entry = det_to_ball(len(balls), *recovered_det)
                if score is not None:
                    cue_ball_entry = dict(cue_ball_entry, cue_id_score=round(score, 3))
                balls.insert(0, cue_ball_entry)
                cue_balls = [cue_ball_entry]
                recovery_warnings.append(
                    f"cue_recovered_by_appearance(bright={bright},sat={sat})"
                )
            else:
                recovery_warnings.append(
                    f"cue_appearance_fallback_rejected_by_classifier"
                    f"(score={score:.3f},bright={bright},sat={sat})" if score is not None else
                    f"cue_appearance_fallback_rejected_by_classifier"
                    f"(bright={bright},sat={sat})"
                )

    # 5b. Status
    cue_balls  = [b for b in balls if b["type"] == "cue_ball"]
    obj_balls  = [b for b in balls if b["type"] == "object_ball"]
    cue_present = len(cue_balls) == 1
    warnings = list(pp_warnings) + recovery_warnings

    if not cue_present:
        warnings.append("cue_missing")
    if len(balls) > 15:
        warnings.append(f"high_ball_count:{len(balls)}")
    if len(obj_balls) == 0:
        warnings.append("no_object_balls")

    ready = cue_present and len(balls) >= 2 and len(balls) <= MAX_BALLS

    # 6. Save balls overlay
    balls_vis = draw_balls(warped, balls, warnings)
    cv2.imwrite(str(img_out / f"{stem}_balls.jpg"), balls_vis)

    # 7. Build output JSON
    result = {
        "image": img_path.name,
        "table": {
            "corners": corners.tolist(),
            "warp_size": [WARP_W, WARP_H],
            "corner_confidence": round(corner_conf, 3) if corner_conf else None,
        },
        "balls": balls,
        "status": {
            "cue_present": cue_present,
            "ball_count": len(balls),
            "cue_ball_count": len(cue_balls),
            "object_ball_count": len(obj_balls),
            "ready_for_planner": ready,
            "warnings": warnings,
        },
    }

    with open(img_out / f"{stem}_perception.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def print_report(results):
    print(f"\n{'=' * 76}")
    print("PERCEPTION PIPELINE REPORT")
    print(f"{'=' * 76}")
    fmt = "{:28s} {:>5} {:>4} {:>4} {:>8} {}"
    print(fmt.format("image", "balls", "cue", "obj", "ready", "warnings"))
    print("-" * 76)

    n_ready = n_cue_missing = n_over = n_lqw = 0
    for r in results:
        if r is None:
            continue
        st = r.get("status", {})
        ready  = st.get("ready_for_planner", False)
        n_ball = st.get("ball_count", 0)
        n_cue  = st.get("cue_ball_count", 0)
        n_obj  = st.get("object_ball_count", 0)
        warns  = ", ".join(st.get("warnings", [])) or "—"
        flag   = "✓" if ready else "✗"

        print(fmt.format(r["image"][:28], str(n_ball), str(n_cue),
                         str(n_obj), flag, warns))

        if ready:
            n_ready += 1
        if "cue_missing" in warns:
            n_cue_missing += 1
        if "low_quality_warp" in warns:
            n_lqw += 1
        if n_ball > MAX_BALLS:
            n_over += 1

    total = len([r for r in results if r])
    print("-" * 76)
    print(f"Planner-ready:     {n_ready}/{total}")
    print(f"Cue missing:       {n_cue_missing}")
    print(f"Low-quality warp:  {n_lqw}  (gated — not sent to planner)")
    print(f"Over-detection:    {n_over}")

    print(f"\n{'─'*76}")
    print("Shot planner eligibility:")
    for r in results:
        if r is None:
            continue
        st    = r.get("status", {})
        ready = st.get("ready_for_planner", False)
        warns = st.get("warnings", [])
        if "low_quality_warp" in warns:
            symbol = "⊘ LQ WARP "
        elif ready:
            symbol = "✓ READY   "
        else:
            symbol = "✗ NOT READY"
        reason = f"  ← {', '.join(warns)}" if warns else ""
        print(f"  {symbol}  {r['image'][:28]}{reason}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None, help="Single source image path")
    parser.add_argument("--all", action="store_true",
                        help="Run on all images in picture/")
    parser.add_argument("--conf", type=float, default=BALL_CONF_THRESH,
                        help=f"Ball detection confidence threshold (default {BALL_CONF_THRESH})")
    parser.add_argument("--ball-ckpt", default=None,
                        help="Override ball detector checkpoint path")
    parser.add_argument("--cue-id-ckpt", default=None,
                        help="Cue identity classifier checkpoint "
                             "(models/checkpoints/cue_id_v1.pt)")
    parser.add_argument("--cue-id-threshold", type=float, default=0.30,
                        help="Classifier score threshold (default 0.30; tuned on audit set)")
    args = parser.parse_args()

    # Wire classifier globals before any processing
    global CUE_ID_CKPT, CUE_ID_THRESHOLD
    if args.cue_id_ckpt:
        CUE_ID_CKPT      = Path(args.cue_id_ckpt)
        CUE_ID_THRESHOLD = args.cue_id_threshold
        print(f"  Cue ID classifier: {CUE_ID_CKPT.name}  threshold={CUE_ID_THRESHOLD}")

    from ultralytics import YOLO
    conf_thresh = args.conf
    ball_ckpt = Path(args.ball_ckpt) if args.ball_ckpt else BALL_CKPT
    corner_model = YOLO(str(CORNER_CKPT))
    ball_model   = YOLO(str(ball_ckpt))

    if args.image:
        imgs = [Path(args.image)]
    elif args.all:
        exts = {".jpg", ".jpeg", ".png"}
        imgs = sorted(p for p in PICTURE_DIR.iterdir()
                      if p.suffix.lower() in exts
                      and p.stem not in SKIP_STEMS
                      and not p.stem.startswith("aug_")
                      and not p.stem.startswith("tgt_"))
    else:
        imgs = []
        for stem in EVAL_STEMS:
            p = find_image(stem)
            if p:
                imgs.append(p)

    print(f"Running perception pipeline on {len(imgs)} image(s)...")
    print(f"  Corner model: {CORNER_CKPT.name}")
    print(f"  Ball model:   {ball_ckpt.name}")
    print(f"  Conf thresh:  {conf_thresh}")
    print()

    results = []
    for img_path in imgs:
        print(f"  {img_path.name}...", end=" ", flush=True)
        r = process_image(corner_model, ball_model, img_path, conf_thresh)
        results.append(r)
        if r:
            st = r.get("status", {})
            ready = "✓" if st.get("ready_for_planner") else "✗"
            print(f"balls={st.get('ball_count','?')}  "
                  f"cue={'yes' if st.get('cue_present') else 'NO'}  "
                  f"ready={ready}  "
                  f"warns={st.get('warnings', [])}")
        else:
            print("FAILED")

    print_report(results)

    summary = {
        "corner_model": CORNER_CKPT.name,
        "ball_model": ball_ckpt.name,
        "conf_threshold": conf_thresh,
        "n_images": len(imgs),
        "results": [r for r in results if r],
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nOutputs  → {OUT_DIR}")
    print(f"Summary  → {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
