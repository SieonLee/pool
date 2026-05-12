"""
Ball detection v2 — staged pipeline on pose-warped 900×450 table.

Stages:
  1. Pose model → warp table to 900×450
  2. Candidate detection  (HoughCircles, low threshold)
  3. Physical filtering   (radius, position, dedup, max 16)
  4. Cue-ball scoring     (HSV/LAB vs local cloth background)
  5. Classification       → cue_ball | object_ball

Debug outputs per image → review/ball_detect_v2/<stem>/
  _warp.jpg           clean pose warp
  _stage1_raw.jpg     all HoughCircles candidates
  _stage2_filtered.jpg   after physical filters
  _stage3_final.jpg   classified balls (cue=white, object=orange)
  _result.json        full per-image data

Summary → review/ball_detect_v2/eval_summary.json

Run:
  python scripts/ball_detect_v2.py
  python scripts/ball_detect_v2.py --image picture/pool_real_007.jpg
"""
import argparse
import cv2
import json
import numpy as np
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

BASE = Path(__file__).parent.parent
CHECKPOINT = BASE / "models" / "checkpoints" / "table_corners_mvp_v1.pt"
PICTURE_DIR = BASE / "picture"
GT_DIR = BASE / "annotations" / "table_corners_gt"
OUT_DIR = BASE / "review" / "ball_detect_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WARP_W, WARP_H = 900, 450
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ── Physical constants (900×450 warp, real table 2700×1350mm) ──────────────
# Ball ⌀ 57mm → 57/2700 × 900 ≈ 19px diameter → ~9.5px radius
# Allow perspective foreshortening ± distortion
BALL_R_MIN    = 7    # px
BALL_R_MAX    = 26   # px
BALL_MIN_DIST = 17   # px  (prevent double-counting same ball)
# Playable-area margin: rail+pocket area to exclude (increased to cut cushion FP)
PLAY_MARGIN_X = 38   # px from left/right edge
PLAY_MARGIN_Y = 35   # px from top/bottom edge
MAX_BALLS     = 16
# Minimum Sobel edge mean inside ball region — rejects low-contrast cushion noise
BALL_MIN_EDGE = 10.0
# Maximum HSV saturation for any ball (pockets/rails can be highly saturated)
BALL_MAX_SAT  = 210

# ── Cue-ball classifier thresholds ─────────────────────────────────────────
CUE_MIN_L       = 155    # LAB L* — cue ball must be bright
CUE_MAX_S       = 90     # HSV S  — relaxed from 55; allows slight shadow/compression
CUE_MIN_DELTA_L = 25     # cue L* must exceed surrounding cloth by this much
CUE_MIN_DELTA_S = 20     # cue must be less saturated than cloth by this much

TARGET_STEMS = [
    "pool_real_002", "pool_real_003", "pool_real_007", "pool_real_010",
    "pool_real_023", "pool_real_024", "pool_real_027", "original", "images",
]
SKIP_SUFFIXES = {"new_uploads_contact_sheet", "search_contact_sheet", "thumb"}


# ── Data structures ─────────────────────────────────────────────────────────
@dataclass
class BallCandidate:
    cx: float
    cy: float
    r:  float
    # Filter flags (sequential — each applied after previous)
    pass_radius:   bool = True
    pass_position: bool = True
    pass_cloth:    bool = True
    pass_edge:     bool = True   # local edge contrast — rejects cushion/texture noise
    pass_sat:      bool = True   # max saturation — rejects pockets/markers
    pass_dedup:    bool = True
    # Cue scoring
    mean_L:        float = 0.0
    mean_S:        float = 0.0
    cloth_L:       float = 0.0
    cloth_S:       float = 0.0
    cue_score:     float = 0.0
    label:         str   = "object_ball"

    def passes_all(self):
        return (self.pass_radius and self.pass_position and self.pass_cloth
                and self.pass_edge and self.pass_sat and self.pass_dedup)

    def reject_reason(self):
        reasons = []
        if not self.pass_radius:   reasons.append("radius")
        if not self.pass_position: reasons.append("pos")
        if not self.pass_cloth:    reasons.append("cloth")
        if not self.pass_edge:     reasons.append("edge")
        if not self.pass_sat:      reasons.append("sat")
        if not self.pass_dedup:    reasons.append("dup")
        return reasons or ["pass"]


# ── Warp helpers ────────────────────────────────────────────────────────────
def get_pose_corners(model, img):
    results = model(img, verbose=False)
    if not results or results[0].keypoints is None:
        return None, None
    kps = results[0].keypoints
    if len(kps) == 0:
        return None, None
    best = int(kps.conf.mean(dim=1).argmax()) if kps.conf is not None else 0
    conf = kps.conf[best].cpu().numpy() if kps.conf is not None else None
    xy   = kps.xy[best].cpu().numpy()
    return xy, conf


def warp_image(img, corners):
    dst = np.array([[0,0],[WARP_W,0],[WARP_W,WARP_H],[0,WARP_H]], dtype=np.float32)
    M   = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (WARP_W, WARP_H)), M


# ── Cloth detection ─────────────────────────────────────────────────────────
def build_cloth_mask(warped: np.ndarray) -> np.ndarray:
    """Binary mask covering playable cloth (green / blue-green felt)."""
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    # Green felt H≈35-85, blue-green H≈85-130
    mask = cv2.inRange(hsv, (30, 35, 35), (135, 255, 255))
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    return mask


# ── Stage 1: Candidate detection ────────────────────────────────────────────
def detect_candidates(warped: np.ndarray) -> list[BallCandidate]:
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 1.5)

    # Adaptive accumulator threshold based on cloth texture
    roi = gray[int(0.15*WARP_H):int(0.85*WARP_H),
               int(0.15*WARP_W):int(0.85*WARP_W)]
    tex = float(roi.std())
    # Low texture → lower param2 (easier to detect)
    # High texture → higher param2 (stricter, fewer FP)
    param2 = max(14, min(22, int(tex * 0.9)))

    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=BALL_MIN_DIST,
        param1=55, param2=param2,
        minRadius=BALL_R_MIN - 2,   # slightly loose → filters trim later
        maxRadius=BALL_R_MAX + 2,
    )
    if circles is None:
        return []
    return [BallCandidate(cx=float(x), cy=float(y), r=float(r))
            for x, y, r in circles[0]]


# ── Stage 2: Physical filtering ─────────────────────────────────────────────
def apply_physical_filters(
    candidates: list[BallCandidate],
    cloth_mask: np.ndarray,
    warped_img: np.ndarray,
) -> list[BallCandidate]:

    # 2a. Radius range
    for c in candidates:
        if not (BALL_R_MIN <= c.r <= BALL_R_MAX):
            c.pass_radius = False

    # 2b. Playable-area position
    for c in candidates:
        if (c.cx < PLAY_MARGIN_X or c.cx > WARP_W - PLAY_MARGIN_X or
                c.cy < PLAY_MARGIN_Y or c.cy > WARP_H - PLAY_MARGIN_Y):
            c.pass_position = False

    # 2c. Cloth mask — center must land on cloth
    for c in candidates:
        xi = min(int(c.cx), WARP_W - 1)
        yi = min(int(c.cy), WARP_H - 1)
        if cloth_mask[yi, xi] == 0:
            c.pass_cloth = False

    # 2d. Edge contrast: real balls have circular edges; cushion noise is low-contrast
    gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    for c in candidates:
        if not (c.pass_radius and c.pass_position and c.pass_cloth):
            continue
        x0 = max(0, int(c.cx - c.r))
        y0 = max(0, int(c.cy - c.r))
        x1 = min(WARP_W, int(c.cx + c.r) + 1)
        y1 = min(WARP_H, int(c.cy + c.r) + 1)
        patch = gray[y0:y1, x0:x1]
        if patch.size < 4:
            c.pass_edge = False
            continue
        sx = cv2.Sobel(patch.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(patch.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        edge_mean = float(np.sqrt(sx**2 + sy**2).mean())
        if edge_mean < BALL_MIN_EDGE:
            c.pass_edge = False

    # 2e. Max saturation filter (pockets/markers are very saturated)
    hsv_img = cv2.cvtColor(warped_img, cv2.COLOR_BGR2HSV)
    for c in candidates:
        if not (c.pass_radius and c.pass_position and c.pass_cloth and c.pass_edge):
            continue
        x0 = max(0, int(c.cx - c.r * 0.8))
        y0 = max(0, int(c.cy - c.r * 0.8))
        x1 = min(WARP_W, int(c.cx + c.r * 0.8) + 1)
        y1 = min(WARP_H, int(c.cy + c.r * 0.8) + 1)
        sat_patch = hsv_img[y0:y1, x0:x1, 1]
        if sat_patch.size > 0 and float(sat_patch.mean()) > BALL_MAX_SAT:
            c.pass_sat = False

    # 2f. Radius consistency: within ±40% of median among survivors so far
    survivors = [c for c in candidates
                 if c.pass_radius and c.pass_position and c.pass_cloth
                 and c.pass_edge and c.pass_sat]
    if survivors:
        med_r = float(np.median([c.r for c in survivors]))
        lo, hi = 0.60 * med_r, 1.40 * med_r
        for c in survivors:
            if not (lo <= c.r <= hi):
                c.pass_radius = False

    # 2g. Dedup: if two survivors overlap (dist < sum_of_radii * 0.7), keep larger r
    alive = [c for c in candidates if c.passes_all()]
    used  = set()
    for i, a in enumerate(alive):
        if i in used:
            continue
        for j, b in enumerate(alive):
            if j <= i or j in used:
                continue
            dist = np.hypot(a.cx - b.cx, a.cy - b.cy)
            if dist < (a.r + b.r) * 0.7:
                # suppress smaller
                (alive[j] if a.r >= b.r else alive[i]).pass_dedup = False
                used.add(j if a.r >= b.r else i)

    # 2f. Cap at MAX_BALLS (keep strongest by accumulator → keep those with larger r)
    final_pass = sorted([c for c in candidates if c.passes_all()],
                        key=lambda c: -c.r)
    if len(final_pass) > MAX_BALLS:
        for c in final_pass[MAX_BALLS:]:
            c.pass_dedup = False

    return candidates


# ── Stage 3: Cue-ball scoring ────────────────────────────────────────────────
def _circular_mean(img_channel: np.ndarray, cx: float, cy: float, r: float,
                   scale: float = 1.0) -> float:
    """Mean value of img_channel inside circle at (cx,cy) with radius r*scale."""
    h, w = img_channel.shape
    x0 = max(0, int(cx - r * scale))
    y0 = max(0, int(cy - r * scale))
    x1 = min(w, int(cx + r * scale) + 1)
    y1 = min(h, int(cy + r * scale) + 1)
    patch = img_channel[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    # Mask to circle
    ys, xs = np.mgrid[y0:y1, x0:x1]
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    valid = patch[dist <= r * scale]
    return float(valid.mean()) if valid.size > 0 else 0.0


def score_cue_ball(warped: np.ndarray, candidates: list[BallCandidate]) -> None:
    lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    L_ch = lab[:, :, 0].astype(np.float32)
    S_ch = hsv[:, :, 1].astype(np.float32)

    for c in candidates:
        if not c.passes_all():
            continue
        # Ball interior (scale=0.85 to avoid edge pixels)
        ball_L = _circular_mean(L_ch, c.cx, c.cy, c.r, scale=0.85)
        ball_S = _circular_mean(S_ch, c.cx, c.cy, c.r, scale=0.85)

        # Surrounding cloth ring (1.5r … 2.5r)
        ring_L_outer = _circular_mean(L_ch, c.cx, c.cy, c.r, scale=2.5)
        ring_L_inner = _circular_mean(L_ch, c.cx, c.cy, c.r, scale=1.5)
        ring_S_outer = _circular_mean(S_ch, c.cx, c.cy, c.r, scale=2.5)
        ring_S_inner = _circular_mean(S_ch, c.cx, c.cy, c.r, scale=1.5)
        # Cloth estimate: weighted average of ring (exclude ball area)
        cloth_L = (ring_L_outer * 2.5 ** 2 - ring_L_inner * 1.5 ** 2) / (2.5 ** 2 - 1.5 ** 2)
        cloth_S = (ring_S_outer * 2.5 ** 2 - ring_S_inner * 1.5 ** 2) / (2.5 ** 2 - 1.5 ** 2)

        c.mean_L   = round(ball_L, 1)
        c.mean_S   = round(ball_S, 1)
        c.cloth_L  = round(cloth_L, 1)
        c.cloth_S  = round(cloth_S, 1)

        delta_L = ball_L - cloth_L
        delta_S = ball_S - cloth_S   # negative = less saturated than cloth

        # Cue score: weighted combination
        score = 0.0
        if ball_L  >= CUE_MIN_L:          score += 0.35
        if ball_S  <= CUE_MAX_S:          score += 0.25
        if delta_L >= CUE_MIN_DELTA_L:    score += 0.25
        if delta_S <= -CUE_MIN_DELTA_S:   score += 0.15

        # Penalty: if extremely saturated (cloth glare or colored ball artifact)
        if ball_S > 120:
            score -= 0.30
        elif ball_S > 100:
            score -= 0.10

        c.cue_score = round(max(0.0, min(1.0, score)), 3)

    # Assign label: only the top cue_score candidate gets cue_ball IF score ≥ 0.5
    alive = [c for c in candidates if c.passes_all()]
    if alive:
        best_cue = max(alive, key=lambda c: c.cue_score)
        if best_cue.cue_score >= 0.50:
            best_cue.label = "cue_ball"


# ── Drawing helpers ──────────────────────────────────────────────────────────
_CUE_COLOR = (255, 255, 255)     # white
_OBJ_COLOR = (0, 165, 255)       # orange
_REJ_COLOR = (60, 60, 60)        # dark gray

def draw_candidates(warped, candidates, show_rejected=True, stage_label=""):
    canvas = warped.copy()
    for i, c in enumerate(candidates):
        if c.passes_all():
            color = _CUE_COLOR if c.label == "cue_ball" else _OBJ_COLOR
            thick = 2
        elif show_rejected:
            color = _REJ_COLOR
            thick = 1
        else:
            continue
        cx, cy, r = int(c.cx), int(c.cy), int(c.r)
        cv2.circle(canvas, (cx, cy), r, color, thick)
        cv2.circle(canvas, (cx, cy), 2, color, -1)
        if c.passes_all():
            lbl = "C" if c.label == "cue_ball" else str(i + 1)
            cv2.putText(canvas, lbl, (cx - 5, cy + 5), FONT, 0.4, color, 1)
        else:
            reason = ",".join(c.reject_reason()[:1])
            cv2.putText(canvas, reason[0], (cx - 4, cy + 4), FONT, 0.35, color, 1)
    n_pass = sum(1 for c in candidates if c.passes_all())
    cv2.putText(canvas, f"{stage_label} final={n_pass}",
                (8, 24), FONT, 0.65, (220, 220, 220), 2)
    return canvas


def draw_raw(warped, candidates):
    canvas = warped.copy()
    for c in candidates:
        cv2.circle(canvas, (int(c.cx), int(c.cy)), int(c.r), (100, 255, 100), 1)
        cv2.circle(canvas, (int(c.cx), int(c.cy)), 2, (100, 255, 100), -1)
    cv2.putText(canvas, f"raw={len(candidates)}", (8, 24), FONT, 0.65, (100, 255, 100), 2)
    return canvas


# ── Main pipeline ────────────────────────────────────────────────────────────
def load_gt(stem):
    jf = GT_DIR / (stem + ".json")
    if not jf.exists():
        return None
    with open(jf) as f:
        d = json.load(f)
    c = d["corners"]
    return np.array([c["TL"], c["TR"], c["BR"], c["BL"]], dtype=np.float32)


def process(model, img_path: Path, gt_corners=None):
    img = cv2.imread(str(img_path))
    if img is None:
        return {"image": img_path.name, "status": "read_error"}

    # ── Warp ──────────────────────────────────────────────────────────────
    pred_corners, confs = get_pose_corners(model, img)
    if pred_corners is None:
        return {"image": img_path.name, "status": "no_pose"}

    # Flag low-resolution sources (upsampling to 900×450 may create artifacts)
    src_h, src_w = img.shape[:2]
    src_diag = (src_w ** 2 + src_h ** 2) ** 0.5
    low_res = src_diag < 600   # ~480×360 equivalent

    warped, _ = warp_image(img, pred_corners)
    avg_conf  = float(np.mean(confs)) if confs is not None else None

    corner_error = None
    if gt_corners is not None:
        errs = np.linalg.norm(pred_corners - gt_corners, axis=1)
        corner_error = {
            "mean_px":    round(float(errs.mean()), 2),
            "max_px":     round(float(errs.max()), 2),
            "TL": round(float(errs[0]),2), "TR": round(float(errs[1]),2),
            "BR": round(float(errs[2]),2), "BL": round(float(errs[3]),2),
        }

    # ── Stage 1: candidates ────────────────────────────────────────────────
    candidates = detect_candidates(warped)

    # ── Stage 2: physical filters ──────────────────────────────────────────
    cmask = build_cloth_mask(warped)
    apply_physical_filters(candidates, cmask, warped)

    # ── Stage 3: cue scoring ───────────────────────────────────────────────
    score_cue_ball(warped, candidates)

    finals = [c for c in candidates if c.passes_all()]

    # ── Output dir ────────────────────────────────────────────────────────
    stem    = img_path.stem
    img_dir = OUT_DIR / stem
    img_dir.mkdir(exist_ok=True)

    # Warp
    cv2.imwrite(str(img_dir / f"{stem}_warp.jpg"), warped)
    # Stage 1 — raw
    cv2.imwrite(str(img_dir / f"{stem}_stage1_raw.jpg"),
                draw_raw(warped, candidates))
    # Stage 2 — filtered (show rejected too)
    cv2.imwrite(str(img_dir / f"{stem}_stage2_filtered.jpg"),
                draw_candidates(warped, candidates, show_rejected=True, stage_label="filtered"))
    # Stage 3 — final classified
    cv2.imwrite(str(img_dir / f"{stem}_stage3_final.jpg"),
                draw_candidates(warped, finals, show_rejected=False, stage_label="final"))

    # JSON result
    result = {
        "image": img_path.name,
        "status": "ok",
        "corner_confidence": round(avg_conf, 3) if avg_conf else None,
        "corner_error": corner_error,
        "source_resolution": [src_w, src_h],
        "low_res_source": low_res,
        "n_raw_candidates": len(candidates),
        "n_final": len(finals),
        "cue_detected": any(c.label == "cue_ball" for c in finals),
        "balls": [
            {
                "id": i + 1,
                "label": c.label,
                "cx": round(c.cx, 1),
                "cy": round(c.cy, 1),
                "r":  round(c.r, 1),
                "cue_score": c.cue_score,
                "mean_L": c.mean_L,
                "mean_S": c.mean_S,
                "cloth_L": c.cloth_L,
                "cloth_S": c.cloth_S,
            }
            for i, c in enumerate(finals)
        ],
        "rejected": [
            {"cx": round(c.cx,1), "cy": round(c.cy,1), "r": round(c.r,1),
             "reasons": c.reject_reason()}
            for c in candidates if not c.passes_all()
        ],
        "filter_counts": {
            "raw":           len(candidates),
            "after_radius":  sum(1 for c in candidates if c.pass_radius),
            "after_pos":     sum(1 for c in candidates if c.pass_radius and c.pass_position),
            "after_cloth":   sum(1 for c in candidates if c.pass_radius and c.pass_position and c.pass_cloth),
            "after_edge":    sum(1 for c in candidates if c.pass_radius and c.pass_position and c.pass_cloth and c.pass_edge),
            "after_sat":     sum(1 for c in candidates if c.pass_radius and c.pass_position and c.pass_cloth and c.pass_edge and c.pass_sat),
            "after_dedup":   sum(1 for c in candidates if c.passes_all()),
            "final":         len(finals),
        },
    }
    with open(img_dir / f"{stem}_result.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


# ── Report printer ───────────────────────────────────────────────────────────
def print_report(results):
    ok = [r for r in results if r.get("status") == "ok"]

    print("\n" + "=" * 78)
    print("BALL DETECTION v2 — Staged Pipeline Report")
    print("=" * 78)
    hdr = f"{'image':22s} {'raw':>4} {'final':>5} {'cue':>4} {'c_mean':>8} {'c_max':>8}  verdict"
    print(hdr)
    print("-" * 78)

    verdicts = {}
    for r in results:
        if r.get("status") != "ok":
            print(f"{r['image']:22s}  —  {r['status']}")
            continue
        ce    = r.get("corner_error") or {}
        c_m   = f"{ce['mean_px']:.1f}px" if ce else "N/A"
        c_x   = f"{ce['max_px']:.1f}px"  if ce else "N/A"
        cue   = "YES" if r["cue_detected"] else "no"
        n     = r["n_final"]
        raw   = r["n_raw_candidates"]
        cmean = ce.get("mean_px", 999)

        lr    = r.get("low_res_source", False)
        if cmean > 18:             v = "BORDERLINE-corner"
        elif n == 0 and lr:        v = "LOW-RES-FAIL"
        elif n == 0:               v = "NO-BALLS"
        elif n > 12 and lr:        v = "LOW-RES-OVERDET"
        elif n > 12:               v = "OVERDETECT"
        else:                      v = "OK"
        verdicts[r["image"]] = v
        print(f"{r['image']:22s} {raw:>4} {n:>5} {cue:>4} {c_m:>8} {c_x:>8}  {v}")

    print("-" * 78)
    avg_n = np.mean([r["n_final"] for r in ok]) if ok else 0
    n_cue = sum(1 for r in ok if r["cue_detected"])
    print(f"Avg final balls: {avg_n:.1f}  |  Cue detected: {n_cue}/{len(ok)}")

    # Per-image filter funnel (sequential / cumulative)
    print("\n── Filter funnel (cumulative) ─────────────────────────────────────────")
    print(f"{'image':22s}  raw →r →pos →cloth →edge →sat →dedup →final")
    for r in ok:
        fc = r["filter_counts"]
        print(f"  {r['image']:20s}  "
              f"{fc['raw']:3}→{fc['after_radius']:2}→{fc['after_pos']:3}"
              f"→{fc['after_cloth']:5}→{fc['after_edge']:4}→{fc['after_sat']:3}"
              f"→{fc['after_dedup']:5}→{fc['final']:5}")

    # Cue scoring detail
    print("\n── Cue-ball scoring ───────────────────────────────────────────────────")
    for r in ok:
        balls = r.get("balls", [])
        if not balls:
            continue
        best = max(balls, key=lambda b: b["cue_score"])
        cue_lbl = "✓ CUE" if r["cue_detected"] else "  obj"
        print(f"  {r['image']:22s}  best_cue_score={best['cue_score']:.3f}  "
              f"L={best['mean_L']:.0f}/cloth{best['cloth_L']:.0f}  "
              f"S={best['mean_S']:.0f}/cloth{best['cloth_S']:.0f}  {cue_lbl}")

    # Summary verdict
    print("\n── Summary verdict ────────────────────────────────────────────────────")
    for img, v in verdicts.items():
        mark = "✓" if v == "OK" else ("~" if "BORDERLINE" in v else "✗")
        print(f"  {mark} {img:22s}  {v}")

    # Readiness for YOLO ball detector
    n_ok = sum(1 for v in verdicts.values() if v == "OK")
    print(f"\n  {n_ok}/{len(verdicts)} images PASS  → "
          f"{'Ready to benchmark YOLO ball detector' if n_ok >= 6 else 'Need more improvement first'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None)
    parser.add_argument("--all",   action="store_true")
    args = parser.parse_args()

    from ultralytics import YOLO
    model = YOLO(str(CHECKPOINT))

    if args.image:
        imgs = [Path(args.image)]
    elif args.all:
        exts = {".jpg", ".jpeg", ".png"}
        imgs = sorted(p for p in PICTURE_DIR.iterdir()
                      if p.suffix.lower() in exts
                      and p.stem not in SKIP_SUFFIXES
                      and not any(p.stem.startswith(x) for x in ("aug_","tgt_")))
    else:
        seen, imgs = set(), []
        for stem in TARGET_STEMS:
            for ext in [".jpg", ".jpeg", ".png"]:
                p = PICTURE_DIR / (stem + ext)
                if p.exists() and stem not in seen:
                    imgs.append(p); seen.add(stem); break

    print(f"Processing {len(imgs)} image(s)  [checkpoint: table_corners_mvp_v1.pt]")
    results = []
    for img_path in imgs:
        print(f"  {img_path.name} ...", end=" ", flush=True)
        gt = load_gt(img_path.stem)
        r  = process(model, img_path, gt)
        results.append(r)
        if r.get("status") == "ok":
            ce = (r.get("corner_error") or {}).get("mean_px")
            print(f"raw={r['n_raw_candidates']}  final={r['n_final']}  "
                  f"cue={'YES' if r['cue_detected'] else 'no'}  "
                  f"corner={ce:.1f}px" if ce else "")
        else:
            print(r.get("status"))

    print_report(results)

    with open(OUT_DIR / "eval_summary.json", "w") as f:
        json.dump({"n_images": len(imgs), "results": results}, f, indent=2)
    print(f"\nOutputs  → {OUT_DIR}")
    print(f"Summary  → {OUT_DIR / 'eval_summary.json'}")


if __name__ == "__main__":
    main()
