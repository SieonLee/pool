"""
Validate a single image as a training candidate.

Runs corner detection, warp, ball detection, and produces a candidate report.
Does NOT write to the training dataset.

Usage:
  python scripts/validate_candidate_image.py --image new_picture/maxresdefault.jpg
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

BASE        = Path(__file__).parent.parent
CORNER_CKPT = BASE / "models" / "checkpoints" / "table_corners_mvp_v1.pt"
BALL_CKPT   = BASE / "models" / "checkpoints" / "ball_yolo_v7_below_baseline.pt"
OUT_DIR     = BASE / "review" / "candidate_validation"

WARP_W, WARP_H = 900, 450
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Class-specific thresholds (match stress_test.py defaults)
CUE_CONF  = 0.25
OBJ_CONF  = 0.35


def get_corners(model, img):
    results = model(img, verbose=False)
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


def draw_corners(img, corners, conf):
    out = img.copy()
    labels = ["TL", "TR", "BR", "BL"]
    colors = [(0,255,0),(0,200,255),(255,100,0),(255,0,200)]
    for i, (x, y) in enumerate(corners):
        xi, yi = int(x), int(y)
        cv2.circle(out, (xi, yi), 12, colors[i], 3)
        cv2.putText(out, labels[i], (xi+14, yi+6), FONT, 0.7, colors[i], 2)
    # Draw quad
    pts = corners.astype(int)
    for i in range(4):
        cv2.line(out, tuple(pts[i]), tuple(pts[(i+1)%4]), (0,255,0), 2)
    c_str = f"{conf:.3f}" if conf is not None else "?"
    cv2.putText(out, f"corner_conf={c_str}", (10, 30), FONT, 0.7, (0,255,255), 2)
    return out


def warp(img, corners):
    dst = np.float32([[0,0],[WARP_W,0],[WARP_W,WARP_H],[0,WARP_H]])
    M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (WARP_W, WARP_H))


def run_ball_detection(model, warped):
    results = model(warped, verbose=False, conf=CUE_CONF)
    raw = []
    if results and results[0].boxes is not None:
        for i in range(len(results[0].boxes)):
            b = results[0].boxes
            cls  = int(b.cls[i].item())
            cf   = float(b.conf[i].item())
            xyxy = b.xyxy[i].cpu().numpy().tolist()
            raw.append((cls, cf, *xyxy))
    # Apply class-specific filtering
    cue_dets = [(c,f,*xy) for (c,f,*xy) in raw if c == 0 and f >= CUE_CONF]
    obj_dets = [(c,f,*xy) for (c,f,*xy) in raw if c == 1 and f >= OBJ_CONF]
    return cue_dets, obj_dets, raw


def draw_detections(warped, cue_dets, obj_dets):
    out = warped.copy()
    for cls, cf, x1, y1, x2, y2 in cue_dets:
        cv2.rectangle(out, (int(x1),int(y1)), (int(x2),int(y2)), (255,255,0), 2)
        cv2.putText(out, f"CUE {cf:.2f}", (int(x1),int(y1)-5), FONT, 0.45, (255,255,0), 1)
    for cls, cf, x1, y1, x2, y2 in obj_dets:
        cv2.rectangle(out, (int(x1),int(y1)), (int(x2),int(y2)), (255,120,0), 2)
        cv2.putText(out, f"OBJ {cf:.2f}", (int(x1),int(y1)-5), FONT, 0.40, (255,120,0), 1)

    n_cue = len(cue_dets)
    n_obj = len(obj_dets)
    label = f"cue={n_cue}  obj={n_obj}  total={n_cue+n_obj}"
    cv2.putText(out, label, (10, 20), FONT, 0.60, (200,255,200), 2)
    return out


def analyze_text_overlay(warped):
    """
    Estimate if Korean subtitle text is inside the playfield.
    Text tends to appear as large bright/white blobs near the bottom of the image.
    Returns (text_detected_in_lower_third, bbox_approx).
    """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # Threshold for very bright regions (text is usually white/yellow on dark bg)
    _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    # Focus on lower third of the warped image
    lower = thresh[int(WARP_H*0.65):, :]
    upper = thresh[:int(WARP_H*0.65), :]
    lower_density = float(lower.sum()) / (lower.size * 255)
    upper_density = float(upper.sum()) / (upper.size * 255)
    # High density of bright pixels in lower third suggests text overlay
    text_likely = lower_density > 0.03 and lower_density > upper_density * 3
    return text_likely, lower_density, upper_density


def warp_quality_check(warped, corners, src_shape):
    """Check for common warp artifacts."""
    issues = []
    h, w = src_shape[:2]
    # Corner proximity to image edges
    for i, (x, y) in enumerate(corners):
        if x < 20 or x > w-20 or y < 20 or y > h-20:
            issues.append(f"corner_{i}_near_edge")
    # Warp area check: ensure the warped image has meaningful content
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    mean_bright = float(gray.mean())
    if mean_bright < 50:
        issues.append("warp_too_dark")
    if mean_bright > 230:
        issues.append("warp_too_bright")
    # Check for large black regions (failed warp artifacts)
    black_ratio = float((gray < 15).sum()) / gray.size
    if black_ratio > 0.15:
        issues.append(f"warp_black_region({black_ratio:.2f})")
    return issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True,
                        help="Path to candidate image (relative to project root or absolute)")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.is_absolute():
        img_path = BASE / img_path
    if not img_path.exists():
        print(f"ERROR: image not found: {img_path}")
        sys.exit(1)

    stem = img_path.stem
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading models...")
    corner_model = YOLO(str(CORNER_CKPT))
    ball_model   = YOLO(str(BALL_CKPT))
    print(f"  Corner: {CORNER_CKPT.name}")
    print(f"  Balls:  {BALL_CKPT.name}")

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"ERROR: cannot read image: {img_path}")
        sys.exit(1)
    src_h, src_w = img.shape[:2]
    print(f"\nSource: {img_path.name}  {src_w}×{src_h}")

    # 1. Corner detection
    print("\n[1/4] Corner detection...")
    corners, corner_conf = get_corners(corner_model, img)
    if corners is None:
        print("  FAIL: no table corners detected")
        corner_img = img.copy()
        cv2.putText(corner_img, "CORNER DETECTION FAILED", (30, 60), FONT, 1.2, (0,0,255), 3)
        cv2.imwrite(str(OUT_DIR / f"{stem}_corners.jpg"), corner_img)
        verdict = "SKIP — corner detection failed"
        print(f"\n{verdict}")
        sys.exit(0)

    print(f"  corner_conf={corner_conf:.3f}")
    corner_img = draw_corners(img, corners, corner_conf)
    cv2.imwrite(str(OUT_DIR / f"{stem}_corners.jpg"), corner_img)
    print(f"  Saved: {OUT_DIR.name}/{stem}_corners.jpg")

    # 2. Warp
    print("\n[2/4] Warping to 900×450...")
    warped = warp(img, corners)
    warp_issues = warp_quality_check(warped, corners, img.shape)
    cv2.imwrite(str(OUT_DIR / f"{stem}_warped.jpg"), warped)
    print(f"  Saved: {OUT_DIR.name}/{stem}_warped.jpg")
    if warp_issues:
        print(f"  Warp issues: {warp_issues}")
    else:
        print("  Warp looks clean (no artifacts detected)")

    # 3. Text overlay analysis
    print("\n[3/4] Text overlay analysis...")
    text_in_lower, lower_dens, upper_dens = analyze_text_overlay(warped)
    print(f"  Lower-third bright density: {lower_dens:.4f}")
    print(f"  Upper bright density:       {upper_dens:.4f}")
    if text_in_lower:
        print("  WARNING: text overlay likely in lower playfield area")
    else:
        print("  Text overlay appears minimal or outside playfield")

    # 4. Ball detection
    print("\n[4/4] Ball detection (cue≥0.25, obj≥0.35)...")
    cue_dets, obj_dets, raw_dets = run_ball_detection(ball_model, warped)
    print(f"  Raw detections (conf≥0.25): {len(raw_dets)}")
    print(f"  After class-specific filter: cue={len(cue_dets)}, obj={len(obj_dets)}, total={len(cue_dets)+len(obj_dets)}")
    for d in cue_dets:
        print(f"    CUE  conf={d[1]:.3f}  cx={int((d[2]+d[4])/2)}  cy={int((d[3]+d[5])/2)}")
    for d in obj_dets:
        print(f"    OBJ  conf={d[1]:.3f}  cx={int((d[2]+d[4])/2)}  cy={int((d[3]+d[5])/2)}")

    det_img = draw_detections(warped, cue_dets, obj_dets)
    cv2.imwrite(str(OUT_DIR / f"{stem}_detections.jpg"), det_img)
    print(f"  Saved: {OUT_DIR.name}/{stem}_detections.jpg")

    # Verdict
    print("\n" + "="*60)
    print("  CANDIDATE EVALUATION SUMMARY")
    print("="*60)

    flags = []
    if corner_conf is not None and corner_conf < 0.6:
        flags.append(f"LOW_CORNER_CONF ({corner_conf:.3f})")
    if warp_issues:
        flags.append(f"WARP_ISSUES: {warp_issues}")
    if text_in_lower:
        flags.append("TEXT_IN_PLAYFIELD")
    n_balls = len(cue_dets) + len(obj_dets)
    if n_balls < 3:
        flags.append(f"TOO_FEW_BALLS ({n_balls})")
    if len(cue_dets) == 0:
        flags.append("NO_CUE_DETECTED (may be present but undetected — inspect warped)")

    report = {
        "image": img_path.name,
        "source_size": [src_w, src_h],
        "corner_conf": round(corner_conf, 3) if corner_conf else None,
        "warp_issues": warp_issues,
        "text_in_lower_third": text_in_lower,
        "lower_bright_density": round(lower_dens, 4),
        "ball_detections": {
            "raw": len(raw_dets),
            "cue": len(cue_dets),
            "obj": len(obj_dets),
            "total_filtered": n_balls,
        },
        "flags": flags,
    }

    print(f"  Source:         {src_w}×{src_h}")
    print(f"  Corner conf:    {corner_conf:.3f}" if corner_conf else "  Corner conf:    N/A")
    print(f"  Warp issues:    {warp_issues or 'none'}")
    print(f"  Text in field:  {'YES — WARNING' if text_in_lower else 'no'}")
    print(f"  Detected balls: cue={len(cue_dets)}, obj={len(obj_dets)}, total={n_balls}")
    print(f"  Flags:          {flags or 'none'}")
    print()

    if not flags:
        print("  VERDICT: LABEL — warp clean, balls clear, no blocking issues")
        report["verdict"] = "LABEL"
    elif any("WARP" in f or "CORNER" in f for f in flags):
        print("  VERDICT: SKIP — warp or corner quality insufficient")
        report["verdict"] = "SKIP"
    else:
        print("  VERDICT: REVIEW — inspect warped image before deciding")
        report["verdict"] = "REVIEW"

    print("="*60)

    report_path = OUT_DIR / f"{stem}_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
