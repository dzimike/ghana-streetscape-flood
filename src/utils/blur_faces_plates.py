"""Blur faces and licence plates in Street View panel images before journal submission.

Face detection uses YOLOv8n (ultralytics) when available, or cv2.FaceDetectorYN
(YuNet, requires a local ONNX model file at models/yunet.onnx).  Google Street View
already blurs faces by default, so residual PII risk is low.  The plate heuristic
(wide rectangular blobs in the lower-half of the image) runs unconditionally.

Usage
-----
  python src/utils/blur_faces_plates.py \
      --input  outputs/figures/panel_sources/ \
      --output outputs/figures/panel_sources_blurred/

  # Or blur images listed in a CSV manifest:
  python src/utils/blur_faces_plates.py --from-manifest data/interim/double_coding/double_coding_manifest.csv

The script writes blurred copies to --output, preserving filenames.
After blurring, re-run: python src/visualization/make_streetview_panel.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


# ── Blur kernel ───────────────────────────────────────────────────────────────

def gaussian_box(img: np.ndarray, x: int, y: int, w: int, h: int,
                 ksize: int = 51) -> np.ndarray:
    """Gaussian-blur a rectangular ROI in-place (clamps to image bounds)."""
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, img.shape[1]), min(y + h, img.shape[0])
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return img
    k = ksize if ksize % 2 == 1 else ksize + 1
    img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
    return img


# ── Detectors ─────────────────────────────────────────────────────────────────

def detect_faces_yunet(img: np.ndarray,
                       model_path: str | None = None) -> list[tuple[int, int, int, int]]:
    """YuNet DNN face detector (cv2.FaceDetectorYN).

    Requires a YuNet ONNX model.  Pass model_path or place the file at
    models/yunet.onnx.  Returns empty list if model file is missing.
    """
    if model_path is None:
        root = Path(__file__).resolve().parents[2]
        model_path = str(root / "models" / "yunet.onnx")
    if not Path(model_path).exists():
        return []
    h, w = img.shape[:2]
    detector = cv2.FaceDetectorYN.create(
        model_path, "", (w, h), score_threshold=0.6, nms_threshold=0.3, top_k=100,
    )
    _, faces = detector.detect(img)
    if faces is None:
        return []
    boxes = []
    for face in faces:
        x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
        boxes.append((x, y, fw, fh))
    return boxes


def detect_faces_yolo(img: np.ndarray) -> list[tuple[int, int, int, int]]:
    """YOLOv8n person detector — high recall.  Requires ultralytics."""
    try:
        from ultralytics import YOLO
    except ImportError:
        return []
    model = YOLO("yolov8n.pt")
    results = model(img, verbose=False, classes=[0])  # class 0 = person
    boxes = []
    for r in results:
        for box in r.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = box[:4].astype(int)
            # Restrict to upper body: crop box to top 60% (head/face region)
            fh = y2 - y1
            boxes.append((x1, y1, x2 - x1, int(fh * 0.45)))
    return boxes


def detect_plates_heuristic(img: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Licence-plate heuristic: wide rectangular blobs in the lower half of the image.
    Tuned for West-African street-view distances (plates 30–200 px wide).
    Returns (x, y, w, h) tuples.
    """
    h_img, w_img = img.shape[:2]
    lower = img[h_img // 2:, :]          # only search lower half
    grey  = cv2.cvtColor(lower, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(grey, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    plates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / max(h, 1)
        area   = w * h
        if 2.0 < aspect < 6.5 and 600 < area < 20_000:
            plates.append((x, y + h_img // 2, w, h))   # restore full-image y
    return plates


# ── Main processing ───────────────────────────────────────────────────────────

def blur_image(src: Path, dst: Path, use_yolo: bool = True,
               pad: float = 0.25) -> int:
    """Blur faces and plates in src, write to dst.  Returns n detections."""
    img = cv2.imread(str(src))
    if img is None:
        print(f"  [SKIP] Cannot read {src}")
        return 0

    detections: list[tuple[int, int, int, int]] = []

    # Faces — try YuNet DNN first, then YOLO fallback
    yunet_boxes = detect_faces_yunet(img)
    if yunet_boxes:
        detections += yunet_boxes
    elif use_yolo:
        detections += detect_faces_yolo(img)

    # Plates
    detections += detect_plates_heuristic(img)

    # De-duplicate (simple non-max suppression by overlap)
    detections = _nms(detections)

    # Expand boxes by pad fraction and blur
    ih, iw = img.shape[:2]
    for x, y, w, h in detections:
        px, py = int(w * pad), int(h * pad)
        gaussian_box(img, x - px, y - py, w + 2 * px, h + 2 * py, ksize=61)

    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), img)
    return len(detections)


def _nms(boxes: list[tuple[int, int, int, int]],
          iou_thresh: float = 0.3) -> list[tuple[int, int, int, int]]:
    """Simple greedy NMS to remove near-duplicate detection boxes."""
    if not boxes:
        return []

    def area(b):
        return b[2] * b[3]

    def iou(a, b):
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ix = max(0, min(ax1 + aw, bx1 + bw) - max(ax1, bx1))
        iy = max(0, min(ay1 + ah, by1 + bh) - max(ay1, by1))
        inter = ix * iy
        union = area(a) + area(b) - inter
        return inter / union if union > 0 else 0.0

    boxes = sorted(boxes, key=area, reverse=True)
    keep = []
    used = set()
    for i, b in enumerate(boxes):
        if i in used:
            continue
        keep.append(b)
        for j in range(i + 1, len(boxes)):
            if iou(b, boxes[j]) > iou_thresh:
                used.add(j)
    return keep


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ROOT = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description="Blur faces and licence plates in Street View panel images."
    )
    parser.add_argument("--input", type=Path,
                        default=ROOT / "outputs/figures/panel_sources",
                        help="Directory of source images (or glob pattern)")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/figures/blurred",
                        help="Directory for blurred output images")
    parser.add_argument("--no-yolo", action="store_true",
                        help="Skip YOLOv8 face detection (use YuNet only if model present)")
    parser.add_argument("--from-manifest", type=Path, default=None,
                        help="Use image_path column from a CSV manifest")
    args = parser.parse_args()

    # Collect source paths
    if args.from_manifest and args.from_manifest.exists():
        import pandas as pd
        mf = pd.read_csv(args.from_manifest)
        sources = [Path(p) for p in mf["image_path"].dropna() if Path(p).exists()]
    else:
        exts = {".jpg", ".jpeg", ".png"}
        sources = [p for p in args.input.iterdir()
                   if p.suffix.lower() in exts] if args.input.is_dir() else []

    if not sources:
        print("No source images found. Check --input or --from-manifest.")
        return

    print(f"Blurring {len(sources)} images → {args.output}")
    total_det = 0
    for src in sorted(sources):
        dst = args.output / src.name
        n = blur_image(src, dst, use_yolo=not args.no_yolo)
        total_det += n
        status = f"{n} detection(s)" if n else "no detections"
        print(f"  {src.name}  →  {dst.name}  [{status}]")

    print(f"\nDone. Total detections blurred: {total_det}")
    print(f"Inspect output in: {args.output}")
    print("Then regenerate fig4: python src/visualization/make_streetview_panel.py")


if __name__ == "__main__":
    main()
