"""Object detection features using YOLOv8n (COCO pre-trained, no fine-tuning).

Runs YOLOv8 nano over all street-level images to count detected objects by COCO class.
No local training — weights downloaded from Ultralytics once and cached.

Parallels the 32 object-detection variables in Zhou et al. (2026, Cities).
Here we export all 80 COCO classes; run_fusion_clean.py selects the 32 most
variable at training time so the feature set is reproducible.

Per-image output:
  obj_<class>   — count of detections above conf threshold for each COCO class

Per-segment output:
  mean detection count per class across the 4 headings.

Output:
  data/processed/object_detection_features.parquet   — per-image
  data/processed/object_detection_segment.parquet    — per-point (mean over headings)

Requirements:
  pip install ultralytics          (conda run -n flood pip install ultralytics)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger

MANIFEST_IN   = ROOT / "data/interim/image_manifest.parquet"
OUT_IMAGE     = ROOT / "data/processed/object_detection_features.parquet"
OUT_SEGMENT   = ROOT / "data/processed/object_detection_segment.parquet"

CONF_THRESHOLD = 0.25   # detection confidence threshold
BATCH_SIZE     = 8      # images per YOLOv8 batch


# COCO 80-class names in standard order
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]
OBJ_COLS = [f"obj_{c.replace(' ', '_')}" for c in COCO_CLASSES]


def run(sample_only: bool = False) -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error(
            "ultralytics not installed. Run:\n"
            "  conda run -n flood pip install ultralytics"
        )
        raise

    logger.info("Loading YOLOv8n (COCO pre-trained, no fine-tuning) …")
    model = YOLO("yolov8n.pt")   # downloads ~6 MB on first run, then cached
    model.conf = CONF_THRESHOLD

    manifest = pd.read_parquet(MANIFEST_IN)
    manifest = manifest[manifest["downloaded"].fillna(False)].copy()
    if sample_only:
        manifest = manifest.sample(200, random_state=42)
    logger.info(f"Images to process: {len(manifest):,}")

    paths  = manifest["local_path"].tolist()
    pids   = manifest["pano_id"].tolist()
    heads  = manifest["heading"].tolist()
    pts    = manifest["point_id"].tolist()

    rows = []
    for start in tqdm(range(0, len(paths), BATCH_SIZE), desc="YOLOv8"):
        batch_paths = paths[start: start + BATCH_SIZE]
        batch_pids  = pids[start: start + BATCH_SIZE]
        batch_heads = heads[start: start + BATCH_SIZE]
        batch_pts   = pts[start: start + BATCH_SIZE]

        # Filter paths that exist
        valid_idx = [i for i, p in enumerate(batch_paths) if Path(p).exists()]
        if not valid_idx:
            continue

        valid_paths = [batch_paths[i] for i in valid_idx]

        try:
            results = model(valid_paths, verbose=False)
        except Exception as e:
            logger.debug(f"Batch error at {start}: {e}")
            continue

        for local_i, global_i in enumerate(valid_idx):
            counts = np.zeros(80, dtype=np.int32)
            r = results[local_i]
            if r.boxes is not None and len(r.boxes):
                cls_ids = r.boxes.cls.cpu().numpy().astype(int)
                for c in cls_ids:
                    if 0 <= c < 80:
                        counts[c] += 1

            row: dict = {
                "pano_id":  batch_pids[global_i],
                "heading":  batch_heads[global_i],
                "point_id": batch_pts[global_i],
            }
            for j, col in enumerate(OBJ_COLS):
                row[col] = int(counts[j])
            rows.append(row)

    df = pd.DataFrame(rows)
    OUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_IMAGE, index=False)
    logger.info(f"Per-image object counts → {OUT_IMAGE}  ({len(df):,} rows × {df.shape[1]} cols)")

    seg = df.groupby("point_id")[OBJ_COLS].mean().reset_index()
    seg.to_parquet(OUT_SEGMENT, index=False)
    logger.info(f"Per-segment object counts → {OUT_SEGMENT}  ({len(seg):,} rows)")

    # Show top detected classes
    mean_counts = df[OBJ_COLS].mean().sort_values(ascending=False)
    nonzero = mean_counts[mean_counts > 0]
    print(f"\n── Top detected COCO classes (mean count/image, threshold={CONF_THRESHOLD}) ──")
    for col, v in nonzero.head(20).items():
        print(f"  {col.replace('obj_', ''):<25} {v:.3f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sample", action="store_true", help="Run on 200 images (test)")
    args = p.parse_args()
    run(sample_only=args.sample)
