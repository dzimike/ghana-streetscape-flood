"""SegFormer semantic segmentation for surface-type fractions — Task 6.

Runs SegFormer-B2 (pretrained on Cityscapes 19 classes) over every image
and maps predictions to 8 flood-relevant surface types. Outputs pixel
fraction per class per image.

Output:
  data/processed/segmentation_fractions.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from loguru import logger

from src.config.settings import load_config

cfg = load_config("accra_pilot")

MANIFEST_IN = ROOT / "data/interim/image_manifest.parquet"
OUT_PATH    = ROOT / "data/processed/segmentation_fractions.parquet"
BATCH_SIZE  = 8
MODEL_NAME  = "nvidia/segformer-b0-finetuned-ade-512-512"

# ADE20K 150-class → flood-relevant surface groups
# Only listing classes relevant to outdoor/street scenes
ADE20K_TO_FLOOD: dict[int, str] = {
    2:   "sky",          # sky
    4:   "vegetation",   # tree
    6:   "road",         # road, route
    9:   "vegetation",   # grass
    11:  "sidewalk",     # sidewalk, pavement
    13:  "bare_ground",  # earth, ground
    17:  "vegetation",   # plant, flora
    21:  "water",        # water
    26:  "water",        # sea
    29:  "bare_ground",  # field
    34:  "bare_ground",  # rock, stone
    46:  "bare_ground",  # sand
    52:  "road",         # path
    54:  "road",         # runway
    60:  "water",        # river
    72:  "vegetation",   # palm tree
    79:  "building",     # hovel, hut, shack (informal)
    88:  "kiosk",        # booth, kiosk, stall
    91:  "road",         # dirt track
    94:  "bare_ground",  # land, ground, soil
    128: "water",        # lake
    138: "waste",        # ashcan, trash can, garbage can
}
# All other ADE20K classes → building or other
CITYSCAPES_TO_FLOOD = ADE20K_TO_FLOOD  # alias used below

FLOOD_CLASSES = ["road", "sidewalk", "building", "vegetation",
                 "bare_ground", "sky", "other", "water", "drain", "waste", "kiosk"]


def run(sample_only: bool = False) -> None:
    device = "cpu"  # SegFormer-B0 on CPU; MPS runs out of memory on Intel Mac
    logger.info(f"Device: {device}")

    logger.info(f"Loading {MODEL_NAME} …")
    processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
    model     = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    manifest = pd.read_parquet(MANIFEST_IN)
    manifest = manifest[manifest["downloaded"].fillna(False)].copy()
    if sample_only:
        manifest = manifest.sample(100, random_state=42)
    logger.info(f"Running segmentation on {len(manifest):,} images")

    rows   = []
    paths  = manifest["local_path"].tolist()
    ids    = manifest["pano_id"].tolist()
    heads  = manifest["heading"].tolist()
    pts    = manifest["point_id"].tolist()

    for batch_start in tqdm(range(0, len(paths), BATCH_SIZE), desc="Segmentation"):
        batch_paths = paths[batch_start: batch_start + BATCH_SIZE]
        batch_ids   = ids[batch_start: batch_start + BATCH_SIZE]
        batch_heads = heads[batch_start: batch_start + BATCH_SIZE]
        batch_pts   = pts[batch_start: batch_start + BATCH_SIZE]

        images = []
        valid  = []
        for i, p in enumerate(batch_paths):
            try:
                images.append(Image.open(p).convert("RGB"))
                valid.append(i)
            except Exception:
                pass

        if not images:
            continue

        with torch.no_grad():
            inputs  = processor(images=images, return_tensors="pt").to(device)
            outputs = model(**inputs)
            # Upsample logits to original image size
            logits  = outputs.logits                     # (B, 19, H/4, W/4)
            upsampled = torch.nn.functional.interpolate(
                logits, size=(inputs["pixel_values"].shape[-2],
                              inputs["pixel_values"].shape[-1]),
                mode="bilinear", align_corners=False,
            )
            preds = upsampled.argmax(dim=1).cpu().numpy()  # (B, H, W)

        for local_i, global_i in enumerate(valid):
            pred  = preds[local_i]
            total = pred.size

            # Map ADE20K classes → flood groups
            fracs: dict[str, float] = {c: 0.0 for c in FLOOD_CLASSES}
            n_classes = int(pred.max()) + 1
            for cs_id in range(n_classes):
                flood_cls = ADE20K_TO_FLOOD.get(cs_id, "other")
                fracs[flood_cls] += float((pred == cs_id).sum() / total)

            row = {
                "pano_id":  batch_ids[global_i],
                "heading":  batch_heads[global_i],
                "point_id": batch_pts[global_i],
            }
            for cls in FLOOD_CLASSES:
                row[f"seg_{cls}"] = round(fracs[cls], 4)
            rows.append(row)

    df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    logger.info(f"Segmentation fractions → {OUT_PATH}  ({len(df):,} rows)")

    print("\n── Segmentation Summary ────────────────────────────────")
    print(f"  Images processed: {len(df):,}")
    seg_cols = [c for c in df.columns if c.startswith("seg_")]
    means = df[seg_cols].mean().sort_values(ascending=False)
    print("  Mean pixel fraction per class:")
    for col, v in means.items():
        print(f"    {col.replace('seg_',''):<20} {v:.3f}  ({v*100:.1f}%)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sample", action="store_true", help="Run on 100 images only (test mode)")
    args = p.parse_args()
    run(sample_only=args.sample)
