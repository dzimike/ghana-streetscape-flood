"""Master inference runner — Task 6.

Runs CLIP and SegFormer sequentially over all images, then merges
predictions into a single per-image feature table.

Output:
  data/processed/image_features.parquet  — merged CLIP + segmentation
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
from loguru import logger


def run(sample_only: bool = False) -> None:
    from src.models.run_clip import run as run_clip
    from src.models.run_segmentation import run as run_seg

    logger.info("=== Step 1/2: CLIP zero-shot classification ===")
    run_clip(sample_only=sample_only)

    logger.info("=== Step 2/2: SegFormer segmentation ===")
    run_seg(sample_only=sample_only)

    logger.info("=== Merging predictions ===")
    clip = pd.read_parquet(ROOT / "data/processed/clip_predictions.parquet")
    seg  = pd.read_parquet(ROOT / "data/processed/segmentation_fractions.parquet")

    merged = clip.merge(seg, on=["pano_id", "heading", "point_id"], how="outer")

    # Add manifest metadata back
    manifest = pd.read_parquet(ROOT / "data/interim/image_manifest.parquet")
    meta_cols = ["pano_id", "heading", "point_id", "capture_date",
                 "pano_lat", "pano_lon", "highway", "flood_stratum", "local_path"]
    merged = merged.merge(
        manifest[meta_cols], on=["pano_id", "heading", "point_id"], how="left"
    )

    out = ROOT / "data/processed/image_features.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)

    print("\n── Image Feature Table ─────────────────────────────────")
    print(f"  Rows:    {len(merged):,}")
    print(f"  Columns: {len(merged.columns)}  "
          f"({len([c for c in merged.columns if c.startswith('clip_')])} CLIP + "
          f"{len([c for c in merged.columns if c.startswith('seg_')])} seg + "
          f"{len([c for c in merged.columns if not c.startswith(('clip_','seg_'))])} meta)")
    print(f"  Saved:   {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sample", action="store_true",
                   help="Run on a small sample for testing")
    args = p.parse_args()
    run(sample_only=args.sample)
