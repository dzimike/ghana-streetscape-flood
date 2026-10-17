"""Pixel-level image features for flood vulnerability pipeline.

Extracts 9 low-level pixel statistics from each street-level image, matching
the pixel feature category (9 variables) in Zhou et al. (2026, Cities):

  mean_r, mean_g, mean_b   — RGB channel means
  mean_h, mean_s, mean_v   — HSV channel means
  contrast                  — grayscale luminance std (texture / edge intensity)
  green_ratio               — fraction of pixels where green channel dominates (vegetation proxy)
  sky_ratio                 — fraction of upper-third pixels where blue dominates (sky / water proxy)

All features are computed from raw PIL pixels — no external model required.

Output:
  data/processed/pixel_features.parquet    — per-image (pano_id × heading)
  data/processed/pixel_segment.parquet     — per-point aggregated (mean over headings)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from loguru import logger

MANIFEST_IN  = ROOT / "data/interim/image_manifest.parquet"
OUT_IMAGE    = ROOT / "data/processed/pixel_features.parquet"
OUT_SEGMENT  = ROOT / "data/processed/pixel_segment.parquet"

PIXEL_COLS = [
    "mean_r", "mean_g", "mean_b",
    "mean_h", "mean_s", "mean_v",
    "contrast", "green_ratio", "sky_ratio",
]


def _extract(path: str) -> dict:
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0   # H × W × 3

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # --- RGB means ---
    mean_r = float(r.mean())
    mean_g = float(g.mean())
    mean_b = float(b.mean())

    # --- HSV (vectorised, no colorsys overhead) ---
    cmax  = arr.max(axis=2)
    cmin  = arr.min(axis=2)
    delta = cmax - cmin

    v = cmax
    s = np.where(cmax > 0, delta / cmax, 0.0)

    h = np.zeros_like(v, dtype=np.float32)
    eps = 1e-8
    mask_r = (cmax == r) & (delta > 0)
    mask_g = (cmax == g) & (delta > 0)
    mask_b = (cmax == b) & (delta > 0)
    h[mask_r] = ((g[mask_r] - b[mask_r]) / (delta[mask_r] + eps) % 6) / 6.0
    h[mask_g] = ((b[mask_g] - r[mask_g]) / (delta[mask_g] + eps) + 2) / 6.0
    h[mask_b] = ((r[mask_b] - g[mask_b]) / (delta[mask_b] + eps) + 4) / 6.0

    mean_h = float(h.mean())
    mean_s = float(s.mean())
    mean_v = float(v.mean())

    # --- Contrast: std of luminance ---
    lum      = 0.299 * r + 0.587 * g + 0.114 * b
    contrast = float(lum.std())

    # --- Green ratio: vegetation proxy ---
    green_ratio = float(((g > r) & (g > b)).mean())

    # --- Sky ratio: blue-dominant in upper third of frame ---
    h3  = max(1, arr.shape[0] // 3)
    top = arr[:h3]
    sky_ratio = float(
        ((top[:, :, 2] > top[:, :, 0]) & (top[:, :, 2] > top[:, :, 1])).mean()
    )

    return {
        "mean_r": mean_r, "mean_g": mean_g, "mean_b": mean_b,
        "mean_h": mean_h, "mean_s": mean_s, "mean_v": mean_v,
        "contrast": contrast, "green_ratio": green_ratio, "sky_ratio": sky_ratio,
    }


def run(sample_only: bool = False) -> None:
    manifest = pd.read_parquet(MANIFEST_IN)
    manifest = manifest[manifest["downloaded"].fillna(False)].copy()
    if sample_only:
        manifest = manifest.sample(200, random_state=42)
    logger.info(f"Images to process: {len(manifest):,}")

    rows = []
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Pixel features"):
        try:
            feats = _extract(row["local_path"])
        except Exception as e:
            logger.debug(f"Skip {row['local_path']}: {e}")
            feats = {c: float("nan") for c in PIXEL_COLS}

        feats["pano_id"]  = row["pano_id"]
        feats["heading"]  = row["heading"]
        feats["point_id"] = row["point_id"]
        rows.append(feats)

    df = pd.DataFrame(rows)
    OUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_IMAGE, index=False)
    logger.info(f"Per-image pixel features → {OUT_IMAGE}  ({len(df):,} rows)")

    seg = df.groupby("point_id")[PIXEL_COLS].mean().reset_index()
    seg.to_parquet(OUT_SEGMENT, index=False)
    logger.info(f"Per-segment pixel features → {OUT_SEGMENT}  ({len(seg):,} rows)")

    print("\n── Pixel feature means ──────────────────────────────────────")
    for col in PIXEL_COLS:
        print(f"  {col:<15} {seg[col].mean():.4f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sample", action="store_true", help="Run on 200 images (test)")
    args = p.parse_args()
    run(sample_only=args.sample)
