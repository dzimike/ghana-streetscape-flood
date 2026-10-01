"""Aggregate image-level CV predictions to road-segment (point) level — Task 7.

Reads image_features.parquet (one row per image), groups by point_id, and
produces a single feature vector per sample point that summarises what the
four Street View headings collectively show.

Composite scores
----------------
drain_detection_score      presence of any drainage infrastructure
drain_obstruction_score    evidence of blocked or waste-choked drains
waste_risk_score           solid waste accumulation visible
water_exposure_score       stagnant water or visible waterways
road_quality_score         surface deterioration and unpaved conditions
informal_encroachment_score  kiosks / informal structures near drainage
streetscape_sensitivity_score  overall S component of SLFVI (0–1)
adaptive_capacity_raw_score    pre-inversion A component (0–1)

Outputs
-------
data/processed/streetscape_features.parquet  — flat feature table
data/processed/streetscape_features.gpkg     — geodataframe with geometry
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from loguru import logger

MANUSCRIPTS = Path("/Users/newuser/Documents/manuscripts/accra-3d-spatial-risk-lab")

IMAGE_FEATURES = ROOT / "data/processed/image_features.parquet"
SAMPLE_POINTS  = ROOT / "data/interim/sample_points.gpkg"
OUT_PARQUET    = ROOT / "data/processed/streetscape_features.parquet"
OUT_GPKG       = ROOT / "data/processed/streetscape_features.gpkg"


def _norm(s: pd.Series) -> pd.Series:
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn) if mx > mn else s * 0.0


def build_streetscape_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-image features to per-point features."""
    clip_cols = [c for c in df.columns if c.startswith("clip_")]
    seg_cols  = [c for c in df.columns if c.startswith("seg_")]

    # Fill missing seg values (not all images may have segmentation output)
    for col in seg_cols:
        df[col] = df[col].fillna(0.0)

    # Mean aggregation across headings for each point
    agg = df.groupby("point_id")[clip_cols + seg_cols].mean().reset_index()

    # ── Composite CLIP scores ──────────────────────────────────────────────
    c = lambda name: f"clip_{name}"

    # Drainage presence (higher = more visible drainage infrastructure)
    drain_presence_cols = [
        c("visible_drain_present"), c("open_gutter_present"),
        c("culvert_or_bridge_visible"), c("visible_waterway_or_stream"),
    ]
    agg["drain_detection_score"] = agg[drain_presence_cols].mean(axis=1)

    # Drain obstruction (blocked + waste near drain)
    obstruction_cols = [c("blocked_drain_present"), c("solid_waste_accumulation")]
    agg["drain_obstruction_score"] = agg[obstruction_cols].mean(axis=1)

    # Waste
    agg["waste_risk_score"] = agg[c("solid_waste_accumulation")]

    # Water exposure (stagnant water or waterway visible)
    water_cols = [c("stagnant_water_visible"), c("visible_waterway_or_stream")]
    agg["water_exposure_score"] = agg[water_cols].mean(axis=1)

    # Road surface quality (poor conditions → higher score)
    road_cols = [
        c("poor_road_condition"), c("unpaved_shoulder"), c("roadside_erosion"),
    ]
    agg["road_quality_score"] = agg[road_cols].mean(axis=1)

    # Informal encroachment near drainage
    encroach_cols = [
        c("informal_structure_near_drainage"), c("pedestrian_exposure"),
    ]
    agg["informal_encroachment_score"] = agg[encroach_cols].mean(axis=1)

    # Low-lying street form (terrain signal visible from street)
    agg["low_lying_score"] = agg[c("low_lying_street_form")]

    # Impervious surface (from segmentation road + sidewalk fractions)
    seg_imp_cols = ["seg_road", "seg_sidewalk"]
    present_imp  = [c for c in seg_imp_cols if c in agg.columns]
    if present_imp:
        agg["impervious_surface_frac"] = agg[present_imp].sum(axis=1)
    else:
        agg["impervious_surface_frac"] = 0.0

    # Vegetation fraction from segmentation
    if "seg_vegetation" in agg.columns:
        agg["vegetation_frac"] = agg["seg_vegetation"]
    else:
        agg["vegetation_frac"] = 0.0

    # Water fraction from segmentation
    if "seg_water" in agg.columns:
        agg["seg_water_frac"] = agg["seg_water"]
    else:
        agg["seg_water_frac"] = 0.0

    # ── Streetscape Sensitivity Score (S) ─────────────────────────────────
    # Combines: blockage, waste, road condition, encroachment, water
    # Weights guided by CLAUDE.md Section 11
    w_obs  = 0.30   # drain obstruction
    w_road = 0.20   # road quality
    w_wst  = 0.20   # waste
    w_wat  = 0.15   # stagnant water / waterway exposure
    w_enc  = 0.10   # informal encroachment
    w_low  = 0.05   # low-lying street form

    agg["streetscape_sensitivity_score"] = (
        w_obs  * agg["drain_obstruction_score"]
        + w_road * agg["road_quality_score"]
        + w_wst  * agg["waste_risk_score"]
        + w_wat  * agg["water_exposure_score"]
        + w_enc  * agg["informal_encroachment_score"]
        + w_low  * agg["low_lying_score"]
    )

    # ── Adaptive Capacity Raw Score (A — before inversion) ────────────────
    # Higher = better adapted (good road, visible drainage, vegetation)
    # Will be inverted when computing SLFVI
    agg["adaptive_capacity_raw_score"] = (
        0.40 * agg["drain_detection_score"]       # drainage infrastructure present
        + 0.30 * (1.0 - agg["road_quality_score"])  # good road surface
        + 0.30 * agg["vegetation_frac"]              # green buffer
    )

    # ── Normalise composite scores to 0–1 ─────────────────────────────────
    for col in [
        "streetscape_sensitivity_score",
        "adaptive_capacity_raw_score",
        "drain_detection_score",
        "drain_obstruction_score",
        "road_quality_score",
        "waste_risk_score",
        "water_exposure_score",
        "informal_encroachment_score",
        "low_lying_score",
        "impervious_surface_frac",
    ]:
        if col in agg.columns:
            agg[col] = _norm(agg[col])

    # Count of valid headings per point
    heading_counts = df.groupby("point_id")["heading"].count().rename("n_headings")
    agg = agg.merge(heading_counts, on="point_id", how="left")

    return agg


def main() -> None:
    if not IMAGE_FEATURES.exists():
        logger.error(
            f"{IMAGE_FEATURES} not found — run src/models/run_all_inference.py first"
        )
        raise SystemExit(1)

    logger.info(f"Loading image features from {IMAGE_FEATURES}")
    df = pd.read_parquet(IMAGE_FEATURES)
    logger.info(f"  {len(df):,} image rows, {df['point_id'].nunique():,} unique points")

    logger.info("Building streetscape feature vectors …")
    features = build_streetscape_features(df)
    logger.info(f"  {len(features):,} point-level rows, {len(features.columns)} columns")

    # ── Join geometry from sample points ──────────────────────────────────
    try:
        import geopandas as gpd
        sp = gpd.read_file(SAMPLE_POINTS)[
            ["point_id", "highway", "flood_stratum",
             "dist_drain_m", "dist_waterway_m",
             "near_drain", "near_waterway",
             "segment_length_m", "latitude", "longitude", "geometry"]
        ]
        gdf = sp.merge(features, on="point_id", how="inner")

        OUT_GPKG.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(OUT_GPKG, driver="GPKG")
        logger.info(f"GeoPackage → {OUT_GPKG}  ({len(gdf):,} rows)")

        # Drop geometry for flat parquet
        features_flat = gdf.drop(columns="geometry")
    except Exception as e:
        logger.warning(f"Could not join geometry: {e}")
        features_flat = features

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    features_flat.to_parquet(OUT_PARQUET, index=False)
    logger.info(f"Parquet → {OUT_PARQUET}  ({len(features_flat):,} rows)")

    # ── Summary ───────────────────────────────────────────────────────────
    score_cols = [
        "streetscape_sensitivity_score", "adaptive_capacity_raw_score",
        "drain_detection_score", "drain_obstruction_score",
        "road_quality_score", "waste_risk_score", "water_exposure_score",
    ]
    print("\n── Streetscape Feature Summary ─────────────────────────────")
    print(f"  Points:   {len(features_flat):,}")
    print(f"  Columns:  {len(features_flat.columns)}")
    print()
    print("  Score means (0–1 normalised):")
    for col in score_cols:
        if col in features_flat.columns:
            print(f"    {col:<45} {features_flat[col].mean():.3f}")
    print()
    print("  By flood stratum:")
    if "flood_stratum" in features_flat.columns:
        for s, grp in features_flat.groupby("flood_stratum"):
            print(f"    {s:<30} sensitivity={grp['streetscape_sensitivity_score'].mean():.3f}")
    print()
    print("  Output files:")
    for p in [OUT_PARQUET, OUT_GPKG]:
        if p.exists():
            kb = p.stat().st_size / 1024
            print(f"    {p.name:<50} {kb:>8.1f} KB")


if __name__ == "__main__":
    main()
