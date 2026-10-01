"""Task 7: Aggregate per-image predictions into per-segment streetscape features.

Merges CLIP + segmentation + EfficientNet predictions, then aggregates
by point_id to produce one row per road point with mean/max probabilities
across all four headings (0°, 90°, 180°, 270°).

Joins with geospatial features (elevation, DEM, drainage distance, etc.)
and recomputes the Street-Level Flood Vulnerability Index.

Outputs
-------
  data/processed/streetscape_features_v2.parquet  — enriched per-segment features
  data/processed/vulnerability_index_v2.parquet   — updated SLFVI
  data/processed/vulnerability_index_v2.gpkg      — GeoPackage for QGIS
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import geopandas as gpd
from loguru import logger

# ── Inputs ────────────────────────────────────────────────────────────────────
CLIP_PREDS    = ROOT / "data/processed/clip_predictions.parquet"
SEG_PREDS     = ROOT / "data/processed/segmentation_fractions.parquet"
ENET_PREDS    = ROOT / "data/processed/efficientnet_predictions.parquet"
MANIFEST      = ROOT / "data/interim/image_manifest.parquet"
GEO_FEATS     = ROOT / "data/processed/geospatial_features.parquet"

# ── Outputs ───────────────────────────────────────────────────────────────────
OUT_SF     = ROOT / "data/processed/streetscape_features_v2.parquet"
OUT_VI     = ROOT / "data/processed/vulnerability_index_v2.parquet"
OUT_VI_GEO = ROOT / "data/processed/vulnerability_index_v2.gpkg"

LABEL_COLS = [
    "visible_drain_present", "open_gutter_present", "blocked_drain_present",
    "stagnant_water_visible", "poor_road_condition", "heavy_impervious_surface",
    "unpaved_shoulder", "informal_structure_near_drainage", "solid_waste_accumulation",
    "visible_waterway_or_stream", "low_lying_street_form", "roadside_erosion",
    "pedestrian_exposure", "culvert_or_bridge_visible", "no_visible_drainage",
]


def norm01(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    lo, hi = s.min(), s.max()
    if hi == lo:
        return s * 0.0
    return (s - lo) / (hi - lo)


def run() -> None:
    OUT_SF.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Load all prediction sources ───────────────────────────────────────
    manifest = pd.read_parquet(MANIFEST)
    clip     = pd.read_parquet(CLIP_PREDS)
    seg      = pd.read_parquet(SEG_PREDS)
    enet     = pd.read_parquet(ENET_PREDS)
    geo      = pd.read_parquet(GEO_FEATS)

    logger.info(f"Manifest:   {len(manifest):,} images")
    logger.info(f"CLIP:       {len(clip):,} rows")
    logger.info(f"Seg:        {len(seg):,} rows")
    logger.info(f"EfficientNet: {len(enet):,} rows")
    logger.info(f"Geo features: {len(geo):,} points")

    # ── 2. Merge per-image predictions ───────────────────────────────────────
    key = ["pano_id", "heading", "point_id"]

    # Start from manifest (ground truth for which images exist)
    img = manifest[key + ["local_path", "flood_stratum", "highway",
                          "pano_lat", "pano_lon", "capture_date"]].copy()
    img = img.merge(clip,  on=key, how="left")
    img = img.merge(seg,   on=key, how="left")
    img = img.merge(enet,  on=key, how="left")

    logger.info(f"Merged image table: {img.shape}")

    # ── 3. Aggregate to point level (mean & max across headings) ─────────────
    clip_cols = [c for c in img.columns if c.startswith("clip_")]
    seg_cols  = [c for c in img.columns if c.startswith("seg_")]
    enet_cols = [c for c in img.columns if c.startswith("enet_") and c != "enet_valid"]

    agg_dict: dict = {}
    for c in clip_cols + seg_cols + enet_cols:
        agg_dict[c] = ["mean", "max"]

    # Metadata aggregations
    agg_dict["local_path"]   = "first"
    agg_dict["flood_stratum"] = "first"
    agg_dict["highway"]      = "first"
    agg_dict["pano_lat"]     = "mean"
    agg_dict["pano_lon"]     = "mean"

    seg_agg = img.groupby("point_id").agg(agg_dict)
    seg_agg.columns = ["_".join(c).rstrip("_") for c in seg_agg.columns]
    seg_agg = seg_agg.reset_index()

    # Rename mean columns to drop _mean suffix for readability
    rename = {}
    for c in seg_agg.columns:
        if c.endswith("_mean"):
            rename[c] = c[:-5]  # strip _mean
    seg_agg = seg_agg.rename(columns=rename)

    # Heading count
    n_headings = img.groupby("point_id")["heading"].count().rename("n_headings")
    seg_agg = seg_agg.merge(n_headings, on="point_id", how="left")

    logger.info(f"Point-level features: {seg_agg.shape}")

    # ── 4. Join geospatial features ───────────────────────────────────────────
    # Geo features index may be on point_id or integer index — try both
    if "point_id" in geo.columns:
        geo_merge = geo
    else:
        geo_merge = geo.reset_index().rename(columns={"index": "point_id"})

    sf = seg_agg.merge(geo_merge, on="point_id", how="left",
                       suffixes=("", "_geo"))

    # Resolve lat/lon: prefer geo (more precise from OSM)
    for coord in ["latitude", "longitude"]:
        geo_col = f"{coord}_geo"
        if geo_col in sf.columns:
            sf[coord] = sf[geo_col].combine_first(sf.get(coord, pd.Series(dtype=float)))
            sf = sf.drop(columns=[geo_col])
        elif coord not in sf.columns:
            sf[coord] = sf.get("pano_lat_mean" if coord == "latitude" else "pano_lon_mean",
                               np.nan)

    logger.info(f"After geo join: {sf.shape}")

    # ── 5. Compute composite streetscape sensitivity score (S) ───────────────
    # Weighted sum of EfficientNet label probabilities — higher weight
    # for indicators most directly linked to flood risk
    WEIGHTS = {
        "visible_drain_present":              0.08,
        "open_gutter_present":                0.08,
        "blocked_drain_present":              0.12,
        "stagnant_water_visible":             0.12,
        "poor_road_condition":                0.08,
        "heavy_impervious_surface":           0.06,
        "unpaved_shoulder":                   0.04,
        "informal_structure_near_drainage":   0.08,
        "solid_waste_accumulation":           0.08,
        "visible_waterway_or_stream":         0.06,
        "low_lying_street_form":              0.08,
        "roadside_erosion":                   0.06,
        "pedestrian_exposure":                0.04,
        "culvert_or_bridge_visible":          0.04,
        "no_visible_drainage":               -0.06,  # penalises hidden/absent drains
    }

    enet_sensitivity = sum(
        w * sf.get(f"enet_{lbl}", pd.Series(0.0, index=sf.index))
        for lbl, w in WEIGHTS.items()
    )
    sf["enet_sensitivity"] = enet_sensitivity.clip(0, 1)

    # Overall streetscape vulnerability (blend EfficientNet + CLIP)
    clip_vuln = sf.get("clip_vulnerability_score",
                       pd.Series(0.5, index=sf.index)).fillna(0.5)
    enet_vuln = sf.get("enet_vuln_prob",
                       pd.Series(0.5, index=sf.index)).fillna(0.5)
    sf["streetscape_vuln"] = 0.6 * enet_vuln + 0.4 * clip_vuln

    sf.to_parquet(OUT_SF, index=False)
    logger.info(f"Streetscape features v2 → {OUT_SF}  ({len(sf):,} points)")

    # ── 6. Recompute SLFVI ───────────────────────────────────────────────────
    # SLFVI = 0.30·H + 0.20·E + 0.35·S + 0.15·A
    # Use geospatial hazard/exposure/adaptive if available, else fallback

    def get_norm(col, default=0.5):
        if col in sf.columns:
            return norm01(sf[col].fillna(sf[col].median()))
        return pd.Series(default, index=sf.index)

    # Hazard components
    elev_inv  = 1 - get_norm("elevation_m")          # lower = more hazard
    twi       = get_norm("twi")                        # higher = more hazard
    rain      = get_norm("rainfall_mean_mm")
    flood_occ = get_norm("flood_occurrence")
    H = (0.35 * elev_inv + 0.25 * twi + 0.20 * rain + 0.20 * flood_occ).clip(0, 1)

    # Exposure
    pop   = get_norm("pop_density")
    bld   = get_norm("building_count")
    E = (0.50 * pop + 0.50 * bld).clip(0, 1)

    # Streetscape sensitivity (use new EfficientNet-enriched score)
    S = norm01(sf["enet_sensitivity"].fillna(sf["enet_sensitivity"].median()))

    # Adaptive capacity (inverse — lower capacity = higher vulnerability)
    road_quality = 1 - get_norm("streetscape_vuln")   # poor roads → low capacity
    drain_access = get_norm("near_drain", 0.3)
    A_inv = (0.60 * (1 - road_quality) + 0.40 * (1 - drain_access)).clip(0, 1)

    slfvi = (0.30 * H + 0.20 * E + 0.35 * S + 0.15 * A_inv).clip(0, 1)

    vi = sf[["point_id", "highway", "flood_stratum", "latitude", "longitude"]].copy()
    vi["H"] = H.round(4)
    vi["E"] = E.round(4)
    vi["S"] = S.round(4)
    vi["A"] = A_inv.round(4)
    vi["slfvi"] = slfvi.round(4)
    vi["enet_vuln_prob"]   = sf["enet_vuln_prob"].round(4)
    vi["enet_sensitivity"] = sf["enet_sensitivity"].round(4)
    vi["streetscape_vuln"] = sf["streetscape_vuln"].round(4)

    bins   = [0, 0.20, 0.40, 0.60, 0.80, 1.01]
    labels = ["very_low", "low", "moderate", "high", "very_high"]
    vi["slfvi_class"] = pd.cut(vi["slfvi"], bins=bins, labels=labels, right=False)

    vi.to_parquet(OUT_VI, index=False)

    # GeoPackage
    gdf = gpd.GeoDataFrame(
        vi,
        geometry=gpd.points_from_xy(vi["longitude"], vi["latitude"]),
        crs="EPSG:4326",
    ).dropna(subset=["geometry"])
    gdf.to_file(OUT_VI_GEO, driver="GPKG")

    print("\n── Streetscape Features v2 ─────────────────────────────────")
    print(f"  Road points:    {len(sf):,}")
    print(f"  Feature columns:{sf.shape[1]}")
    print()
    print("── Updated SLFVI Distribution ──────────────────────────────")
    print(f"  {'Class':<20} {'N':>6}  {'%':>6}")
    for cls, grp in vi.groupby("slfvi_class", observed=True):
        print(f"  {str(cls):<20} {len(grp):>6,}  {100*len(grp)/len(vi):>5.1f}%")
    print()
    print(f"  Mean SLFVI:     {vi['slfvi'].mean():.3f}")
    print(f"  High+VeryHigh:  {(vi['slfvi']>=0.60).sum():,}  "
          f"({100*(vi['slfvi']>=0.60).mean():.1f}%)")
    print()
    print("── Output Files ────────────────────────────────────────────")
    for p in [OUT_SF, OUT_VI, OUT_VI_GEO]:
        kb = p.stat().st_size / 1024
        tag = f"{kb/1024:.1f} MB" if kb > 1024 else f"{kb:.1f} KB"
        print(f"  {p.relative_to(ROOT)!s:<55} {tag}")


if __name__ == "__main__":
    run()
