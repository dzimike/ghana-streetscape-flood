"""Build sampling frame from existing Accra road data.

Reads the pre-cleaned road network, filters to driveable road classes,
splits into 100m segments, generates centroid sample points, and
stratifies by flood-risk proximity using the waterways layer.

Outputs:
  data/interim/road_segments.gpkg   — 100m road segments
  data/interim/sample_points.gpkg   — centroid sample points
  data/interim/sample_points.csv    — same, for API queries
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger
from shapely.ops import substring, unary_union

from src.config.settings import load_config

# ── Config ──────────────────────────────────────────────────────────────────
cfg = load_config("accra_pilot")

ROADS_SRC  = ROOT / "data/external/accra_core_roads.gpkg"
WATER_SRC  = ROOT / "data/external/accra_core_waterways.gpkg"
SEG_OUT    = ROOT / "data/interim/road_segments.gpkg"
PTS_OUT    = ROOT / "data/interim/sample_points.gpkg"
PTS_CSV    = ROOT / "data/interim/sample_points.csv"

SEGMENT_LEN   = cfg["sampling"]["segment_length_m"]    # 100 m
TARGET_POINTS = cfg["sampling"]["target_points"]        # 5000
PROX_DRAIN_M  = cfg["flood_risk"]["proximity_drain_m"] # 100 m
CRS_METRIC    = cfg["project"]["crs"]                  # EPSG:32630
CRS_WGS84     = cfg["project"]["output_crs"]           # EPSG:4326

KEEP_HIGHWAY = set(cfg["road_classes"]["include"])
DROP_HIGHWAY = set(cfg["road_classes"]["exclude"])


# ── 1. Load and filter roads ─────────────────────────────────────────────────
def load_and_filter_roads() -> gpd.GeoDataFrame:
    logger.info(f"Loading roads from {ROADS_SRC}")
    roads = gpd.read_file(ROADS_SRC)
    logger.info(f"Loaded {len(roads):,} edges (CRS: {roads.crs})")

    # Ensure metric CRS
    if roads.crs.to_epsg() != 32630:
        roads = roads.to_crs(CRS_METRIC)

    # Normalise highway column (can be list in some exports)
    def _norm(v):
        return v[0] if isinstance(v, list) else v

    roads["highway"] = roads["highway"].apply(_norm)

    before = len(roads)
    roads = roads[
        roads["highway"].isin(KEEP_HIGHWAY) &
        ~roads["highway"].isin(DROP_HIGHWAY)
    ].copy()
    roads["length_m"] = roads.geometry.length
    logger.info(f"Filtered {before:,} → {len(roads):,} driveable edges")
    return roads


# ── 2. Split into 100m segments ──────────────────────────────────────────────
def split_segments(roads: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    logger.info(f"Splitting edges into ≤{SEGMENT_LEN}m segments …")
    records = []
    for idx, row in roads.iterrows():
        geom = row.geometry
        total = geom.length
        if total < 1:
            continue
        n = max(1, int(np.ceil(total / SEGMENT_LEN)))
        cuts = np.linspace(0, total, n + 1)
        for i in range(n):
            seg = substring(geom, cuts[i], cuts[i + 1])
            records.append({
                **{c: row[c] for c in ["highway", "name", "lanes", "maxspeed", "oneway"]
                   if c in roads.columns},
                "geometry": seg,
                "segment_id": f"{idx}_{i}",
                "parent_idx": idx,
                "segment_length_m": seg.length,
            })

    segs = gpd.GeoDataFrame(records, crs=roads.crs)
    logger.info(f"Created {len(segs):,} segments")
    return segs


# ── 3. Stratify by flood proximity ──────────────────────────────────────────
def add_flood_proximity(segs: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    logger.info("Computing proximity to drains/waterways …")
    water = gpd.read_file(WATER_SRC).to_crs(CRS_METRIC)

    # Separate drainage vs natural waterways
    drain_types = {"drain", "ditch"}
    water_types = {"stream", "river", "canal"}

    drains = water[water["waterway"].isin(drain_types)]
    natural = water[water["waterway"].isin(water_types)]

    drain_union   = unary_union(drains.geometry) if len(drains) else None
    natural_union = unary_union(natural.geometry) if len(natural) else None

    centroids = segs.geometry.centroid

    segs["dist_drain_m"]     = centroids.distance(drain_union)   if drain_union   else np.inf
    segs["dist_waterway_m"]  = centroids.distance(natural_union) if natural_union else np.inf
    segs["near_drain"]       = segs["dist_drain_m"]    <= PROX_DRAIN_M
    segs["near_waterway"]    = segs["dist_waterway_m"] <= 100

    # Flood-risk stratum
    def _stratum(row):
        if row["near_drain"] or row["near_waterway"]:
            return "high_risk_proximity"
        return "low_risk_reference"

    segs["flood_stratum"] = segs.apply(_stratum, axis=1)
    counts = segs["flood_stratum"].value_counts()
    logger.info(f"Flood strata: {counts.to_dict()}")
    return segs


# ── 4. Generate sample points ────────────────────────────────────────────────
def generate_sample_points(segs: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    logger.info(f"Generating stratified sample of {TARGET_POINTS:,} points …")

    # Stratified sampling: oversample high-risk (60/40 split)
    strata = segs["flood_stratum"].unique()
    n_high = int(TARGET_POINTS * 0.60)
    n_low  = TARGET_POINTS - n_high

    alloc = {"high_risk_proximity": n_high, "low_risk_reference": n_low}
    parts = []
    for stratum, n in alloc.items():
        pool = segs[segs["flood_stratum"] == stratum]
        n = min(n, len(pool))
        parts.append(pool.sample(n=n, random_state=42))

    sample_segs = pd.concat(parts).reset_index(drop=True)

    # Centroid points in WGS84 for API queries
    pts = sample_segs.copy()
    pts["geometry"] = pts.geometry.centroid
    pts = pts.to_crs(CRS_WGS84)
    pts["point_id"]  = range(len(pts))
    pts["longitude"] = pts.geometry.x
    pts["latitude"]  = pts.geometry.y

    logger.info(f"Sample: {len(pts):,} points "
                f"({(pts['flood_stratum']=='high_risk_proximity').sum()} high-risk, "
                f"{(pts['flood_stratum']=='low_risk_reference').sum()} reference)")
    return pts


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    for d in [SEG_OUT.parent, PTS_OUT.parent]:
        d.mkdir(parents=True, exist_ok=True)

    roads = load_and_filter_roads()
    segs  = split_segments(roads)
    segs  = add_flood_proximity(segs)

    logger.info(f"Saving segments → {SEG_OUT}")
    segs.to_file(SEG_OUT, driver="GPKG", layer="road_segments")

    pts = generate_sample_points(segs)

    logger.info(f"Saving sample points → {PTS_OUT}")
    pts.to_file(PTS_OUT, driver="GPKG", layer="sample_points")

    csv_cols = ["point_id", "segment_id", "latitude", "longitude",
                "highway", "flood_stratum", "dist_drain_m", "dist_waterway_m"]
    pts[[c for c in csv_cols if c in pts.columns]].to_csv(PTS_CSV, index=False)
    logger.info(f"Saved CSV → {PTS_CSV}")

    # Summary table
    print("\n── Sampling Frame Summary ──────────────────────────────")
    print(f"  Road edges loaded:     {len(roads):>8,}")
    print(f"  100m segments:         {len(segs):>8,}")
    print(f"  Sample points:         {len(pts):>8,}")
    print(f"  Near drain (<100m):    {segs['near_drain'].sum():>8,} segments")
    print(f"  High-risk stratum:     {(pts['flood_stratum']=='high_risk_proximity').sum():>8,} points")
    print(f"  Reference stratum:     {(pts['flood_stratum']=='low_risk_reference').sum():>8,} points")
    print(f"\n  Segments saved:  {SEG_OUT}")
    print(f"  Points saved:    {PTS_OUT}")
    print(f"  CSV saved:       {PTS_CSV}")


if __name__ == "__main__":
    main()
