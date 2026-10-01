"""OSM road download and sampling frame construction."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
from loguru import logger
from shapely.geometry import Point


ROAD_TAGS = {
    "highway": [
        "motorway", "trunk", "primary", "secondary", "tertiary",
        "residential", "unclassified", "living_street", "service",
    ]
}

EXCLUDE_TYPES = {"footway", "cycleway", "path", "steps"}


def download_roads(
    bbox: tuple[float, float, float, float],
    output_path: Path | None = None,
    network_type: str = "drive",
) -> gpd.GeoDataFrame:
    """Download OSM road network for a bounding box.

    Args:
        bbox: (west, south, east, north) in WGS84 degrees.
        output_path: If given, save result as GeoPackage.
        network_type: OSMnx network type.

    Returns:
        GeoDataFrame of road edges.
    """
    west, south, east, north = bbox
    logger.info(f"Downloading OSM roads for bbox {bbox}")
    G = ox.graph_from_bbox(
        bbox=(north, south, east, west),
        network_type=network_type,
        retain_all=False,
        simplify=True,
    )
    _, edges = ox.graph_to_gdfs(G)
    edges = edges.reset_index()
    logger.info(f"Downloaded {len(edges):,} road edges")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        edges.to_file(output_path, driver="GPKG", layer="roads_raw")
        logger.info(f"Saved raw roads to {output_path}")

    return edges


def clean_roads(
    roads: gpd.GeoDataFrame,
    include_types: Sequence[str] | None = None,
    target_crs: str = "EPSG:32630",
) -> gpd.GeoDataFrame:
    """Filter road classes, drop excluded types, reproject.

    Args:
        roads: Raw OSM road edges.
        include_types: Highway types to keep. Defaults to ROAD_TAGS["highway"].
        target_crs: Metric CRS for length calculations.

    Returns:
        Cleaned GeoDataFrame in target_crs.
    """
    include = set(include_types or ROAD_TAGS["highway"])
    logger.info(f"Cleaning roads — keeping {len(include)} highway types")

    # Normalise highway column (can be list or string in OSM data)
    def _normalise(val):
        if isinstance(val, list):
            return val[0] if val else None
        return val

    roads = roads.copy()
    roads["highway"] = roads["highway"].apply(_normalise)
    mask = roads["highway"].isin(include) & ~roads["highway"].isin(EXCLUDE_TYPES)
    cleaned = roads[mask].copy()

    # Reproject to metric CRS
    cleaned = cleaned.to_crs(target_crs)
    cleaned["length_m"] = cleaned.geometry.length
    logger.info(f"Retained {len(cleaned):,} edges after cleaning")
    return cleaned


def split_into_segments(
    roads: gpd.GeoDataFrame,
    segment_length_m: float = 100.0,
) -> gpd.GeoDataFrame:
    """Split road linestrings into fixed-length segments.

    Args:
        roads: Cleaned roads in a metric CRS.
        segment_length_m: Target segment length in metres.

    Returns:
        GeoDataFrame with one row per segment, including parent edge attributes.
    """
    from shapely.ops import substring

    logger.info(f"Splitting roads into {segment_length_m} m segments")
    records = []

    for _, row in roads.iterrows():
        geom = row.geometry
        total = geom.length
        if total == 0:
            continue
        n = max(1, int(np.ceil(total / segment_length_m)))
        for i in range(n):
            start = i * (total / n)
            end = (i + 1) * (total / n)
            seg = substring(geom, start, end, normalized=False)
            rec = row.to_dict()
            rec["geometry"] = seg
            rec["segment_id"] = f"{row.get('osmid', row.name)}_{i}"
            rec["segment_length_m"] = seg.length
            records.append(rec)

    segments = gpd.GeoDataFrame(records, crs=roads.crs)
    logger.info(f"Created {len(segments):,} road segments")
    return segments


def generate_sample_points(
    segments: gpd.GeoDataFrame,
    target_n: int | None = None,
    output_crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """Generate centroid sample points from road segments.

    Args:
        segments: Road segments GeoDataFrame.
        target_n: If set, randomly subsample to this many points.
        output_crs: CRS for output points (default WGS84 for API queries).

    Returns:
        GeoDataFrame of sample points with segment attributes.
    """
    logger.info("Generating sample points at segment centroids")
    pts = segments.copy()
    pts["geometry"] = pts.geometry.centroid
    pts = pts.to_crs(output_crs)
    pts["longitude"] = pts.geometry.x
    pts["latitude"] = pts.geometry.y
    pts["point_id"] = range(len(pts))

    if target_n and target_n < len(pts):
        pts = pts.sample(n=target_n, random_state=42).reset_index(drop=True)
        logger.info(f"Subsampled to {target_n:,} points")

    logger.info(f"Generated {len(pts):,} sample points")
    return pts
