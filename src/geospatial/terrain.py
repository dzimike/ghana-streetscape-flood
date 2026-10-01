"""Terrain and hydrological feature extraction.

Uses rasterio / xarray for DEM processing and Google Earth Engine
(via the earthengine-api) for satellite-derived flood proxies.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import rowcol
import geopandas as gpd
import pandas as pd
from loguru import logger


def compute_slope(dem_path: Path) -> np.ndarray:
    """Compute percent slope from a DEM raster."""
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(float)
        res_x, res_y = src.res
    dy, dx = np.gradient(dem, res_y, res_x)
    slope = np.sqrt(dx**2 + dy**2)
    return slope


def compute_twi(dem_path: Path) -> np.ndarray:
    """Compute Topographic Wetness Index: ln(flow_acc / tan(slope)).

    Note: requires a pre-computed flow accumulation raster at the same
    resolution as the DEM. Simplified single-grid version here.
    """
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(float)
        res = src.res[0]
    dy, dx = np.gradient(dem, res)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_rad = np.where(slope_rad < 1e-4, 1e-4, slope_rad)
    # Placeholder flow accumulation (constant 1) — replace with real FAC raster
    flow_acc = np.ones_like(dem)
    twi = np.log(flow_acc / np.tan(slope_rad))
    return twi


def sample_raster_at_points(
    raster_path: Path,
    points: gpd.GeoDataFrame,
    band: int = 1,
    col_name: str = "raster_value",
) -> gpd.GeoDataFrame:
    """Sample raster values at point locations.

    Args:
        raster_path: Path to raster file.
        points: GeoDataFrame of points (must match raster CRS or be reprojected).
        band: Band index to sample (1-based).
        col_name: Output column name.

    Returns:
        Copy of points with sampled values added.
    """
    with rasterio.open(raster_path) as src:
        pts = points.to_crs(src.crs)
        coords = [(geom.x, geom.y) for geom in pts.geometry]
        values = [v[0] for v in src.sample(coords, indexes=band)]
    result = points.copy()
    result[col_name] = values
    return result


def distance_to_layer(
    points: gpd.GeoDataFrame,
    layer: gpd.GeoDataFrame,
    col_name: str = "distance_m",
) -> gpd.GeoDataFrame:
    """Compute minimum distance from each point to nearest geometry in layer.

    Both GeoDataFrames must be in the same metric CRS.
    """
    from shapely.ops import unary_union
    target = unary_union(layer.geometry)
    result = points.copy()
    result[col_name] = result.geometry.distance(target)
    return result
