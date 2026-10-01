"""Build geospatial features for each sample point — Task 8.

Derives terrain, hydrology, building exposure, and population features
from existing raster and vector datasets. Joins them to the sample-point
frame and exports a flat feature table and a GeoPackage.

Features computed
-----------------
Terrain (from Copernicus DEM):
  elevation_m          Elevation at point (metres)
  slope_deg            Slope in degrees (2-D gradient of local DEM patch)
  local_depression_m   Point elevation minus 500 m neighbourhood mean
                        (negative = below surroundings → depression)

Hydrology (already in sample_points.gpkg, forwarded here):
  dist_drain_m         Distance to nearest OSM drain/waterway channel
  dist_waterway_m      Distance to nearest OSM waterway polyline

Flood exposure (from flood model):
  flood_exposed_count_100m  Buildings flagged as flood-exposed within 100 m
  any_flood_exposed_100m    Binary: ≥1 flood-exposed building within 100 m

Urban exposure:
  building_count_100m  OSM buildings within 100 m buffer
  building_count_250m  OSM buildings within 250 m buffer
  building_density_100m  buildings per km²

Population:
  pop_density_per_km2  WorldPop 2020 population within 100 m buffer

Outputs
-------
data/processed/geospatial_features.parquet
data/processed/geospatial_features.gpkg
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import warnings
import numpy as np
import pandas as pd
from loguru import logger

MANUSCRIPTS = Path("/Users/newuser/Documents/manuscripts/accra-3d-spatial-risk-lab")

# ── Input paths ───────────────────────────────────────────────────────────────
SAMPLE_POINTS  = ROOT / "data/interim/sample_points.gpkg"
DEM_TIF        = MANUSCRIPTS / "data/raw/dem/accra_core_copernicus_dem_20260411.tif"
BUILDINGS_GPKG = MANUSCRIPTS / "data/raw/osm/accra_core_buildings_20260411.gpkg"
WORLDPOP_TIF   = MANUSCRIPTS / "data/raw/population/greater_accra_worldpop_2020_20260411.tif"
FLOOD_GPKG     = MANUSCRIPTS / "data/processed/flood_accra_accra_core_20260411.gpkg"
WATERWAYS_GPKG = ROOT / "data/external/accra_core_waterways.gpkg"

# ── Output paths ──────────────────────────────────────────────────────────────
OUT_PARQUET = ROOT / "data/processed/geospatial_features.parquet"
OUT_GPKG    = ROOT / "data/processed/geospatial_features.gpkg"

# Buffer radii (metres)
BUFFER_100  = 100
BUFFER_250  = 250

# Local DEM window for depression calculation (cells around the point)
DEPRESSION_WINDOW = 15   # ≈ 500 m at ~30 m resolution


# ── Terrain features from DEM ────────────────────────────────────────────────
def _sample_dem_at_points(gdf, dem_path: Path) -> pd.DataFrame:
    """Sample elevation and compute slope + local depression for each point."""
    import rasterio
    from rasterio.transform import rowcol

    logger.info(f"Sampling DEM: {dem_path.name}")
    elevations   = []
    slopes       = []
    depressions  = []

    with rasterio.open(dem_path) as src:
        dem_array = src.read(1).astype(np.float32)
        nodata    = src.nodata or -9999
        dem_array[dem_array == nodata] = np.nan

        # Pixel size in degrees → approximate metres
        res_lon = abs(src.transform.a)   # degrees per pixel
        res_lat = abs(src.transform.e)
        meters_per_deg_lat = 111_111.0
        px_m_x = res_lon * meters_per_deg_lat  # ~30 m for Copernicus DEM
        px_m_y = res_lat * meters_per_deg_lat

        rows_arr, cols_arr = rowcol(
            src.transform,
            gdf.geometry.x.values,
            gdf.geometry.y.values,
        )

        nrows, ncols = dem_array.shape
        hw = DEPRESSION_WINDOW // 2

        for r, c in zip(rows_arr, cols_arr):
            r, c = int(r), int(c)
            if 0 <= r < nrows and 0 <= c < ncols:
                elev = float(dem_array[r, c]) if not np.isnan(dem_array[r, c]) else np.nan
                elevations.append(elev)

                # Slope: gradient of local patch
                r0, r1 = max(0, r - 2), min(nrows, r + 3)
                c0, c1 = max(0, c - 2), min(ncols, c + 3)
                patch = dem_array[r0:r1, c0:c1]
                if patch.size > 4 and not np.all(np.isnan(patch)):
                    dy, dx = np.gradient(np.nan_to_num(patch, nan=np.nanmean(patch)),
                                         px_m_y, px_m_x)
                    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
                    slopes.append(float(np.degrees(slope_rad).mean()))
                else:
                    slopes.append(np.nan)

                # Local depression: point elevation minus neighbourhood mean
                r0d = max(0, r - hw); r1d = min(nrows, r + hw + 1)
                c0d = max(0, c - hw); c1d = min(ncols, c + hw + 1)
                nbhd = dem_array[r0d:r1d, c0d:c1d]
                nbhd_mean = float(np.nanmean(nbhd)) if not np.all(np.isnan(nbhd)) else np.nan
                depressions.append(
                    (elev - nbhd_mean) if (not np.isnan(elev) and nbhd_mean is not np.nan)
                    else np.nan
                )
            else:
                elevations.append(np.nan)
                slopes.append(np.nan)
                depressions.append(np.nan)

    return pd.DataFrame({
        "point_id":          gdf["point_id"].values,
        "elevation_m":       elevations,
        "slope_deg":         slopes,
        "local_depression_m": depressions,
    })


# ── Population from WorldPop raster ──────────────────────────────────────────
def _sample_worldpop(gdf, wp_path: Path, buffer_m: float = 100) -> pd.Series:
    """Estimate population within buffer_m of each point using WorldPop raster."""
    import rasterio
    from rasterio.mask import mask as rio_mask
    from shapely.ops import transform as shp_transform
    import pyproj

    logger.info(f"Sampling WorldPop raster (buffer={buffer_m} m) …")

    # Project to UTM 30N for accurate buffering
    wgs84  = pyproj.CRS("EPSG:4326")
    utm30n = pyproj.CRS("EPSG:32630")
    project_fwd = pyproj.Transformer.from_crs(wgs84, utm30n, always_xy=True).transform
    project_inv = pyproj.Transformer.from_crs(utm30n, wgs84, always_xy=True).transform

    pop_values = []
    # Pixel area in km² (WorldPop ~100 m resolution)
    with rasterio.open(wp_path) as src:
        nodata  = src.nodata or -99999
        res     = abs(src.transform.a)           # degrees
        px_km2  = (res * 111.111) ** 2           # approximate km²

        for _, row in gdf.iterrows():
            try:
                pt_utm  = shp_transform(project_fwd, row.geometry)
                buf_utm = pt_utm.buffer(buffer_m)
                buf_wgs = shp_transform(project_inv, buf_utm)
                arr, _  = rio_mask(src, [buf_wgs], crop=True, nodata=nodata, filled=True)
                vals    = arr[arr != nodata].astype(float)
                vals    = vals[vals >= 0]
                # WorldPop stores pop/pixel; sum over buffer then scale to /km²
                buf_km2 = np.pi * (buffer_m / 1000) ** 2
                pop_values.append(float(vals.sum()) / buf_km2 if buf_km2 > 0 else np.nan)
            except Exception:
                pop_values.append(np.nan)

    return pd.Series(pop_values, index=gdf.index, name="pop_density_per_km2")


# ── Building count within buffer ─────────────────────────────────────────────
def _count_buildings_in_buffer(
    gdf_pts,
    bld_gdf,
    buffer_m_list: list[int],
) -> pd.DataFrame:
    """Count buildings within each buffer radius for all sample points."""
    import geopandas as gpd
    import pyproj
    from shapely.ops import transform as shp_transform

    logger.info(f"Counting buildings within {buffer_m_list} m buffers …")

    wgs84  = pyproj.CRS("EPSG:4326")
    utm30n = pyproj.CRS("EPSG:32630")
    project_fwd = pyproj.Transformer.from_crs(wgs84, utm30n, always_xy=True).transform
    project_inv = pyproj.Transformer.from_crs(utm30n, wgs84, always_xy=True).transform

    # Reproject building centroids to WGS84 if needed
    if bld_gdf.crs and bld_gdf.crs != "EPSG:4326":
        bld_wgs = bld_gdf.to_crs("EPSG:4326")
    else:
        bld_wgs = bld_gdf.copy()
    bld_centroids = bld_wgs.geometry.centroid

    # Build a spatial index over building centroids
    bld_sindex = bld_centroids.sindex

    results = {f"building_count_{b}m": [] for b in buffer_m_list}
    results["point_id"] = []

    for _, row in gdf_pts.iterrows():
        results["point_id"].append(row["point_id"])
        pt_utm  = shp_transform(project_fwd, row.geometry)

        for buf_m in buffer_m_list:
            buf_utm = pt_utm.buffer(buf_m)
            buf_wgs = shp_transform(project_inv, buf_utm)
            candidates = list(bld_sindex.intersection(buf_wgs.bounds))
            count = int(sum(
                1 for i in candidates
                if bld_centroids.iloc[i].within(buf_wgs)
            ))
            results[f"building_count_{buf_m}m"].append(count)

    df = pd.DataFrame(results)
    df["building_density_100m"] = (
        df["building_count_100m"] / (np.pi * (BUFFER_100 / 1000) ** 2)
    ).round(1)
    return df


# ── Flood-exposed buildings within buffer ────────────────────────────────────
def _count_flood_exposed_in_buffer(gdf_pts, flood_gdf, buffer_m: int = 100) -> pd.DataFrame:
    """Count flood-exposed buildings within buffer_m of each point."""
    import pyproj
    from shapely.ops import transform as shp_transform

    logger.info(f"Counting flood-exposed buildings within {buffer_m} m …")

    if flood_gdf.crs and str(flood_gdf.crs) != "EPSG:4326":
        flood_wgs = flood_gdf.to_crs("EPSG:4326")
    else:
        flood_wgs = flood_gdf.copy()

    # Only flood-exposed buildings
    exposed = flood_wgs[flood_wgs["flood_exposed"] == True].copy()
    exp_centroids = exposed.geometry.centroid
    exp_sindex    = exp_centroids.sindex

    wgs84  = pyproj.CRS("EPSG:4326")
    utm30n = pyproj.CRS("EPSG:32630")
    project_fwd = pyproj.Transformer.from_crs(wgs84, utm30n, always_xy=True).transform
    project_inv = pyproj.Transformer.from_crs(utm30n, wgs84, always_xy=True).transform

    counts = []
    for _, row in gdf_pts.iterrows():
        pt_utm  = shp_transform(project_fwd, row.geometry)
        buf_utm = pt_utm.buffer(buffer_m)
        buf_wgs = shp_transform(project_inv, buf_utm)
        candidates = list(exp_sindex.intersection(buf_wgs.bounds))
        count = int(sum(
            1 for i in candidates
            if exp_centroids.iloc[i].within(buf_wgs)
        ))
        counts.append(count)

    return pd.DataFrame({
        "point_id":                   gdf_pts["point_id"].values,
        "flood_exposed_count_100m":   counts,
        "any_flood_exposed_100m":     [int(c > 0) for c in counts],
    })


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    import geopandas as gpd

    logger.info("Loading sample points …")
    gdf = gpd.read_file(SAMPLE_POINTS)
    logger.info(f"  {len(gdf):,} points, CRS: {gdf.crs}")

    # ── 1. Terrain features ───────────────────────────────────────────────
    terrain = _sample_dem_at_points(gdf, DEM_TIF)

    # ── 2. Building counts ────────────────────────────────────────────────
    logger.info(f"Loading buildings from {BUILDINGS_GPKG.name} …")
    buildings = gpd.read_file(BUILDINGS_GPKG)
    logger.info(f"  {len(buildings):,} building footprints")
    bld_counts = _count_buildings_in_buffer(gdf, buildings, [BUFFER_100, BUFFER_250])

    # ── 3. Flood-exposed buildings ────────────────────────────────────────
    logger.info(f"Loading flood exposure from {FLOOD_GPKG.name} …")
    flood = gpd.read_file(FLOOD_GPKG)
    flood_counts = _count_flood_exposed_in_buffer(gdf, flood, BUFFER_100)

    # ── 4. Population density ─────────────────────────────────────────────
    pop = _sample_worldpop(gdf, WORLDPOP_TIF, buffer_m=BUFFER_100)

    # ── 5. Hydrology — already in sample_points ───────────────────────────
    hydro = gdf[["point_id", "dist_drain_m", "dist_waterway_m",
                  "near_drain", "near_waterway"]].copy()

    # ── 6. Merge all features ─────────────────────────────────────────────
    logger.info("Merging feature tables …")
    result = (
        gdf[["point_id", "highway", "flood_stratum",
              "latitude", "longitude", "geometry"]]
        .merge(terrain,    on="point_id", how="left")
        .merge(bld_counts, on="point_id", how="left")
        .merge(flood_counts, on="point_id", how="left")
        .merge(hydro,      on="point_id", how="left")
    )
    result["pop_density_per_km2"] = pop.values

    # ── 7. Export ─────────────────────────────────────────────────────────
    OUT_GPKG.parent.mkdir(parents=True, exist_ok=True)
    result.to_file(OUT_GPKG, driver="GPKG")
    logger.info(f"GeoPackage → {OUT_GPKG}  ({len(result):,} rows)")

    flat = result.drop(columns="geometry")
    flat.to_parquet(OUT_PARQUET, index=False)
    logger.info(f"Parquet → {OUT_PARQUET}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n── Geospatial Feature Summary ──────────────────────────────")
    print(f"  Points:   {len(flat):,}")
    print(f"  Columns:  {len(flat.columns)}")
    print()
    feature_stats = {
        "elevation_m":              flat["elevation_m"],
        "slope_deg":                flat["slope_deg"],
        "local_depression_m":       flat["local_depression_m"],
        "dist_drain_m":             flat["dist_drain_m"],
        "dist_waterway_m":          flat["dist_waterway_m"],
        "building_count_100m":      flat["building_count_100m"],
        "building_count_250m":      flat["building_count_250m"],
        "flood_exposed_count_100m": flat["flood_exposed_count_100m"],
        "pop_density_per_km2":      flat["pop_density_per_km2"],
    }
    print(f"  {'Feature':<35} {'Mean':>10} {'Min':>10} {'Max':>10}")
    print(f"  {'-'*65}")
    for name, s in feature_stats.items():
        print(f"  {name:<35} {s.mean():>10.1f} {s.min():>10.1f} {s.max():>10.1f}")

    print()
    print("  Output files:")
    for p in [OUT_PARQUET, OUT_GPKG]:
        if p.exists():
            mb = p.stat().st_size / (1024 * 1024)
            print(f"    {p.name:<50} {mb:>6.1f} MB")


if __name__ == "__main__":
    main()
