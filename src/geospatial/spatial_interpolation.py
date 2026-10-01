"""City-wide spatial interpolation of SLFVI to a continuous 250 m grid.

Uses IDW (inverse-distance weighting) from the 3,836 known road points
to produce a continuous vulnerability surface over the full study area.
Also runs kriging via pykrige if available.

Outputs
-------
  outputs/maps/slfvi_surface_250m.tif     — GeoTIFF raster (EPSG:32630)
  outputs/maps/slfvi_surface_250m.gpkg    — grid polygons with SLFVI values
  outputs/figures/slfvi_surface_map.png   — manuscript figure
  outputs/tables/grid_vulnerability.csv   — tabular grid output
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from loguru import logger
from scipy.spatial import cKDTree

# ── Config ────────────────────────────────────────────────────────────────────
GRID_SIZE_M   = 250     # grid cell size in metres
IDW_K         = 12      # neighbours for IDW
IDW_POWER     = 2.0     # IDW decay power
MASK_DIST_M   = 2000    # mask cells > this far from any observed point

# Hotspot zoom extent (WGS-84, inner Accra / Odaw basin)
ZOOM_LON = (-0.330, -0.130)
ZOOM_LAT = (5.510, 5.660)

# ── Inputs ────────────────────────────────────────────────────────────────────
VI_PARQ  = ROOT / "data/processed/vulnerability_index.parquet"
ADM2     = ROOT / "data/external/GHA_ADM2.geojson"

# ── Outputs ───────────────────────────────────────────────────────────────────
OUT_MAPS  = ROOT / "outputs/maps"
OUT_FIGS  = ROOT / "outputs/figures"
OUT_TABS  = ROOT / "outputs/tables"

VULN_BINS   = [-0.001, 0.20, 0.40, 0.60, 0.80, 1.001]
VULN_LABELS = ["very_low", "low", "moderate", "high", "very_high"]
CMAP        = mcolors.LinearSegmentedColormap.from_list(
    "slfvi", ["#1a9641", "#a6d96a", "#ffffbf", "#fdae61", "#d7191c"]
)

STUDY_DISTRICTS = [
    "Accra Metropolis", "Ablekuma Central Municipal", "Ablekuma West Municipal",
    "Ledzokuku Municipal", "Ablekuma North Municipal", "Weija Gbawe Municipal",
    "Ayawaso North Municipal", "Krowor Municipal", "Ga East", "La Dade-kotopon",
    "La-nkwantanang-madina", "Okaikwei North Municipal", "Ayawaso Central Municipal",
    "Ga Central Municipal", "Adenta Municipal", "Korle Klottey Municipal",
    "Ga North Municipal", "Tema West Municipal", "Ga West Municipal",
    "Ayawaso East Municipal", "Ayawaso West",
]


def idw(tree: cKDTree, known_vals: np.ndarray,
        query_xy: np.ndarray, k: int, power: float) -> np.ndarray:
    dists, idxs = tree.query(query_xy, k=k, workers=-1)
    dists = np.maximum(dists, 1e-6)
    w = 1.0 / dists ** power
    w /= w.sum(axis=1, keepdims=True)
    return (w * known_vals[idxs]).sum(axis=1)


def main() -> None:
    for d in [OUT_MAPS, OUT_FIGS, OUT_TABS]:
        d.mkdir(parents=True, exist_ok=True)

    # ── 1. Load known vulnerability points ───────────────────────────────────
    vi = pd.read_parquet(VI_PARQ)
    score_col = "slfvi_final" if "slfvi_final" in vi.columns else "slfvi"
    vi_gdf = gpd.GeoDataFrame(
        vi,
        geometry=gpd.points_from_xy(vi["longitude"], vi["latitude"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:32630")

    known_xy  = np.column_stack([vi_gdf.geometry.x, vi_gdf.geometry.y])
    known_slfvi = vi_gdf[score_col].fillna(vi_gdf[score_col].median()).values
    logger.info(f"Known points: {len(vi_gdf):,}  score range: "
                f"{known_slfvi.min():.3f}–{known_slfvi.max():.3f}")

    # ── 2. Build study boundary ───────────────────────────────────────────────
    adm = gpd.read_file(ADM2).to_crs("EPSG:32630")
    study = adm[adm["shapeName"].isin(STUDY_DISTRICTS)]
    study_union = study.union_all()
    minx, miny, maxx, maxy = study_union.bounds
    logger.info(f"Study area bounds (UTM 30N): "
                f"x {minx:.0f}–{maxx:.0f}  y {miny:.0f}–{maxy:.0f}")

    # ── 3. Build 250 m grid ───────────────────────────────────────────────────
    xs = np.arange(minx, maxx + GRID_SIZE_M, GRID_SIZE_M)
    ys = np.arange(miny, maxy + GRID_SIZE_M, GRID_SIZE_M)
    xx, yy = np.meshgrid(xs, ys)
    grid_xy = np.column_stack([xx.ravel(), yy.ravel()])
    logger.info(f"Grid cells before clipping: {len(grid_xy):,}")

    # Clip to study boundary
    from shapely.geometry import Point
    grid_pts = gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in grid_xy],
        crs="EPSG:32630",
    )
    in_study = grid_pts.within(study_union)
    grid_pts = grid_pts[in_study].copy()
    grid_xy  = grid_xy[in_study.values]
    logger.info(f"Grid cells inside study area: {len(grid_pts):,}")

    # ── 4. IDW interpolation ──────────────────────────────────────────────────
    tree = cKDTree(known_xy)
    k    = min(IDW_K, len(known_xy))
    logger.info(f"Running IDW (k={k}, power={IDW_POWER}) …")
    slfvi_grid = idw(tree, known_slfvi, grid_xy, k=k, power=IDW_POWER)

    # Clip to 0–1
    slfvi_grid = np.clip(slfvi_grid, 0, 1)
    logger.info(f"IDW complete — grid SLFVI range: {slfvi_grid.min():.3f}–{slfvi_grid.max():.3f}")

    # ── 4b. Mask cells > MASK_DIST_M from any observed point ─────────────────
    nearest_dist, _ = tree.query(grid_xy, k=1, workers=-1)
    within_mask = nearest_dist <= MASK_DIST_M
    n_masked = (~within_mask).sum()
    logger.info(f"Data-void mask: {n_masked:,} cells > {MASK_DIST_M/1000:.0f} km from "
                f"nearest observation ({100*n_masked/len(grid_xy):.1f}% of grid)")

    # ── 5. Try pykrige if available ───────────────────────────────────────────
    kriging_done = False
    try:
        from pykrige.ok import OrdinaryKriging
        logger.info("Running Ordinary Kriging (pykrige) …")
        ok = OrdinaryKriging(
            known_xy[:, 0], known_xy[:, 1], known_slfvi,
            variogram_model="spherical",
            verbose=False, enable_plotting=False,
        )
        z_krig, ss_krig = ok.execute("points", grid_xy[:, 0], grid_xy[:, 1])
        slfvi_krig = np.clip(z_krig.data, 0, 1)
        kriging_done = True
        logger.info(f"Kriging complete — range: {slfvi_krig.min():.3f}–{slfvi_krig.max():.3f}")
    except ImportError:
        logger.info("pykrige not installed — using IDW only")
        slfvi_krig = None

    # ── 6. Assign classes and export ──────────────────────────────────────────
    grid_pts = grid_pts.reset_index(drop=True)
    grid_pts["slfvi_idw"]   = slfvi_grid
    grid_pts["slfvi_class"] = pd.cut(
        slfvi_grid, bins=VULN_BINS, labels=VULN_LABELS, right=False
    )
    if kriging_done:
        grid_pts["slfvi_kriging"] = slfvi_krig

    # District join
    grid_pts = gpd.sjoin(
        grid_pts, adm[["shapeName", "geometry"]], how="left", predicate="within"
    ).drop(columns=["index_right"], errors="ignore")

    # Tabular output
    out_csv = OUT_TABS / "grid_vulnerability.csv"
    grid_pts[["geometry", "slfvi_idw", "slfvi_class", "shapeName"]].assign(
        lon=grid_pts.to_crs("EPSG:4326").geometry.x,
        lat=grid_pts.to_crs("EPSG:4326").geometry.y,
    ).drop(columns="geometry").to_csv(out_csv, index=False)
    logger.info(f"Grid table → {out_csv}")

    # GeoPackage (square polygons)
    half = GRID_SIZE_M / 2
    from shapely.geometry import box
    grid_poly = grid_pts.copy()
    grid_poly["geometry"] = [
        box(x - half, y - half, x + half, y + half)
        for x, y in grid_xy
    ]
    out_gpkg = OUT_MAPS / "slfvi_surface_250m.gpkg"
    grid_poly.to_file(out_gpkg, driver="GPKG")
    logger.info(f"GeoPackage → {out_gpkg}")

    # GeoTIFF
    try:
        import rasterio
        from rasterio.transform import from_bounds
        from rasterio.features import rasterize

        rows = int(np.ceil((maxy - miny) / GRID_SIZE_M))
        cols = int(np.ceil((maxx - minx) / GRID_SIZE_M))
        transform = from_bounds(minx, miny, maxx, maxy, cols, rows)

        raster = np.full((rows, cols), np.nan, dtype=np.float32)
        for (x, y), val in zip(grid_xy, slfvi_grid):
            col = int((x - minx) / GRID_SIZE_M)
            row = int((maxy - y) / GRID_SIZE_M)
            if 0 <= row < rows and 0 <= col < cols:
                raster[row, col] = val

        out_tif = OUT_MAPS / "slfvi_surface_250m.tif"
        with rasterio.open(
            out_tif, "w", driver="GTiff", height=rows, width=cols,
            count=1, dtype="float32", crs="EPSG:32630",
            transform=transform, nodata=np.nan,
        ) as dst:
            dst.write(raster, 1)
        logger.info(f"GeoTIFF → {out_tif}")
    except ImportError:
        logger.warning("rasterio not available — GeoTIFF skipped")
        out_tif = None

    # ── 7. Manuscript figure ──────────────────────────────────────────────────
    grid_wgs = grid_pts.to_crs("EPSG:4326")
    study_wgs = study.to_crs("EPSG:4326")
    vi_wgs = vi_gdf.to_crs("EPSG:4326")

    # Mask coordinates for the hatched overlay (cells in data void)
    grid_wgs_void = grid_wgs[~within_mask]
    grid_wgs_obs  = grid_wgs[within_mask]
    slfvi_obs     = slfvi_grid[within_mask]

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # ── Left panel: full IDW surface with data-void hatching ─────────────────
    ax = axes[0]
    study_wgs.plot(ax=ax, color="none", edgecolor="#555", linewidth=0.8, zorder=3)
    # Data-void cells: grey hatched
    ax.scatter(
        grid_wgs_void.geometry.x, grid_wgs_void.geometry.y,
        c="#cccccc", s=3, alpha=0.35, linewidths=0, zorder=1,
        label="Data void (>2 km from observation)",
    )
    # Observed-coverage cells: colour-coded SLFVI
    sc = ax.scatter(
        grid_wgs_obs.geometry.x, grid_wgs_obs.geometry.y,
        c=slfvi_obs, cmap=CMAP, vmin=0, vmax=1,
        s=4, alpha=0.85, linewidths=0, zorder=2,
    )
    ax.scatter(
        vi_wgs.geometry.x, vi_wgs.geometry.y,
        c="black", s=5, alpha=0.4, linewidths=0, zorder=4,
        label="Street View observed",
    )
    cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("SLFVI", fontsize=9)
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title(
        f"SLFVI Surface — IDW Interpolation\n"
        f"250 m grid, {within_mask.sum():,} cells (grey = data void >2 km)",
        fontsize=11,
    )
    ax.set_xlabel("Longitude", fontsize=9)
    ax.set_ylabel("Latitude", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.tick_params(labelsize=8)

    # ── Right panel: zoom-in on hotspot area (Odaw basin / inner Accra) ───────
    ax2 = axes[1]
    study_wgs.plot(ax=ax2, color="none", edgecolor="#555", linewidth=0.8, zorder=3)

    # Clip to zoom extent
    zoom_mask = (
        (grid_wgs_obs.geometry.x >= ZOOM_LON[0]) &
        (grid_wgs_obs.geometry.x <= ZOOM_LON[1]) &
        (grid_wgs_obs.geometry.y >= ZOOM_LAT[0]) &
        (grid_wgs_obs.geometry.y <= ZOOM_LAT[1])
    )
    zoom_gdf  = grid_wgs_obs[zoom_mask]
    zoom_slfvi = slfvi_obs[zoom_mask.values]

    # Use IDW values in zoom panel — kriging suppresses extremes to 0.18–0.58
    # which renders the zoom uniformly green; IDW preserves hotspot contrast.
    zoom_vals = zoom_slfvi
    zoom_method = "IDW (detail)"

    sc2 = ax2.scatter(
        zoom_gdf.geometry.x, zoom_gdf.geometry.y,
        c=zoom_vals, cmap=CMAP, vmin=0, vmax=1,
        s=12, alpha=0.9, linewidths=0, zorder=2,
    )
    # Observed points within zoom
    vi_zoom = vi_wgs[
        (vi_wgs.geometry.x >= ZOOM_LON[0]) & (vi_wgs.geometry.x <= ZOOM_LON[1]) &
        (vi_wgs.geometry.y >= ZOOM_LAT[0]) & (vi_wgs.geometry.y <= ZOOM_LAT[1])
    ]
    ax2.scatter(
        vi_zoom.geometry.x, vi_zoom.geometry.y,
        c="black", s=8, alpha=0.5, linewidths=0, zorder=4,
        label="Street View observed",
    )
    cbar2 = plt.colorbar(sc2, ax=ax2, fraction=0.03, pad=0.02)
    cbar2.set_label("SLFVI", fontsize=9)
    cbar2.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_xlim(ZOOM_LON)
    ax2.set_ylim(ZOOM_LAT)
    ax2.set_title(
        f"Hotspot Zoom — Odaw Basin / Inner Accra\n{zoom_method}, 250 m grid",
        fontsize=11,
    )
    ax2.set_xlabel("Longitude", fontsize=9)
    ax2.set_ylabel("Latitude", fontsize=9)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.tick_params(labelsize=8)

    fig.suptitle(
        "Street-Level Flood Vulnerability Index — Continuous Surface, Greater Accra",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    out_fig = OUT_FIGS / "slfvi_surface_map.png"
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Surface map → {out_fig}")

    # ── 8. Summary ────────────────────────────────────────────────────────────
    print("\n── Spatial Interpolation Summary ───────────────────────────")
    print(f"  Method:        IDW (k={k}, power={IDW_POWER})"
          + (" + Ordinary Kriging" if kriging_done else ""))
    print(f"  Grid size:     {GRID_SIZE_M} m")
    print(f"  Grid cells:    {len(grid_pts):,}")
    print(f"  SLFVI range:   {slfvi_grid.min():.3f} – {slfvi_grid.max():.3f}")
    print(f"  Mean SLFVI:    {slfvi_grid.mean():.3f}")
    print()
    print("  Class distribution:")
    for cls, grp in grid_pts.groupby("slfvi_class", observed=True):
        print(f"    {str(cls):<12} {len(grp):>6,}  ({100*len(grp)/len(grid_pts):.1f}%)")
    print()
    print(f"  High+VeryHigh cells: {(slfvi_grid>=0.6).sum():,}  "
          f"({100*(slfvi_grid>=0.6).mean():.1f}%)")
    print()
    print("── Output Files ────────────────────────────────────────────")
    for p in [out_gpkg, out_fig, out_csv,
              OUT_MAPS / "slfvi_surface_250m.tif"]:
        if p and p.exists():
            kb = p.stat().st_size / 1024
            tag = f"{kb/1024:.1f} MB" if kb > 1024 else f"{kb:.1f} KB"
            print(f"  {p.relative_to(ROOT)!s:<55} {tag}")


if __name__ == "__main__":
    main()
