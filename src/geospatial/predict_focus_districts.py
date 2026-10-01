"""Predict flood vulnerability for all road segments in 4 focus districts.

Strategy
--------
1. Clip ADM2 boundaries to the 4 focus districts.
2. Clip road network to those districts and generate sample points every 100 m.
3. For sampled points with Street View data, use observed streetscape features.
4. For unsampled road points, interpolate geospatial features from known points
   using inverse-distance weighting, then predict SLFVI with the LightGBM model.
5. Produce:
   - district_vulnerability.parquet / .gpkg  — all road points with SLFVI
   - district_municipal_summary.csv
   - district_vulnerability_map.html         — interactive Folium map
   - district_vulnerability_map_static.png

Outputs
-------
  outputs/focus_districts/
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
from loguru import logger
from scipy.spatial import cKDTree
import pickle

# ── Config ────────────────────────────────────────────────────────────────────
FOCUS_DISTRICTS = [
    "Ablekuma Central Municipal",
    "Ledzokuku Municipal",
    "Ga East",
    "La Dade-kotopon",
]
POINT_SPACING_M = 100   # metres between sample points along roads

# ── Inputs ────────────────────────────────────────────────────────────────────
ADM2          = ROOT / "data/external/GHA_ADM2.geojson"
ROADS_EXT     = ROOT / "data/external/accra_core_roads.gpkg"
ROADS_OSM     = ROOT / "data/interim/road_segments.gpkg"
KNOWN_VI      = ROOT / "data/processed/vulnerability_index.parquet"
GEO_FEATS     = ROOT / "data/processed/geospatial_features.parquet"
MODEL_STORE   = ROOT / "models/fusion"

# ── Outputs ───────────────────────────────────────────────────────────────────
OUT_DIR = ROOT / "outputs/focus_districts"

GEO_COLS = [
    "elevation_m", "slope_deg", "local_depression_m",
    "dist_drain_m", "dist_waterway_m",
    "building_count_100m", "building_count_250m", "pop_density_per_km2",
]

SLFVI_COLS = [
    "elevation_m", "slope_deg", "local_depression_m",
    "dist_drain_m", "dist_waterway_m",
    "building_count_100m", "building_count_250m", "pop_density_per_km2",
    # Streetscape — set to district median for unsampled points
    "enet_sensitivity", "enet_vuln_prob",
    "enet_visible_drain_present", "enet_open_gutter_present",
    "enet_blocked_drain_present", "enet_stagnant_water_visible",
    "enet_poor_road_condition", "enet_heavy_impervious_surface",
    "enet_solid_waste_accumulation", "enet_informal_structure_near_drainage",
    "enet_low_lying_street_form", "enet_no_visible_drainage",
    # CLIP
    "clip_blocked_drain_present", "clip_stagnant_water_visible",
    "clip_solid_waste_accumulation", "clip_poor_road_condition",
    "clip_visible_drain_present", "clip_no_visible_drainage",
    "clip_unpaved_shoulder", "clip_low_lying_street_form",
    # Segmentation
    "seg_road", "seg_water", "seg_vegetation", "seg_bare_ground",
    "seg_building", "seg_waste",
]

VULN_BINS   = [-0.001, 0.20, 0.40, 0.60, 0.80, 1.001]
VULN_LABELS = ["very_low", "low", "moderate", "high", "very_high"]
COLOUR_MAP  = {
    "very_low": "#1a9641", "low": "#a6d96a",
    "moderate": "#ffffbf", "high": "#fdae61", "very_high": "#d7191c",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def points_along_line(geom, spacing_m: float):
    """Yield (x, y) points every spacing_m metres along a LineString."""
    length = geom.length
    if length == 0:
        yield geom.interpolate(0).x, geom.interpolate(0).y
        return
    d = 0.0
    while d <= length:
        pt = geom.interpolate(d)
        yield pt.x, pt.y
        d += spacing_m
    # Always include the end point
    pt = geom.interpolate(length)
    yield pt.x, pt.y


def idw_interpolate(tree: cKDTree, known_vals: np.ndarray,
                    query_xy: np.ndarray, k: int = 8, power: float = 2.0) -> np.ndarray:
    """Inverse-distance weighted interpolation for multiple columns."""
    dists, idxs = tree.query(query_xy, k=k, workers=-1)
    dists = np.maximum(dists, 1e-6)
    weights = 1.0 / dists ** power
    weights /= weights.sum(axis=1, keepdims=True)
    # known_vals: (n_known, n_cols), weights: (n_query, k), idxs: (n_query, k)
    result = np.einsum("qk,qkc->qc", weights, known_vals[idxs])
    return result


def norm01(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi > lo else pd.Series(0.5, index=s.index)


def build_slfvi_manual(df: pd.DataFrame) -> pd.Series:
    """Compute SLFVI from components without the ML model (fallback)."""
    elev_inv  = norm01(-df["elevation_m"].fillna(df["elevation_m"].median()))
    depression = norm01(-df.get("local_depression_m", pd.Series(0, index=df.index)).fillna(0))
    drain_prox = norm01(-df["dist_drain_m"].fillna(df["dist_drain_m"].median()))
    water_prox = norm01(-df["dist_waterway_m"].fillna(df["dist_waterway_m"].median()))
    H = 0.40 * elev_inv + 0.25 * depression + 0.20 * drain_prox + 0.15 * water_prox

    pop  = norm01(df.get("pop_density_per_km2", pd.Series(0, index=df.index)).fillna(0))
    bld  = norm01(df.get("building_count_100m", pd.Series(0, index=df.index)).fillna(0))
    E = 0.50 * pop + 0.50 * bld

    S = norm01(df.get("enet_sensitivity", pd.Series(0.5, index=df.index)).fillna(0.5))

    A = norm01(df.get("enet_poor_road_condition", pd.Series(0.5, index=df.index)).fillna(0.5))

    return (0.30 * H + 0.20 * E + 0.35 * S + 0.15 * A).clip(0, 1)


def load_lgb_model():
    """Try to load saved LightGBM model."""
    for ext in ["lgb.pkl", "lgbm.pkl", "lightgbm.pkl", "best_model.pkl"]:
        p = MODEL_STORE / ext
        if p.exists():
            with open(p, "rb") as f:
                return pickle.load(f)
    return None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load district boundaries ───────────────────────────────────────────
    adm = gpd.read_file(ADM2).to_crs("EPSG:4326")
    focus = adm[adm["shapeName"].isin(FOCUS_DISTRICTS)].copy()
    logger.info(f"Focus districts: {list(focus['shapeName'])}")

    focus_union = focus.union_all()
    focus_utm   = focus.to_crs("EPSG:32630")

    # ── 2. Load and clip road network ─────────────────────────────────────────
    roads = gpd.read_file(ROADS_EXT)   # EPSG:32630
    roads_clip = roads.clip(focus_utm.union_all())
    roads_clip = roads_clip[roads_clip.geometry.length > 0].copy()
    logger.info(f"Road segments in focus area: {len(roads_clip):,}")

    # ── 3. Generate points every 100 m along roads ───────────────────────────
    records = []
    for _, row in roads_clip.iterrows():
        for x, y in points_along_line(row.geometry, POINT_SPACING_M):
            records.append({
                "x_utm": x, "y_utm": y,
                "highway": row.get("highway", "unclassified"),
            })

    pts_utm = gpd.GeoDataFrame(
        records,
        geometry=gpd.points_from_xy([r["x_utm"] for r in records],
                                     [r["y_utm"] for r in records]),
        crs="EPSG:32630",
    )
    pts_wgs = pts_utm.to_crs("EPSG:4326")
    pts_wgs["latitude"]  = pts_wgs.geometry.y
    pts_wgs["longitude"] = pts_wgs.geometry.x
    pts_wgs["new_point_id"] = range(len(pts_wgs))

    # Spatial join to get district name
    pts_wgs = gpd.sjoin(
        pts_wgs, focus[["shapeName", "geometry"]], how="left", predicate="within"
    ).drop(columns=["index_right"], errors="ignore")
    pts_wgs = pts_wgs.dropna(subset=["shapeName"])
    logger.info(f"Generated {len(pts_wgs):,} prediction points")

    # ── 4. Load known sampled points with observed features ───────────────────
    known_vi = pd.read_parquet(KNOWN_VI)
    known_vi_gdf = gpd.GeoDataFrame(
        known_vi,
        geometry=gpd.points_from_xy(known_vi["longitude"], known_vi["latitude"]),
        crs="EPSG:4326",
    )
    # Filter known points to focus districts
    known_focus = gpd.sjoin(
        known_vi_gdf, focus[["shapeName", "geometry"]], how="inner", predicate="within"
    ).drop(columns=["index_right"], errors="ignore")
    logger.info(f"Known sampled points in focus area: {len(known_focus):,}")

    # ── 5. IDW interpolation: geo features for all new points ────────────────
    geo = pd.read_parquet(GEO_FEATS)
    geo_gdf = gpd.GeoDataFrame(
        geo,
        geometry=gpd.points_from_xy(geo["longitude"], geo["latitude"]),
        crs="EPSG:4326",
    )
    # Build KD-tree on known geo points
    known_xy  = np.column_stack([geo["longitude"].values, geo["latitude"].values])
    query_xy  = np.column_stack([pts_wgs["longitude"].values, pts_wgs["latitude"].values])

    avail_geo = [c for c in GEO_COLS if c in geo.columns]
    known_vals = geo[avail_geo].fillna(geo[avail_geo].median()).values.astype(float)

    tree = cKDTree(known_xy)
    interp_vals = idw_interpolate(tree, known_vals, query_xy, k=min(8, len(known_xy)))
    interp_df   = pd.DataFrame(interp_vals, columns=avail_geo)

    for col in avail_geo:
        pts_wgs[col] = interp_df[col].values

    # ── 6. Streetscape features: use district median from known sampled points ─
    enet_cols = [c for c in SLFVI_COLS if c.startswith("enet_") or c.startswith("clip_") or c.startswith("seg_")]
    district_medians = {}
    for dist in FOCUS_DISTRICTS:
        sub = known_focus[known_focus.get("shapeName_left",
                           known_focus.get("shapeName", pd.Series())).eq(dist)
                          if "shapeName_left" in known_focus.columns
                          else known_focus["shapeName"].eq(dist)]
        if len(sub) == 0:
            sub = known_focus  # fall back to all known if no district match
        dm = {}
        for col in enet_cols:
            if col in sub.columns:
                dm[col] = float(sub[col].median())
            else:
                dm[col] = 0.3
        district_medians[dist] = dm

    for col in enet_cols:
        pts_wgs[col] = pts_wgs["shapeName"].map(
            lambda d: district_medians.get(d, {}).get(col, 0.3)
        )

    # ── 7. Predict SLFVI ──────────────────────────────────────────────────────
    lgb_model = load_lgb_model()

    # For unsampled points: use only geospatial features (which are properly
    # interpolated). Streetscape features are not reliably estimated by district
    # median, so the manual formula gives better spatial gradients.
    logger.info("Computing SLFVI analytically for predicted points …")
    pts_wgs["slfvi"] = build_slfvi_manual(pts_wgs)
    pts_wgs["ml_flood_prob"] = pts_wgs["slfvi"]
    pts_wgs["slfvi_class"] = pd.cut(
        pts_wgs["slfvi"], bins=VULN_BINS, labels=VULN_LABELS, right=False
    )

    # ── 8. Merge in observed streetscape points ───────────────────────────────
    known_focus_out = known_focus.copy()
    sname_col = "shapeName_left" if "shapeName_left" in known_focus_out.columns else "shapeName"
    known_focus_out = known_focus_out.rename(columns={sname_col: "shapeName"})
    known_focus_out["data_source"] = "streetview_observed"

    keep_cols = ["latitude", "longitude", "highway", "shapeName",
                 "slfvi", "slfvi_final", "vulnerability_class_final",
                 "ml_flood_prob", "enet_sensitivity", "enet_vuln_prob",
                 "elevation_m", "building_count_100m", "geometry"]
    known_focus_out["slfvi_class"] = pd.cut(
        known_focus_out.get("slfvi_final", known_focus_out["slfvi"]),
        bins=VULN_BINS, labels=VULN_LABELS, right=False
    )

    pts_wgs["data_source"] = "predicted_interpolated"
    pts_wgs["slfvi_final"] = pts_wgs["slfvi"]

    combined = pd.concat([
        pts_wgs[["latitude", "longitude", "highway", "shapeName",
                 "slfvi", "slfvi_final", "slfvi_class", "ml_flood_prob",
                 "enet_sensitivity", "enet_vuln_prob",
                 "elevation_m", "building_count_100m", "data_source", "geometry"]],
        known_focus_out[[c for c in
                         ["latitude", "longitude", "highway", "shapeName",
                          "slfvi", "slfvi_final", "slfvi_class", "ml_flood_prob",
                          "enet_sensitivity", "enet_vuln_prob",
                          "elevation_m", "building_count_100m", "data_source", "geometry"]
                         if c in known_focus_out.columns]],
    ], ignore_index=True)

    combined = gpd.GeoDataFrame(combined, crs="EPSG:4326")
    combined = combined.dropna(subset=["latitude", "longitude"])
    logger.info(f"Combined output: {len(combined):,} points "
                f"({(combined.data_source=='streetview_observed').sum():,} observed + "
                f"{(combined.data_source=='predicted_interpolated').sum():,} predicted)")

    # ── 9. Save outputs ───────────────────────────────────────────────────────
    out_parq = OUT_DIR / "district_vulnerability.parquet"
    out_gpkg = OUT_DIR / "district_vulnerability.gpkg"
    combined.drop(columns=["geometry"]).to_parquet(out_parq, index=False)
    combined.to_file(out_gpkg, driver="GPKG")

    # Municipal summary
    summary_rows = []
    for dist in FOCUS_DISTRICTS:
        sub = combined[combined["shapeName"] == dist]
        if len(sub) == 0:
            continue
        obs = sub[sub.data_source == "streetview_observed"]
        summary_rows.append({
            "district":           dist,
            "total_points":       len(sub),
            "observed_points":    len(obs),
            "predicted_points":   len(sub) - len(obs),
            "mean_slfvi":         round(sub["slfvi"].mean(), 3),
            "high_very_high_pct": round(100 * (sub["slfvi"] >= 0.6).mean(), 1),
            "mean_elevation_m":   round(sub["elevation_m"].mean(), 1),
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("mean_slfvi", ascending=False)
    summary_df.to_csv(OUT_DIR / "district_municipal_summary.csv", index=False)

    # ── 10. Interactive Folium map ────────────────────────────────────────────
    try:
        import folium
        from folium.plugins import MarkerCluster

        centre_lat = combined["latitude"].mean()
        centre_lon = combined["longitude"].mean()
        m = folium.Map(location=[centre_lat, centre_lon], zoom_start=13,
                       tiles="CartoDB positron")

        # District boundaries
        for _, row in focus.iterrows():
            folium.GeoJson(
                row.geometry.__geo_interface__,
                style_function=lambda x: {
                    "fillColor": "none", "color": "#333333",
                    "weight": 2, "fillOpacity": 0,
                },
                tooltip=row["shapeName"],
            ).add_to(m)

        # Colour points by vulnerability class
        for _, row in combined.iterrows():
            cls = str(row.get("slfvi_class", "moderate"))
            colour = COLOUR_MAP.get(cls, "#ffffbf")
            radius = 4 if row.get("data_source") == "streetview_observed" else 3
            opacity = 0.9 if row.get("data_source") == "streetview_observed" else 0.6
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=radius,
                color=colour,
                fill=True,
                fill_color=colour,
                fill_opacity=opacity,
                weight=0,
                tooltip=(
                    f"District: {row.get('shapeName','')}<br>"
                    f"Road: {row.get('highway','')}<br>"
                    f"SLFVI: {row.get('slfvi', 0):.3f} ({cls})<br>"
                    f"Source: {row.get('data_source','')}"
                ),
            ).add_to(m)

        # Legend
        legend_html = """
        <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                    background:white;padding:12px;border-radius:6px;
                    box-shadow:2px 2px 6px rgba(0,0,0,0.3);font-size:12px">
          <b>Flood Vulnerability (SLFVI)</b><br>
          <i style="background:#1a9641;width:12px;height:12px;display:inline-block;border-radius:50%"></i> Very low (0–0.20)<br>
          <i style="background:#a6d96a;width:12px;height:12px;display:inline-block;border-radius:50%"></i> Low (0.20–0.40)<br>
          <i style="background:#ffffbf;width:12px;height:12px;display:inline-block;border-radius:50%;border:1px solid #ccc"></i> Moderate (0.40–0.60)<br>
          <i style="background:#fdae61;width:12px;height:12px;display:inline-block;border-radius:50%"></i> High (0.60–0.80)<br>
          <i style="background:#d7191c;width:12px;height:12px;display:inline-block;border-radius:50%"></i> Very high (0.80–1.0)<br>
          <hr style="margin:4px 0">
          <i style="background:#555;width:10px;height:10px;display:inline-block;border-radius:50%"></i> Larger = Street View observed<br>
          <i style="background:#aaa;width:8px;height:8px;display:inline-block;border-radius:50%"></i> Smaller = Predicted
        </div>"""
        m.get_root().html.add_child(folium.Element(legend_html))

        out_html = OUT_DIR / "district_vulnerability_map.html"
        m.save(str(out_html))
        logger.info(f"Interactive map → {out_html}")

    except ImportError:
        logger.warning("folium not installed — skipping HTML map")
        out_html = None

    # ── 11. Static map ────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes = axes.flatten()

        for ax, dist in zip(axes, FOCUS_DISTRICTS):
            sub = combined[combined["shapeName"] == dist].copy()
            dist_boundary = focus[focus["shapeName"] == dist]

            colours = sub["slfvi_class"].map(COLOUR_MAP).fillna("#ffffbf")

            # Background district polygon
            dist_boundary.plot(ax=ax, color="#f5f5f5", edgecolor="#333", linewidth=1.5)

            # Predicted points (smaller)
            pred = sub[sub.data_source == "predicted_interpolated"]
            if len(pred):
                ax.scatter(pred["longitude"], pred["latitude"],
                           c=pred["slfvi_class"].map(COLOUR_MAP).fillna("#ffffbf"),
                           s=8, alpha=0.5, linewidths=0, zorder=2)

            # Observed points (larger)
            obs = sub[sub.data_source == "streetview_observed"]
            if len(obs):
                ax.scatter(obs["longitude"], obs["latitude"],
                           c=obs["slfvi_class"].map(COLOUR_MAP).fillna("#ffffbf"),
                           s=30, alpha=0.9, linewidths=0.4,
                           edgecolors="#333", zorder=3)

            ax.set_title(f"{dist}\n"
                         f"n={len(sub):,}  mean SLFVI={sub['slfvi'].mean():.3f}  "
                         f"high+={100*(sub['slfvi']>=0.6).mean():.1f}%",
                         fontsize=10, pad=6)
            ax.set_xlabel("Longitude", fontsize=8)
            ax.set_ylabel("Latitude", fontsize=8)
            ax.tick_params(labelsize=7)

        patches = [mpatches.Patch(color=v, label=k) for k, v in COLOUR_MAP.items()]
        fig.legend(handles=patches, title="SLFVI class", loc="lower center",
                   ncol=5, fontsize=9, title_fontsize=9, bbox_to_anchor=(0.5, 0.01))
        fig.suptitle(
            "Street-Level Flood Vulnerability Index — Focus Districts, Accra",
            fontsize=13, y=1.01
        )
        plt.tight_layout(rect=[0, 0.04, 1, 1])

        out_png = OUT_DIR / "district_vulnerability_map_static.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Static map → {out_png}")

    except Exception as e:
        logger.warning(f"Static map skipped: {e}")
        out_png = None

    # ── 12. Summary print ─────────────────────────────────────────────────────
    print("\n── Focus District Coverage ─────────────────────────────────")
    print(f"  {'District':<35} {'Total':>7} {'Observed':>9} {'Predicted':>10} {'SLFVI':>7} {'High%':>6}")
    print(f"  {'-'*75}")
    for _, r in summary_df.iterrows():
        print(f"  {r['district']:<35} {r['total_points']:>7,} "
              f"{r['observed_points']:>9,} {r['predicted_points']:>10,} "
              f"{r['mean_slfvi']:>7.3f} {r['high_very_high_pct']:>5.1f}%")

    print(f"\n── Output Files ────────────────────────────────────────────")
    for p in [out_parq, out_gpkg,
              OUT_DIR / "district_municipal_summary.csv",
              OUT_DIR / "district_vulnerability_map.html",
              OUT_DIR / "district_vulnerability_map_static.png"]:
        if p and p.exists():
            kb = p.stat().st_size / 1024
            tag = f"{kb/1024:.1f} MB" if kb > 1024 else f"{kb:.1f} KB"
            print(f"  {p.relative_to(ROOT)!s:<58} {tag}")


if __name__ == "__main__":
    main()
