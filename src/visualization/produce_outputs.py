"""Produce all Task 10 outputs — maps, figures, tables, dashboard — Task 10.

Outputs
-------
outputs/maps/vulnerability_map.html        — interactive Folium map
outputs/maps/vulnerability_map_static.png  — static PNG for manuscript
outputs/figures/shap_importance.png        — SHAP feature importance chart
outputs/figures/slfvi_distribution.png     — vulnerability class distribution
outputs/figures/vulnerability_by_highway.png
outputs/tables/municipal_summary.csv       — district-level vulnerability profile
outputs/tables/top100_maintenance.csv      — ranked drainage priority segments
outputs/dashboard/vulnerability.geojson    — dashboard-ready GeoJSON
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from loguru import logger

# ── Paths ─────────────────────────────────────────────────────────────────────
VULN_GPKG   = ROOT / "data/processed/vulnerability_index.gpkg"
VULN_PARQ   = ROOT / "data/processed/vulnerability_index.parquet"
SHAP_CSV    = ROOT / "outputs/tables/shap_feature_importance.csv"
ADM2        = ROOT / "data/external/GHA_ADM2.geojson"
PRIORITY    = ROOT / "outputs/tables/drainage_maintenance_priority.csv"

OUT_MAP_HTML   = ROOT / "outputs/maps/vulnerability_map.html"
OUT_MAP_PNG    = ROOT / "outputs/maps/vulnerability_map_static.png"
OUT_SHAP_FIG   = ROOT / "outputs/figures/shap_importance.png"
OUT_DIST_FIG   = ROOT / "outputs/figures/slfvi_distribution.png"
OUT_HWY_FIG    = ROOT / "outputs/figures/vulnerability_by_highway.png"
OUT_MUNI_CSV   = ROOT / "outputs/tables/municipal_summary.csv"
OUT_GEOJSON    = ROOT / "outputs/dashboard/vulnerability.geojson"

# Ensure output dirs exist
for d in [OUT_MAP_HTML.parent, OUT_SHAP_FIG.parent, OUT_MUNI_CSV.parent,
          OUT_GEOJSON.parent]:
    d.mkdir(parents=True, exist_ok=True)

# Colour palette (green → yellow → orange → red → dark red)
VULN_COLORS = {
    "very_low":  "#1a9641",
    "low":       "#a6d96a",
    "moderate":  "#ffffbf",
    "high":      "#fdae61",
    "very_high": "#d7191c",
}
CLASS_ORDER = ["very_low", "low", "moderate", "high", "very_high"]


# ── 1. Interactive Folium map ─────────────────────────────────────────────────
def make_folium_map(gdf: gpd.GeoDataFrame) -> None:
    try:
        import folium
        from folium.plugins import MarkerCluster, HeatMap
    except ImportError:
        logger.warning("folium not installed — skipping interactive map")
        return

    center = [gdf.geometry.y.mean(), gdf.geometry.x.mean()]
    m = folium.Map(location=center, zoom_start=12,
                   tiles="CartoDB positron")

    # Choropleth circle markers coloured by vulnerability class
    vc_col = "vulnerability_class_final"
    for _, row in gdf.iterrows():
        vc  = str(row.get(vc_col, "low"))
        col = VULN_COLORS.get(vc, "#aaaaaa")
        slfvi = row.get("slfvi_final", 0)
        popup_html = f"""
        <b>Point {int(row['point_id'])}</b><br>
        Road: {row.get('highway','')}<br>
        SLFVI: {slfvi:.3f}<br>
        Class: <span style='color:{col};font-weight:bold'>{vc}</span><br>
        Stratum: {row.get('flood_stratum','')}<br>
        Elevation: {row.get('elevation_m', 0):.1f} m<br>
        Buildings (100m): {int(row.get('building_count_100m', 0))}<br>
        Flood-exposed bldgs: {int(row.get('flood_exposed_count_100m', 0))}
        """
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=4 + slfvi * 6,
            color=col,
            fill=True,
            fill_color=col,
            fill_opacity=0.75,
            weight=0.5,
            popup=folium.Popup(popup_html, max_width=250),
        ).add_to(m)

    # Legend
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:12px;border-radius:8px;
                border:1px solid #ccc;font-size:13px;line-height:1.8">
    <b>Street-Level Flood Vulnerability Index</b><br>
    """
    labels = {"very_low": "Very Low", "low": "Low", "moderate": "Moderate",
              "high": "High", "very_high": "Very High"}
    for vc in CLASS_ORDER:
        legend_html += (
            f'<span style="background:{VULN_COLORS[vc]};display:inline-block;'
            f'width:14px;height:14px;margin-right:6px;border:1px solid #999"></span>'
            f'{labels[vc]}<br>'
        )
    legend_html += "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(str(OUT_MAP_HTML))
    logger.info(f"Interactive map → {OUT_MAP_HTML}")


# ── 2. Static vulnerability map (matplotlib) ─────────────────────────────────
def make_static_map(gdf: gpd.GeoDataFrame) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_facecolor("#f0f4f8")
    fig.patch.set_facecolor("#ffffff")

    # District boundaries
    if ADM2.exists():
        adm = gpd.read_file(ADM2)
        # Clip to study area bbox
        bbox = gdf.total_bounds  # minx, miny, maxx, maxy
        pad  = 0.05
        adm_clip = adm.cx[bbox[0]-pad:bbox[2]+pad, bbox[1]-pad:bbox[3]+pad]
        adm_clip.boundary.plot(ax=ax, color="#bbbbbb", linewidth=0.5, zorder=1)

    # Points coloured by vulnerability class
    vc_col = "vulnerability_class_final"
    for vc in CLASS_ORDER:
        subset = gdf[gdf[vc_col].astype(str) == vc]
        if len(subset) == 0:
            continue
        subset.plot(ax=ax, color=VULN_COLORS[vc], markersize=3,
                    alpha=0.8, zorder=2 + CLASS_ORDER.index(vc))

    # Legend
    patches = [
        mpatches.Patch(color=VULN_COLORS[vc],
                       label=vc.replace("_", " ").title())
        for vc in CLASS_ORDER
    ]
    ax.legend(handles=patches, title="Vulnerability Class",
              loc="lower right", fontsize=9, title_fontsize=10,
              framealpha=0.9)

    ax.set_title("Street-Level Flood Vulnerability Index — Accra, Ghana",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    ax.tick_params(labelsize=8)

    # North arrow
    ax.annotate("N", xy=(0.97, 0.94), xycoords="axes fraction",
                fontsize=14, ha="center", va="bottom", fontweight="bold")
    ax.annotate("▲", xy=(0.97, 0.91), xycoords="axes fraction",
                fontsize=16, ha="center", va="top")

    plt.tight_layout()
    plt.savefig(OUT_MAP_PNG, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Static map → {OUT_MAP_PNG}")


# ── 3. SHAP feature importance chart ─────────────────────────────────────────
def make_shap_figure() -> None:
    if not SHAP_CSV.exists():
        logger.warning("SHAP CSV not found — skipping")
        return

    df = pd.read_csv(SHAP_CSV).head(15)
    df["feature_label"] = df["feature"].str.replace(
        r"^(clip_|seg_)", "", regex=True
    ).str.replace("_", " ").str.title()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(df)))[::-1]
    bars = ax.barh(df["feature_label"][::-1], df["shap_importance"][::-1],
                   color=colors[::-1], edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
    ax.set_title("Feature Importance — LightGBM Flood Vulnerability Model\n"
                 "(Spatial Cross-Validation, Accra, Ghana)",
                 fontsize=12, fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    for bar, val in zip(bars, df["shap_importance"][::-1]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_SHAP_FIG, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"SHAP figure → {OUT_SHAP_FIG}")


# ── 4. SLFVI distribution chart ───────────────────────────────────────────────
def make_distribution_figure(df: pd.DataFrame) -> None:
    vc_col = "vulnerability_class_final"
    counts = df[vc_col].astype(str).value_counts().reindex(CLASS_ORDER).fillna(0)
    pcts   = 100 * counts / counts.sum()
    labels = [c.replace("_", " ").title() for c in CLASS_ORDER]
    colors = [VULN_COLORS[c] for c in CLASS_ORDER]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Bar chart
    ax = axes[0]
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_ylabel("Number of Road Points", fontsize=11)
    ax.set_title("SLFVI Vulnerability Class Distribution", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f"{pct:.1f}%", ha="center", fontsize=9)
    ax.tick_params(axis="x", rotation=15)

    # High + very_high by flood stratum
    ax2 = axes[1]
    high_mask = df[vc_col].astype(str).isin(["high", "very_high"])
    strat_high = df[high_mask].groupby("flood_stratum").size()
    strat_all  = df.groupby("flood_stratum").size()
    strat_pct  = (100 * strat_high / strat_all).fillna(0)
    strat_pct.plot(kind="bar", ax=ax2, color=["#d7191c", "#fdae61"][:len(strat_pct)],
                   edgecolor="white")
    ax2.set_ylabel("% Points in High/Very-High Class", fontsize=11)
    ax2.set_title("High Vulnerability Rate by Flood Stratum", fontsize=12, fontweight="bold")
    ax2.set_xlabel("")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(axis="x", rotation=10)
    for bar in ax2.patches:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{bar.get_height():.1f}%", ha="center", fontsize=9)

    plt.suptitle("Street-Level Flood Vulnerability Index — Accra Pilot Study",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIST_FIG, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Distribution figure → {OUT_DIST_FIG}")


# ── 5. Vulnerability by highway type ─────────────────────────────────────────
def make_highway_figure(df: pd.DataFrame) -> None:
    vc_col = "vulnerability_class_final"
    hw_order = ["motorway", "trunk", "primary", "secondary", "tertiary",
                "unclassified", "residential", "service"]
    df_hw = df[df["highway"].isin(hw_order)].copy()

    pivot = (
        df_hw.groupby(["highway", vc_col])
        .size()
        .unstack(fill_value=0)
    )
    # Normalise to percentages
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    pivot_pct = pivot_pct.reindex(
        [h for h in hw_order if h in pivot_pct.index]
    )
    pivot_pct = pivot_pct.reindex(columns=CLASS_ORDER, fill_value=0)

    fig, ax = plt.subplots(figsize=(11, 6))
    bottom = np.zeros(len(pivot_pct))
    for vc in CLASS_ORDER:
        if vc in pivot_pct.columns:
            vals = pivot_pct[vc].values
            ax.bar(pivot_pct.index, vals, bottom=bottom,
                   color=VULN_COLORS[vc], label=vc.replace("_", " ").title(),
                   edgecolor="white", linewidth=0.5)
            bottom += vals

    ax.set_ylabel("% of Road Points", fontsize=11)
    ax.set_title("Flood Vulnerability Distribution by Road Type — Accra",
                 fontsize=12, fontweight="bold")
    ax.legend(title="Vulnerability Class", bbox_to_anchor=(1.01, 1),
              loc="upper left", fontsize=9)
    ax.tick_params(axis="x", rotation=20)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(OUT_HWY_FIG, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Highway figure → {OUT_HWY_FIG}")


# ── 6. Municipal summary table ────────────────────────────────────────────────
def make_municipal_summary(gdf: gpd.GeoDataFrame) -> None:
    if not ADM2.exists():
        logger.warning("ADM2 boundaries not found — skipping municipal summary")
        return

    adm = gpd.read_file(ADM2).to_crs("EPSG:4326")
    name_col = next(
        (c for c in ["ADM2_EN", "shapeName", "ADM2_NAME", "name"] if c in adm.columns),
        adm.columns[0],
    )
    joined = gpd.sjoin(gdf, adm[[name_col, "geometry"]], how="left",
                       predicate="within")

    vc_col = "vulnerability_class_final"
    grp = joined.groupby(name_col)

    summary = pd.DataFrame({
        "n_points":           grp.size(),
        "mean_slfvi":         grp["slfvi_final"].mean().round(3),
        "high_very_high_pct": (
            grp.apply(lambda g: 100 * g[vc_col].astype(str)
                      .isin(["high","very_high"]).mean()).round(1)
        ),
        "mean_drain_obstruction": grp["drain_obstruction_score"].mean().round(3),
        "mean_elevation_m":       grp["elevation_m"].mean().round(1),
        "mean_flood_exposed_100m":grp["flood_exposed_count_100m"].mean().round(1),
        "top_vulnerability_class":(
            grp[vc_col].apply(
                lambda s: s.astype(str).value_counts().idxmax()
            )
        ),
    }).sort_values("mean_slfvi", ascending=False).reset_index()
    summary.index += 1
    summary.index.name = "rank"
    summary.to_csv(OUT_MUNI_CSV)
    logger.info(f"Municipal summary → {OUT_MUNI_CSV}  ({len(summary)} districts)")

    print("\n── Municipal Vulnerability Ranking ─────────────────────────")
    print(f"  {'District':<30} {'SLFVI':>6} {'High%':>6} {'Points':>7}")
    print(f"  {'-'*54}")
    dist_col = summary.columns[0]
    for _, row in summary.head(15).iterrows():
        print(f"  {str(row[dist_col]):<30} {row['mean_slfvi']:>6.3f} "
              f"{row['high_very_high_pct']:>5.1f}% {int(row['n_points']):>7,}")


# ── 7. Dashboard GeoJSON ──────────────────────────────────────────────────────
def make_geojson(gdf: gpd.GeoDataFrame) -> None:
    cols = [
        "point_id", "highway", "flood_stratum",
        "slfvi_final", "vulnerability_class_final",
        "H", "E", "S", "A",
        "ml_flood_prob", "maintenance_priority_score",
        "drain_obstruction_score", "waste_risk_score",
        "elevation_m", "building_count_100m",
        "flood_exposed_count_100m", "geometry",
    ]
    cols = [c for c in cols if c in gdf.columns]
    export = gdf[cols].copy()
    export["slfvi_final"] = export["slfvi_final"].round(4)
    export["ml_flood_prob"] = export["ml_flood_prob"].round(4)
    export.to_file(OUT_GEOJSON, driver="GeoJSON")
    size_mb = OUT_GEOJSON.stat().st_size / (1024*1024)
    logger.info(f"Dashboard GeoJSON → {OUT_GEOJSON}  ({size_mb:.1f} MB)")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    logger.info("Loading vulnerability index …")
    gdf = gpd.read_file(VULN_GPKG)
    df  = pd.read_parquet(VULN_PARQ)
    logger.info(f"  {len(gdf):,} points loaded")

    logger.info("1/7  Interactive Folium map …")
    make_folium_map(gdf)

    logger.info("2/7  Static PNG map …")
    make_static_map(gdf)

    logger.info("3/7  SHAP importance figure …")
    make_shap_figure()

    logger.info("4/7  SLFVI distribution figure …")
    make_distribution_figure(df)

    logger.info("5/7  Highway vulnerability figure …")
    make_highway_figure(df)

    logger.info("6/7  Municipal summary table …")
    make_municipal_summary(gdf)

    logger.info("7/7  Dashboard GeoJSON …")
    make_geojson(gdf)

    print("\n── All Task 10 Outputs ──────────────────────────────────────")
    outputs = [
        OUT_MAP_HTML, OUT_MAP_PNG, OUT_SHAP_FIG,
        OUT_DIST_FIG, OUT_HWY_FIG, OUT_MUNI_CSV, OUT_GEOJSON,
    ]
    for p in outputs:
        if p.exists():
            kb = p.stat().st_size / 1024
            tag = "MB" if kb > 1000 else "KB"
            val = kb/1024 if kb > 1000 else kb
            print(f"  {p.relative_to(ROOT)!s:<55} {val:>6.1f} {tag}")
        else:
            print(f"  {p.relative_to(ROOT)!s:<55}  (not produced)")


if __name__ == "__main__":
    main()
