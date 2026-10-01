"""Map generation utilities: Folium, Plotly, and static Matplotlib maps."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


SLFVI_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "slfvi", ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]
)

CLASS_COLORS = {
    "very_low":  "#2ecc71",
    "low":       "#f1c40f",
    "moderate":  "#e67e22",
    "high":      "#e74c3c",
    "very_high": "#8e44ad",
}


def plot_vulnerability_map(
    gdf: gpd.GeoDataFrame,
    score_col: str = "slfvi",
    title: str = "Street-Level Flood Vulnerability Index — Accra",
    output_path: Path | None = None,
    figsize: tuple = (14, 10),
) -> plt.Figure:
    """Static choropleth map of SLFVI scores."""
    fig, ax = plt.subplots(figsize=figsize)
    gdf_wgs = gdf.to_crs("EPSG:4326")
    gdf_wgs.plot(
        column=score_col,
        cmap=SLFVI_CMAP,
        linewidth=0.5,
        ax=ax,
        legend=True,
        vmin=0,
        vmax=1,
        legend_kwds={"label": "SLFVI (0–1)", "shrink": 0.6},
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_axis_off()
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def export_folium_map(
    gdf: gpd.GeoDataFrame,
    score_col: str = "slfvi",
    output_path: Path | None = None,
) -> "folium.Map":
    """Interactive Folium choropleth map."""
    import folium
    from folium.plugins import HeatMap

    gdf_wgs = gdf.to_crs("EPSG:4326")
    center = [gdf_wgs.geometry.centroid.y.mean(), gdf_wgs.geometry.centroid.x.mean()]
    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")

    heat_data = [
        [row.geometry.centroid.y, row.geometry.centroid.x, row[score_col]]
        for _, row in gdf_wgs.iterrows()
        if row[score_col] is not None
    ]
    HeatMap(heat_data, radius=12, blur=10, max_zoom=16).add_to(m)

    if output_path:
        m.save(str(output_path))
    return m
