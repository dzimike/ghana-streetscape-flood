"""Generate supplementary figures for Papers 2 and 3.

Paper 2:
  fig2_paper2_coverage_map.png       — covered vs gap points across Accra (basemap)
  fig3_paper2_gap_rates.png          — gap rate by road class (bar chart)
  fig4_paper2_geo_comparison.png     — geospatial property comparison (covered vs gap)

Paper 3:
  fig4_paper3_spatial_archetypes.png — spatial map of 6 CLIP-based archetypes (basemap)
  fig5_paper3_vuln_by_archetype.png  — sensitivity score distributions by archetype
  typology_cluster_heatmap_k6.png    — CLIP label heatmap across 6 archetypes
"""
from __future__ import annotations
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import contextily as ctx

# ── Load local district boundaries (offline basemap) ──────────────────────────
_GHA_ADM2 = Path(__file__).resolve().parents[2] / "data/external/GHA_ADM2.geojson"
_ACCRA_BBOX = dict(minx=-0.60, maxx=0.50, miny=5.30, maxy=6.00)

def _load_accra_districts() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(_GHA_ADM2)
    bb  = gdf.bounds
    mask = ((bb.minx > _ACCRA_BBOX["minx"]) & (bb.maxx < _ACCRA_BBOX["maxx"]) &
            (bb.miny > _ACCRA_BBOX["miny"]) & (bb.maxy < _ACCRA_BBOX["maxy"]))
    return gdf[mask].copy()

_districts = _load_accra_districts()

# ── Publication style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        9,
    "axes.titlesize":   9,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.facecolor":   "white",
    "figure.facecolor": "white",
    "axes.grid":        True,
    "grid.alpha":       0.2,
    "grid.linestyle":   "--",
})

ROOT    = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "outputs/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
meta  = pd.read_parquet(ROOT / "data/interim/streetview_metadata.parquet")
geo   = pd.read_parquet(ROOT / "data/processed/geospatial_features.parquet")
c6    = pd.read_csv(ROOT / "outputs/tables/streetscape_typology_clusters_k6.csv")

meta["covered"] = (meta["status"] == "OK").astype(int)
geo2 = geo.drop(columns=["highway", "flood_stratum", "dist_drain_m"], errors="ignore")

# Merge coordinates into cluster table (join on point_id)
seg = meta[meta["covered"] == 1][["point_id", "pano_lat", "pano_lon",
                                   "highway", "flood_stratum"]].copy()
c6_geo = c6.merge(seg, on="point_id", how="left")

# ── CLIP-based archetype palette ───────────────────────────────────────────────
# Cluster IDs sorted by mean streetscape_sensitivity_score, high → low:
# [2 (0.687), 5 (0.504), 6 (0.434), 1 (0.383), 4 (0.290), 3 (0.244)]
ARCHETYPE_ORDER = [2, 5, 6, 1, 4, 3]

ARCHETYPE_LABELS = {
    2: "Type I   Low-Lying Flood Accumulation Zone",
    5: "Type II  Informal Settlement Drainage Deficit",
    6: "Type III Watercourse-Adjacent Corridor",
    1: "Type IV  Mixed Residential Moderate Risk",
    4: "Type V   Waste-Laden Impervious Corridor",
    3: "Type VI  Low-Risk Formal Network",
}
ARCHETYPE_COLORS = {
    2: "#d73027",   # deep red      Type I
    5: "#fc8d59",   # orange        Type II
    6: "#fee090",   # yellow        Type III
    1: "#91bfdb",   # light blue    Type IV
    4: "#4575b4",   # medium blue   Type V
    3: "#313695",   # deep blue     Type VI
}
ARCHETYPE_NAMES_SHORT = {
    2: "Type I\nLow-Lying Flood\nAccumulation Zone\n(n=190)",
    5: "Type II\nInformal Settlement\nDrainage Deficit\n(n=816)",
    6: "Type III\nWatercourse-\nAdjacent Corridor\n(n=560)",
    1: "Type IV\nMixed Residential\nModerate Risk\n(n=1,076)",
    4: "Type V\nWaste-Laden\nImpervious Corridor\n(n=562)",
    3: "Type VI\nLow-Risk\nFormal Network\n(n=632)",
}


def _add_district_basemap(ax: plt.Axes) -> None:
    """Plot Accra district polygons as a clean offline basemap."""
    _districts.plot(ax=ax, color="#f0f0f0", edgecolor="#b0b0b0", linewidth=0.5,
                    zorder=1)


def _fmt_axes(ax: plt.Axes) -> None:
    """Format lat/lon axes with degree labels."""
    import matplotlib.ticker as mticker

    def lon_fmt(x, _):
        if x < 0:
            return f"{abs(x):.2f}°W"
        return f"{x:.2f}°E"

    def lat_fmt(y, _):
        if y < 0:
            return f"{abs(y):.2f}°S"
        return f"{y:.2f}°N"

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lon_fmt))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lat_fmt))
    ax.tick_params(labelsize=7.5)


# ══════════════════════════════════════════════════════════════════════════════
# PAPER 2 — Figure 2: Coverage map with basemap
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig2_paper2_coverage_map.png …")

gap = meta[meta["covered"] == 0]
cov = meta[meta["covered"] == 1]

fig, ax = plt.subplots(figsize=(9, 8))
_add_district_basemap(ax)

ax.scatter(cov["query_lon"], cov["query_lat"],
           s=2, alpha=0.45, color="#2166ac", linewidths=0,
           label=f"Covered (n={len(cov):,})", zorder=3)
ax.scatter(gap["query_lon"], gap["query_lat"],
           s=5, alpha=0.85, color="#d73027", linewidths=0,
           label=f"No coverage (n={len(gap):,})", zorder=4)

_fmt_axes(ax)
ax.set_xlabel("Longitude", fontsize=9)
ax.set_ylabel("Latitude", fontsize=9)

# Zoom to data extent + 10% padding
buf = 0.04
ax.set_xlim(meta["query_lon"].min() - buf, meta["query_lon"].max() + buf)
ax.set_ylim(meta["query_lat"].min() - buf, meta["query_lat"].max() + buf)

legend = ax.legend(fontsize=9, markerscale=3, framealpha=0.95,
                   loc="lower left", edgecolor="#cccccc", fancybox=False)
legend.get_frame().set_linewidth(0.6)

plt.tight_layout()
out = FIG_DIR / "fig2_paper2_coverage_map.png"
plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  Saved → {out}  ({out.stat().st_size//1024} KB)")


# ══════════════════════════════════════════════════════════════════════════════
# PAPER 2 — Figure 3: Gap rate by road class
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig3_paper2_gap_rates.png …")

hwy = (meta.groupby("highway")
           .agg(n=("covered", "count"), covered=("covered", "sum"))
           .assign(gap_rate=lambda d: (1 - d["covered"] / d["n"]) * 100)
           .sort_values("gap_rate", ascending=False))

fig, ax = plt.subplots(figsize=(9, 5))

bar_colors = ["#d73027" if r > 15 else "#fc8d59" if r > 5 else "#4575b4"
              for r in hwy["gap_rate"]]
bars = ax.barh(hwy.index, hwy["gap_rate"], color=bar_colors, edgecolor="white",
               height=0.65)

for bar, (_, row) in zip(bars, hwy.iterrows()):
    w = bar.get_width()
    ax.text(w + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{w:.1f}%  (n={int(row['n']):,})",
            va="center", fontsize=8, color="#333333")

ax.set_xlabel("Gap rate (%)", fontsize=9)
ax.set_xlim(0, 42)
overall_gap = (1 - meta["covered"].mean()) * 100
ax.axvline(overall_gap, color="#555555", ls="--", lw=1.2, alpha=0.7,
           label=f"Overall mean ({overall_gap:.1f}%)")
ax.legend(fontsize=9)
ax.invert_yaxis()

plt.tight_layout()
out = FIG_DIR / "fig3_paper2_gap_rates.png"
plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  Saved → {out}  ({out.stat().st_size//1024} KB)")


# ══════════════════════════════════════════════════════════════════════════════
# PAPER 2 — Figure 4: Geospatial property comparison (4 panels)
# Population density dropped: extreme outliers and negligible effect (d=0.042)
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig4_paper2_geo_comparison.png …")

cov_geo = meta[meta["covered"] == 1].merge(geo2, on="point_id", how="left")
gap_geo = meta[meta["covered"] == 0].merge(geo2, on="point_id", how="left")

features = [
    ("elevation_m",         "Elevation (m)",             False),
    ("slope_deg",           "Slope (°)",                 False),
    ("building_count_100m", "Building count (100 m)",    False),
    ("dist_waterway_m",     "Distance to waterway (m)",  True),
]

fig, axes = plt.subplots(1, 4, figsize=(12, 4.5))

BLUE = "#2166ac"
RED  = "#d73027"

for ax, (feat, label, key_feat) in zip(axes, features):
    c = cov_geo[feat].dropna()
    g = gap_geo[feat].dropna()

    pooled = np.sqrt((c.std() ** 2 + g.std() ** 2) / 2)
    d = abs((g.mean() - c.mean()) / max(pooled, 1e-9))
    effect_tag = ("large" if d >= 0.8 else "medium" if d >= 0.5
                  else "small" if d >= 0.2 else "negligible")

    bp = ax.violinplot(
        [c.values, g.values],
        positions=[1, 2],
        widths=0.55,
        showmedians=True,
        showextrema=False,
    )
    bp["bodies"][0].set_facecolor(BLUE);  bp["bodies"][0].set_alpha(0.70)
    bp["bodies"][1].set_facecolor(RED);   bp["bodies"][1].set_alpha(0.70)
    for body in bp["bodies"]:
        body.set_edgecolor("#333333");  body.set_linewidth(0.6)
    bp["cmedians"].set_color("black");    bp["cmedians"].set_linewidth(2)

    title_color = RED if key_feat else "#333333"
    ax.set_title(f"d = {d:.3f}  ({effect_tag})", fontsize=8, color=title_color,
                 fontweight="bold" if key_feat else "normal")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Covered", "Gap"], fontsize=8)
    ax.set_ylabel(label, fontsize=8.5)

legend_handles = [
    mpatches.Patch(facecolor=BLUE, alpha=0.70, label="Covered (n=3,912)"),
    mpatches.Patch(facecolor=RED,  alpha=0.70, label="No coverage (n=1,088)"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=2,
           fontsize=9, bbox_to_anchor=(0.5, -0.06), frameon=False)

plt.tight_layout()
out = FIG_DIR / "fig4_paper2_geo_comparison.png"
plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  Saved → {out}  ({out.stat().st_size//1024} KB)")


# ══════════════════════════════════════════════════════════════════════════════
# PAPER 3 — Figure 4: Spatial archetype map with basemap
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig4_paper3_spatial_archetypes.png …")

valid = c6_geo.dropna(subset=["pano_lon", "pano_lat"])

fig, ax = plt.subplots(figsize=(10, 9))
_add_district_basemap(ax)

# Plot low-vulnerability types first so high-vulnerability sits on top
for cid in reversed(ARCHETYPE_ORDER):
    subset = valid[valid["cluster"] == cid]
    if subset.empty:
        continue
    ax.scatter(subset["pano_lon"], subset["pano_lat"],
               s=4, alpha=0.72, linewidths=0,
               color=ARCHETYPE_COLORS[cid],
               label=ARCHETYPE_LABELS[cid],
               zorder=ARCHETYPE_ORDER.index(cid) + 2)

_fmt_axes(ax)
ax.set_xlabel("Longitude", fontsize=9)
ax.set_ylabel("Latitude", fontsize=9)

# Zoom to data extent + 10% padding
buf = 0.04
ax.set_xlim(valid["pano_lon"].min() - buf, valid["pano_lon"].max() + buf)
ax.set_ylim(valid["pano_lat"].min() - buf, valid["pano_lat"].max() + buf)

# Legend: reverse so highest-risk (Type I) appears first
handles, labels = ax.get_legend_handles_labels()
legend = ax.legend(
    handles[::-1], labels[::-1],
    fontsize=8,
    markerscale=2.5,
    framealpha=0.95,
    title="Archetype  (N = 3,836 segments)",
    title_fontsize=8.5,
    loc="lower right",
    edgecolor="#cccccc",
    fancybox=False,
)
legend.get_frame().set_linewidth(0.6)

plt.tight_layout()
out = FIG_DIR / "fig4_paper3_spatial_archetypes.png"
plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  Saved → {out}  ({out.stat().st_size//1024} KB)")


# ══════════════════════════════════════════════════════════════════════════════
# PAPER 3 — Figure 5: CLIP sensitivity score distributions by archetype
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig5_paper3_vuln_by_archetype.png …")

VULN_COL = "streetscape_sensitivity_score"

fig, ax = plt.subplots(figsize=(13, 6))
fig.subplots_adjust(bottom=0.30)

positions  = list(range(1, 7))
vuln_data  = [c6[c6["cluster"] == cid][VULN_COL].values for cid in ARCHETYPE_ORDER]

vp = ax.violinplot(vuln_data, positions=positions, widths=0.68,
                   showmedians=True, showextrema=False)

for body, cid in zip(vp["bodies"], ARCHETYPE_ORDER):
    body.set_facecolor(ARCHETYPE_COLORS[cid])
    body.set_alpha(0.80)
    body.set_edgecolor("#333333")
    body.set_linewidth(0.8)

vp["cmedians"].set_color("black")
vp["cmedians"].set_linewidth(2.5)
vp["cmedians"].set_zorder(6)

# IQR box overlay
for cid, pos in zip(ARCHETYPE_ORDER, positions):
    vals = c6[c6["cluster"] == cid][VULN_COL]
    q1, q3 = np.percentile(vals, [25, 75])
    ax.add_patch(plt.Rectangle(
        (pos - 0.10, q1), 0.20, q3 - q1,
        facecolor="white", edgecolor="#333333",
        linewidth=0.9, zorder=5, alpha=0.70,
    ))

# Mean diamond + annotation
for cid, pos in zip(ARCHETYPE_ORDER, positions):
    vals = c6[c6["cluster"] == cid][VULN_COL]
    mean_v = vals.mean()
    med_v  = np.median(vals.values)

    ax.scatter(pos, mean_v, s=80, color="white",
               edgecolors=ARCHETYPE_COLORS[cid], linewidths=2.0,
               zorder=7, marker="D")
    ax.annotate(
        f"μ={mean_v:.3f}",
        xy=(pos, mean_v), xytext=(pos, mean_v + 0.040),
        ha="center", va="bottom", fontsize=7.5, fontweight="bold",
        color="#111111", zorder=8,
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                  edgecolor="#cccccc", linewidth=0.6, alpha=0.92),
        arrowprops=dict(arrowstyle="-", color="#888888", lw=0.7),
    )
    ax.text(pos + 0.13, med_v, f"  m={med_v:.3f}",
            va="center", ha="left", fontsize=6.5, color="#444444", zorder=8)

ax.set_xticks(positions)
ax.set_xticklabels([ARCHETYPE_NAMES_SHORT[cid] for cid in ARCHETYPE_ORDER],
                   fontsize=8.0, linespacing=1.4)
ax.set_ylabel("CLIP-derived streetscape sensitivity score", fontsize=9)
ax.set_ylim(0.0, 0.90)

ax.axhline(0.5, color="#666666", ls=":", lw=1.0, alpha=0.7, zorder=1)
ax.text(0.55, 0.503, "0.50", va="bottom", ha="left",
        fontsize=7, color="#666666", style="italic")

ax.spines[["top", "right"]].set_visible(False)

legend_handles = [
    Line2D([0], [0], color="black", lw=2.5, label="Median"),
    plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#333333",
                  linewidth=0.9, alpha=0.70, label="IQR (Q1–Q3)"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor="white",
           markeredgecolor="#666666", markeredgewidth=2, markersize=8,
           label="Mean (μ, annotated)"),
]
ax.legend(handles=legend_handles, fontsize=8, loc="upper right",
          framealpha=0.93, edgecolor="#cccccc", fancybox=False)

plt.tight_layout()
out = FIG_DIR / "fig5_paper3_vuln_by_archetype.png"
plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  Saved → {out}  ({out.stat().st_size//1024} KB)")


# ══════════════════════════════════════════════════════════════════════════════
# PAPER 3 — Heatmap: CLIP label means across 6 archetypes
# ══════════════════════════════════════════════════════════════════════════════
print("Regenerating typology_cluster_heatmap_k6.png …")

profiles = pd.read_csv(ROOT / "outputs/tables/typology_cluster_profiles_k6.csv",
                        index_col=0)
LABEL_COLS = [
    "clip_visible_drain_present", "clip_open_gutter_present",
    "clip_blocked_drain_present", "clip_stagnant_water_visible",
    "clip_poor_road_condition", "clip_heavy_impervious_surface",
    "clip_unpaved_shoulder", "clip_informal_structure_near_drainage",
    "clip_solid_waste_accumulation", "clip_visible_waterway_or_stream",
    "clip_low_lying_street_form", "clip_roadside_erosion",
    "clip_pedestrian_exposure", "clip_culvert_or_bridge_visible",
    "clip_no_visible_drainage",
]
SHORT = {c: c.replace("clip_", "").replace("_", " ") for c in LABEL_COLS}

profiles_ord = profiles.loc[ARCHETYPE_ORDER]
row_labels   = [ARCHETYPE_LABELS[int(c)] for c in profiles_ord.index]

fig2, ax2 = plt.subplots(figsize=(14, 4.5))
mat  = profiles_ord[LABEL_COLS].values
im   = ax2.imshow(mat, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1)

ax2.set_xticks(range(len(LABEL_COLS)))
ax2.set_xticklabels([SHORT[c] for c in LABEL_COLS], rotation=40, ha="right", fontsize=8)
ax2.set_yticks(range(len(ARCHETYPE_ORDER)))
ax2.set_yticklabels(row_labels, fontsize=8.5)

for i in range(len(ARCHETYPE_ORDER)):
    for j in range(len(LABEL_COLS)):
        val = mat[i, j]
        txt_color = "white" if val > 0.65 or val < 0.25 else "#222222"
        ax2.text(j, i, f"{val:.2f}", ha="center", va="center",
                 fontsize=6.5, color=txt_color)

plt.colorbar(im, ax=ax2, label="Mean CLIP probability", shrink=0.85)
ax2.spines[:].set_visible(False)
plt.tight_layout()
out2 = FIG_DIR / "typology_cluster_heatmap_k6.png"
plt.savefig(out2, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  Saved → {out2}  ({out2.stat().st_size//1024} KB)")

print("\nAll figures done.")
