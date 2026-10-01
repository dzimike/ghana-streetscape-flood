"""Generate supplementary figures for Papers 2 and 3.

Paper 2:
  fig2_paper2_coverage_map.png  — covered vs gap points across Accra
  fig3_paper2_gap_rates.png     — gap rate by road class (bar chart)
  fig4_paper2_geo_comparison.png — geospatial property comparison (covered vs gap)

Paper 3:
  fig2_paper3_cluster_validation.png — already exists; copy reference only
  fig3_paper3_heatmap.png            — already exists as typology_cluster_heatmap_k6.png
  fig4_paper3_spatial_archetypes.png — spatial map of 6 archetypes
"""
from __future__ import annotations
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

ROOT    = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "outputs/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
meta  = pd.read_parquet(ROOT / "data/interim/streetview_metadata.parquet")
geo   = pd.read_parquet(ROOT / "data/processed/geospatial_features.parquet")
c6    = pd.read_csv(ROOT / "outputs/tables/streetscape_typology_clusters_k6.csv")

meta["covered"] = (meta["status"] == "OK").astype(int)
geo2 = geo.drop(columns=["highway", "flood_stratum", "dist_drain_m"], errors="ignore")

# Merge coordinates into cluster table
seg = meta[meta["covered"] == 1][["pano_id", "pano_lat", "pano_lon", "highway",
                                   "flood_stratum", "point_id"]].copy()
c6_geo = c6.merge(seg, on="pano_id", how="left")

# ── Archetype colour palette (Types I–VI) ─────────────────────────────────────
ARCHETYPE_ORDER = [4, 2, 1, 6, 3, 5]   # sorted by mean vuln (high → low)
ARCHETYPE_LABELS = {
    4: "Type I   Dense Impervious Drainage Void",
    2: "Type II  Compound Drainage Failure",
    1: "Type III Watercourse-Adjacent Mixed",
    6: "Type IV  Open Gutter Waste Corridor",
    3: "Type V   Residential Maintenance Deficit",
    5: "Type VI  Low-Risk Formal Corridor",
}
ARCHETYPE_COLORS = {
    4: "#d73027",   # deep red      Type I
    2: "#fc8d59",   # orange        Type II
    1: "#fee090",   # yellow        Type III
    6: "#91bfdb",   # light blue    Type IV
    3: "#4575b4",   # medium blue   Type V
    5: "#313695",   # deep blue     Type VI
}


# ══════════════════════════════════════════════════════════════════════════════
# PAPER 2 — Figure 2: Coverage map
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig2_paper2_coverage_map.png …")

fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor("white")

gap = meta[meta["covered"] == 0]
cov = meta[meta["covered"] == 1]

ax.scatter(cov["query_lon"], cov["query_lat"],
           s=3, alpha=0.45, color="#2166ac", linewidths=0, label=f"Covered (n={len(cov):,})")
ax.scatter(gap["query_lon"], gap["query_lat"],
           s=6, alpha=0.75, color="#d73027", linewidths=0, label=f"Gap (n={len(gap):,})")

ax.set_xlabel("Longitude", fontsize=10)
ax.set_ylabel("Latitude", fontsize=10)

legend = ax.legend(fontsize=9, markerscale=3, framealpha=0.9)
ax.grid(alpha=0.2)
ax.set_facecolor("#f8f8f8")

plt.tight_layout()
out = FIG_DIR / "fig2_paper2_coverage_map.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
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
fig.patch.set_facecolor("white")

bar_colors = ["#d73027" if r > 15 else "#fc8d59" if r > 5 else "#4575b4"
              for r in hwy["gap_rate"]]
bars = ax.barh(hwy.index, hwy["gap_rate"], color=bar_colors, edgecolor="white",
               height=0.65)

for bar, (_, row) in zip(bars, hwy.iterrows()):
    w = bar.get_width()
    ax.text(w + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{w:.1f}%  (n={int(row['n']):,})",
            va="center", fontsize=8.5, color="#333333")

ax.set_xlabel("Gap rate (%)", fontsize=10)
ax.set_xlim(0, 42)
ax.axvline(meta["covered"].apply(lambda x: 1 - x).mean() * 100,
           color="#555555", ls="--", lw=1.2, alpha=0.7, label="Overall mean (21.8%)")
ax.legend(fontsize=9)
ax.grid(axis="x", alpha=0.25)
ax.set_facecolor("#f8f8f8")
ax.invert_yaxis()

plt.tight_layout()
out = FIG_DIR / "fig3_paper2_gap_rates.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  Saved → {out}  ({out.stat().st_size//1024} KB)")


# ══════════════════════════════════════════════════════════════════════════════
# PAPER 2 — Figure 4: Geospatial property comparison
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig4_paper2_geo_comparison.png …")

cov_geo = meta[meta["covered"] == 1].merge(geo2, on="point_id", how="left")
gap_geo = meta[meta["covered"] == 0].merge(geo2, on="point_id", how="left")

features = [
    ("elevation_m",          "Elevation (m)",             False),
    ("slope_deg",            "Slope (°)",                 False),
    ("building_count_100m",  "Building count\n(100 m)",   False),
    ("dist_waterway_m",      "Distance to\nwaterway (m)", True),
    ("pop_density_per_km2",  "Population density\n(per km²)", False),
]

fig, axes = plt.subplots(1, len(features), figsize=(14, 5))
fig.patch.set_facecolor("white")

for ax, (feat, label, key_feat) in zip(axes, features):
    c = cov_geo[feat].dropna()
    g = gap_geo[feat].dropna()

    # Violin-style comparison using boxplot
    bp = ax.boxplot(
        [c.values, g.values],
        patch_artist=True,
        widths=0.55,
        medianprops=dict(color="white", lw=2),
        whiskerprops=dict(color="#555555"),
        capprops=dict(color="#555555"),
        flierprops=dict(marker=".", color="#aaaaaa", markersize=2, alpha=0.4),
    )
    bp["boxes"][0].set_facecolor("#2166ac")
    bp["boxes"][0].set_alpha(0.75)
    bp["boxes"][1].set_facecolor("#d73027")
    bp["boxes"][1].set_alpha(0.75)

    pooled = np.sqrt((c.std()**2 + g.std()**2) / 2)
    d = abs((g.mean() - c.mean()) / max(pooled, 1e-9))
    effect_tag = "large" if d >= 0.8 else "medium" if d >= 0.5 else "small" if d >= 0.2 else "negligible"

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Covered", "Gap"], fontsize=8.5)
    ax.set_ylabel(label, fontsize=8.5)
    ax.set_title(f"d = {d:.3f}\n({effect_tag})",
                 fontsize=8, color="#d73027" if key_feat else "#333333",
                 fontweight="bold" if key_feat else "normal")
    ax.grid(axis="y", alpha=0.2)
    ax.set_facecolor("#f8f8f8")

legend_handles = [
    mpatches.Patch(facecolor="#2166ac", alpha=0.75, label="Covered (n=3,912)"),
    mpatches.Patch(facecolor="#d73027", alpha=0.75, label="Gap (n=1,088)"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=2,
           fontsize=9, bbox_to_anchor=(0.5, -0.04))

plt.tight_layout()
out = FIG_DIR / "fig4_paper2_geo_comparison.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  Saved → {out}  ({out.stat().st_size//1024} KB)")


# ══════════════════════════════════════════════════════════════════════════════
# PAPER 3 — Figure 4: Spatial archetype map
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig4_paper3_spatial_archetypes.png …")

fig, ax = plt.subplots(figsize=(11, 9))
fig.patch.set_facecolor("white")

# Plot by archetype in vulnerability order (low first so high sits on top)
for cid in reversed(ARCHETYPE_ORDER):
    subset = c6_geo[c6_geo["cluster"] == cid]
    ax.scatter(
        subset["pano_lon"], subset["pano_lat"],
        s=5, alpha=0.65, linewidths=0,
        color=ARCHETYPE_COLORS[cid],
        label=ARCHETYPE_LABELS[cid],
        zorder=ARCHETYPE_ORDER.index(cid) + 2,
    )

ax.set_xlabel("Longitude", fontsize=10)
ax.set_ylabel("Latitude", fontsize=10)

legend = ax.legend(
    fontsize=8.2,
    markerscale=3,
    framealpha=0.93,
    title="Archetype (n = 3,836 road segments)",
    title_fontsize=8.5,
    loc="lower right",
    edgecolor="#cccccc",
)
ax.grid(alpha=0.18)
ax.set_facecolor("#f5f5f5")

plt.tight_layout()
out = FIG_DIR / "fig4_paper3_spatial_archetypes.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  Saved → {out}  ({out.stat().st_size//1024} KB)")

print(f"  Saved → {out}  ({out.stat().st_size//1024} KB)")


# ══════════════════════════════════════════════════════════════════════════════
# PAPER 3 — Figure 5: Vulnerability distribution by archetype
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig5_paper3_vuln_by_archetype.png …")

# x-tick labels include n= so nothing falls outside the axes
ARCHETYPE_NAMES_SHORT = {
    4: "Type I\nDense Impervious\nDrainage Void\n(n=741)",
    2: "Type II\nCompound\nDrainage Failure\n(n=406)",
    1: "Type III\nWatercourse-\nAdjacent Mixed\n(n=217)",
    6: "Type IV\nOpen Gutter\nWaste Corridor\n(n=655)",
    3: "Type V\nResidential\nMaint. Deficit\n(n=737)",
    5: "Type VI\nLow-Risk\nFormal Corridor\n(n=1,080)",
}

fig, ax = plt.subplots(figsize=(13, 7))
fig.patch.set_facecolor("white")
fig.subplots_adjust(bottom=0.26)   # room for 4-line tick labels

positions = list(range(1, 7))
vuln_data = [c6[c6["cluster"] == cid]["enet_vuln_prob"].values for cid in ARCHETYPE_ORDER]

vp = ax.violinplot(vuln_data, positions=positions, widths=0.68,
                   showmedians=True, showextrema=False)

for body, cid in zip(vp["bodies"], ARCHETYPE_ORDER):
    body.set_facecolor(ARCHETYPE_COLORS[cid])
    body.set_alpha(0.80)
    body.set_edgecolor("#333333")
    body.set_linewidth(0.9)

# Median as thick black bar — clearly visible on any colour
vp["cmedians"].set_color("black")
vp["cmedians"].set_linewidth(2.8)
vp["cmedians"].set_zorder(6)

# IQR box overlay for Q1–Q3
for cid, pos in zip(ARCHETYPE_ORDER, positions):
    vals = c6[c6["cluster"] == cid]["enet_vuln_prob"]
    q1, q3 = np.percentile(vals, [25, 75])
    ax.add_patch(plt.Rectangle(
        (pos - 0.10, q1), 0.20, q3 - q1,
        facecolor="white", edgecolor="#333333",
        linewidth=1.0, zorder=5, alpha=0.70,
    ))

# Mean diamond — white face with archetype-coloured edge so it pops on any background
for cid, pos in zip(ARCHETYPE_ORDER, positions):
    mean_v = c6[c6["cluster"] == cid]["enet_vuln_prob"].mean()
    med_v  = np.median(c6[c6["cluster"] == cid]["enet_vuln_prob"].values)

    # Diamond: white fill, coloured border
    ax.scatter(pos, mean_v, s=90, color="white",
               edgecolors=ARCHETYPE_COLORS[cid], linewidths=2.2,
               zorder=7, marker="D")

    # Mean value label in a white box
    ax.annotate(
        f"mean {mean_v:.3f}",
        xy=(pos, mean_v), xytext=(pos, mean_v + 0.075),
        ha="center", va="bottom", fontsize=7.8, fontweight="bold",
        color="#111111", zorder=8,
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                  edgecolor="#cccccc", linewidth=0.7, alpha=0.92),
        arrowprops=dict(arrowstyle="-", color="#888888", lw=0.8),
    )

    # Median value as small label, right-offset from the IQR box
    ax.text(pos + 0.14, med_v, f"  med {med_v:.3f}",
            va="center", ha="left", fontsize=6.8, color="#444444", zorder=8)

ax.set_xticks(positions)
ax.set_xticklabels([ARCHETYPE_NAMES_SHORT[cid] for cid in ARCHETYPE_ORDER],
                   fontsize=8.0, linespacing=1.4)
ax.set_ylabel("EfficientNet composite vulnerability probability", fontsize=10)
ax.set_ylim(0.0, 1.25)   # generous top headroom for mean labels

# 0.50 threshold line — label anchored to left margin
ax.axhline(0.5, color="#666666", ls=":", lw=1.1, alpha=0.7, zorder=1)
ax.text(0.55, 0.502, "0.50 threshold", va="bottom", ha="left",
        fontsize=7.5, color="#666666", style="italic")

ax.grid(axis="y", alpha=0.20)
ax.set_facecolor("#f7f7f7")
ax.spines[["top", "right"]].set_visible(False)

# Legend for plot elements
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], color="black", lw=2.8, label="Median"),
    plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#333333",
                  linewidth=1.0, alpha=0.70, label="IQR (Q1–Q3)"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor="white",
           markeredgecolor="#666666", markeredgewidth=2, markersize=8,
           label="Mean (annotated)"),
]
ax.legend(handles=legend_handles, fontsize=8, loc="upper right",
          framealpha=0.93, edgecolor="#cccccc")

plt.tight_layout()
out = FIG_DIR / "fig5_paper3_vuln_by_archetype.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  Saved → {out}  ({out.stat().st_size//1024} KB)")


# ══════════════════════════════════════════════════════════════════════════════
# Regenerate heatmap without embedded title
# ══════════════════════════════════════════════════════════════════════════════
print("Regenerating typology_cluster_heatmap_k6.png (no title) …")

profiles = pd.read_csv(ROOT / "outputs/tables/typology_cluster_profiles_k6.csv", index_col=0)
LABEL_COLS = [
    "enet_visible_drain_present", "enet_open_gutter_present",
    "enet_blocked_drain_present", "enet_stagnant_water_visible",
    "enet_poor_road_condition", "enet_heavy_impervious_surface",
    "enet_unpaved_shoulder", "enet_informal_structure_near_drainage",
    "enet_solid_waste_accumulation", "enet_visible_waterway_or_stream",
    "enet_low_lying_street_form", "enet_roadside_erosion",
    "enet_pedestrian_exposure", "enet_culvert_or_bridge_visible",
    "enet_no_visible_drainage",
]
SHORT = {c: c.replace("enet_", "").replace("_", " ") for c in LABEL_COLS}

# Reorder rows by vuln (high to low)
profiles_ord = profiles.loc[ARCHETYPE_ORDER]
row_labels = [ARCHETYPE_LABELS[int(c)].split(":")[1].strip()
              if ":" in ARCHETYPE_LABELS[int(c)] else ARCHETYPE_LABELS[int(c)]
              for c in profiles_ord.index]

fig2, ax2 = plt.subplots(figsize=(14, 4.5))
fig2.patch.set_facecolor("white")
mat = profiles_ord[LABEL_COLS].values
im = ax2.imshow(mat, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1)
ax2.set_xticks(range(len(LABEL_COLS)))
ax2.set_xticklabels([SHORT[c] for c in LABEL_COLS], rotation=40, ha="right", fontsize=8)
ax2.set_yticks(range(len(ARCHETYPE_ORDER)))
ax2.set_yticklabels(row_labels, fontsize=8.5)

# Annotate cells
for i in range(len(ARCHETYPE_ORDER)):
    for j in range(len(LABEL_COLS)):
        val = mat[i, j]
        txt_color = "white" if val > 0.65 or val < 0.25 else "#222222"
        ax2.text(j, i, f"{val:.2f}", ha="center", va="center",
                 fontsize=6.5, color=txt_color)

plt.colorbar(im, ax=ax2, label="Mean label probability", shrink=0.85)
plt.tight_layout()
out2 = FIG_DIR / "typology_cluster_heatmap_k6.png"
plt.savefig(out2, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  Saved → {out2}  ({out2.stat().st_size//1024} KB)")

print("\nAll figures done.")