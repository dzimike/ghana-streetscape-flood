"""Validation of SLFVI against flood-exposed building locations.

Produces:
  outputs/tables/validation_metrics.csv
  outputs/tables/validation_by_class.csv
  outputs/tables/validation_by_district.csv
  outputs/figures/validation_roc_pr.png
  outputs/figures/validation_by_class.png
  outputs/figures/validation_map.png
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
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from loguru import logger
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    confusion_matrix,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
FLOOD_BLD  = ROOT / "data/external/accra_flood_exposure.gpkg"
VI_PARQ    = ROOT / "data/processed/vulnerability_index.parquet"
ADM2       = ROOT / "data/external/GHA_ADM2.geojson"
FOCUS_VI   = ROOT / "outputs/focus_districts/district_vulnerability.parquet"

OUT_TABLES = ROOT / "outputs/tables"
OUT_FIGS   = ROOT / "outputs/figures"

FOCUS_DISTRICTS = [
    "Ablekuma Central Municipal",
    "Ledzokuku Municipal",
    "Ga East",
    "La Dade-kotopon",
]
VULN_BINS   = [-0.001, 0.20, 0.40, 0.60, 0.80, 1.001]
VULN_LABELS = ["very_low", "low", "moderate", "high", "very_high"]
CLASS_COLOURS = {
    "very_low": "#1a9641", "low": "#a6d96a",
    "moderate": "#ffffbf", "high": "#fdae61", "very_high": "#d7191c",
}


def build_validation_gdf(buffer_m: int = 100) -> gpd.GeoDataFrame:
    """Join road points with flood-exposed buildings within buffer_m metres."""
    flood = gpd.read_file(FLOOD_BLD)   # EPSG:32630
    vi    = pd.read_parquet(VI_PARQ)

    vi_gdf = gpd.GeoDataFrame(
        vi,
        geometry=gpd.points_from_xy(vi["longitude"], vi["latitude"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:32630")

    vi_buf = vi_gdf.copy()
    vi_buf["geometry"] = vi_buf.geometry.buffer(buffer_m)

    joined = gpd.sjoin(
        vi_buf, flood[["flood_exposed", "geometry"]], how="left", predicate="intersects"
    )
    agg = (
        joined.groupby(level=0)
        .agg(n_flood_exposed=("flood_exposed", "sum"))
        .reindex(vi_gdf.index)
        .fillna(0)
    )
    vi_gdf["n_flood_exposed_100m"] = agg["n_flood_exposed"].astype(int)
    vi_gdf["any_flood_bld_100m"]   = (vi_gdf["n_flood_exposed_100m"] > 0).astype(int)

    # SLFVI class
    score_col = "slfvi_final" if "slfvi_final" in vi_gdf.columns else "slfvi"
    vi_gdf["slfvi_class"] = pd.cut(
        vi_gdf[score_col].fillna(vi_gdf[score_col].median()),
        bins=VULN_BINS, labels=VULN_LABELS, right=False,
    )

    # District
    adm = gpd.read_file(ADM2).to_crs("EPSG:32630")
    vi_gdf = gpd.sjoin(
        vi_gdf, adm[["shapeName", "geometry"]], how="left", predicate="within"
    ).drop(columns=["index_right"], errors="ignore")

    logger.info(
        f"Validation set: {len(vi_gdf):,} road points | "
        f"{vi_gdf.any_flood_bld_100m.sum():,} near flood buildings "
        f"({100*vi_gdf.any_flood_bld_100m.mean():.1f}%)"
    )
    return vi_gdf


def compute_metrics(vi_gdf: gpd.GeoDataFrame) -> dict:
    y_true = vi_gdf["any_flood_bld_100m"].values
    results = {}
    for col in ["slfvi", "slfvi_final", "ml_flood_prob"]:
        if col not in vi_gdf.columns:
            continue
        y_sc = vi_gdf[col].fillna(vi_gdf[col].median()).values
        results[col] = {
            "roc_auc": round(roc_auc_score(y_true, y_sc), 3),
            "pr_auc":  round(average_precision_score(y_true, y_sc), 3),
            "n_positive": int(y_true.sum()),
            "n_total":    int(len(y_true)),
            "positive_rate_pct": round(100 * y_true.mean(), 1),
        }
    return results, y_true


def main() -> None:
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_FIGS.mkdir(parents=True, exist_ok=True)

    # ── 1. Build validation dataset ───────────────────────────────────────────
    vi_gdf = build_validation_gdf(buffer_m=100)
    y_true = vi_gdf["any_flood_bld_100m"].values

    # ── 2. Overall metrics ────────────────────────────────────────────────────
    metrics_rows = []
    curves = {}
    for col in ["slfvi", "slfvi_final", "ml_flood_prob"]:
        if col not in vi_gdf.columns:
            continue
        y_sc  = vi_gdf[col].fillna(vi_gdf[col].median()).values
        auc   = roc_auc_score(y_true, y_sc)
        ap    = average_precision_score(y_true, y_sc)
        fpr, tpr, _ = roc_curve(y_true, y_sc)
        prec, rec, _ = precision_recall_curve(y_true, y_sc)
        curves[col]  = dict(fpr=fpr, tpr=tpr, prec=prec, rec=rec, auc=auc, ap=ap)
        metrics_rows.append({
            "score":          col,
            "roc_auc":        round(auc, 3),
            "pr_auc":         round(ap, 3),
            "n_positive":     int(y_true.sum()),
            "n_total":        int(len(y_true)),
            "positive_rate":  round(100 * y_true.mean(), 1),
        })
        logger.info(f"  {col:<22} ROC-AUC={auc:.3f}  PR-AUC={ap:.3f}")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(OUT_TABLES / "validation_metrics.csv", index=False)

    # ── 3. By SLFVI class ─────────────────────────────────────────────────────
    class_rows = []
    for cls, grp in vi_gdf.groupby("slfvi_class", observed=True):
        class_rows.append({
            "slfvi_class":          str(cls),
            "n_road_points":        len(grp),
            "n_near_flood_bld":     int(grp["any_flood_bld_100m"].sum()),
            "pct_near_flood_bld":   round(100 * grp["any_flood_bld_100m"].mean(), 1),
            "mean_slfvi":           round(grp["slfvi"].mean(), 3),
        })
    class_df = pd.DataFrame(class_rows)
    class_df.to_csv(OUT_TABLES / "validation_by_class.csv", index=False)

    # ── 4. By district ────────────────────────────────────────────────────────
    dist_rows = []
    for dist in FOCUS_DISTRICTS:
        sub = vi_gdf[vi_gdf.get("shapeName", vi_gdf.get("shapeName_left",
              pd.Series())).eq(dist) if "shapeName" in vi_gdf.columns
              else vi_gdf.index < 0]
        if "shapeName" in vi_gdf.columns:
            sub = vi_gdf[vi_gdf["shapeName"] == dist]
        if len(sub) == 0:
            continue
        score_col = "slfvi_final" if "slfvi_final" in sub.columns else "slfvi"
        dist_rows.append({
            "district":              dist,
            "n_road_points":         len(sub),
            "n_near_flood_bld":      int(sub["any_flood_bld_100m"].sum()),
            "pct_near_flood_bld":    round(100 * sub["any_flood_bld_100m"].mean(), 1),
            "mean_slfvi":            round(sub["slfvi"].mean(), 3),
            "mean_slfvi_final":      round(sub[score_col].mean(), 3),
            "pct_high_very_high":    round(
                100 * sub["slfvi_class"].isin(["high", "very_high"]).mean(), 1
            ),
        })
    dist_df = pd.DataFrame(dist_rows)
    dist_df.to_csv(OUT_TABLES / "validation_by_district.csv", index=False)

    # ── 5. Figure 1: ROC + PR curves ─────────────────────────────────────────
    labels = {
        "slfvi":        "SLFVI formula",
        "slfvi_final":  "SLFVI final (blended)",
        "ml_flood_prob":"ML flood probability",
    }
    colours = {"slfvi": "#4575b4", "slfvi_final": "#d73027", "ml_flood_prob": "#1a9641"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ROC
    ax1.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    for col, c in curves.items():
        ax1.plot(c["fpr"], c["tpr"], color=colours[col], lw=2,
                 label=f"{labels[col]}  (AUC={c['auc']:.3f})")
    ax1.set_xlabel("False Positive Rate", fontsize=11)
    ax1.set_ylabel("True Positive Rate", fontsize=11)
    ax1.set_title("ROC Curve — SLFVI vs Flood-Exposed Buildings", fontsize=12)
    ax1.legend(fontsize=9, loc="lower right")
    ax1.grid(alpha=0.3)

    # PR
    baseline = y_true.mean()
    ax2.axhline(baseline, color="k", ls="--", lw=0.8, alpha=0.5,
                label=f"Baseline (prevalence={baseline:.3f})")
    for col, c in curves.items():
        ax2.plot(c["rec"], c["prec"], color=colours[col], lw=2,
                 label=f"{labels[col]}  (AP={c['ap']:.3f})")
    ax2.set_xlabel("Recall", fontsize=11)
    ax2.set_ylabel("Precision", fontsize=11)
    ax2.set_title("Precision-Recall Curve", fontsize=12)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT_FIGS / "validation_roc_pr.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC/PR figure → {OUT_FIGS}/validation_roc_pr.png")

    # ── 6. Figure 2: Flood-exposed rate by SLFVI class ───────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: bar chart
    ax = axes[0]
    cls_names = class_df["slfvi_class"].tolist()
    pcts      = class_df["pct_near_flood_bld"].tolist()
    counts    = class_df["n_road_points"].tolist()
    bar_colours = [CLASS_COLOURS.get(c, "#aaa") for c in cls_names]

    bars = ax.bar(cls_names, pcts, color=bar_colours, edgecolor="#333", linewidth=0.8)
    for bar, pct, n in zip(bars, pcts, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{pct:.0f}%\n(n={n:,})", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Road points near flood-exposed building (%)", fontsize=10)
    ax.set_xlabel("SLFVI Vulnerability Class", fontsize=10)
    ax.set_title("Flood Validation by SLFVI Class", fontsize=11)
    ax.set_ylim(0, 115)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=15)

    # Right: district comparison (observed Street View points only)
    ax2 = axes[1]
    x  = np.arange(len(dist_df))
    w  = 0.35
    b1 = ax2.bar(x - w/2, dist_df["mean_slfvi"] * 100, w,
                 label="Mean SLFVI (×100)", color="#4575b4", alpha=0.8)
    b2 = ax2.bar(x + w/2, dist_df["pct_near_flood_bld"], w,
                 label="% near flood building", color="#d73027", alpha=0.8)

    # Add n= count labels above each bar group
    for i, (_, row) in enumerate(dist_df.iterrows()):
        ax2.text(i, max(row["mean_slfvi"] * 100, row["pct_near_flood_bld"]) + 1.5,
                 f"n={row['n_road_points']:,}", ha="center", va="bottom",
                 fontsize=7, color="#333")

    ax2.set_xticks(x)
    ax2.set_xticklabels(
        [d.replace(" Municipal", "").replace(" ", "\n") for d in dist_df["district"]],
        fontsize=8,
    )
    ax2.set_ylabel("Score / Percentage", fontsize=10)
    ax2.set_title(
        "Focus Districts: SLFVI vs Flood Exposure\n"
        "(observed Street View points only — see Table 6 for full road network)",
        fontsize=10,
    )
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(0, ax2.get_ylim()[1] * 1.12)   # headroom for n= labels

    plt.suptitle(
        "Validation: Street-Level Flood Vulnerability Index vs Flood-Exposed Buildings",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    fig.savefig(OUT_FIGS / "validation_by_class.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Class validation figure → {OUT_FIGS}/validation_by_class.png")

    # ── 7. Figure 3: Spatial validation map (focus districts) ─────────────────
    flood     = gpd.read_file(FLOOD_BLD).to_crs("EPSG:4326")
    flood_exp = flood[flood["flood_exposed"]].copy()
    adm4326   = gpd.read_file(ADM2)
    focus_bnd = adm4326[adm4326["shapeName"].isin(FOCUS_DISTRICTS)]

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    for ax, dist in zip(axes, FOCUS_DISTRICTS):
        dist_poly = focus_bnd[focus_bnd["shapeName"] == dist]
        bounds    = dist_poly.total_bounds  # [minx, miny, maxx, maxy]
        pad       = 0.005

        # Background district boundary
        dist_poly.plot(ax=ax, color="#f5f5f5", edgecolor="#555", linewidth=1.5, zorder=1)

        # Flood-exposed buildings (red footprints)
        clip_exp = flood_exp.cx[
            bounds[0]-pad:bounds[2]+pad,
            bounds[1]-pad:bounds[3]+pad,
        ]
        if len(clip_exp):
            clip_exp.plot(ax=ax, color="#d7191c", alpha=0.4, linewidth=0, zorder=2,
                          label="Flood-exposed building")

        # Road points coloured by SLFVI class
        sub_vi = vi_gdf.to_crs("EPSG:4326")
        sub_vi = sub_vi[sub_vi.geometry.within(dist_poly.union_all())]
        if len(sub_vi):
            colours_pt = sub_vi["slfvi_class"].map(CLASS_COLOURS).fillna("#ffffbf")
            ax.scatter(
                sub_vi.geometry.x, sub_vi.geometry.y,
                c=colours_pt, s=25, zorder=3, linewidths=0.3,
                edgecolors="#333", alpha=0.9,
            )

        ax.set_xlim(bounds[0]-pad, bounds[2]+pad)
        ax.set_ylim(bounds[1]-pad, bounds[3]+pad)

        n_near = sub_vi["any_flood_bld_100m"].sum() if len(sub_vi) else 0
        ax.set_title(
            f"{dist}\n"
            f"n={len(sub_vi):,} road pts | "
            f"{100*sub_vi['any_flood_bld_100m'].mean():.1f}% near flood bld | "
            f"mean SLFVI={sub_vi['slfvi'].mean():.3f}",
            fontsize=9, pad=5,
        )
        ax.set_xlabel("Longitude", fontsize=8)
        ax.set_ylabel("Latitude", fontsize=8)
        ax.tick_params(labelsize=7)

    # Shared legend
    slfvi_patches = [mpatches.Patch(color=v, label=k) for k, v in CLASS_COLOURS.items()]
    flood_patch   = mpatches.Patch(color="#d7191c", alpha=0.4, label="Flood-exposed building")
    fig.legend(
        handles=slfvi_patches + [flood_patch],
        title="SLFVI class / Flood exposure",
        loc="lower center", ncol=6, fontsize=8,
        title_fontsize=9, bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle(
        "Spatial Validation — SLFVI Road Points vs Flood-Exposed Buildings\nAccra Focus Districts",
        fontsize=13, y=1.01,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(OUT_FIGS / "validation_map.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Spatial validation map → {OUT_FIGS}/validation_map.png")

    # ── 8. Print summary ──────────────────────────────────────────────────────
    print("\n── Overall Validation Metrics ──────────────────────────────")
    print(f"  {'Score':<22} {'ROC-AUC':>8}  {'PR-AUC':>8}")
    print(f"  {'-'*42}")
    for _, r in metrics_df.iterrows():
        print(f"  {r['score']:<22} {r['roc_auc']:>8.3f}  {r['pr_auc']:>8.3f}")

    print("\n── Flood-Exposed Rate by SLFVI Class ───────────────────────")
    print(f"  {'Class':<12}  {'N pts':>6}  {'Near flood bld%':>15}")
    for _, r in class_df.iterrows():
        bar = "█" * int(r["pct_near_flood_bld"] / 2.5)
        print(f"  {r['slfvi_class']:<12}  {r['n_road_points']:>6,}  "
              f"{r['pct_near_flood_bld']:>13.1f}%  {bar}")

    print("\n── Focus District Validation ────────────────────────────────")
    print(f"  {'District':<35}  {'N':>5}  {'SLFVI':>6}  {'Near flood%':>11}  {'High+%':>7}")
    for _, r in dist_df.iterrows():
        print(f"  {r['district']:<35}  {r['n_road_points']:>5,}  "
              f"{r['mean_slfvi']:>6.3f}  {r['pct_near_flood_bld']:>10.1f}%  "
              f"{r['pct_high_very_high']:>6.1f}%")

    print("\n── Output Files ────────────────────────────────────────────")
    for p in [
        OUT_TABLES / "validation_metrics.csv",
        OUT_TABLES / "validation_by_class.csv",
        OUT_TABLES / "validation_by_district.csv",
        OUT_FIGS   / "validation_roc_pr.png",
        OUT_FIGS   / "validation_by_class.png",
        OUT_FIGS   / "validation_map.png",
    ]:
        if p.exists():
            kb = p.stat().st_size / 1024
            tag = f"{kb/1024:.1f} MB" if kb > 1024 else f"{kb:.1f} KB"
            print(f"  {p.relative_to(ROOT)!s:<55} {tag}")


if __name__ == "__main__":
    main()
