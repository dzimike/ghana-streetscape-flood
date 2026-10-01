"""Assemble fig6_mapillary_panel.png.

Selects 4 highest and 4 lowest EfficientNet vulnerability-probability images
from the Mapillary dataset, assembles a 2×4 panel with per-label captions,
and adds CC BY-SA 4.0 attribution.

Outputs:
  outputs/figures/fig6_mapillary_panel.png

Usage:
  python src/visualization/make_mapillary_panel.py
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
from matplotlib.gridspec import GridSpec
from PIL import Image

ROOT     = Path(__file__).resolve().parents[2]
PREDS    = ROOT / "data/processed/mapillary_efficientnet_predictions.parquet"
MANIFEST = ROOT / "data/interim/mapillary_manifest.csv"
OUT      = ROOT / "outputs/figures/fig6_mapillary_panel.png"

LABEL_COLS = [
    "enet_visible_drain_present",
    "enet_open_gutter_present",
    "enet_blocked_drain_present",
    "enet_stagnant_water_visible",
    "enet_poor_road_condition",
    "enet_heavy_impervious_surface",
    "enet_unpaved_shoulder",
    "enet_informal_structure_near_drainage",
    "enet_solid_waste_accumulation",
    "enet_visible_waterway_or_stream",
    "enet_low_lying_street_form",
    "enet_roadside_erosion",
    "enet_pedestrian_exposure",
    "enet_culvert_or_bridge_visible",
    "enet_no_visible_drainage",
]
LABEL_SHORT = {
    "enet_visible_drain_present":            "drain present",
    "enet_open_gutter_present":              "open gutter",
    "enet_blocked_drain_present":            "blocked drain",
    "enet_stagnant_water_visible":           "stagnant water",
    "enet_poor_road_condition":              "poor road",
    "enet_heavy_impervious_surface":         "impervious surf.",
    "enet_unpaved_shoulder":                 "unpaved shoulder",
    "enet_informal_structure_near_drainage": "informal struct.",
    "enet_solid_waste_accumulation":         "solid waste",
    "enet_visible_waterway_or_stream":       "waterway",
    "enet_low_lying_street_form":            "low-lying street",
    "enet_roadside_erosion":                 "erosion",
    "enet_pedestrian_exposure":              "pedestrian exp.",
    "enet_culvert_or_bridge_visible":        "culvert/bridge",
    "enet_no_visible_drainage":              "no drainage",
}

# Thresholds for caption highlighting
HIGH_THRESH = 0.55
LOW_THRESH  = 0.25


# Curated pano_ids selected after visual review (round 9 → round 10 revision).
# Hi-1 replaced: 688279250501231 had visible functional drainage, weakening the
#   high-vulnerability message. 872046640193304 has strong blocked-drain, open-
#   gutter, stagnant-water, and solid-waste signals (drain_score 2.38).
# Lo-1 replaced: 1235989674946089 showed unpaved road with no drainage — visually
#   inconsistent with "low vulnerability". 461722814906435 is a clearly paved,
#   formal corridor (vuln=0.0024, poor_road=0.004, unpaved=0.004).
# Lo-2 replaced: 521504222187390 had poor_road_condition=0.45, visually ambiguous.
#   1202466384126795 is a paved surface with low road-problem scores (vuln=0.005).
FIXED_HI_IDS = [
    872046640193304,   # Hi-1: blocked drain + open gutter + stagnant water + waste
    323786925933787,   # Hi-2: unpaved + erosion + waste + informal structures
    624769990553876,   # Hi-3: impervious surface + informal structures
    504432675906007,   # Hi-4: erosion + poor road + informal structures + waste
]
FIXED_LO_IDS = [
    461722814906435,   # Lo-1: paved formal corridor, north study area (vuln=0.0024)
    1202466384126795,  # Lo-2: paved surface, south (vuln=0.005, poor_road=0.054)
    223409402555706,   # Lo-3: visible drain/gutter, south-central
    1318297132518328,  # Lo-4: moderate indicators but overall low vulnerability
]


def select_images(df: pd.DataFrame, n: int = 4, high: bool = True) -> pd.DataFrame:
    """Return the curated fixed selections (ordered, not re-selected algorithmically).

    Fixed IDs were chosen after visual review to ensure model scores and visual
    appearance are consistent: high-vulnerability images show clear drainage
    distress; low-vulnerability images show paved, formally maintained roads.
    """
    ids = FIXED_HI_IDS if high else FIXED_LO_IDS
    sel = df[df["pano_id"].isin(ids)].copy()
    # Preserve the curated order
    sel["_order"] = sel["pano_id"].map({pid: i for i, pid in enumerate(ids)})
    return sel.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def build_caption(row: pd.Series, max_labels: int = 5) -> str:
    """Return top-scoring label names as a compact caption string."""
    scores = {LABEL_SHORT[c]: row[c] for c in LABEL_COLS}
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:max_labels]
    parts = [f"{name} ({v:.2f})" for name, v in top if v >= LOW_THRESH]
    return "\n".join(parts) if parts else "no indicators above threshold"


def main():
    mly  = pd.read_parquet(PREDS)
    mani = pd.read_csv(MANIFEST)

    df = mly.merge(mani[["pano_id", "local_path"]], on="pano_id", how="left")
    df["file_exists"] = df["local_path"].apply(
        lambda p: Path(p).exists() if pd.notna(p) else False
    )
    df = df[df["file_exists"]].copy()
    print(f"Images available: {len(df)}  vuln range: "
          f"{df['enet_vuln_prob'].min():.3f}–{df['enet_vuln_prob'].max():.3f}")

    hi_sel = select_images(df, n=4, high=True)
    lo_sel = select_images(df, n=4, high=False)

    print("\nHIGH vulnerability selected:")
    print(hi_sel[["pano_id", "pano_lat", "pano_lon", "enet_vuln_prob"]].to_string(index=False))
    print("\nLOW vulnerability selected:")
    print(lo_sel[["pano_id", "pano_lat", "pano_lon", "enet_vuln_prob"]].to_string(index=False))

    # ── Layout ────────────────────────────────────────────────────────────────
    N_COLS   = 4
    N_ROWS   = 2
    IMG_H    = 3.2   # inches per image row
    CAP_H    = 1.1   # inches for caption row
    COL_W    = 3.4   # inches per column
    TOP_H    = 0.35  # row-label strip

    fig_w = N_COLS * COL_W
    fig_h = N_ROWS * (IMG_H + CAP_H) + TOP_H + 0.4  # + bottom margin for attribution

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    # GridSpec: 2 bands (high / low), each band = image row + caption row, plus header
    gs = GridSpec(
        nrows=N_ROWS * 2 + 1,
        ncols=N_COLS,
        figure=fig,
        height_ratios=[TOP_H] + [IMG_H, CAP_H] * N_ROWS,
        hspace=0.04,
        wspace=0.03,
        left=0.01, right=0.99,
        top=0.96, bottom=0.07,
    )

    ROW_LABELS = ["High vulnerability", "Low vulnerability"]
    ROW_COLORS = ["#d73027", "#4575b4"]
    selections = [hi_sel, lo_sel]

    for band, (sel, rlabel, rcol) in enumerate(zip(selections, ROW_LABELS, ROW_COLORS)):
        img_gs_row = 1 + band * 2       # GridSpec row index for images
        cap_gs_row = img_gs_row + 1     # GridSpec row index for captions

        # Row label spanning full width (in the header or as a text)
        ax_label = fig.add_subplot(gs[img_gs_row, :])
        ax_label.set_visible(False)
        fig.text(
            0.01, (1.0 - (img_gs_row) / (N_ROWS * 2 + 1)) * 0.96 + 0.02,
            rlabel,
            va="bottom", ha="left",
            fontsize=9, fontweight="bold", color=rcol,
            transform=fig.transFigure,
        )

        for col, (_, row) in enumerate(sel.iterrows()):
            # Image
            ax_img = fig.add_subplot(gs[img_gs_row, col])
            try:
                img = Image.open(row["local_path"]).convert("RGB")
                # Crop to 4:3 centre
                w, h = img.size
                target_h = int(w * 3 / 4)
                if target_h < h:
                    top_px = (h - target_h) // 2
                    img = img.crop((0, top_px, w, top_px + target_h))
                ax_img.imshow(img, aspect="auto")
            except Exception as e:
                ax_img.set_facecolor("#cccccc")
                ax_img.text(0.5, 0.5, f"[image error]\n{e}", ha="center", va="center",
                            fontsize=6, transform=ax_img.transAxes)

            # Vulnerability score badge
            score = row["enet_vuln_prob"]
            badge_col = "#d73027" if score > 0.5 else "#4575b4"
            ax_img.text(
                0.97, 0.97, f"p={score:.3f}",
                transform=ax_img.transAxes,
                ha="right", va="top", fontsize=7.5, fontweight="bold",
                color="white",
                bbox=dict(facecolor=badge_col, alpha=0.85, boxstyle="round,pad=0.2",
                          edgecolor="none"),
            )

            # Mapillary attribution watermark
            ax_img.text(
                0.03, 0.03, f"© Mapillary / CC BY-SA 4.0\nID: {row['pano_id']}",
                transform=ax_img.transAxes,
                ha="left", va="bottom", fontsize=4.5, color="white",
                bbox=dict(facecolor="black", alpha=0.45, boxstyle="round,pad=0.15",
                          edgecolor="none"),
            )

            ax_img.axis("off")

            # Caption
            ax_cap = fig.add_subplot(gs[cap_gs_row, col])
            caption = build_caption(row)
            ax_cap.text(
                0.5, 0.95, caption,
                transform=ax_cap.transAxes,
                ha="center", va="top",
                fontsize=6.2, linespacing=1.5,
                color="#222222",
            )
            ax_cap.set_facecolor("#f8f8f8")
            ax_cap.axis("off")
            for spine in ax_cap.spines.values():
                spine.set_visible(False)

    # Global attribution footer
    fig.text(
        0.5, 0.005,
        "Images: Mapillary contributors, licensed under CC BY-SA 4.0 "
        "(https://www.mapillary.com). Probabilities are EfficientNet-B0 model predictions. "
        "Captions list top-scoring flood-indicator labels; values are per-label probabilities.",
        ha="center", va="bottom", fontsize=5.5, color="#555555",
        wrap=True,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    size_kb = OUT.stat().st_size // 1024
    print(f"\nSaved → {OUT}  ({size_kb} KB)")


if __name__ == "__main__":
    main()
