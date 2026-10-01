"""Assemble fig4_streetview_panel.png.

Selects the 4 highest and 4 lowest EfficientNet vulnerability-probability images,
copies them to outputs/figures/panel_sources/, and assembles a 2×4 panel with
per-label captions derived from the model's own probabilities.

Blur workflow (run separately before this script):
  python src/utils/blur_faces_plates.py \
    --input  outputs/figures/panel_sources \
    --output outputs/figures/panel_sources_blurred

If blurred images exist they are used automatically; otherwise raw images are used
and a warning is printed.

Usage:
  python src/visualization/make_streetview_panel.py          # full run
  python src/visualization/make_streetview_panel.py --no-copy  # skip file copy
"""
from __future__ import annotations

import argparse
import shutil
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

PREDS_PARQ  = ROOT / "data/processed/efficientnet_predictions.parquet"
MANIF_PARQ  = ROOT / "data/interim/image_manifest.parquet"
PANEL_SRC   = ROOT / "outputs/figures/panel_sources"
PANEL_BLUR  = ROOT / "outputs/figures/panel_sources_blurred"
OUT_FIG4    = ROOT / "outputs/figures/fig4_streetview_panel.png"

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
    "enet_visible_drain_present":            "drain",
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


def _caption(row: pd.Series, top_n: int = 4) -> str:
    """Return a short caption listing the top-n detected flood indicators."""
    scores = {col: float(row[col]) for col in LABEL_COLS if col in row.index}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    parts = [f"{LABEL_SHORT[col]} {v:.2f}" for col, v in ranked if v >= 0.10]
    return " | ".join(parts) if parts else "(no dominant indicator)"


def select_images(preds: pd.DataFrame, manif: pd.DataFrame) -> pd.DataFrame:
    """Return 4 highest and 4 lowest vulnerability images with paths."""
    merged = preds.merge(
        manif[["pano_id", "heading", "local_path"]],
        on=["pano_id", "heading"],
        how="left",
    )
    merged = merged[merged["local_path"].notna()].copy()
    merged["local_path"] = merged["local_path"].astype(str)

    # Keep only images that exist on disk
    merged = merged[merged["local_path"].apply(lambda p: Path(p).exists())]

    top4 = merged.nlargest(4, "enet_vuln_prob").reset_index(drop=True)
    bot4 = merged.nsmallest(4, "enet_vuln_prob").reset_index(drop=True)
    top4["row_label"] = "high"
    bot4["row_label"] = "low"
    return pd.concat([top4, bot4], ignore_index=True)


def copy_sources(selection: pd.DataFrame) -> None:
    PANEL_SRC.mkdir(parents=True, exist_ok=True)
    for _, row in selection.iterrows():
        src = Path(row["local_path"])
        dst = PANEL_SRC / f"{row['pano_id']}_{row['heading']}.jpg"
        if not dst.exists():
            shutil.copy2(src, dst)
    print(f"  Panel sources → {PANEL_SRC}  ({len(selection)} images)")


def _load_image(row: pd.Series) -> Image.Image:
    """Load blurred version if available, else raw."""
    fname = f"{row['pano_id']}_{row['heading']}.jpg"
    blurred = PANEL_BLUR / fname
    if blurred.exists():
        return Image.open(blurred).convert("RGB")
    raw = PANEL_SRC / fname
    if raw.exists():
        return Image.open(raw).convert("RGB")
    # Fallback: original path
    return Image.open(row["local_path"]).convert("RGB")


def build_panel(selection: pd.DataFrame) -> None:
    blurred_available = PANEL_BLUR.exists() and any(PANEL_BLUR.iterdir())
    if not blurred_available:
        print(
            "  WARNING: blurred images not found — using raw images.\n"
            "  Run: python src/utils/blur_faces_plates.py "
            "--input outputs/figures/panel_sources "
            "--output outputs/figures/panel_sources_blurred\n"
            "  Then re-run this script."
        )

    high_rows = selection[selection["row_label"] == "high"]
    low_rows  = selection[selection["row_label"] == "low"]

    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("#1a1a1a")

    # Header text
    fig.text(
        0.5, 0.97,
        "Street View Examples — High vs. Low Flood Vulnerability (EfficientNet-B0)",
        ha="center", va="top", color="white", fontsize=13, fontweight="bold",
    )
    fig.text(
        0.5, 0.93,
        "Captions show top-4 model-predicted flood indicators (probability ≥ 0.10)",
        ha="center", va="top", color="#cccccc", fontsize=9.5,
    )

    gs = GridSpec(
        2, 4,
        figure=fig,
        left=0.01, right=0.99,
        top=0.88, bottom=0.08,
        hspace=0.22, wspace=0.03,
    )

    row_configs = [
        (high_rows, "#d7191c", "HIGH VULNERABILITY"),
        (low_rows,  "#1a9641", "LOW VULNERABILITY"),
    ]

    for r_idx, (subset, border_color, row_title) in enumerate(row_configs):
        fig.text(
            0.005, 0.88 - r_idx * 0.40,
            row_title,
            va="top", color=border_color, fontsize=10, fontweight="bold",
            rotation=90,
        )
        for c_idx, (_, irow) in enumerate(subset.iterrows()):
            ax = fig.add_subplot(gs[r_idx, c_idx])
            img = _load_image(irow)
            ax.imshow(np.array(img))
            ax.axis("off")

            # Coloured border
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(2.5)

            # Probability badge
            prob = irow["enet_vuln_prob"]
            badge_bg = "#d7191c" if irow["row_label"] == "high" else "#1a9641"
            ax.text(
                0.03, 0.97, f"p = {prob:.4f}",
                transform=ax.transAxes,
                va="top", ha="left", fontsize=8.5, fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=badge_bg, alpha=0.85, edgecolor="none"),
            )

            # Caption below each image
            caption = _caption(irow)
            ax.text(
                0.5, -0.03, caption,
                transform=ax.transAxes,
                ha="center", va="top", fontsize=6.5, color="#cccccc",
                wrap=True,
            )

    # Footer
    fig.text(
        0.5, 0.01,
        ("Source: Google Street View (API-compliant access). "
         "Faces and licence plates blurred for privacy. "
         "Captions derived from EfficientNet-B0 per-label probabilities; "
         "no visual impression used."),
        ha="center", va="bottom", color="#888888", fontsize=7.5,
        style="italic",
    )

    plt.savefig(OUT_FIG4, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    kb = OUT_FIG4.stat().st_size / 1024
    print(f"  fig4 → {OUT_FIG4}  ({kb:.0f} KB)")
    if not blurred_available:
        print("  NOTE: fig4 was built from UNBLURRED images. Blur first before submission.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fig4_streetview_panel.png")
    parser.add_argument("--no-copy", action="store_true",
                        help="Skip copying source images (use existing panel_sources/)")
    args = parser.parse_args()

    print("Loading EfficientNet predictions …")
    preds  = pd.read_parquet(PREDS_PARQ)
    manif  = pd.read_parquet(MANIF_PARQ)

    print("Selecting top-4 / bottom-4 images …")
    selection = select_images(preds, manif)
    print(f"  High: {len(selection[selection['row_label']=='high'])}  "
          f"Low: {len(selection[selection['row_label']=='low'])}")

    if not args.no_copy:
        copy_sources(selection)

    print("Assembling panel …")
    build_panel(selection)

    print("\nNext steps:")
    print("  1. Run blur script on panel_sources/:")
    print("     python src/utils/blur_faces_plates.py "
          "--input outputs/figures/panel_sources "
          "--output outputs/figures/panel_sources_blurred")
    print("  2. Re-run this script to rebuild from blurred images.")


if __name__ == "__main__":
    main()
