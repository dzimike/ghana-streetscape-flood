"""Generate Figure 1: seven-stage methodological workflow.

Output: outputs/figures/fig1_workflow.png
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT  = ROOT / "outputs/figures/fig1_workflow.png"

C_PROC  = "#2166ac"
C_DATA  = "#f2f2f2"
C_INPUT = "#d4e9f7"
C_OUT   = "#1a9641"
C_STAGE = "#762a83"
C_ARROW = "#555555"
C_LT    = "white"
C_DK    = "#1a1a1a"


def _tc(fill):
    return C_LT if fill not in (C_DATA, C_INPUT) else C_DK


def box(ax, cx, cy, w, h, line1, line2=None, fill=C_DATA, rounded=False, fs=8.5):
    style = "round,pad=0.08" if rounded else "square,pad=0.0"
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=style, linewidth=1.1, zorder=3,
        edgecolor="#999999" if fill in (C_DATA, C_INPUT) else fill,
        facecolor=fill,
    )
    ax.add_patch(patch)
    c = _tc(fill)
    if line2:
        ax.text(cx, cy + h * 0.15, line1, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=c, zorder=4)
        ax.text(cx, cy - h * 0.22, line2, ha="center", va="center",
                fontsize=fs - 1.8, color=c, zorder=4, fontstyle="italic",
                multialignment="center")
    else:
        ax.text(cx, cy, line1, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=c, zorder=4,
                multialignment="center")


def badge(ax, cx, cy, n):
    ax.text(cx, cy, f"Stage {n}", ha="center", va="center",
            fontsize=6.5, color="white", fontweight="bold", zorder=5,
            bbox=dict(boxstyle="round,pad=0.22", facecolor=C_STAGE, edgecolor="none"))


def _vline(ax, x, y1, y2):
    ax.plot([x, x], [y1, y2], color=C_ARROW, lw=1.4, zorder=2, solid_capstyle="round")


def _hline(ax, x1, x2, y):
    ax.plot([x1, x2], [y, y], color=C_ARROW, lw=1.4, zorder=2, solid_capstyle="round")


def _arrowhead_down(ax, x, y_from, y_to):
    """Vertical line from y_from to just above y_to, then annotate arrowhead."""
    _vline(ax, x, y_from, y_to + 0.015)
    ax.annotate("", xy=(x, y_to), xytext=(x, y_to + 0.015), zorder=2,
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.4,
                                mutation_scale=11, connectionstyle="arc3,rad=0"))


def arrow(ax, x1, y1, x2, y2):
    """Straight arrow (any direction)."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=2,
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.4, mutation_scale=11))


def split(ax, x, y_src, xl, xr, y_dst, ymid=None):
    """One source → horizontal bar → two arrows to branch boxes."""
    if ymid is None:
        ymid = (y_src + y_dst) / 2
    _vline(ax, x, y_src, ymid)
    _hline(ax, xl, xr, ymid)
    _arrowhead_down(ax, xl, ymid, y_dst)
    _arrowhead_down(ax, xr, ymid, y_dst)


def merge(ax, xl, xr, y_src, x, y_dst, ymid=None):
    """Two branch bottoms → horizontal bar → one arrow to target."""
    if ymid is None:
        ymid = (y_src + y_dst) / 2
    _vline(ax, xl, y_src, ymid)
    _vline(ax, xr, y_src, ymid)
    _hline(ax, xl, xr, ymid)
    _arrowhead_down(ax, x, ymid, y_dst)


# ── Canvas ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11.5, 16))
fig.patch.set_facecolor("white")
ax.set_xlim(0, 11.5)
ax.set_ylim(0, 16)
ax.axis("off")

CX  = 5.0    # centre x of main column
WM  = 5.8    # main box width
H   = 0.62   # standard box height
HS  = 0.52   # small box height
WB  = 2.3    # branch box width
XL  = 2.85   # left branch x
XR  = 7.15   # right branch x

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(CX, 15.72, "Figure 1. Methodological Workflow",
        ha="center", va="center", fontsize=12, fontweight="bold", color=C_DK)

# ── External inputs ───────────────────────────────────────────────────────────
y_in = 14.95
box(ax, 2.8,  y_in, 2.3, HS, "OSM Roads · DEM\nRainfall · Hydrology", fill=C_INPUT, fs=7.8)
box(ax, 7.2,  y_in, 2.3, HS, "Street View API\nMapillary Tiles",       fill=C_INPUT, fs=7.8)

# ── Stage 1: Sampling ─────────────────────────────────────────────────────────
y1 = 13.78
arrow(ax, 2.8, y_in - HS/2, 2.8, y1 + H/2)
arrow(ax, 7.2, y_in - HS/2, 7.2, y1 + H/2)
box(ax, CX, y1, WM, H,
    "Stage 1 — Stratified Road-Segment Sampling",
    "5,000 points · 4 flood strata × 3 settlement types · 100 m segments",
    fill=C_PROC, rounded=True)
badge(ax, CX - WM/2 - 0.52, y1, 1)

# ── Stage 2: Metadata query ───────────────────────────────────────────────────
y2 = 12.58
arrow(ax, CX, y1 - H/2, CX, y2 + H/2)
box(ax, CX, y2, WM, H,
    "Stage 2 — Street View Metadata Query",
    "Metadata API · Mapillary gap fill · deduplication · spatial join",
    fill=C_PROC, rounded=True)
badge(ax, CX - WM/2 - 0.52, y2, 2)

# ── Split: valid panoramas vs coverage gaps ───────────────────────────────────
y3 = 11.5
split(ax, CX, y2 - H/2, XL, XR, y3 + HS/2, ymid=11.92)
box(ax, XL, y3, WB, HS, "3,836 valid\npanoramas",     fill=C_DATA, fs=8)
box(ax, XR, y3, WB, HS, "1,088 coverage\ngap points", fill=C_DATA, fs=8)

# ── Image download / Mapillary gap fill ───────────────────────────────────────
y4 = 10.47
arrow(ax, XL, y3 - HS/2, XL, y4 + H/2)
arrow(ax, XR, y3 - HS/2, XR, y4 + H/2)
box(ax, XL, y4, WB, H, "Image Download\n(4 headings per point)", fill=C_PROC, rounded=True, fs=8)
box(ax, XR, y4, WB, H, "Mapillary\nGap Query",                   fill=C_PROC, rounded=True, fs=8)

# ── Stage 3: Annotation — merge two branches ──────────────────────────────────
y5 = 9.32
merge(ax, XL, XR, y4 - H/2, CX, y5 + H/2, ymid=9.77)
box(ax, CX, y5, WM, H,
    "Stage 3 — Annotation & Hybrid Dataset Construction",
    "400 human-labelled + 1,600 pseudo-labelled · 15 binary flood-indicator labels",
    fill=C_PROC, rounded=True)
badge(ax, CX - WM/2 - 0.52, y5, 3)

# ── Stage 4: EfficientNet ────────────────────────────────────────────────────
y6 = 8.1
arrow(ax, CX, y5 - H/2, CX, y6 + H/2)
box(ax, CX, y6, WM, H,
    "Stage 4 — EfficientNet-B0 Training & Inference",
    "Fine-tuned on hybrid dataset · inference over 15,934 images · 15 label probabilities",
    fill=C_PROC, rounded=True)
badge(ax, CX - WM/2 - 0.52, y6, 4)

# ── Streetscape features ──────────────────────────────────────────────────────
y7 = 7.02
arrow(ax, CX, y6 - H/2, CX, y7 + HS/2)
box(ax, CX, y7, WM, HS,
    "Streetscape Features per Road Segment  (26 columns)",
    fill=C_DATA, fs=8)

# ── Stage 5: Terrain / exposure (side box, clear of main column) ──────────────
S5X   = 9.55   # centre x — well clear of main column right edge (7.9)
S5W   = 2.6    # width  — right edge at 9.55+1.3 = 10.85, fits within 11.5
S5H   = 0.72   # height — matches streetscape row comfortably
box(ax, S5X, y7, S5W, S5H,
    "Terrain · Hydrology ·\nExposure Features",
    "DEM · MERIT Hydro · WorldPop\nOSM buildings · GHSL",
    fill=C_PROC, rounded=True, fs=7.4)
# Badge sits above the Stage 5 box, not overlapping it
badge(ax, S5X, y7 + S5H/2 + 0.21, 5)
# Horizontal arrow: Stage 5 left edge → Streetscape features right edge
arrow(ax, S5X - S5W/2, y7, CX + WM/2, y7)

# ── Stage 6: LightGBM fusion ─────────────────────────────────────────────────
y8 = 5.85
arrow(ax, CX, y7 - HS/2, CX, y8 + H/2)
box(ax, CX, y8, WM, H,
    "Stage 6 — LightGBM Fusion Model",
    "34 features · spatial block 5-fold CV · Random Forest & XGBoost comparison",
    fill=C_PROC, rounded=True)
badge(ax, CX - WM/2 - 0.52, y8, 6)

# ── SLFVI formula ─────────────────────────────────────────────────────────────
y9 = 4.75
arrow(ax, CX, y8 - H/2, CX, y9 + HS/2)
box(ax, CX, y9, WM, HS,
    "SLFVI  =  0.30 · H  +  0.20 · E  +  0.35 · S  +  0.15 · (1 − A)",
    fill=C_DATA, fs=8.5)

# ── Split: IDW interpolation vs ordinary kriging ─────────────────────────────
y10 = 3.68
split(ax, CX, y9 - HS/2, XL, XR, y10 + HS/2, ymid=4.13)
box(ax, XL, y10, WB, HS, "IDW\nInterpolation", fill=C_PROC, rounded=True, fs=8)
box(ax, XR, y10, WB, HS, "Ordinary\nKriging",  fill=C_PROC, rounded=True, fs=8)

# ── 250 m grid — merge two interpolations ─────────────────────────────────────
y11 = 2.78
merge(ax, XL, XR, y10 - HS/2, CX, y11 + HS/2, ymid=3.18)
box(ax, CX, y11, WM, HS,
    "250 m Vulnerability Grid  ·  Kriging Prediction Variance",
    fill=C_DATA, fs=8)

# ── Stage 7: Outputs ─────────────────────────────────────────────────────────
y12 = 1.85
arrow(ax, CX, y11 - HS/2, CX, y12 + H/2)
box(ax, CX, y12, WM, H,
    "Stage 7 — Validation, Maps & Decision Outputs",
    "FloodSpots spot-check · AUC/PR · SLFVI surface map · Drainage maintenance rankings",
    fill=C_OUT, rounded=True)
badge(ax, CX - WM/2 - 0.52, y12, 7)

# ── Legend ────────────────────────────────────────────────────────────────────
leg = [
    (C_PROC,  "Processing step",      True),
    (C_DATA,  "Data / intermediate",  False),
    (C_INPUT, "External input",       False),
    (C_OUT,   "Final output",         True),
]
lx, ly = 0.55, 1.0
for fill, label, rnd in leg:
    p = FancyBboxPatch(
        (lx, ly - 0.095), 0.3, 0.19,
        boxstyle="round,pad=0.06" if rnd else "square,pad=0.0",
        linewidth=0.8, facecolor=fill, zorder=5,
        edgecolor="#aaaaaa" if fill in (C_DATA, C_INPUT) else fill,
    )
    ax.add_patch(p)
    ax.text(lx + 0.42, ly, label, fontsize=7, va="center", color="#333333", zorder=5)
    lx += 2.45

plt.tight_layout(pad=0.3)
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved → {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
