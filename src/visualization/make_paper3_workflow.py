"""Generate Figure 1 for Paper 3: streetscape typology workflow.

Output: outputs/figures/fig1_paper3_workflow.png
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT  = ROOT / "outputs/figures/fig1_paper3_workflow.png"

C_PROC  = "#d6604d"   # red-orange for typology paper
C_DATA  = "#f2f2f2"
C_INPUT = "#fddbc7"
C_OUT   = "#762a83"
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


def badge(ax, cx, cy, text, color="#d6604d"):
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=6.5, color="white", fontweight="bold", zorder=5,
            bbox=dict(boxstyle="round,pad=0.22", facecolor=color, edgecolor="none"))


def _vline(ax, x, y1, y2):
    ax.plot([x, x], [y1, y2], color=C_ARROW, lw=1.4, zorder=2, solid_capstyle="round")


def _hline(ax, x1, x2, y):
    ax.plot([x1, x2], [y, y], color=C_ARROW, lw=1.4, zorder=2, solid_capstyle="round")


def _arrowhead_down(ax, x, y_from, y_to):
    _vline(ax, x, y_from, y_to + 0.015)
    ax.annotate("", xy=(x, y_to), xytext=(x, y_to + 0.015), zorder=2,
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.4,
                                mutation_scale=11, connectionstyle="arc3,rad=0"))


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=2,
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.4, mutation_scale=11))


def split(ax, x, y_src, xl, xr, y_dst, ymid=None):
    if ymid is None:
        ymid = (y_src + y_dst) / 2
    _vline(ax, x, y_src, ymid)
    _hline(ax, xl, xr, ymid)
    _arrowhead_down(ax, xl, ymid, y_dst)
    _arrowhead_down(ax, xr, ymid, y_dst)


def merge(ax, xl, xr, y_src, x, y_dst, ymid=None):
    if ymid is None:
        ymid = (y_src + y_dst) / 2
    _vline(ax, xl, y_src, ymid)
    _vline(ax, xr, y_src, ymid)
    _hline(ax, xl, xr, ymid)
    _arrowhead_down(ax, x, ymid, y_dst)


# ── Canvas ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11.5, 15))
fig.patch.set_facecolor("white")
ax.set_xlim(0, 11.5)
ax.set_ylim(0, 15)
ax.axis("off")

CX  = 5.0
WM  = 5.8
H   = 0.62
HS  = 0.52
WB  = 2.3
XL  = 2.85
XR  = 7.15

# ── External input ─────────────────────────────────────────────────────────────
y_in = 14.25
box(ax, CX, y_in, WM, HS,
    "EfficientNet-B0 Predictions  (15,934 images · 15 flood-indicator labels)",
    fill=C_INPUT, fs=8)

# ── Stage 1: Segment aggregation ──────────────────────────────────────────────
y1 = 12.78
arrow(ax, CX, y_in - HS/2, CX, y1 + H/2)
box(ax, CX, y1, WM, H,
    "Stage 1 — Road-Segment Aggregation",
    "Mean across 4 headings per panorama · 3,836 road-point feature vectors",
    fill=C_PROC, rounded=True)
badge(ax, CX - WM/2 - 0.52, y1, "S1")

# ── Stage 2: Feature standardisation ─────────────────────────────────────────
y2 = 11.58
arrow(ax, CX, y1 - H/2, CX, y2 + H/2)
box(ax, CX, y2, WM, H,
    "Stage 2 — Feature Standardisation and Dimensionality Assessment",
    "z-score normalisation (StandardScaler) · PCA: 5 components explain 95% variance",
    fill=C_PROC, rounded=True)
badge(ax, CX - WM/2 - 0.52, y2, "S2")

# ── Stage 3: k selection ──────────────────────────────────────────────────────
y3 = 10.38
arrow(ax, CX, y2 - H/2, CX, y3 + H/2)
box(ax, CX, y3, WM, H,
    "Stage 3 — Cluster Number Selection (k = 2 to 8)",
    "Silhouette coefficient · Calinski-Harabasz index · n_init=20 for stability",
    fill=C_PROC, rounded=True)
badge(ax, CX - WM/2 - 0.52, y3, "S3")

# ── Split: k=2 optimal vs k=6 chosen ─────────────────────────────────────────
y4 = 9.3
split(ax, CX, y3 - H/2, XL, XR, y4 + HS/2, ymid=9.72)
box(ax, XL, y4, WB, HS, "k=2 optimal\n(silhouette 0.3081)", fill=C_DATA, fs=8)
box(ax, XR, y4, WB, HS, "k=6 chosen\n(silhouette 0.2301)", fill=C_DATA, fs=8)

# ── Merge into Stage 4 ────────────────────────────────────────────────────────
y5 = 8.18
merge(ax, XL, XR, y4 - HS/2, CX, y5 + H/2, ymid=8.62)
box(ax, CX, y5, WM, H,
    "Stage 4 — Six-Archetype k-means Typology (k = 6)",
    "n_init=50 · 6 archetypes ranging from vuln_prob 0.734 to 0.150",
    fill=C_PROC, rounded=True)
badge(ax, CX - WM/2 - 0.52, y5, "S4")

# ── Archetype profiles ────────────────────────────────────────────────────────
y6 = 7.08
arrow(ax, CX, y5 - H/2, CX, y6 + HS/2)
box(ax, CX, y6, WM, HS,
    "6 Archetype Mean Profiles  (15-label probability vectors + silhouette per point)",
    fill=C_DATA, fs=8)

# ── Stage 5: Spatial mapping ──────────────────────────────────────────────────
y7 = 5.95
arrow(ax, CX, y6 - HS/2, CX, y7 + H/2)
box(ax, CX, y7, WM, H,
    "Stage 5 — Spatial Mapping and Cross-tabulation",
    "Archetype assignment to road segments · choropleth by municipality",
    fill=C_PROC, rounded=True)
badge(ax, CX - WM/2 - 0.52, y7, "S5")

# ── Split: municipal distribution vs SLFVI comparison ─────────────────────────
y8 = 4.88
split(ax, CX, y7 - H/2, XL, XR, y8 + HS/2, ymid=5.30)
box(ax, XL, y8, WB, HS, "Municipal archetype\ndistribution",  fill=C_PROC, rounded=True, fs=7.8)
box(ax, XR, y8, WB, HS, "SLFVI comparison\nacross archetypes", fill=C_PROC, rounded=True, fs=7.8)

# ── Merge into Stage 6 ────────────────────────────────────────────────────────
y9 = 3.72
merge(ax, XL, XR, y8 - HS/2, CX, y9 + H/2, ymid=4.18)
box(ax, CX, y9, WM, H,
    "Stage 6 — Policy Outputs and Intervention Typology",
    "Drainage maintenance rankings by archetype · intervention priority matrix",
    fill=C_OUT, rounded=True)
badge(ax, CX - WM/2 - 0.52, y9, "S6")

# ── Legend ────────────────────────────────────────────────────────────────────
leg = [
    (C_PROC,  "Processing step",      True),
    (C_DATA,  "Data / intermediate",  False),
    (C_INPUT, "External input",       False),
    (C_OUT,   "Final output",         True),
]
lx, ly = 0.55, 2.88
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
