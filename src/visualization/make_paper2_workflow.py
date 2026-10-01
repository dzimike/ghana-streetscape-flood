"""Generate Figure 1 for Paper 2: coverage gap analysis workflow.

Output: outputs/figures/fig1_paper2_workflow.png
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT  = ROOT / "outputs/figures/fig1_paper2_workflow.png"

C_PROC  = "#1b7837"   # green for this paper (coverage theme)
C_DATA  = "#f2f2f2"
C_INPUT = "#d4edda"
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


def badge(ax, cx, cy, text, color="#1b7837"):
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
fig, ax = plt.subplots(figsize=(11.5, 14))
fig.patch.set_facecolor("white")
ax.set_xlim(0, 11.5)
ax.set_ylim(0, 14)
ax.axis("off")

CX  = 5.0
WM  = 5.8
H   = 0.62
HS  = 0.52
WB  = 2.3
XL  = 2.85
XR  = 7.15

# ── Title ──────────────────────────────────────────────────────────────────────
# ── External inputs ────────────────────────────────────────────────────────────
y_in = 13.25
box(ax, 2.8, y_in, 2.3, HS, "OSM Roads\nFlood strata (DEM)", fill=C_INPUT, fs=7.8)
box(ax, 7.2, y_in, 2.3, HS, "Street View\nMetadata API",     fill=C_INPUT, fs=7.8)

# ── Stage 1: Road segment sampling ────────────────────────────────────────────
y1 = 11.78
arrow(ax, 2.8, y_in - HS/2, 2.8, y1 + H/2)
arrow(ax, 7.2, y_in - HS/2, 7.2, y1 + H/2)
box(ax, CX, y1, WM, H,
    "Stage 1 — Stratified Road-Segment Sampling",
    "5,000 points · 4 flood strata × 3 settlement types · 100 m segments",
    fill=C_PROC, rounded=True)
badge(ax, CX - WM/2 - 0.52, y1, "S1")

# ── Stage 2: Metadata query → split ────────────────────────────────────────────
y2 = 10.58
arrow(ax, CX, y1 - H/2, CX, y2 + H/2)
box(ax, CX, y2, WM, H,
    "Stage 2 — Street View Metadata API Query",
    "5,000 queries · status codes OK / ZERO_RESULTS · deduplication",
    fill=C_PROC, rounded=True)
badge(ax, CX - WM/2 - 0.52, y2, "S2")

# ── Split: covered vs gap ──────────────────────────────────────────────────────
y3 = 9.5
split(ax, CX, y2 - H/2, XL, XR, y3 + HS/2, ymid=9.92)
box(ax, XL, y3, WB, HS, "Covered\n3,912 (78.2%)", fill=C_DATA, fs=8)
box(ax, XR, y3, WB, HS, "Coverage gap\n1,088 (21.8%)", fill="#fde0d9", fs=8)

# ── Stage 3: Coverage bias characterisation ────────────────────────────────────
y4 = 8.47
arrow(ax, XL, y3 - HS/2, XL, y4 + H/2)
arrow(ax, XR, y3 - HS/2, XR, y4 + H/2)
box(ax, XL, y4, WB, H, "Coverage-rate\nmodelling", fill=C_PROC, rounded=True, fs=8)
box(ax, XR, y4, WB, H, "Spatial property\ncomparison", fill=C_PROC, rounded=True, fs=8)

# ── Stage 3 outputs ────────────────────────────────────────────────────────────
y4b = 7.38
arrow(ax, XL, y4 - H/2, XL, y4b + HS/2)
arrow(ax, XR, y4 - H/2, XR, y4b + HS/2)
box(ax, XL, y4b, WB, HS, "Logistic regression\nchi-square (χ²=276.6, p<0.001)", fill=C_DATA, fs=7.2)
box(ax, XR, y4b, WB, HS, "Elevation · dist_waterway\nbuilding density · Cohen's d", fill=C_DATA, fs=7.2)

# ── Merge into Stage 4 ────────────────────────────────────────────────────────
y5 = 6.28
merge(ax, XL, XR, y4b - HS/2, CX, y5 + H/2, ymid=6.75)
box(ax, CX, y5, WM, H,
    "Stage 3 — Coverage Inequality Characterisation",
    "Logistic regression · chi-square · geospatial property comparison · Cohen's d",
    fill=C_PROC, rounded=True)
badge(ax, CX - WM/2 - 0.52, y5, "S3")

# ── Stage 4: Supplementary coverage ───────────────────────────────────────────
y6 = 5.1
arrow(ax, CX, y5 - H/2, CX, y6 + H/2)
box(ax, CX, y6, WM, H,
    "Stage 4 — Supplementary Gap Coverage",
    "Mapillary query for gap points · 159 partial fills (14.6%) · IDW spatial imputation",
    fill=C_PROC, rounded=True)
badge(ax, CX - WM/2 - 0.52, y6, "S4")

# ── Bias index ────────────────────────────────────────────────────────────────
y7 = 3.95
arrow(ax, CX, y6 - H/2, CX, y7 + HS/2)
box(ax, CX, y7, WM, HS,
    "Coverage Bias Index  B = p̂_gap(x) − p̂_covered(x)  ·  Cohen's d per feature",
    fill=C_DATA, fs=8.2)

# ── Stage 5: Adjusted outputs ─────────────────────────────────────────────────
y8 = 2.88
arrow(ax, CX, y7 - HS/2, CX, y8 + H/2)
box(ax, CX, y8, WM, H,
    "Stage 5 — Adjusted Uncertainty and Decision Outputs",
    "Coverage-weighted SLFVI · kriging prediction variance · gap-flagged hotspot map",
    fill=C_OUT, rounded=True)
badge(ax, CX - WM/2 - 0.52, y8, "S5")

# ── Legend ────────────────────────────────────────────────────────────────────
leg = [
    (C_PROC,    "Processing step",      True),
    (C_DATA,    "Data / intermediate",  False),
    (C_INPUT,   "External input",       False),
    ("#fde0d9",  "Coverage gap",        False),
    (C_OUT,     "Final output",         True),
]
lx, ly = 0.25, 2.05
for fill, label, rnd in leg:
    p = FancyBboxPatch(
        (lx, ly - 0.095), 0.28, 0.19,
        boxstyle="round,pad=0.06" if rnd else "square,pad=0.0",
        linewidth=0.8, facecolor=fill, zorder=5,
        edgecolor="#aaaaaa" if fill in (C_DATA, C_INPUT, "#fde0d9") else fill,
    )
    ax.add_patch(p)
    ax.text(lx + 0.38, ly, label, fontsize=7, va="center", color="#333333", zorder=5)
    lx += 2.1

plt.tight_layout(pad=0.3)
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved → {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
