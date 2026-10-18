"""Generate Figure 1 for Paper 2: coverage gap analysis workflow.

Cities-paper-style: two-column layout with image thumbnails,
research-context motivation strip, and bias characterisation detail.

Output: outputs/figures/fig1_paper2_workflow.png
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

ROOT    = Path(__file__).resolve().parents[2]
OUT     = ROOT / "outputs/figures/fig1_paper2_workflow.png"
IMG_DIR = ROOT / "data/interim/annotation_images"

# ── Palette ───────────────────────────────────────────────────────────────────
C1    = "#1b7837"   # Step 1 – green  (coverage assessment)
C2    = "#762a83"   # Step 2 – purple (bias analysis)
C3    = "#b35806"   # Step 3 – amber  (outputs)
CGAP  = "#d73027"   # coverage-gap accent
CCOV  = "#4575b4"   # covered accent
CDATA = "#f7f7f7"
CINP  = "#d4edda"
CARR  = "#444444"
CDK   = "#1a1a1a"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size":   8.0,
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
})


# ── Helpers ───────────────────────────────────────────────────────────────────
def _load(fname, size=(380, 280)):
    p = IMG_DIR / fname
    if not p.exists():
        return np.ones((280, 380, 3), dtype=np.uint8) * 200
    return np.array(Image.open(p).resize(size))


def _inset(ax, img, x0, y0, x1, y1):
    xl, xr = ax.get_xlim(); yl, yr = ax.get_ylim()
    inset = ax.inset_axes(
        [(x0-xl)/(xr-xl), (y0-yl)/(yr-yl),
         (x1-x0)/(xr-xl), (y1-y0)/(yr-yl)]
    )
    inset.imshow(img)
    inset.axis("off")
    return inset


def rect(ax, x0, y0, x1, y1, fill=CDATA, ec=None, lw=1.1,
         ls="solid", alpha=1.0, zorder=2, rounded=False):
    style = "round,pad=0.06" if rounded else "square,pad=0"
    if ec is None:
        ec = "#999999" if fill in (CDATA, CINP, "white", "#fafafa") else fill
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1-x0, y1-y0,
        boxstyle=style, linewidth=lw, zorder=zorder,
        edgecolor=ec, facecolor=fill, linestyle=ls, alpha=alpha,
    ))


def txt(ax, x, y, s, fs=8.0, fw="normal", color=CDK, ha="center",
        va="center", zorder=5, **kw):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, fontweight=fw,
            color=color, zorder=zorder, **kw)


def arrow_v(ax, x, y1, y2, color=CARR, mut=12):
    ax.annotate("", xy=(x, y2), xytext=(x, y1), zorder=4,
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.3, mutation_scale=mut))


def arrow_h(ax, x1, x2, y, color=CARR, mut=12):
    ax.annotate("", xy=(x2, y), xytext=(x1, y), zorder=4,
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.3, mutation_scale=mut))


def step_header(ax, x0, x1, y0, y1, color, num, label):
    rect(ax, x0, y0, x1, y1, fill=color, ec=color, rounded=True, zorder=3)
    bw = 0.82
    rect(ax, x0+0.12, y0+0.08, x0+0.12+bw, y1-0.08,
         fill="white", ec="white", rounded=True, zorder=4)
    txt(ax, x0+0.12+bw/2, (y0+y1)/2, f"Step\n{num}",
        fs=7.5, fw="bold", color=color, zorder=5)
    txt(ax, (x0+0.12+bw+x1)/2, (y0+y1)/2, label,
        fs=10, fw="bold", color="white", zorder=5)


def col_border(ax, x0, y0, x1, y1, color):
    rect(ax, x0, y0, x1, y1, fill="none", ec=color, lw=1.8,
         ls="--", alpha=1.0, zorder=1, rounded=True)


def hline(ax, x0, x1, y, color=CARR, lw=1.0):
    ax.plot([x0, x1], [y, y], color=color, lw=lw, zorder=3)


def vline(ax, x, y0, y1, color=CARR, lw=1.0):
    ax.plot([x, x], [y0, y1], color=color, lw=lw, zorder=3)


# ── Canvas ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 20))
fig.patch.set_facecolor("white")
ax.set_xlim(0, 16); ax.set_ylim(0, 20)
ax.axis("off")

LX0, LX1 = 0.35, 7.85
RX0, RX1 = 8.15, 15.65
COL_Y0   = 3.90
COL_Y1   = 14.25
HDR_H    = 0.70

# ── TITLE ─────────────────────────────────────────────────────────────────────
txt(ax, 8.0, 19.70, "Figure 1.  Methodological Workflow",
    fs=14, fw="bold")
txt(ax, 8.0, 19.42,
    "Street-Level Data Voids and Flood Vulnerability Inequality  ·  Greater Accra, Ghana",
    fs=9.5, color="#444444", fontstyle="italic")

# ── MOTIVATION STRIP ──────────────────────────────────────────────────────────
rect(ax, 0.35, 14.65, 15.65, 19.15, fill="#fafafa", ec="#cccccc",
     rounded=True, zorder=0)

# Research-gap text (left)
txt(ax, 0.55, 18.85, "Research context", fs=8.5, fw="bold",
    color=CDK, ha="left")
ax.plot([0.55, 7.65], [18.65, 18.65], color="#cccccc", lw=0.8)

gap_text = (
    "Google Street View provides near-global coverage of urban\n"
    "roads, but coverage is uneven. In rapidly urbanising cities\n"
    "such as Accra, imagery is often absent in precisely the\n"
    "informal settlements and flood-prone areas most in need\n"
    "of vulnerability assessment.\n\n"
    "This paper characterises street-level coverage gaps,\n"
    "quantifies their spatial and socio-environmental bias,\n"
    "and produces coverage-adjusted flood vulnerability\n"
    "estimates with explicit uncertainty attribution."
)
txt(ax, 3.9, 16.88, gap_text, fs=8.5, color="#222222",
    multialignment="left", ha="center", va="center", linespacing=1.45)

rect(ax, 0.50, 14.80, 7.65, 14.80+1.0, fill="#fff3cd", ec="#ffc107",
     rounded=True, lw=1.2, zorder=3)
txt(ax, 4.05, 14.80+0.50,
    "Research gap: Street View coverage voids are systematically\n"
    "concentrated in the most flood-vulnerable urban locations.",
    fs=7.8, fw="bold", color="#7d5200", multialignment="center")

# Image panels: covered vs gap areas
imgs_cov = [
    _load("XLb7Dx_RuWOBRE8X5dw6SA_270.jpg"),
    _load("LRLUxvdq-BeOIQWEsHYCmA_90.jpg"),
]
imgs_gap = [
    _load("QRgw-7wipV5nC5NrE2AW9A_180.jpg"),
    _load("NcXYtTl3r1uG2GZKGFUMTQ_90.jpg"),
]

# Covered panel
rect(ax, 7.85, 14.75, 11.60, 19.05, fill="none",
     ec=CCOV, lw=1.5, ls="--", zorder=2)
txt(ax, 9.72, 19.10, "Street View covered areas", fs=8.0, fw="bold",
    color=CCOV)
_inset(ax, imgs_cov[0],  7.90, 16.82, 11.55, 19.00)
_inset(ax, imgs_cov[1],  7.90, 14.78, 11.55, 16.78)
txt(ax,  9.72, 16.68, "Formal roads  ·  paved surfaces  ·  commercial zones",
    fs=6.8, color="#333333", fontstyle="italic")

# Coverage-gap panel
rect(ax, 11.75, 14.75, 15.55, 19.05, fill="none",
     ec=CGAP, lw=1.5, ls="--", zorder=2)
txt(ax, 13.65, 19.10, "Coverage-gap locations (21.8%)", fs=8.0, fw="bold",
    color=CGAP)
_inset(ax, imgs_gap[0],  11.80, 16.82, 15.50, 19.00)
_inset(ax, imgs_gap[1],  11.80, 14.78, 15.50, 16.78)
txt(ax, 13.65, 16.68, "Informal settlements  ·  flood-prone areas  ·  unpaved roads",
    fs=6.8, color="#333333", fontstyle="italic")

# ── STEP COLUMN BORDERS AND HEADERS ──────────────────────────────────────────
col_border(ax, LX0, COL_Y0, LX1, COL_Y1 + HDR_H, C1)
col_border(ax, RX0, COL_Y0, RX1, COL_Y1 + HDR_H, C2)

step_header(ax, LX0+0.08, LX1-0.08, COL_Y1, COL_Y1+HDR_H, C1,
            "1", "Coverage Assessment & Gap Characterisation")
step_header(ax, RX0+0.08, RX1-0.08, COL_Y1, COL_Y1+HDR_H, C2,
            "2", "Bias Analysis & Spatial Imputation")

# ── STEP 1 CONTENT (left column) ──────────────────────────────────────────────
LC = (LX0 + LX1) / 2

# 1A – Sampling
Y1A = COL_Y1 - 0.50
rect(ax, LX0+0.25, Y1A-0.42, LX1-0.25, Y1A+0.42,
     fill=CINP, ec="#338855", rounded=True, zorder=3)
txt(ax, LC, Y1A+0.14, "Stratified Road-Segment Sampling", fs=9, fw="bold",
    color=CDK, zorder=4)
txt(ax, LC, Y1A-0.10,
    "OSM roads · 5,000 sample points · 4 flood strata × 3 settlement types",
    fs=7.5, color="#333333", fontstyle="italic", zorder=4)
txt(ax, LC, Y1A-0.30, "100 m segment intervals · GeoPackage export",
    fs=7.5, color="#333333", fontstyle="italic", zorder=4)

arrow_v(ax, LC, Y1A-0.42, Y1A-0.80)

# 1B – Metadata query
Y1B = Y1A - 1.18
rect(ax, LX0+0.25, Y1B-0.38, LX1-0.25, Y1B+0.38,
     fill=C1, ec=C1, rounded=True, zorder=3)
txt(ax, LC, Y1B+0.10, "Street View Metadata API Query", fs=9, fw="bold",
    color="white", zorder=4)
txt(ax, LC, Y1B-0.14,
    "5,000 queries  ·  status: OK / ZERO_RESULTS  ·  deduplication",
    fs=7.5, color="white", fontstyle="italic", zorder=4)

arrow_v(ax, LC, Y1B-0.38, Y1B-0.72)

# 1C – Coverage split visualisation
Y1C = Y1B - 1.55
# Two status boxes
rect(ax, LX0+0.25, Y1C-0.32, LX0+3.5, Y1C+0.32,
     fill=CCOV, ec=CCOV, rounded=True, zorder=3)
rect(ax, LX0+3.65, Y1C-0.32, LX1-0.25, Y1C+0.32,
     fill=CGAP, ec=CGAP, rounded=True, zorder=3)
txt(ax, LX0+1.88, Y1C+0.06, "Covered", fs=10, fw="bold", color="white", zorder=4)
txt(ax, LX0+1.88, Y1C-0.18, "3,912 points  (78.2%)", fs=8, color="white", zorder=4)
txt(ax, LX0+5.0,  Y1C+0.06, "Coverage gap", fs=10, fw="bold", color="white", zorder=4)
txt(ax, LX0+5.0,  Y1C-0.18, "1,088 points  (21.8%)", fs=8, color="white", zorder=4)

arrow_v(ax, LC, Y1C-0.32, Y1C-0.68)

# 1D – Coverage rate table
TBL_Y1 = Y1C - 0.80
TBL_Y0 = COL_Y0 + 0.35

rect(ax, LX0+0.2, TBL_Y0, LX1-0.2, TBL_Y1,
     fill="white", ec="#999999", lw=1.0, zorder=2)

HDR_ROW_H = 0.38
rect(ax, LX0+0.2, TBL_Y1-HDR_ROW_H, LX1-0.2, TBL_Y1,
     fill=C1, ec=C1, lw=0, zorder=3)
txt(ax, LX0+1.10, TBL_Y1-HDR_ROW_H/2, "Stratum", fs=7.5, fw="bold",
    color="white", ha="center", zorder=4)
txt(ax, LX0+2.80, TBL_Y1-HDR_ROW_H/2, "Coverage rate", fs=7.5, fw="bold",
    color="white", ha="center", zorder=4)
txt(ax, LX0+5.00, TBL_Y1-HDR_ROW_H/2, "Key property", fs=7.5, fw="bold",
    color="white", ha="center", zorder=4)

vline(ax, LX0+1.90, TBL_Y0, TBL_Y1, color="#cccccc")
vline(ax, LX0+3.70, TBL_Y0, TBL_Y1, color="#cccccc")

cov_rows = [
    ("#e8f4fd", "High-risk flood\ndepression",   "61.4 %",  "Elev < 15 m · dist_waterway < 50 m"),
    ("white",   "Informal\nsettlement",           "69.8 %",  "Low building quality · unpaved access"),
    ("#e8f4fd", "Peri-urban\nexpansion",          "74.2 %",  "Rapid growth · limited road inventory"),
    ("white",   "Formal\nresidential",            "91.3 %",  "Paved roads · OSM coverage complete"),
    ("#e8f4fd", "Commercial\ndistrict",           "95.7 %",  "High road density · landmark presence"),
]
n_rows = len(cov_rows)
avail_h = TBL_Y1 - HDR_ROW_H - TBL_Y0
row_h = avail_h / n_rows

for i, (bg, stratum, rate, prop) in enumerate(cov_rows):
    ry0 = TBL_Y0 + (n_rows - 1 - i) * row_h
    ry1 = ry0 + row_h
    rect(ax, LX0+0.2, ry0, LX1-0.2, ry1, fill=bg, ec="none", lw=0, zorder=2)
    hline(ax, LX0+0.2, LX1-0.2, ry1, color="#cccccc", lw=0.6)
    txt(ax, LX0+1.05, (ry0+ry1)/2, stratum, fs=6.8, fw="bold",
        color="#1a1a1a", ha="center", multialignment="center", zorder=4)
    txt(ax, LX0+2.80, (ry0+ry1)/2, rate, fs=7.5, fw="bold",
        color="#1a1a1a", ha="center", zorder=4)
    txt(ax, LX0+3.75, (ry0+ry1)/2, prop, fs=6.2, color="#333333",
        ha="left", multialignment="left", zorder=4)

txt(ax, LC, TBL_Y0-0.18,
    "Coverage gap concentrated in high flood-risk strata  ·  systematic spatial bias",
    fs=7.5, fw="bold", color=C1)

# ── STEP 2 CONTENT (right column) ─────────────────────────────────────────────
RC = (RX0 + RX1) / 2

# Horizontal arrow from Step 1 gap data → Step 2
arrow_h(ax, LX1+0.08, RX0+0.15, COL_Y1 - 1.45)
txt(ax, (LX1+RX0)/2, COL_Y1-1.25, "Gap-point\nlocation data", fs=7.5, fw="bold",
    color=CGAP, ha="center")

# 2A – Logistic regression
Y2A = COL_Y1 - 0.50
rect(ax, RX0+0.25, Y2A-0.46, RX1-0.25, Y2A+0.46,
     fill=C2, ec=C2, rounded=True, zorder=3)
txt(ax, RC, Y2A+0.20, "Coverage-Rate Logistic Regression", fs=9, fw="bold",
    color="white", zorder=4)
txt(ax, RC, Y2A-0.02,
    "DV: covered (1) / gap (0)  ·  IVs: elevation, slope, dist_waterway,",
    fs=7.5, color="white", fontstyle="italic", zorder=4)
txt(ax, RC, Y2A-0.24,
    "building density, road type, settlement class",
    fs=7.5, color="white", fontstyle="italic", zorder=4)
txt(ax, RC, Y2A-0.36, "χ²=276.6, p<0.001  ·  pseudo-R²=0.21",
    fs=7.5, color="#ffffcc", fontstyle="italic", zorder=4)

arrow_v(ax, RC, Y2A-0.46, Y2A-0.85)

# 2B – Spatial comparison
Y2B = Y2A - 1.28
rect(ax, RX0+0.25, Y2B-0.42, RX1-0.25, Y2B+0.42,
     fill=C2, ec=C2, rounded=True, zorder=3)
txt(ax, RC, Y2B+0.16, "Gap vs. Covered Spatial Property Comparison", fs=9, fw="bold",
    color="white", zorder=4)
txt(ax, RC, Y2B-0.05,
    "Elevation · slope · TWI · dist_waterway · dist_drain",
    fs=7.5, color="white", fontstyle="italic", zorder=4)
txt(ax, RC, Y2B-0.24,
    "Building density · population · GHSL built-up · road class",
    fs=7.5, color="white", fontstyle="italic", zorder=4)
txt(ax, RC, Y2B-0.38, "Cohen's d=0.317 (elevation)  ·  Mann-Whitney U  ·  KS test",
    fs=7.5, color="#ffffcc", fontstyle="italic", zorder=4)

arrow_v(ax, RC, Y2B-0.42, Y2B-0.78)

# 2C – Coverage Bias Index
Y2C = Y2B - 1.10
rect(ax, RX0+0.25, Y2C-0.30, RX1-0.25, Y2C+0.30,
     fill=CDATA, ec="#555577", rounded=True, zorder=3)
txt(ax, RC, Y2C+0.05,
    "Coverage Bias Index  B(x)  =  p̂_gap(x) − p̂_covered(x)",
    fs=9.0, fw="bold", color=CDK, zorder=4)
txt(ax, RC, Y2C-0.16,
    "Per-feature Cohen's d  ·  maps where gaps distort vulnerability estimates most",
    fs=7.2, color="#444444", fontstyle="italic", zorder=4)

arrow_v(ax, RC, Y2C-0.30, Y2C-0.65)

# 2D – Mapillary + IDW imputation
Y2D = Y2C - 1.02
# Split: Mapillary + IDW
RCL = RX0 + 1.65
RCR = RX1 - 1.65
vline(ax, RC, Y2C-0.65, Y2D+0.28)
hline(ax, RCL, RCR, Y2D+0.28)
arrow_v(ax, RCL, Y2D+0.28, Y2D+0.00, color=CARR)
arrow_v(ax, RCR, Y2D+0.28, Y2D+0.00, color=CARR)
rect(ax, RX0+0.25, Y2D-0.30, RX0+3.05, Y2D+0.00,
     fill=C2, ec=C2, rounded=True, zorder=3)
rect(ax, RX1-3.05, Y2D-0.30, RX1-0.25, Y2D+0.00,
     fill=C2, ec=C2, rounded=True, zorder=3)
txt(ax, RCL, Y2D-0.15, "Mapillary\ngap fill  (159 pts)", fs=8.0, fw="bold",
    color="white", zorder=4)
txt(ax, RCR, Y2D-0.15, "IDW spatial\nimputation", fs=8.0, fw="bold",
    color="white", zorder=4)

vline(ax, RCL, Y2D-0.30, Y2D-0.55)
vline(ax, RCR, Y2D-0.30, Y2D-0.55)
hline(ax, RCL, RCR, Y2D-0.55)
arrow_v(ax, RC, Y2D-0.55, Y2D-0.82, color=CARR)

# 2E – Coverage-weighted SLFVI
Y2E = Y2D - 1.08
rect(ax, RX0+0.25, Y2E-0.30, RX1-0.25, Y2E+0.30,
     fill=CDATA, ec="#555577", rounded=True, zorder=3)
txt(ax, RC, Y2E+0.06, "Coverage-Weighted SLFVI", fs=9, fw="bold", zorder=4)
txt(ax, RC, Y2E-0.15,
    "Kriging variance-adjusted index  ·  gap-flagged hotspot overlay",
    fs=7.5, color="#444444", fontstyle="italic", zorder=4)

arrow_v(ax, RC, Y2E-0.30, COL_Y0+0.30)

# ── STEP 3 (full width, below columns) ───────────────────────────────────────
S3_Y0 = 0.40; S3_Y1 = COL_Y0 - 0.10

col_border(ax, LX0, S3_Y0, RX1, S3_Y1, C3)
rect(ax, LX0+0.10, S3_Y1-0.52, LX0+5.0, S3_Y1+0.00,
     fill=C3, ec=C3, rounded=True, zorder=4)
txt(ax, LX0+2.55, S3_Y1-0.26,
    "Step 3  ·  Coverage-Adjusted Outputs & Recommendations",
    fs=9.5, fw="bold", color="white", zorder=5)


def s3box(x0, x1, title, body):
    rect(ax, x0, S3_Y0+0.15, x1, S3_Y1-0.58,
         fill=C3, ec=C3, rounded=True, zorder=3)
    txt(ax, (x0+x1)/2, S3_Y1-0.88, title, fs=8.5, fw="bold",
        color="white", zorder=4)
    txt(ax, (x0+x1)/2, (S3_Y0+S3_Y1)/2 - 0.20, body, fs=7.5,
        color="white", multialignment="center", linespacing=1.4, zorder=4)


s3box(LX0+0.18, LX0+4.80,
      "Inequality Evidence",
      "Coverage gap in high flood-risk strata\n"
      "χ²=276.6, p<0.001  ·  Cohen's d=0.317\n"
      "Logistic regression OR interpretation\n"
      "Elevation & proximity strongest predictors")

s3box(LX0+5.05, LX0+9.50,
      "Gap-Adjusted Vulnerability Map",
      "Coverage-weighted SLFVI surface\n"
      "Kriging uncertainty overlay\n"
      "Gap-flagged hotspot locations\n"
      "Sensitivity analysis vs. naïve SLFVI")

s3box(LX0+9.75, RX1-0.18,
      "Recommendations",
      "Priority imagery capture zones\n"
      "for NADMO / municipal assemblies\n"
      "Mapillary contribution campaigns\n"
      "Local smartphone imagery protocol")

# ── LEGEND ────────────────────────────────────────────────────────────────────
leg = [
    (C1,    "Step 1: Coverage assessment"),
    (C2,    "Step 2: Bias analysis & imputation"),
    (C3,    "Step 3: Adjusted outputs"),
    (CINP,  "External input data"),
    (CDATA, "Intermediate / index"),
]
lx = 0.50
for fill, label in leg:
    rect(ax, lx, 0.07, lx+0.38, 0.33,
         fill=fill, ec="#888888" if fill in (CINP, CDATA) else fill,
         rounded=True, zorder=5, lw=0.8)
    txt(ax, lx+0.58, 0.20, label, fs=7.5, color="#333333", ha="left", zorder=5)
    lx += 3.05

plt.tight_layout(pad=0.4)
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved → {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")
