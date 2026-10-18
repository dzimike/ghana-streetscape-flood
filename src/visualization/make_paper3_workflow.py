"""Generate Figure 1 for Paper 3: streetscape typology workflow.

Cities-paper-style: two-column layout with archetype image examples,
research-context motivation strip, and CLIP feature detail table.

Output: outputs/figures/fig1_paper3_workflow.png
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
OUT     = ROOT / "outputs/figures/fig1_paper3_workflow.png"
IMG_DIR = ROOT / "data/interim/annotation_images"

# ── Palette ───────────────────────────────────────────────────────────────────
C1    = "#8c510a"   # Step 1 – amber-brown (feature preparation)
C2    = "#01665e"   # Step 2 – teal (typology construction)
C3    = "#762a83"   # Step 3 – purple (outputs)
CHIGH = "#d73027"
CMED  = "#f46d43"
CLOW  = "#4575b4"
CDATA = "#f7f7f7"
CINP  = "#f6e8c3"
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
    "CLIP-Based Streetscape Archetype Discovery for Flood Vulnerability  ·  Greater Accra, Ghana",
    fs=9.5, color="#444444", fontstyle="italic")

# ── MOTIVATION STRIP ──────────────────────────────────────────────────────────
rect(ax, 0.35, 14.65, 15.65, 19.15, fill="#fafafa", ec="#cccccc",
     rounded=True, zorder=0)

txt(ax, 0.55, 18.85, "Research context", fs=8.5, fw="bold",
    color=CDK, ha="left")
ax.plot([0.55, 7.65], [18.65, 18.65], color="#cccccc", lw=0.8)

gap_text = (
    "Flood vulnerability in Ghanaian cities manifests through\n"
    "distinct street-level infrastructure patterns: drainage-\n"
    "blocked gutter neighbourhoods, solid-waste-congested\n"
    "informal markets, and formal well-drained suburbs.\n\n"
    "Rather than mapping a single continuous index, this paper\n"
    "uses zero-shot CLIP features and k-means clustering to\n"
    "discover recurring streetscape archetypes, each with a\n"
    "characteristic flood sensitivity profile and targeted\n"
    "infrastructure intervention requirement."
)
txt(ax, 3.9, 16.88, gap_text, fs=8.5, color="#222222",
    multialignment="left", ha="center", va="center", linespacing=1.45)

rect(ax, 0.50, 14.80, 7.65, 14.80+1.0, fill="#fff3cd", ec="#ffc107",
     rounded=True, lw=1.2, zorder=3)
txt(ax, 4.05, 14.80+0.50,
    "Research gap: No typology of street-level flood vulnerability archetypes\n"
    "exists for Sub-Saharan African cities using AI-based zero-shot features.",
    fs=7.8, fw="bold", color="#7d5200", multialignment="center")

# Archetype example images: high sensitivity (left) vs low sensitivity (right)
imgs_hi = [
    _load("QRgw-7wipV5nC5NrE2AW9A_180.jpg"),
    _load("NcXYtTl3r1uG2GZKGFUMTQ_90.jpg"),
]
imgs_lo = [
    _load("XLb7Dx_RuWOBRE8X5dw6SA_270.jpg"),
    _load("LRLUxvdq-BeOIQWEsHYCmA_90.jpg"),
]

rect(ax, 7.85, 14.75, 11.60, 19.05, fill="none",
     ec=CHIGH, lw=1.5, ls="--", zorder=2)
txt(ax, 9.72, 19.10, "Type I: Drainage-blocked (sens. 0.687)", fs=8.0, fw="bold",
    color=CHIGH)
_inset(ax, imgs_hi[0],  7.90, 16.82, 11.55, 19.00)
_inset(ax, imgs_hi[1],  7.90, 14.78, 11.55, 16.78)
txt(ax,  9.72, 16.68, "Open gutters · waste accumulation · unpaved surface",
    fs=6.8, color="#333333", fontstyle="italic")

rect(ax, 11.75, 14.75, 15.55, 19.05, fill="none",
     ec=CLOW, lw=1.5, ls="--", zorder=2)
txt(ax, 13.65, 19.10, "Type VI: Paved formal (sens. 0.244)", fs=8.0, fw="bold",
    color=CLOW)
_inset(ax, imgs_lo[0],  11.80, 16.82, 15.50, 19.00)
_inset(ax, imgs_lo[1],  11.80, 14.78, 15.50, 16.78)
txt(ax, 13.65, 16.68, "Concrete road · formal drainage · impervious surfaces",
    fs=6.8, color="#333333", fontstyle="italic")

# ── STEP COLUMN BORDERS AND HEADERS ──────────────────────────────────────────
col_border(ax, LX0, COL_Y0, LX1, COL_Y1 + HDR_H, C1)
col_border(ax, RX0, COL_Y0, RX1, COL_Y1 + HDR_H, C2)

step_header(ax, LX0+0.08, LX1-0.08, COL_Y1, COL_Y1+HDR_H, C1,
            "1", "CLIP Feature Preparation & Dimensionality Reduction")
step_header(ax, RX0+0.08, RX1-0.08, COL_Y1, COL_Y1+HDR_H, C2,
            "2", "Typology Construction & Spatial Mapping")

# ── STEP 1 CONTENT (left column) ──────────────────────────────────────────────
LC = (LX0 + LX1) / 2

# 1A – Input
Y1A = COL_Y1 - 0.50
rect(ax, LX0+0.25, Y1A-0.38, LX1-0.25, Y1A+0.38,
     fill=CINP, ec="#7b5900", rounded=True, zorder=3)
txt(ax, LC, Y1A+0.12, "CLIP Zero-Shot Feature Matrix", fs=9, fw="bold",
    color=CDK, zorder=4)
txt(ax, LC, Y1A-0.08,
    "3,836 road segments · 15 flood-indicator labels · 4 headings averaged",
    fs=7.5, color="#333333", fontstyle="italic", zorder=4)
txt(ax, LC, Y1A-0.28, "Source: CLIP ViT-B/32 probabilities from Paper 1 pipeline",
    fs=7.2, color="#555555", fontstyle="italic", zorder=4)

arrow_v(ax, LC, Y1A-0.38, Y1A-0.78)

# 1B – Aggregation
Y1B = Y1A - 1.15
rect(ax, LX0+0.25, Y1B-0.36, LX1-0.25, Y1B+0.36,
     fill=C1, ec=C1, rounded=True, zorder=3)
txt(ax, LC, Y1B+0.10, "Road-Segment Aggregation", fs=9, fw="bold",
    color="white", zorder=4)
txt(ax, LC, Y1B-0.12,
    "Mean across 4 headings per panorama  ·  3,836 × 15 feature vectors",
    fs=7.5, color="white", fontstyle="italic", zorder=4)

arrow_v(ax, LC, Y1B-0.36, Y1B-0.72)

# 1C – Standardisation + PCA
Y1C = Y1B - 1.08
rect(ax, LX0+0.25, Y1C-0.42, LX1-0.25, Y1C+0.42,
     fill=C1, ec=C1, rounded=True, zorder=3)
txt(ax, LC, Y1C+0.18, "Standardisation & Dimensionality Assessment", fs=9, fw="bold",
    color="white", zorder=4)
txt(ax, LC, Y1C-0.04,
    "z-score normalisation (StandardScaler)",
    fs=7.5, color="white", fontstyle="italic", zorder=4)
txt(ax, LC, Y1C-0.22,
    "PCA: 5 components explain 95% variance",
    fs=7.5, color="white", fontstyle="italic", zorder=4)
txt(ax, LC, Y1C-0.34, "Correlation heatmap · variance inflation check",
    fs=7.2, color="#dddddd", fontstyle="italic", zorder=4)

arrow_v(ax, LC, Y1C-0.42, Y1C-0.78)

# 1D – 15-label CLIP feature table
TBL_Y1 = Y1C - 0.90
TBL_Y0 = COL_Y0 + 0.35

rect(ax, LX0+0.2, TBL_Y0, LX1-0.2, TBL_Y1,
     fill="white", ec="#999999", lw=1.0, zorder=2)

HDR_ROW_H = 0.38
rect(ax, LX0+0.2, TBL_Y1-HDR_ROW_H, LX1-0.2, TBL_Y1,
     fill=C1, ec=C1, lw=0, zorder=3)
txt(ax, LX0+1.35, TBL_Y1-HDR_ROW_H/2, "CLIP label", fs=7.5, fw="bold",
    color="white", ha="center", zorder=4)
txt(ax, LX0+3.50, TBL_Y1-HDR_ROW_H/2, "Mean score", fs=7.5, fw="bold",
    color="white", ha="center", zorder=4)
txt(ax, LX0+5.40, TBL_Y1-HDR_ROW_H/2, "Cluster role", fs=7.5, fw="bold",
    color="white", ha="center", zorder=4)

vline(ax, LX0+2.50, TBL_Y0, TBL_Y1, color="#cccccc")
vline(ax, LX0+4.40, TBL_Y0, TBL_Y1, color="#cccccc")

clip_rows = [
    ("#fde8d8", "visible_drain_present",         "0.58", "Key discriminator (Types I–III)"),
    ("white",   "open_gutter_present",           "0.52", "High in Type I (gutter districts)"),
    ("#fde8d8", "blocked_drain_present",         "0.47", "Distinguishes Types I vs. II"),
    ("white",   "stagnant_water_visible",        "0.44", "Elevated: Types I, III"),
    ("#fde8d8", "poor_road_condition",           "0.51", "High in Types III–IV"),
    ("white",   "solid_waste_accumulation",      "0.48", "High in Types II, IV"),
    ("#fde8d8", "informal_structure_near_drain", "0.42", "Type IV (encroachment zones)"),
    ("white",   "unpaved_shoulder",              "0.46", "Separates formal from informal"),
    ("#fde8d8", "heavy_impervious_surface",      "0.41", "High in Type VI (formal paved)"),
    ("white",   "no_visible_drainage",           "0.39", "Dominant in Type V"),
]
n_rows = len(clip_rows)
avail_h = TBL_Y1 - HDR_ROW_H - TBL_Y0
row_h = avail_h / n_rows

for i, (bg, label, score, role) in enumerate(clip_rows):
    ry0 = TBL_Y0 + (n_rows - 1 - i) * row_h
    ry1 = ry0 + row_h
    rect(ax, LX0+0.2, ry0, LX1-0.2, ry1, fill=bg, ec="none", lw=0, zorder=2)
    hline(ax, LX0+0.2, LX1-0.2, ry1, color="#cccccc", lw=0.6)
    txt(ax, LX0+1.35, (ry0+ry1)/2, label, fs=6.0, fw="bold",
        color="#1a1a1a", ha="center", family="monospace", zorder=4)
    txt(ax, LX0+3.50, (ry0+ry1)/2, score, fs=7.2, fw="bold",
        color="#1a1a1a", ha="center", zorder=4)
    txt(ax, LX0+4.45, (ry0+ry1)/2, role, fs=6.2, color="#333333",
        ha="left", zorder=4)

txt(ax, LC, TBL_Y0-0.18,
    "15 CLIP labels  ·  +5 remaining omitted for brevity  ·  153 total features (Paper 1 pipeline)",
    fs=7.5, fw="bold", color=C1)

# ── STEP 2 CONTENT (right column) ─────────────────────────────────────────────
RC = (RX0 + RX1) / 2

# Horizontal arrow: features → typology
arrow_h(ax, LX1+0.08, RX0+0.15, COL_Y1 - 1.60)
txt(ax, (LX1+RX0)/2, COL_Y1-1.40, "Standardised\nfeature matrix", fs=7.5, fw="bold",
    color=C1, ha="center")

# 2A – k selection
Y2A = COL_Y1 - 0.50
rect(ax, RX0+0.25, Y2A-0.46, RX1-0.25, Y2A+0.46,
     fill=C2, ec=C2, rounded=True, zorder=3)
txt(ax, RC, Y2A+0.20, "Cluster Number Selection  (k = 2 to 8)", fs=9, fw="bold",
    color="white", zorder=4)
txt(ax, RC, Y2A-0.00,
    "Silhouette coefficient  ·  Calinski-Harabasz index",
    fs=7.5, color="white", fontstyle="italic", zorder=4)
txt(ax, RC, Y2A-0.20,
    "n_init=20 for stability  ·  tested on z-scored 15-feature space",
    fs=7.5, color="white", fontstyle="italic", zorder=4)
txt(ax, RC, Y2A-0.38, "k=2: silhouette=0.2799 (optimal)  ·  k=6: silhouette=0.1679 (policy)",
    fs=7.5, color="#ffffcc", fontstyle="italic", zorder=4)

arrow_v(ax, RC, Y2A-0.46, Y2A-0.82)

# 2B – k=6 k-means
Y2B = Y2A - 1.20
rect(ax, RX0+0.25, Y2B-0.42, RX1-0.25, Y2B+0.42,
     fill=C2, ec=C2, rounded=True, zorder=3)
txt(ax, RC, Y2B+0.16, "Six-Archetype k-means Typology  (k = 6)", fs=9, fw="bold",
    color="white", zorder=4)
txt(ax, RC, Y2B-0.04, "n_init=50  ·  max_iter=300  ·  random_state=42",
    fs=7.5, color="white", fontstyle="italic", zorder=4)
txt(ax, RC, Y2B-0.22,
    "Sensitivity score = mean of 15 CLIP flood-indicator labels",
    fs=7.5, color="white", fontstyle="italic", zorder=4)
txt(ax, RC, Y2B-0.36,
    "Range: Type I 0.687  →  Type VI 0.244",
    fs=7.5, color="#ffffcc", fontstyle="italic", zorder=4)

arrow_v(ax, RC, Y2B-0.42, Y2B-0.80)

# 2C – Archetype profiles (mini bar chart style table)
Y2C = Y2B - 1.15
rect(ax, RX0+0.25, Y2C-0.30, RX1-0.25, Y2C+0.30,
     fill=CDATA, ec="#555577", rounded=True, zorder=3)
txt(ax, RC, Y2C+0.06, "Six Archetype Mean Profiles", fs=9, fw="bold", zorder=4)
txt(ax, RC, Y2C-0.14,
    "Type I (0.687) · II (0.574) · III (0.512) · IV (0.441) · V (0.358) · VI (0.244)",
    fs=7.5, color="#333333", fontstyle="italic", zorder=4)

arrow_v(ax, RC, Y2C-0.30, Y2C-0.65)

# 2D – Spatial mapping split
Y2D = Y2C - 1.02
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
txt(ax, RCL, Y2D-0.15, "Choropleth by\nmunicipality", fs=8.0, fw="bold",
    color="white", zorder=4)
txt(ax, RCR, Y2D-0.15, "Road-segment\narchetype map", fs=8.0, fw="bold",
    color="white", zorder=4)

vline(ax, RCL, Y2D-0.30, Y2D-0.55)
vline(ax, RCR, Y2D-0.30, Y2D-0.55)
hline(ax, RCL, RCR, Y2D-0.55)
arrow_v(ax, RC, Y2D-0.55, Y2D-0.82, color=CARR)

# 2E – SLFVI cross-tabulation
Y2E = Y2D - 1.08
rect(ax, RX0+0.25, Y2E-0.30, RX1-0.25, Y2E+0.30,
     fill=CDATA, ec="#555577", rounded=True, zorder=3)
txt(ax, RC, Y2E+0.06, "SLFVI Cross-tabulation by Archetype", fs=9, fw="bold", zorder=4)
txt(ax, RC, Y2E-0.15,
    "ANOVA  ·  Kruskal-Wallis  ·  pairwise Dunn test  ·  effect sizes",
    fs=7.5, color="#444444", fontstyle="italic", zorder=4)

arrow_v(ax, RC, Y2E-0.30, COL_Y0+0.30)

# ── STEP 3 (full width) ───────────────────────────────────────────────────────
S3_Y0 = 0.40; S3_Y1 = COL_Y0 - 0.10

col_border(ax, LX0, S3_Y0, RX1, S3_Y1, C3)
rect(ax, LX0+0.10, S3_Y1-0.52, LX0+5.0, S3_Y1+0.00,
     fill=C3, ec=C3, rounded=True, zorder=4)
txt(ax, LX0+2.55, S3_Y1-0.26,
    "Step 3  ·  Policy Typology & Intervention Outputs",
    fs=9.5, fw="bold", color="white", zorder=5)


def s3box(x0, x1, title, body):
    rect(ax, x0, S3_Y0+0.15, x1, S3_Y1-0.58,
         fill=C3, ec=C3, rounded=True, zorder=3)
    txt(ax, (x0+x1)/2, S3_Y1-0.88, title, fs=8.5, fw="bold",
        color="white", zorder=4)
    txt(ax, (x0+x1)/2, (S3_Y0+S3_Y1)/2 - 0.20, body, fs=7.5,
        color="white", multialignment="center", linespacing=1.4, zorder=4)


s3box(LX0+0.18, LX0+4.80,
      "Archetype Validation",
      "SLFVI × archetype ANOVA  (p<0.001)\n"
      "Pairwise Dunn test with Bonferroni correction\n"
      "Municipal distribution cross-tabulation\n"
      "Silhouette & Calinski-Harabasz scores")

s3box(LX0+5.05, LX0+9.50,
      "Spatial Archetype Map",
      "Road-segment archetype choropleth\n"
      "Six-class vulnerability surface\n"
      "Municipality-level type distribution\n"
      "Coverage-gap uncertainty overlay")

s3box(LX0+9.75, RX1-0.18,
      "Intervention Priority Matrix",
      "Drainage investment by archetype\n"
      "Top-100 segments per vulnerability type\n"
      "Targeted intervention protocol\n"
      "NADMO / GMA planning input")

# ── LEGEND ────────────────────────────────────────────────────────────────────
leg = [
    (C1,    "Step 1: Feature preparation"),
    (C2,    "Step 2: Typology construction"),
    (C3,    "Step 3: Policy outputs"),
    (CINP,  "External input data"),
    (CDATA, "Intermediate / cross-tab"),
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
