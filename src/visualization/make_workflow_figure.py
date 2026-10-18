"""Generate Figure 1 for Paper 1: SLFVI methodological workflow.

Cities-paper-style: two-column layout with street view thumbnails,
feature extraction table, and research-context motivation strip.

Output: outputs/figures/fig1_workflow.png
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
OUT     = ROOT / "outputs/figures/fig1_workflow.png"
IMG_DIR = ROOT / "data/interim/annotation_images"

# ── Palette ───────────────────────────────────────────────────────────────────
C1    = "#2166ac"   # Step 1 – blue  (data collection / feature extraction)
C2    = "#1b7837"   # Step 2 – green (geospatial fusion)
C3    = "#762a83"   # Step 3 – purple (validation / outputs)
CHIGH = "#d73027"   # high-vulnerability accent
CLOW  = "#4575b4"   # low-vulnerability accent
CDATA = "#f7f7f7"   # data / intermediate boxes
CINP  = "#d1e5f0"   # external-input boxes
CARR  = "#444444"   # arrows
CDK   = "#1a1a1a"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size":   8.0,
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
})


# ── Helpers ───────────────────────────────────────────────────────────────────
def _load(fname, size=(380, 280)):
    return np.array(Image.open(IMG_DIR / fname).resize(size))


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
    """Colored column header with Step badge."""
    rect(ax, x0, y0, x1, y1, fill=color, ec=color, rounded=True, zorder=3)
    # Badge
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

# column geometry
LX0, LX1 = 0.35, 7.85   # Step 1 left column
RX0, RX1 = 8.15, 15.65  # Step 2 right column
COL_Y0   = 3.90          # bottom of step columns
COL_Y1   = 14.25         # top of step column content (below header)
HDR_H    = 0.70          # step header height

# ── TITLE ─────────────────────────────────────────────────────────────────────
txt(ax, 8.0, 19.70, "Figure 1.  Methodological Workflow",
    fs=14, fw="bold")
txt(ax, 8.0, 19.42,
    "Street-Level Flood Vulnerability Index (SLFVI)  ·  Greater Accra, Ghana",
    fs=9.5, color="#444444", fontstyle="italic")

# ── MOTIVATION STRIP ──────────────────────────────────────────────────────────
# Outer frame
rect(ax, 0.35, 14.65, 15.65, 19.15, fill="#fafafa", ec="#cccccc",
     rounded=True, zorder=0)

# Research-gap text (left half)
txt(ax, 0.55, 18.85, "Research context", fs=8.5, fw="bold",
    color=CDK, ha="left", va="center")
ax.plot([0.55, 7.65], [18.65, 18.65], color="#cccccc", lw=0.8)

gap_text = (
    "Conventional flood mapping relies on satellite imagery and\n"
    "digital elevation models, which cannot capture the street-\n"
    "level infrastructure conditions that drive urban flooding:\n"
    "blocked drains, open gutters, solid-waste accumulation,\n"
    "unpaved roads, and encroachment near waterways.\n\n"
    "This study uses AI applied to Google Street View imagery\n"
    "to detect these visible vulnerability indicators and fuse\n"
    "them with hydrological and exposure data into a\n"
    "neighborhood-scale flood vulnerability index."
)
txt(ax, 3.9, 16.90, gap_text, fs=8.5, color="#222222",
    multialignment="left", ha="center", va="center", linespacing=1.45)

# Research gap box
rect(ax, 0.50, 14.80, 7.65, 14.80+1.0, fill="#fff3cd", ec="#ffc107",
     rounded=True, lw=1.2, zorder=3)
txt(ax, 4.05, 14.80+0.50,
    "Research gap: AI-based street-level flood vulnerability mapping\n"
    "has not been systematically applied in Sub-Saharan African cities.",
    fs=7.8, fw="bold", color="#7d5200", multialignment="center")

# Image grid (right half): 2 high-vuln + 2 low-vuln
imgs_high = [
    _load("QRgw-7wipV5nC5NrE2AW9A_180.jpg"),
    _load("NcXYtTl3r1uG2GZKGFUMTQ_90.jpg"),
]
imgs_low = [
    _load("XLb7Dx_RuWOBRE8X5dw6SA_270.jpg"),
    _load("LRLUxvdq-BeOIQWEsHYCmA_90.jpg"),
]
# High-vuln panel (left half of image area)
rect(ax, 7.85, 14.75, 11.60, 19.05, fill="none",
     ec=CHIGH, lw=1.5, ls="--", zorder=2)
txt(ax, 9.72, 19.10, "High flood vulnerability", fs=8.0, fw="bold",
    color=CHIGH)
_inset(ax, imgs_high[0],  7.90, 16.82, 11.55, 19.00)
_inset(ax, imgs_high[1],  7.90, 14.78, 11.55, 16.78)
txt(ax,  9.72, 16.68, "Solid waste · unpaved surface · drainage obstruction",
    fs=6.8, color="#333333", fontstyle="italic")

# Low-vuln panel (right half)
rect(ax, 11.75, 14.75, 15.55, 19.05, fill="none",
     ec=CLOW, lw=1.5, ls="--", zorder=2)
txt(ax, 13.65, 19.10, "Low flood vulnerability", fs=8.0, fw="bold",
    color=CLOW)
_inset(ax, imgs_low[0],  11.80, 16.82, 15.50, 19.00)
_inset(ax, imgs_low[1],  11.80, 14.78, 15.50, 16.78)
txt(ax, 13.65, 16.68, "Paved roads · formal drainage · impervious surfaces",
    fs=6.8, color="#333333", fontstyle="italic")

# ── STEP COLUMN BORDERS AND HEADERS ──────────────────────────────────────────
# Column dashed borders
col_border(ax, LX0, COL_Y0, LX1, COL_Y1 + HDR_H, C1)
col_border(ax, RX0, COL_Y0, RX1, COL_Y1 + HDR_H, C2)

# Step headers
step_header(ax, LX0+0.08, LX1-0.08, COL_Y1, COL_Y1+HDR_H, C1,
            "1", "Street-Level Data Collection & Feature Extraction")
step_header(ax, RX0+0.08, RX1-0.08, COL_Y1, COL_Y1+HDR_H, C2,
            "2", "Geospatial Fusion & Vulnerability Index")

# ── STEP 1 CONTENT (left column) ──────────────────────────────────────────────
LC = (LX0 + LX1) / 2   # left column centre x
LW = LX1 - LX0 - 0.5   # box width

# 1A – Sampling
Y1A = COL_Y1 - 0.48
rect(ax, LX0+0.25, Y1A-0.38, LX1-0.25, Y1A+0.38,
     fill=C1, ec=C1, rounded=True, zorder=3)
txt(ax, LC, Y1A+0.10, "Stratified Road-Segment Sampling", fs=9, fw="bold",
    color="white", zorder=4)
txt(ax, LC, Y1A-0.14,
    "5,000 points  ·  4 flood strata × 3 settlement types  ·  100 m segments",
    fs=7.5, color="white", fontstyle="italic", zorder=4)

arrow_v(ax, LC, Y1A-0.38, Y1A-0.85)

# 1B – Metadata query
Y1B = Y1A - 1.28
rect(ax, LX0+0.25, Y1B-0.38, LX1-0.25, Y1B+0.38,
     fill=C1, ec=C1, rounded=True, zorder=3)
txt(ax, LC, Y1B+0.10, "Street View Metadata API Query", fs=9, fw="bold",
    color="white", zorder=4)
txt(ax, LC, Y1B-0.14,
    "5,000 queries  →  3,836 valid panoramas  ·  1,088 coverage-gap points",
    fs=7.5, color="white", fontstyle="italic", zorder=4)

arrow_v(ax, LC, Y1B-0.38, Y1B-0.78)

# 1C – Images
Y1C = Y1B - 1.52
# Two image thumbnails side-by-side
img_mid = _load("OqWCH03o7Z85lpG-etMKfw_180.jpg")   # medium
img_hi2  = _load("2JT6uPGX5ao-uNs-H8oiGQ_270.jpg",  size=(360,270))
rect(ax, LX0+0.2, Y1C-0.42, LX0+3.8, Y1C+0.82,
     fill=CDATA, ec="#aaaaaa", rounded=False, zorder=2)
rect(ax, LX0+3.9, Y1C-0.42, LX1-0.2, Y1C+0.82,
     fill=CDATA, ec="#aaaaaa", rounded=False, zorder=2)
_inset(ax, img_mid, LX0+0.22, Y1C-0.40, LX0+3.78, Y1C+0.80)
_inset(ax, img_hi2,  LX0+3.92, Y1C-0.40, LX1-0.22, Y1C+0.80)
txt(ax, LC, Y1C+1.00, "4 headings per panorama  ·  15,344 images collected",
    fs=7.5, color="#333333")

arrow_v(ax, LC, Y1C-0.42, Y1C-0.88)

# 1D – Feature extraction table header
TBL_Y1 = Y1C - 1.00   # table top
TBL_Y0 = COL_Y0 + 0.35  # table bottom

# Table outer border
rect(ax, LX0+0.2, TBL_Y0, LX1-0.2, TBL_Y1,
     fill="white", ec="#999999", lw=1.0, zorder=2)

# Column header row
HDR_ROW_H = 0.38
rect(ax, LX0+0.2, TBL_Y1 - HDR_ROW_H, LX1-0.2, TBL_Y1,
     fill=C1, ec=C1, lw=0, zorder=3)
txt(ax, LX0+1.05, TBL_Y1-HDR_ROW_H/2, "Model", fs=7.5, fw="bold",
    color="white", ha="center", zorder=4)
txt(ax, LX0+2.45, TBL_Y1-HDR_ROW_H/2, "Category", fs=7.5, fw="bold",
    color="white", ha="center", zorder=4)
txt(ax, LX0+4.70, TBL_Y1-HDR_ROW_H/2, "Sample outputs", fs=7.5, fw="bold",
    color="white", ha="center", zorder=4)

# Vertical dividers
vline(ax, LX0+1.90, TBL_Y0, TBL_Y1, color="#cccccc")
vline(ax, LX0+3.10, TBL_Y0, TBL_Y1, color="#cccccc")

feature_rows = [
    ("#e8f4fd", "CLIP\nViT-B/32",
     "Flood\nindicators\n(15 labels)",
     "drain_present  0.58\nblocked_drain  0.47\nwaste_visible   0.51\nstagnant_water 0.44\n…  →  15 scores"),
    ("white",   "SegFormer-B0\nADE20K",
     "Semantic\nsegmentation\n(150 classes)",
     "road           0.42\nvegetation    0.31\nbuilding       0.12\nsky             0.28\n…  →  pixel %"),
    ("#e8f4fd", "Places365\nWRN-18",
     "Scene\ncategory\n(365 scenes)",
     "residential    0.87\npaved road     0.04\nsquare         0.03\n…  →  top-5 prob."),
    ("white",   "YOLOv8n\nCOCO",
     "Object\ndetection\n(80 classes)",
     "person    2 instances\ncar         1 instance\nbicycle          0\n…  →  count / class"),
    ("#e8f4fd", "Pixel\nstatistics",
     "Image\nquality\n(9 metrics)",
     "brightness     142\nsaturation      58\nblur score     0.61\n…  →  raw values"),
]

n_rows = len(feature_rows)
avail_h = TBL_Y1 - HDR_ROW_H - TBL_Y0
row_h = avail_h / n_rows

for i, (bg, model, cat, feats) in enumerate(feature_rows):
    ry0 = TBL_Y0 + (n_rows - 1 - i) * row_h
    ry1 = ry0 + row_h
    rect(ax, LX0+0.2, ry0, LX1-0.2, ry1, fill=bg, ec="none", lw=0, zorder=2)
    hline(ax, LX0+0.2, LX1-0.2, ry1, color="#cccccc", lw=0.6)
    txt(ax, LX0+1.05, (ry0+ry1)/2, model, fs=6.8, fw="bold",
        color="#1a1a1a", ha="center", multialignment="center", zorder=4)
    txt(ax, LX0+2.50, (ry0+ry1)/2, cat,   fs=6.5, color="#333333",
        ha="center", multialignment="center", zorder=4)
    txt(ax, LX0+3.15, (ry0+ry1)/2, feats, fs=6.0, color="#333333",
        ha="left", multialignment="left", zorder=4, family="monospace")

# Total features label below table
txt(ax, LC, TBL_Y0-0.18,
    "Total: 153 features per image  →  15 CLIP segment-level scores (mean of 4 headings)",
    fs=7.5, fw="bold", color=C1)

# ── STEP 2 CONTENT (right column) ─────────────────────────────────────────────
RC = (RX0 + RX1) / 2   # right column centre

# 2A – Terrain/hydrology/exposure inputs (side inputs)
Y2A = COL_Y1 - 0.48
rect(ax, RX0+0.25, Y2A-0.48, RX1-0.25, Y2A+0.48,
     fill=CINP, ec="#555577", rounded=True, zorder=3)
txt(ax, RC, Y2A+0.20, "Terrain · Hydrology · Urban Exposure", fs=9, fw="bold",
    color=CDK, zorder=4)
txt(ax, RC, Y2A-0.05,
    "FABDEM elevation · slope · TWI · flow accumulation · depression depth",
    fs=7.5, color="#333333", fontstyle="italic", zorder=4)
txt(ax, RC, Y2A-0.28,
    "MERIT Hydro drains · JRC surface water · WorldPop · OSM buildings · GHSL",
    fs=7.5, color="#333333", fontstyle="italic", zorder=4)

# Horizontal arrow: CLIP scores from Step 1 → enters Step 2
arrow_h(ax, LX1+0.08, RX0+0.15, Y2A-1.52)
txt(ax, (LX1+RX0)/2, Y2A-1.32, "Streetscape\nfeatures", fs=7.5, fw="bold",
    color=C1, ha="center")

# 2B – LightGBM fusion
Y2B = Y2A - 1.95
arrow_v(ax, RC, Y2A-0.48, Y2B+0.42)
rect(ax, RX0+0.25, Y2B-0.42, RX1-0.25, Y2B+0.42,
     fill=C2, ec=C2, rounded=True, zorder=3)
txt(ax, RC, Y2B+0.15, "LightGBM Fusion Model", fs=9, fw="bold",
    color="white", zorder=4)
txt(ax, RC, Y2B-0.05,
    "34 features (streetscape + terrain)  ·  spatial block 5-fold CV",
    fs=7.5, color="white", fontstyle="italic", zorder=4)
txt(ax, RC, Y2B-0.26,
    "ROC-AUC 0.988  ·  PR-AUC 0.832  ·  F1 0.758",
    fs=7.5, color="white", fontstyle="italic", zorder=4)

arrow_v(ax, RC, Y2B-0.42, Y2B-0.85)

# 2C – SLFVI formula
Y2C = Y2B - 1.22
rect(ax, RX0+0.25, Y2C-0.30, RX1-0.25, Y2C+0.30,
     fill=CDATA, ec="#555577", rounded=True, zorder=3)
txt(ax, RC, Y2C+0.04,
    "SLFVI  =  0.30·H  +  0.20·E  +  0.35·S  +  0.15·(1−A)",
    fs=9.5, fw="bold", color=CDK, zorder=4)
txt(ax, RC, Y2C-0.18,
    "H=Hazard · E=Exposure · S=Streetscape sensitivity · A=Adaptive capacity",
    fs=7.0, color="#444444", fontstyle="italic", zorder=4)

arrow_v(ax, RC, Y2C-0.30, Y2C-0.68)

# 2D – Spatial interpolation (split)
Y2D = Y2C - 1.05
RCL = RX0 + 1.65   # left branch centre
RCR = RX1 - 1.65   # right branch centre
# split arrow
vline(ax, RC, Y2C-0.68, Y2D+0.28)
hline(ax, RCL, RCR, Y2D+0.28)
arrow_v(ax, RCL, Y2D+0.28, Y2D+0.00, color=CARR)
arrow_v(ax, RCR, Y2D+0.28, Y2D+0.00, color=CARR)
rect(ax, RX0+0.25, Y2D-0.30, RX0+3.05, Y2D+0.00,
     fill=C2, ec=C2, rounded=True, zorder=3)
rect(ax, RX1-3.05, Y2D-0.30, RX1-0.25, Y2D+0.00,
     fill=C2, ec=C2, rounded=True, zorder=3)
txt(ax, RCL, Y2D-0.15, "IDW Interpolation", fs=8.0, fw="bold",
    color="white", zorder=4)
txt(ax, RCR, Y2D-0.15, "Ordinary Kriging", fs=8.0, fw="bold",
    color="white", zorder=4)

# merge
vline(ax, RCL, Y2D-0.30, Y2D-0.58)
vline(ax, RCR, Y2D-0.30, Y2D-0.58)
hline(ax, RCL, RCR, Y2D-0.58)
arrow_v(ax, RC, Y2D-0.58, Y2D-0.85, color=CARR)

# 2E – 250 m grid
Y2E = Y2D - 1.10
rect(ax, RX0+0.25, Y2E-0.28, RX1-0.25, Y2E+0.28,
     fill=CDATA, ec="#555577", rounded=True, zorder=3)
txt(ax, RC, Y2E+0.06, "250 m Vulnerability Grid", fs=9, fw="bold", zorder=4)
txt(ax, RC, Y2E-0.15,
    "SLFVI surface  ·  Kriging prediction variance (uncertainty map)",
    fs=7.5, color="#444444", fontstyle="italic", zorder=4)

arrow_v(ax, RC, Y2E-0.28, COL_Y0+0.30)

# ── STEP 3 (full width, below columns) ───────────────────────────────────────
S3_Y0 = 0.40
S3_Y1 = COL_Y0 - 0.10
S3_MID = (S3_Y0 + S3_Y1) / 2

# Step 3 dashed border
col_border(ax, LX0, S3_Y0, RX1, S3_Y1, C3)
# Step 3 header tab
rect(ax, LX0+0.10, S3_Y1-0.52, LX0+5.0, S3_Y1+0.00,
     fill=C3, ec=C3, rounded=True, zorder=4)
txt(ax, LX0+2.55, S3_Y1-0.26, "Step 3  ·  Validation & Decision Outputs",
    fs=9.5, fw="bold", color="white", zorder=5)

# Three output columns inside Step 3
def s3box(x0, x1, title, body):
    rect(ax, x0, S3_Y0+0.15, x1, S3_Y1-0.58,
         fill=C3, ec=C3, rounded=True, zorder=3)
    txt(ax, (x0+x1)/2, S3_Y1-0.88, title, fs=8.5, fw="bold",
        color="white", zorder=4)
    txt(ax, (x0+x1)/2, (S3_Y0+S3_Y1)/2 - 0.20, body, fs=7.5,
        color="white", multialignment="center", linespacing=1.4, zorder=4)

s3box(LX0+0.18, LX0+4.80,
      "Accuracy Validation",
      "FloodSpots field-spot comparison\n"
      "ROC-AUC 0.988  ·  PR-AUC 0.832\n"
      "Spatial block 5-fold CV\n"
      "Brier score  ·  calibration curve")

s3box(LX0+5.05, LX0+9.50,
      "SLFVI Surface Map",
      "250 m flood vulnerability grid\n"
      "Road-segment risk scores\n"
      "Kriging uncertainty overlay\n"
      "Municipal vulnerability profiles")

s3box(LX0+9.75, RX1-0.18,
      "Decision Outputs",
      "Top-100 drainage maintenance\n"
      "priority road segments\n"
      "Coverage-gap-flagged hotspots\n"
      "Policy brief for NADMO / GMA")

# ── LEGEND ────────────────────────────────────────────────────────────────────
leg = [
    (C1,    "Step 1: Data collection & feature extraction"),
    (C2,    "Step 2: Geospatial fusion"),
    (C3,    "Step 3: Validation & outputs"),
    (CINP,  "External input data"),
    (CDATA, "Intermediate / formula"),
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
