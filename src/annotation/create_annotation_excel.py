"""Create annotation Excel workbook for human review via Google Drive.

Generates a formatted Excel file with:
  - Sheet 1: Instructions for annotators
  - Sheet 2: Annotation form (400 rows, pre-filled by GPT-4o-mini, editable)
  - Dropdown validation for vulnerability class
  - Colour-coded AI pre-labels for quick review

Upload to Google Drive:
  1. The annotation_review.xlsx file
  2. The annotation_images/ folder (400 images)
  Annotators open each image in Drive and mark their labels in the sheet.

Usage
-----
  python src/annotation/create_annotation_excel.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
from loguru import logger
from openpyxl import Workbook
from openpyxl.styles import (
    Alignment, Border, Fill, Font, GradientFill, PatternFill, Side
)
from openpyxl.styles.differential import DifferentialStyle
from openpyxl.formatting.rule import ColorScaleRule, DataBarRule, Rule
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

REVIEW_CSV = ROOT / "data/interim/human_review_sample.csv"
SAMPLE_CSV = ROOT / "data/interim/annotation_sample.csv"
OUT_EXCEL  = ROOT / "outputs/annotation_review.xlsx"
OUT_IMAGES = ROOT / "outputs/annotation_images_for_drive"

IMAGE_LABELS = [
    "visible_drain_present",
    "open_gutter_present",
    "blocked_drain_present",
    "stagnant_water_visible",
    "poor_road_condition",
    "heavy_impervious_surface",
    "unpaved_shoulder",
    "informal_structure_near_drainage",
    "solid_waste_accumulation",
    "visible_waterway_or_stream",
    "low_lying_street_form",
    "roadside_erosion",
    "pedestrian_exposure",
    "culvert_or_bridge_visible",
    "no_visible_drainage",
]

LABEL_DISPLAY = {
    "visible_drain_present":              "Drain visible",
    "open_gutter_present":                "Open gutter",
    "blocked_drain_present":              "Blocked drain",
    "stagnant_water_visible":             "Stagnant water",
    "poor_road_condition":                "Poor road",
    "heavy_impervious_surface":           "Impervious surface",
    "unpaved_shoulder":                   "Unpaved shoulder",
    "informal_structure_near_drainage":   "Informal structure",
    "solid_waste_accumulation":           "Solid waste",
    "visible_waterway_or_stream":         "Waterway",
    "low_lying_street_form":              "Low-lying street",
    "roadside_erosion":                   "Erosion",
    "pedestrian_exposure":                "Pedestrian exposure",
    "culvert_or_bridge_visible":          "Culvert/bridge",
    "no_visible_drainage":                "No drainage",
}

VULN_COLORS = {
    "low_flood_vulnerability":        "C6EFCE",  # green
    "moderate_flood_vulnerability":   "FFEB9C",  # yellow
    "high_flood_vulnerability":       "FFC7CE",  # red
    "uncertain_requires_field_check": "DDDDDD",  # grey
}

HEADER_FILL   = PatternFill("solid", fgColor="1F4E79")
SUBHEAD_FILL  = PatternFill("solid", fgColor="2E75B6")
ALTROW_FILL   = PatternFill("solid", fgColor="EEF3F9")
LABEL_YES_FILL = PatternFill("solid", fgColor="C6EFCE")
LABEL_NO_FILL  = PatternFill("solid", fgColor="F8F8F8")

THIN = Side(style="thin", color="BBBBBB")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

WHITE_BOLD  = Font(color="FFFFFF", bold=True, size=10)
DARK_BOLD   = Font(bold=True, size=10)
NORMAL_FONT = Font(size=9)
CENTER      = Alignment(horizontal="center", vertical="center", wrap_text=True)
LEFT        = Alignment(horizontal="left",   vertical="center", wrap_text=True)


def write_instructions(ws) -> None:
    ws.column_dimensions["A"].width = 100
    ws.row_dimensions[1].height = 30

    title = ws.cell(1, 1, "Accra Flood Vulnerability — Annotator Instructions")
    title.font = Font(bold=True, size=14, color="1F4E79")
    title.alignment = LEFT

    lines = [
        "",
        "OVERVIEW",
        "You are reviewing 400 street-level images from Accra, Ghana to identify flood vulnerability indicators.",
        "Each image has been pre-labelled by an AI model (GPT-4o-mini). Your job is to confirm or correct the labels.",
        "",
        "HOW TO USE THIS FILE",
        "1. Open the 'Annotation Form' sheet.",
        "2. For each row, click the image filename link to open the image in Google Drive.",
        "3. Look carefully at the image.",
        "4. Check the AI Vulnerability Class — change it if wrong (use the dropdown).",
        "5. Review each flood indicator column — change Y/N if the AI label is incorrect.",
        "6. Add any notes in the 'Notes' column.",
        "7. Set 'Review Status' to 'Done' when you finish a row.",
        "",
        "VULNERABILITY CLASSES",
        "  low_flood_vulnerability          — No visible drainage problems, good road, no waste accumulation",
        "  moderate_flood_vulnerability     — Minor issues: partial blockage, some waste, slight road damage",
        "  high_flood_vulnerability         — Clear problems: blocked drain, standing water, heavy waste, poor road",
        "  uncertain_requires_field_check   — Image is unclear, obstructed, or ambiguous",
        "",
        "FLOOD INDICATOR DEFINITIONS",
        "  Drain visible         — Any drain, gutter, or channel is clearly visible",
        "  Open gutter           — Open-top roadside gutter (concrete, earth, or stone-lined)",
        "  Blocked drain         — Drain or gutter blocked with waste, silt, or debris",
        "  Stagnant water        — Standing or ponded water on road or in drain",
        "  Poor road             — Potholes, cracking, rutting, erosion, or surface failure",
        "  Impervious surface    — Scene dominated by concrete/asphalt, little vegetation",
        "  Unpaved shoulder      — Road shoulder/verge is unpaved (earth, gravel, bare soil)",
        "  Informal structure    — Kiosks, containers, or market stalls next to a drain",
        "  Solid waste           — Visible heap or scatter of plastic bags, refuse, or waste",
        "  Waterway              — Natural or semi-natural stream, river, or channel visible",
        "  Low-lying street      — Road sits in a depression below surrounding land",
        "  Erosion               — Erosion gullies, scour marks, or undercutting at road edge",
        "  Pedestrian exposure   — Pedestrians near flood-risk features (drain edge, flooded road)",
        "  Culvert/bridge        — Culvert pipe or bridge crossing a drainage channel",
        "  No drainage           — No drain, gutter, or waterway infrastructure visible",
        "",
        "IMPORTANT",
        "  • Only label what you can clearly see — do not guess from location or context.",
        "  • If you are unsure about a label, leave the AI label unchanged and add a note.",
        "  • Aim to review 40–50 images per session. Take breaks to maintain accuracy.",
        "  • Contact the research team if you have questions about any image.",
    ]
    for i, line in enumerate(lines, start=2):
        cell = ws.cell(i, 1, line)
        if line in ("OVERVIEW", "HOW TO USE THIS FILE", "VULNERABILITY CLASSES",
                    "FLOOD INDICATOR DEFINITIONS", "IMPORTANT"):
            cell.font = Font(bold=True, size=10, color="1F4E79")
        else:
            cell.font = Font(size=9)
        cell.alignment = LEFT
        ws.row_dimensions[i].height = 15


def write_annotation_sheet(ws, df: pd.DataFrame, drive_folder_url: str = "") -> None:
    # ── Fixed columns ──────────────────────────────────────────────────────────
    fixed_cols = [
        ("No.",             4),
        ("Image File",     28),
        ("Flood Stratum",  18),
        ("Road Type",      14),
        ("AI Class\n(pre-filled)", 22),
        ("Your Class\n(review)",   22),
    ]
    label_cols = [(LABEL_DISPLAY[l], 10) for l in IMAGE_LABELS]
    end_cols = [
        ("Notes",          25),
        ("Review Status",  14),
        ("Annotator",      16),
    ]
    all_cols = fixed_cols + label_cols + end_cols

    # Header row 1 — group labels
    ws.row_dimensions[1].height = 18
    ws.row_dimensions[2].height = 40

    for col_idx, (header, width) in enumerate(all_cols, start=1):
        ws.column_dimensions[get_column_letter(col_idx)].width = width
        cell = ws.cell(2, col_idx, header)
        cell.fill    = HEADER_FILL if col_idx <= len(fixed_cols) else SUBHEAD_FILL
        cell.font    = WHITE_BOLD
        cell.border  = BORDER
        cell.alignment = CENTER

    # Group header for labels
    label_start = len(fixed_cols) + 1
    label_end   = len(fixed_cols) + len(IMAGE_LABELS)
    ws.merge_cells(start_row=1, start_column=label_start,
                   end_row=1,   end_column=label_end)
    grp = ws.cell(1, label_start, "Flood Indicators  (Y = visible, N = not visible)")
    grp.fill      = SUBHEAD_FILL
    grp.font      = WHITE_BOLD
    grp.alignment = CENTER

    # Freeze panes after fixed columns
    ws.freeze_panes = ws.cell(3, len(fixed_cols) + 1)

    # Data validation — vulnerability class dropdown
    vc_col_idx = len(fixed_cols)   # "Your Class" column
    dv = DataValidation(
        type="list",
        formula1='"low_flood_vulnerability,moderate_flood_vulnerability,'
                 'high_flood_vulnerability,uncertain_requires_field_check"',
        allow_blank=True,
        showDropDown=False,
    )
    dv.error      = "Please choose from the dropdown list"
    dv.errorTitle = "Invalid value"
    ws.add_data_validation(dv)
    dv.sqref = f"{get_column_letter(vc_col_idx)}3:{get_column_letter(vc_col_idx)}{len(df)+2}"

    # Status dropdown
    status_col_idx = len(all_cols) - 1
    dv2 = DataValidation(type="list", formula1='"Pending,Done,Skip"',
                         allow_blank=True, showDropDown=False)
    ws.add_data_validation(dv2)
    dv2.sqref = (f"{get_column_letter(status_col_idx)}3:"
                 f"{get_column_letter(status_col_idx)}{len(df)+2}")

    # Y/N validation for label columns
    dv3 = DataValidation(type="list", formula1='"Y,N"',
                         allow_blank=True, showDropDown=False)
    ws.add_data_validation(dv3)
    dv3.sqref = (f"{get_column_letter(label_start)}3:"
                 f"{get_column_letter(label_end)}{len(df)+2}")

    # ── Data rows ──────────────────────────────────────────────────────────────
    for row_idx, (_, row) in enumerate(df.iterrows(), start=3):
        fill = ALTROW_FILL if row_idx % 2 == 0 else PatternFill("solid", fgColor="FFFFFF")
        ws.row_dimensions[row_idx].height = 18

        def c(col, value, **kwargs):
            cell = ws.cell(row_idx, col, value)
            cell.border    = BORDER
            cell.alignment = kwargs.get("align", CENTER)
            cell.font      = kwargs.get("font", NORMAL_FONT)
            cell.fill      = kwargs.get("fill", fill)
            if "color" in kwargs:
                cell.fill = PatternFill("solid", fgColor=kwargs["color"])
            return cell

        img_name = Path(row["local_path"]).name

        c(1, row_idx - 2)                          # No.
        # Image link
        img_cell = ws.cell(row_idx, 2, img_name)
        if drive_folder_url:
            img_cell.hyperlink = drive_folder_url
            img_cell.font = Font(size=9, color="1155CC", underline="single")
        else:
            img_cell.font = NORMAL_FONT
        img_cell.border    = BORDER
        img_cell.alignment = LEFT
        img_cell.fill      = fill

        c(3, row["flood_stratum"])
        c(4, row.get("highway", ""))

        # AI vulnerability class (pre-filled, coloured)
        vc = str(row.get("vulnerability_class", ""))
        ai_color = VULN_COLORS.get(vc, "FFFFFF")
        c(5, vc, color=ai_color, align=LEFT,
          font=Font(size=8, italic=True))

        # Human review class (pre-filled with AI label, editable)
        c(6, vc, color=ai_color, align=LEFT)

        # Label columns (Y/N)
        for li, lbl in enumerate(IMAGE_LABELS):
            val = "Y" if row.get(lbl, 0) == 1 else "N"
            lc = label_start + li
            lbl_fill = LABEL_YES_FILL if val == "Y" else LABEL_NO_FILL
            c(lc, val, color=lbl_fill.fgColor.rgb if val == "Y" else "F8F8F8")

        # Notes, Status, Annotator
        c(len(all_cols) - 2, "", align=LEFT)        # Notes
        c(len(all_cols) - 1, "Pending")             # Status
        c(len(all_cols),     "")                     # Annotator


def copy_images(df: pd.DataFrame) -> None:
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    n = 0
    for _, row in df.iterrows():
        src = Path(row["local_path"])
        dst = OUT_IMAGES / src.name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            n += 1
    logger.info(f"Copied {n} images → {OUT_IMAGES}  ({len(list(OUT_IMAGES.iterdir()))} total)")


def main() -> None:
    review = pd.read_csv(REVIEW_CSV)
    sample = pd.read_csv(SAMPLE_CSV)[["annotation_id", "local_path"]]

    # Merge to get local_path if not already present
    if "local_path" not in review.columns:
        review = review.merge(sample, on="annotation_id")

    logger.info(f"Building Excel for {len(review)} images …")

    wb = Workbook()
    ws_inst = wb.active
    ws_inst.title = "Instructions"
    write_instructions(ws_inst)

    ws_ann = wb.create_sheet("Annotation Form")
    write_annotation_sheet(ws_ann, review, drive_folder_url="")

    OUT_EXCEL.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT_EXCEL)
    size_mb = OUT_EXCEL.stat().st_size / (1024 * 1024)
    logger.info(f"Excel saved → {OUT_EXCEL}  ({size_mb:.1f} MB)")

    logger.info("Copying 400 images for Drive upload …")
    copy_images(review)

    img_size_mb = sum(f.stat().st_size for f in OUT_IMAGES.iterdir()) / (1024 * 1024)

    print("\n── Annotation Package Ready ────────────────────────────────")
    print(f"  Excel file:   outputs/annotation_review.xlsx         ({size_mb:.1f} MB)")
    print(f"  Images:       outputs/annotation_images_for_drive/   ({img_size_mb:.0f} MB, {len(review)} files)")
    print()
    print("── Upload to Google Drive ───────────────────────────────────")
    print("  1. Open Google Drive → New → Folder → name it 'Accra Flood Annotation'")
    print("  2. Upload the folder:  outputs/annotation_images_for_drive/  (drag & drop)")
    print("  3. Upload the file:    outputs/annotation_review.xlsx")
    print("  4. Right-click annotation_review.xlsx → Open with → Google Sheets")
    print("     (Google Sheets lets multiple annotators work simultaneously)")
    print("  5. Share the Google Sheet + image folder with your annotators")
    print()
    print("── Optional: Add image hyperlinks ──────────────────────────")
    print("  After uploading images to Drive:")
    print("  1. Right-click the annotation_images_for_drive folder → Get link → Copy")
    print("  2. Re-run this script with the folder URL:")
    print("     python src/annotation/create_annotation_excel.py --drive-url <url>")
    print("  This adds clickable links in column B so annotators can jump to each image.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--drive-url", default="",
                   help="Google Drive folder URL to hyperlink image filenames")
    args = p.parse_args()

    # Re-run with drive URL if provided
    review = pd.read_csv(REVIEW_CSV)
    sample = pd.read_csv(SAMPLE_CSV)[["annotation_id", "local_path"]]
    if "local_path" not in review.columns:
        review = review.merge(sample, on="annotation_id")

    wb = Workbook()
    ws_inst = wb.active
    ws_inst.title = "Instructions"
    write_instructions(ws_inst)
    ws_ann = wb.create_sheet("Annotation Form")
    write_annotation_sheet(ws_ann, review, drive_folder_url=args.drive_url)
    OUT_EXCEL.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT_EXCEL)

    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    for _, row in review.iterrows():
        src = Path(row["local_path"])
        dst = OUT_IMAGES / src.name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    size_mb = OUT_EXCEL.stat().st_size / (1024*1024)
    img_count = len(list(OUT_IMAGES.iterdir()))
    print(f"Excel → {OUT_EXCEL}  ({size_mb:.1f} MB)")
    print(f"Images → {OUT_IMAGES}  ({img_count} files)")
