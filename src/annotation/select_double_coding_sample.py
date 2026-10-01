"""Select a stratified 60-image subset for double-coding inter-annotator agreement.

Stratification axes:
  - flood_stratum (high_risk_proximity / low_risk_reference)
  - vulnerability_class (high / moderate / low / uncertain)

Output files
------------
data/interim/double_coding/
  double_coding_manifest.csv        — full metadata + lead annotator labels (KEEP PRIVATE)
  annotator_sheet_blank.csv         — blank label columns for second / third annotators
  annotator_sheet_blank.xlsx        — Excel version with dropdown validation (Y/N)
  image_list.txt                    — plain list of local image paths for easy viewing
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
RAW_IMAGES = ROOT / "data/raw/images"
OUT_DIR    = ROOT / "data/interim/double_coding"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GOLD_CSV   = ROOT / "data/interim/gold_labels.csv"
REVIEW_CSV = ROOT / "data/interim/human_review_sample.csv"

N_SAMPLE   = 60
SEED       = 42

LABEL_COLS = [
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

VULN_ORDER = [
    "high_flood_vulnerability",
    "moderate_flood_vulnerability",
    "low_flood_vulnerability",
    "uncertain_requires_field_check",
]


def load_data() -> pd.DataFrame:
    gold = pd.read_csv(GOLD_CSV)
    human = gold[gold["label_source"] == "human"].copy()
    human["pano_id"] = human["image_file"].str.rsplit("_", n=1).str[0]

    review = pd.read_csv(REVIEW_CSV)[
        ["pano_id", "flood_stratum", "highway", "point_id", "local_path"]
    ]
    df = human.merge(review, on="pano_id", how="left")
    df["image_path"] = df["image_file"].apply(
        lambda f: str(RAW_IMAGES / f)
    )
    return df


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Proportional stratified sample across flood_stratum × vulnerability_class."""
    strata_col = "stratum_key"
    df = df.copy()
    df[strata_col] = df["flood_stratum"] + "__" + df["vulnerability_class"]

    counts = df[strata_col].value_counts()
    total  = len(df)
    alloc: dict[str, int] = {}
    for key, cnt in counts.items():
        alloc[key] = max(1, round(n * cnt / total))

    # Adjust to exactly n
    diff = n - sum(alloc.values())
    if diff != 0:
        # Add/remove from the largest strata
        sorted_keys = sorted(alloc, key=lambda k: alloc[k], reverse=(diff > 0))
        for i in range(abs(diff)):
            alloc[sorted_keys[i % len(sorted_keys)]] += int(math.copysign(1, diff))

    rng = np.random.default_rng(seed)
    parts = []
    for key, k in alloc.items():
        pool = df[df[strata_col] == key]
        k = min(k, len(pool))
        idx = rng.choice(len(pool), size=k, replace=False)
        parts.append(pool.iloc[idx])

    sample = pd.concat(parts, ignore_index=True)
    sample = sample.drop(columns=[strata_col])
    # Shuffle so strata order is not visible to annotators
    sample = sample.sample(frac=1, random_state=seed).reset_index(drop=True)
    sample.insert(0, "image_no", range(1, len(sample) + 1))
    return sample


def build_manifest(sample: pd.DataFrame) -> pd.DataFrame:
    """Full record including lead annotator labels — keep private."""
    cols = (
        ["image_no", "image_file", "image_path", "pano_id", "point_id",
         "flood_stratum", "highway", "vulnerability_class"]
        + LABEL_COLS
    )
    return sample[cols].copy()


def build_annotator_sheet(sample: pd.DataFrame) -> pd.DataFrame:
    """Blank sheet for a second or third annotator — no lead labels."""
    meta_cols = ["image_no", "image_file", "flood_stratum", "highway"]
    sheet = sample[meta_cols].copy()
    sheet["vulnerability_class"] = ""
    for label in LABEL_COLS:
        sheet[label] = ""
    sheet["notes"] = ""
    return sheet


def write_excel(sheet: pd.DataFrame, path: Path) -> None:
    """Write blank annotator sheet with Y/N data validation dropdowns."""
    try:
        import openpyxl
        from openpyxl.utils import get_column_letter
        from openpyxl.worksheet.datavalidation import DataValidation
    except ImportError:
        print("  openpyxl not installed — skipping Excel output.")
        return

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        sheet.to_excel(writer, index=False, sheet_name="Annotations")
        ws = writer.sheets["Annotations"]

        # Freeze top row
        ws.freeze_panes = "A2"

        # Auto-width for meta columns
        for col_idx, col_name in enumerate(sheet.columns, start=1):
            width = max(len(str(col_name)), 12)
            ws.column_dimensions[get_column_letter(col_idx)].width = width

        # Y/N dropdown on label columns
        label_start = sheet.columns.tolist().index(LABEL_COLS[0]) + 1
        label_end   = sheet.columns.tolist().index(LABEL_COLS[-1]) + 1
        n_rows = len(sheet) + 1

        dv_yn = DataValidation(type="list", formula1='"Y,N"', allow_blank=True)
        dv_yn.sqref = (
            f"{get_column_letter(label_start)}2:{get_column_letter(label_end)}{n_rows}"
        )
        ws.add_data_validation(dv_yn)

        # Vulnerability class dropdown
        vc_col = get_column_letter(sheet.columns.tolist().index("vulnerability_class") + 1)
        dv_vc = DataValidation(
            type="list",
            formula1='"high_flood_vulnerability,moderate_flood_vulnerability,'
                     'low_flood_vulnerability,uncertain_requires_field_check"',
            allow_blank=True,
        )
        dv_vc.sqref = f"{vc_col}2:{vc_col}{n_rows}"
        ws.add_data_validation(dv_vc)

    print(f"  Excel → {path}")


def print_allocation(sample: pd.DataFrame) -> None:
    print("\n── Sample allocation ───────────────────────────────────────────────────────")
    ct = pd.crosstab(sample["flood_stratum"], sample["vulnerability_class"])
    # Reorder columns
    ordered = [c for c in VULN_ORDER if c in ct.columns]
    print(ct[ordered].to_string())
    print(f"\n  Total: {len(sample)} images  (target: {N_SAMPLE})")


def main() -> None:
    print("Loading annotation data …")
    df = load_data()
    print(f"  Human-labelled pool: {len(df)} images")

    print(f"Stratified sampling (n={N_SAMPLE}, seed={SEED}) …")
    sample = stratified_sample(df, N_SAMPLE, SEED)
    print_allocation(sample)

    manifest = build_manifest(sample)
    annotator_sheet = build_annotator_sheet(sample)

    # Write manifest (private — contains lead labels)
    manifest_path = OUT_DIR / "double_coding_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"\n  Manifest (private) → {manifest_path}")

    # Write blank annotator sheet CSV
    blank_csv = OUT_DIR / "annotator_sheet_blank.csv"
    annotator_sheet.to_csv(blank_csv, index=False)
    print(f"  Blank sheet (CSV)  → {blank_csv}")

    # Write blank annotator sheet Excel
    blank_xlsx = OUT_DIR / "annotator_sheet_blank.xlsx"
    write_excel(annotator_sheet, blank_xlsx)

    # Write plain image path list
    img_list = OUT_DIR / "image_list.txt"
    manifest["image_path"].to_csv(img_list, index=False, header=False)
    print(f"  Image list         → {img_list}")

    # Print kappa computation reminder
    print("\n── Next steps ──────────────────────────────────────────────────────────────")
    print("  1. Give annotator_sheet_blank.xlsx (or .csv) to Annotator 2 and 3.")
    print("  2. Each annotator works independently — do NOT share lead labels.")
    print("  3. Once complete, save as:")
    print("       data/interim/double_coding/annotator_2_completed.csv")
    print("       data/interim/double_coding/annotator_3_completed.csv")
    print("  4. Run: python src/annotation/compute_kappa.py")
    print("     → produces per-label Fleiss' κ and pairwise Cohen's κ table.")


if __name__ == "__main__":
    main()
