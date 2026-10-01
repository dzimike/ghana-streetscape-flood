"""Compute inter-annotator agreement after double-coding is complete.

Expects three completed annotation sheets in data/interim/double_coding/:
  double_coding_manifest.csv          — lead annotator labels (created by select_double_coding_sample.py)
  annotator_2_completed.csv           — second annotator (same format as blank sheet)
  annotator_3_completed.csv           — third annotator  (same format as blank sheet)

Outputs
-------
outputs/tables/inter_annotator_agreement.csv   — per-label Fleiss' κ and pairwise Cohen's κ
outputs/tables/inter_annotator_summary.txt     — plain-text summary for manuscript
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT   = Path(__file__).resolve().parents[2]
DC_DIR = ROOT / "data/interim/double_coding"
OUT_DIR = ROOT / "outputs/tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

VULN_COL = "vulnerability_class"


# ── Agreement functions ────────────────────────────────────────────────────────

def _to_binary(series: pd.Series) -> pd.Series:
    """Normalise Y/N/1/0/True/False to integer 0/1."""
    s = series.astype(str).str.strip().str.upper()
    return s.map({"Y": 1, "YES": 1, "1": 1, "TRUE": 1,
                  "N": 0, "NO":  0, "0": 0, "FALSE": 0}).astype("Int64")


def cohens_kappa(r1: np.ndarray, r2: np.ndarray) -> float:
    """Pairwise Cohen's κ for binary labels (0/1). Drops NaN pairs."""
    mask = ~(np.isnan(r1.astype(float)) | np.isnan(r2.astype(float)))
    a, b = r1[mask].astype(int), r2[mask].astype(int)
    n = len(a)
    if n == 0:
        return float("nan")
    po = (a == b).mean()
    pe = (a.mean() * b.mean()) + ((1 - a.mean()) * (1 - b.mean()))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def fleiss_kappa(ratings: np.ndarray) -> float:
    """Fleiss' κ for binary ratings matrix (n_subjects × n_raters)."""
    n, k = ratings.shape
    n_cat = 2  # binary: 0 and 1

    # Drop rows with any NaN
    mask = ~np.isnan(ratings.astype(float)).any(axis=1)
    ratings = ratings[mask].astype(int)
    n = len(ratings)
    if n == 0:
        return float("nan")

    # Proportion matrix P_ij
    p_ij = np.zeros((n, n_cat))
    for c in range(n_cat):
        p_ij[:, c] = (ratings == c).sum(axis=1) / k

    p_j = p_ij.mean(axis=0)          # overall proportion per category
    pe  = (p_j ** 2).sum()

    p_bar = (p_ij ** 2).sum(axis=1).mean()
    po    = (k * p_bar - 1) / (k - 1)

    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def interpret_kappa(k: float) -> str:
    if np.isnan(k):
        return "n/a"
    if k < 0:
        return "poor"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"


# ── Data loading ───────────────────────────────────────────────────────────────

def load_rater(path: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in LABEL_COLS:
        if col not in df.columns:
            raise ValueError(f"[{name}] missing column: {col}")
        df[col] = _to_binary(df[col])
    df = df.rename(columns={c: f"{c}__{name}" for c in LABEL_COLS + [VULN_COL]
                             if c in df.columns})
    return df[["image_no"] + [f"{c}__{name}" for c in LABEL_COLS
                               if f"{c}__{name}" in df.columns]]


def main() -> None:
    manifest_path = DC_DIR / "double_coding_manifest.csv"
    ann2_path     = DC_DIR / "annotator_2_completed.csv"
    ann3_path     = DC_DIR / "annotator_3_completed.csv"

    missing = [p for p in [manifest_path, ann2_path, ann3_path] if not p.exists()]
    if missing:
        print("ERROR — missing files:")
        for p in missing:
            print(f"  {p}")
        sys.exit(1)

    print("Loading annotations …")
    manifest = pd.read_csv(manifest_path)
    for col in LABEL_COLS:
        manifest[col] = _to_binary(manifest[col])
    lead = manifest[["image_no"] + LABEL_COLS].copy()
    lead = lead.rename(columns={c: f"{c}__lead" for c in LABEL_COLS})

    ann2 = load_rater(ann2_path, "ann2")
    ann3 = load_rater(ann3_path, "ann3")

    merged = lead.merge(ann2, on="image_no", how="inner").merge(ann3, on="image_no", how="inner")
    print(f"  Merged: {len(merged)} images across 3 raters")

    # ── Per-label agreement ───────────────────────────────────────────────────
    rows = []
    for label in LABEL_COLS:
        r_lead = merged[f"{label}__lead"].to_numpy(dtype=float)
        r_ann2 = merged[f"{label}__ann2"].to_numpy(dtype=float)
        r_ann3 = merged[f"{label}__ann3"].to_numpy(dtype=float)

        fleiss_k  = fleiss_kappa(np.column_stack([r_lead, r_ann2, r_ann3]))
        k_12      = cohens_kappa(r_lead, r_ann2)
        k_13      = cohens_kappa(r_lead, r_ann3)
        k_23      = cohens_kappa(r_ann2, r_ann3)
        mean_pair = np.nanmean([k_12, k_13, k_23])

        # Prevalence of positive label
        prevalence = np.nanmean(np.concatenate([r_lead, r_ann2, r_ann3]))

        rows.append({
            "label":             label,
            "prevalence":        round(prevalence, 3),
            "fleiss_kappa":      round(fleiss_k, 3),
            "interpretation":    interpret_kappa(fleiss_k),
            "cohens_k_lead_ann2": round(k_12, 3),
            "cohens_k_lead_ann3": round(k_13, 3),
            "cohens_k_ann2_ann3": round(k_23, 3),
            "mean_pairwise_k":   round(mean_pair, 3),
        })

    results = pd.DataFrame(rows).sort_values("fleiss_kappa")

    # ── Overall ───────────────────────────────────────────────────────────────
    mean_fleiss = results["fleiss_kappa"].mean()
    min_fleiss  = results["fleiss_kappa"].min()
    max_fleiss  = results["fleiss_kappa"].max()
    worst_label = results.iloc[0]["label"]
    best_label  = results.iloc[-1]["label"]

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_out = OUT_DIR / "inter_annotator_agreement.csv"
    results.to_csv(csv_out, index=False)
    print(f"\n  Agreement table → {csv_out}")

    # ── Print table ───────────────────────────────────────────────────────────
    print("\n── Per-label Fleiss' κ ─────────────────────────────────────────────────────")
    print(f"  {'Label':<38} {'Prev':>5} {'Fleiss κ':>9} {'Interp':>15} {'Mean pair κ':>12}")
    print(f"  {'-'*83}")
    for _, row in results.sort_values("fleiss_kappa", ascending=False).iterrows():
        print(f"  {row['label']:<38} {row['prevalence']:>5.3f} "
              f"{row['fleiss_kappa']:>9.3f} {row['interpretation']:>15} "
              f"{row['mean_pairwise_k']:>12.3f}")
    print(f"\n  Overall mean Fleiss' κ: {mean_fleiss:.3f}  "
          f"(range {min_fleiss:.3f}–{max_fleiss:.3f})")
    print(f"  Lowest: {worst_label}  |  Highest: {best_label}")

    # ── Manuscript-ready summary text ─────────────────────────────────────────
    summary_path = OUT_DIR / "inter_annotator_summary.txt"
    summary = f"""Inter-annotator agreement (Fleiss' κ, n=60 double-coded images, 3 raters)

Mean Fleiss' κ across 15 binary flood-indicator labels: {mean_fleiss:.2f}
Range: {min_fleiss:.2f} (lowest: {worst_label}) to {max_fleiss:.2f} (highest: {best_label})

Per-label results:
{results[['label','prevalence','fleiss_kappa','interpretation','mean_pairwise_k']].to_string(index=False)}

Manuscript sentence (insert into Section 4.4.1 annotation paragraph):
  "Inter-annotator agreement on the 60 double-coded images yielded a mean Fleiss' κ of
  {mean_fleiss:.2f} across the 15 binary labels (range {min_fleiss:.2f}–{max_fleiss:.2f}),
  with the lowest agreement on {worst_label.replace('_',' ')} (κ = {min_fleiss:.2f}),
  reflecting ambiguous boundary conditions in the annotation codebook that were refined
  after the pilot annotation round."
"""
    summary_path.write_text(summary)
    print(f"\n  Manuscript summary → {summary_path}")


if __name__ == "__main__":
    main()
