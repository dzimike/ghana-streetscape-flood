"""Process completed human annotations from Google Sheets.

Computes:
  - AI vs Human agreement (Cohen's kappa, % agreement per label)
  - Gold label reconciliation (human labels take precedence)
  - Disagreement flags for third review
  - Manuscript-ready tables

Outputs
-------
  data/interim/gold_labels.csv                     — final reconciled labels
  outputs/tables/annotation_agreement.csv          — AI vs Human agreement metrics
  outputs/tables/annotation_label_comparison.csv   — per-label AI vs Human rates
  outputs/tables/annotation_disagreements.csv      — images where AI ≠ Human
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report

COMPLETED_CSV = ROOT / "data/interim/human_annotations_completed.csv"
AI_CSV        = ROOT / "data/interim/openai_annotation_summary.csv"
REVIEW_CSV    = ROOT / "data/interim/human_review_sample.csv"
SAMPLE_CSV    = ROOT / "data/interim/annotation_sample.csv"

OUT_GOLD      = ROOT / "data/interim/gold_labels.csv"
OUT_AGREEMENT = ROOT / "outputs/tables/annotation_agreement.csv"
OUT_LABEL_CMP = ROOT / "outputs/tables/annotation_label_comparison.csv"
OUT_DISAGREE  = ROOT / "outputs/tables/annotation_disagreements.csv"

IMAGE_LABELS = [
    "visible_drain_present", "open_gutter_present", "blocked_drain_present",
    "stagnant_water_visible", "poor_road_condition", "heavy_impervious_surface",
    "unpaved_shoulder", "informal_structure_near_drainage", "solid_waste_accumulation",
    "visible_waterway_or_stream", "low_lying_street_form", "roadside_erosion",
    "pedestrian_exposure", "culvert_or_bridge_visible", "no_visible_drainage",
]

# Map short display names back to canonical label names
DISPLAY_TO_LABEL = {
    "Drain visible":        "visible_drain_present",
    "Open gutter":          "open_gutter_present",
    "Blocked drain":        "blocked_drain_present",
    "Stagnant water":       "stagnant_water_visible",
    "Poor road":            "poor_road_condition",
    "Impervious surface":   "heavy_impervious_surface",
    "Unpaved shoulder":     "unpaved_shoulder",
    "Informal structure":   "informal_structure_near_drainage",
    "Solid waste":          "solid_waste_accumulation",
    "Waterway":             "visible_waterway_or_stream",
    "Low-lying street":     "low_lying_street_form",
    "Erosion":              "roadside_erosion",
    "Pedestrian exposure":  "pedestrian_exposure",
    "Culvert/bridge":       "culvert_or_bridge_visible",
    "No drainage":          "no_visible_drainage",
}

VULN_ORDER = [
    "low_flood_vulnerability",
    "moderate_flood_vulnerability",
    "high_flood_vulnerability",
    "uncertain_requires_field_check",
]


def load_human(path: Path) -> pd.DataFrame:
    """Load the completed Google Sheets export (row 1 is merged header, row 2 is real header)."""
    df = pd.read_csv(path, header=1)
    df.columns = df.columns.str.strip().str.replace(r"\n.*", "", regex=True)

    # Rename columns
    df = df.rename(columns={
        "Image File":    "image_file",
        "Flood Stratum": "flood_stratum",
        "Road Type":     "highway",
        "AI Class":      "ai_class_excel",
        "Your Class":    "human_class",
        "Notes":         "notes",
        "Review Status": "review_status",
        "Annotator":     "annotator",
    })
    df.columns = [DISPLAY_TO_LABEL.get(c, c) for c in df.columns]

    # Keep only Done rows
    done = df[df["review_status"].str.strip().str.lower() == "done"].copy()
    logger.info(f"Human annotations: {len(done)} Done / {len(df)} total rows")

    # Convert Y/N label columns to 0/1
    for lbl in IMAGE_LABELS:
        if lbl in done.columns:
            done[lbl] = done[lbl].map({"Y": 1, "N": 0, "y": 1, "n": 0}).fillna(0).astype(int)

    return done


def load_ai(done: pd.DataFrame) -> pd.DataFrame:
    """Load AI labels for the same images."""
    ai  = pd.read_csv(AI_CSV)
    smpl = pd.read_csv(SAMPLE_CSV)[["annotation_id", "local_path"]]
    ai  = ai.merge(smpl, on="annotation_id")
    ai["image_file"] = ai["local_path"].apply(lambda p: Path(p).name)
    return ai.merge(done[["image_file"]], on="image_file")


def kappa(y_true, y_pred) -> float:
    try:
        if len(np.unique(y_true)) < 2 and len(np.unique(y_pred)) < 2:
            return 1.0 if (y_true == y_pred).all() else 0.0
        return cohen_kappa_score(y_true, y_pred)
    except Exception:
        return float("nan")


def pct_agree(y_true, y_pred) -> float:
    return float((y_true == y_pred).mean() * 100)


def main() -> None:
    OUT_AGREEMENT.parent.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    human = load_human(COMPLETED_CSV)
    ai    = load_ai(human)
    merged = human.merge(ai, on="image_file", suffixes=("_human", "_ai"))
    logger.info(f"Matched {len(merged)} images between human and AI annotations")

    # ── 1. Vulnerability class agreement ─────────────────────────────────────
    # Normalise classes
    vc_human = merged["human_class"].str.strip()
    vc_ai    = merged["vulnerability_class"].str.strip()

    vc_agree_pct = pct_agree(vc_human, vc_ai)
    vc_kappa     = kappa(vc_human, vc_ai)

    print("\n── Vulnerability Class Agreement (AI vs Human) ─────────────")
    print(f"  % Agreement:   {vc_agree_pct:.1f}%")
    print(f"  Cohen's kappa: {vc_kappa:.3f}  ", end="")
    if vc_kappa >= 0.80:   print("(almost perfect)")
    elif vc_kappa >= 0.60: print("(substantial)")
    elif vc_kappa >= 0.40: print("(moderate)")
    elif vc_kappa >= 0.20: print("(fair)")
    else:                  print("(slight)")

    print()
    print("  Confusion matrix (rows=AI, cols=Human):")
    labels_present = sorted(set(vc_ai) | set(vc_human))
    cm = confusion_matrix(vc_ai, vc_human, labels=labels_present)
    cm_df = pd.DataFrame(cm, index=[f"AI:{l[:8]}" for l in labels_present],
                             columns=[f"H:{l[:8]}" for l in labels_present])
    print(cm_df.to_string())

    # ── 2. Per-label agreement ────────────────────────────────────────────────
    label_rows = []
    for lbl in IMAGE_LABELS:
        h_col = f"{lbl}_human" if f"{lbl}_human" in merged.columns else lbl
        a_col = f"{lbl}_ai"   if f"{lbl}_ai"   in merged.columns else lbl

        if h_col not in merged.columns or a_col not in merged.columns:
            continue

        h = merged[h_col].fillna(0).astype(int)
        a = merged[a_col].fillna(0).astype(int)

        n_pos_human = int(h.sum())
        n_pos_ai    = int(a.sum())
        agree       = pct_agree(h, a)
        k           = kappa(h, a)

        label_rows.append({
            "label":          lbl,
            "n_pos_ai":       n_pos_ai,
            "pct_pos_ai":     round(100 * n_pos_ai / len(merged), 1),
            "n_pos_human":    n_pos_human,
            "pct_pos_human":  round(100 * n_pos_human / len(merged), 1),
            "pct_agreement":  round(agree, 1),
            "cohen_kappa":    round(k, 3),
        })

    label_df = pd.DataFrame(label_rows).sort_values("cohen_kappa", ascending=False)
    label_df.to_csv(OUT_LABEL_CMP, index=False)

    print("\n── Per-Label Agreement (AI vs Human) ───────────────────────")
    print(f"  {'Label':<45} {'AI%':>5} {'H%':>5} {'Agree%':>7} {'κ':>7}")
    print(f"  {'-'*72}")
    for _, r in label_df.iterrows():
        print(f"  {r['label']:<45} {r['pct_pos_ai']:>5.1f} {r['pct_pos_human']:>5.1f} "
              f"{r['pct_agreement']:>6.1f}% {r['cohen_kappa']:>7.3f}")

    # ── 3. Disagreements ──────────────────────────────────────────────────────
    disagree_mask = vc_human != vc_ai
    pick_cols = ["image_file"]
    for c in ["flood_stratum_human", "flood_stratum", "highway_human", "highway"]:
        if c in merged.columns:
            pick_cols.append(c)
            break
    for c in ["highway_human", "highway"]:
        if c in merged.columns and c not in pick_cols:
            pick_cols.append(c)
            break
    pick_cols += ["vulnerability_class", "human_class"]
    if "notes" in merged.columns:
        pick_cols.append("notes")

    disagree_df = merged[disagree_mask][pick_cols].copy()
    disagree_df.columns = (["image_file", "flood_stratum", "highway",
                             "ai_class", "human_class"] +
                            (["notes"] if "notes" in pick_cols else []))
    disagree_df.to_csv(OUT_DISAGREE, index=False)
    logger.info(f"Disagreements → {OUT_DISAGREE}  ({len(disagree_df)} images)")

    print(f"\n── Vulnerability Class Disagreements ────────────────────────")
    print(f"  Total disagreements: {len(disagree_df)} / {len(merged)} ({100*len(disagree_df)/len(merged):.1f}%)")
    if len(disagree_df) > 0:
        print("\n  Most common disagreement patterns:")
        patterns = (disagree_df.groupby(["ai_class", "human_class"])
                    .size().reset_index(name="count")
                    .sort_values("count", ascending=False).head(8))
        for _, r in patterns.iterrows():
            print(f"    AI={r['ai_class'][:25]:<26} → Human={r['human_class'][:25]:<26} ({r['count']})")

    # ── 4. Build gold labels ──────────────────────────────────────────────────
    # Human label is gold; fill remaining 1,600 images with AI labels
    gold_human = human[["image_file", "human_class"] + IMAGE_LABELS].copy()
    gold_human = gold_human.rename(columns={"human_class": "vulnerability_class"})
    gold_human["label_source"] = "human"

    # AI labels for non-reviewed images
    ai_full = pd.read_csv(AI_CSV)
    ai_full = ai_full.merge(
        pd.read_csv(SAMPLE_CSV)[["annotation_id", "local_path"]], on="annotation_id"
    )
    ai_full["image_file"] = ai_full["local_path"].apply(lambda p: Path(p).name)
    ai_rest = ai_full[~ai_full["image_file"].isin(gold_human["image_file"])].copy()
    ai_rest = ai_rest[["image_file", "vulnerability_class"] + IMAGE_LABELS]
    ai_rest["label_source"] = "ai_gpt4o_mini"

    gold = pd.concat([gold_human, ai_rest], ignore_index=True)
    gold.to_csv(OUT_GOLD, index=False)
    logger.info(f"Gold labels → {OUT_GOLD}  ({len(gold)} images, "
                f"{(gold.label_source=='human').sum()} human + "
                f"{(gold.label_source=='ai_gpt4o_mini').sum()} AI)")

    # ── 5. Agreement summary CSV ──────────────────────────────────────────────
    summary = pd.DataFrame([{
        "metric":                    "vulnerability_class_agreement_pct",
        "value":                     round(vc_agree_pct, 1),
    }, {
        "metric":                    "vulnerability_class_cohen_kappa",
        "value":                     round(vc_kappa, 3),
    }, {
        "metric":                    "n_images_reviewed",
        "value":                     len(merged),
    }, {
        "metric":                    "n_disagreements",
        "value":                     len(disagree_df),
    }, {
        "metric":                    "disagreement_rate_pct",
        "value":                     round(100 * len(disagree_df) / len(merged), 1),
    }, {
        "metric":                    "mean_label_agreement_pct",
        "value":                     round(label_df["pct_agreement"].mean(), 1),
    }, {
        "metric":                    "mean_label_cohen_kappa",
        "value":                     round(label_df["cohen_kappa"].mean(), 3),
    }])
    summary.to_csv(OUT_AGREEMENT, index=False)

    print(f"\n── Gold Label Dataset ───────────────────────────────────────")
    print(f"  Human-reviewed:  {(gold.label_source=='human').sum():,} images")
    print(f"  AI-labelled:     {(gold.label_source=='ai_gpt4o_mini').sum():,} images")
    print(f"  Total:           {len(gold):,} images")
    print()
    print("  Vulnerability class distribution (gold):")
    for vc, n in gold["vulnerability_class"].value_counts().items():
        print(f"    {vc:<42} {n:>5,}  ({100*n/len(gold):.1f}%)")

    print(f"\n── Output Files ─────────────────────────────────────────────")
    for p in [OUT_GOLD, OUT_AGREEMENT, OUT_LABEL_CMP, OUT_DISAGREE]:
        kb = p.stat().st_size / 1024
        print(f"  {p.relative_to(ROOT)!s:<55} {kb:>6.1f} KB")

    print("\n── Next Steps ───────────────────────────────────────────────")
    print("  1. Review disagreements: outputs/tables/annotation_disagreements.csv")
    print("  2. Retrain CV classifier with gold labels:")
    print("     .venv/bin/python3 src/models/train_cv_classifier.py --use-gold")
    print("  3. Use agreement metrics in manuscript (Table 3)")


if __name__ == "__main__":
    main()
