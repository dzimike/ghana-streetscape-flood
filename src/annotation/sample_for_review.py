"""Sample 400 images for human review, stratified by flood_stratum and vulnerability_class.

Outputs
-------
  data/interim/human_review_sample.csv         — 400-image sample with local paths
  data/interim/human_review_tasks.json         — Label Studio import (with pre-annotations)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
from loguru import logger

ANNOTATION_CSV = ROOT / "data/interim/openai_annotation_summary.csv"
SAMPLE_CSV     = ROOT / "data/interim/annotation_sample.csv"
OUT_SAMPLE     = ROOT / "data/interim/human_review_sample.csv"
OUT_TASKS      = ROOT / "data/interim/human_review_tasks.json"

N_REVIEW = 400

IMAGE_LABELS = [
    "visible_drain_present", "open_gutter_present", "blocked_drain_present",
    "stagnant_water_visible", "poor_road_condition", "heavy_impervious_surface",
    "unpaved_shoulder", "informal_structure_near_drainage", "solid_waste_accumulation",
    "visible_waterway_or_stream", "low_lying_street_form", "roadside_erosion",
    "pedestrian_exposure", "culvert_or_bridge_visible", "no_visible_drainage",
]

VULN_CLASSES = [
    "low_flood_vulnerability", "moderate_flood_vulnerability",
    "high_flood_vulnerability", "uncertain_requires_field_check",
]


def stratified_sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Sample n rows stratified by flood_stratum × vulnerability_class."""
    df = df.copy()
    # Collapse high+uncertain into moderate (only 9 high images)
    df["strata"] = (
        df["flood_stratum"].fillna("unknown") + "__" +
        df["vulnerability_class"].replace({
            "high_flood_vulnerability":        "moderate_flood_vulnerability",
            "uncertain_requires_field_check":  "moderate_flood_vulnerability",
        })
    )
    strata_counts = df["strata"].value_counts()
    strata_props  = strata_counts / len(df)
    n_per_stratum = (strata_props * n).round().astype(int)

    # Adjust rounding to hit exactly n
    diff = n - n_per_stratum.sum()
    if diff != 0:
        top = n_per_stratum.idxmax()
        n_per_stratum[top] += diff

    parts = []
    for stratum, k in n_per_stratum.items():
        pool = df[df["strata"] == stratum]
        k = min(k, len(pool))
        parts.append(pool.sample(n=k, random_state=seed))

    return pd.concat(parts).drop(columns=["strata"]).sample(frac=1, random_state=seed)


def build_ls_tasks(df: pd.DataFrame) -> list[dict]:
    """Build Label Studio tasks with GPT-4o-mini pre-annotations."""
    tasks = []
    for i, (_, row) in enumerate(df.iterrows()):
        # Pre-annotation results
        checked = [lbl for lbl in IMAGE_LABELS if row.get(lbl, 0) == 1]
        vc      = str(row.get("vulnerability_class", ""))
        conf    = str(row.get("confidence", "medium"))
        score   = {"high": 0.88, "medium": 0.68, "low": 0.45}.get(conf, 0.65)

        ls_results = []
        if checked:
            ls_results.append({
                "from_name": "image_labels",
                "to_name":   "image",
                "type":      "choices",
                "value":     {"choices": checked},
            })
        if vc:
            ls_results.append({
                "from_name": "vulnerability_class",
                "to_name":   "image",
                "type":      "choices",
                "value":     {"choices": [vc]},
            })

        tasks.append({
            "id": i + 1,
            "data": {
                # Label Studio local file serving path
                "image":         f"/data/local-files/?d={row['local_path']}",
                "annotation_id": row["annotation_id"],
                "pano_id":       row["pano_id"],
                "heading":       int(row["heading"]),
                "highway":       str(row.get("highway", "")),
                "flood_stratum": str(row.get("flood_stratum", "")),
                "point_id":      int(row.get("point_id", 0)),
                "ai_class":      vc,
                "ai_confidence": conf,
            },
            "predictions": [{
                "result":        ls_results,
                "score":         score,
                "model_version": "gpt-4o-mini",
            }],
        })
    return tasks


def main() -> None:
    ann    = pd.read_csv(ANNOTATION_CSV)
    sample = pd.read_csv(SAMPLE_CSV)
    df     = ann.merge(sample[["annotation_id", "local_path", "flood_stratum", "highway"]], on="annotation_id", suffixes=("", "_sample"))

    logger.info(f"Full annotated set: {len(df):,} images")
    logger.info(f"Vulnerability distribution:\n{df['vulnerability_class'].value_counts().to_string()}")
    logger.info(f"Flood stratum distribution:\n{df['flood_stratum'].value_counts().to_string()}")

    review_df = stratified_sample(df, N_REVIEW)
    logger.info(f"\nSampled {len(review_df)} images for human review")
    logger.info(f"Stratum breakdown:\n{review_df['flood_stratum'].value_counts().to_string()}")
    logger.info(f"Vulnerability breakdown:\n{review_df['vulnerability_class'].value_counts().to_string()}")

    review_df.to_csv(OUT_SAMPLE, index=False)
    logger.info(f"Sample CSV → {OUT_SAMPLE}")

    tasks = build_ls_tasks(review_df)
    OUT_TASKS.write_text(json.dumps(tasks, indent=2))
    logger.info(f"Label Studio tasks → {OUT_TASKS}  ({len(tasks)} tasks)")

    print("\n── Human Review Sample Summary ─────────────────────────────")
    print(f"  Total images selected:  {len(review_df)}")
    print(f"  % of full dataset:      {100*len(review_df)/len(df):.1f}%")
    print()
    print("  By flood stratum:")
    for s, n in review_df["flood_stratum"].value_counts().items():
        print(f"    {s:<35} {n:>4} ({100*n/len(review_df):.0f}%)")
    print()
    print("  By vulnerability class (AI pre-label):")
    for v, n in review_df["vulnerability_class"].value_counts().items():
        print(f"    {v:<42} {n:>4} ({100*n/len(review_df):.0f}%)")
    print()
    print("  Output files:")
    for p in [OUT_SAMPLE, OUT_TASKS]:
        kb = p.stat().st_size / 1024
        print(f"    {p.name:<45} {kb:>6.1f} KB")


if __name__ == "__main__":
    main()
