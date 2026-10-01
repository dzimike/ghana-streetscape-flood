"""GPT-4o-mini batch annotation for flood vulnerability images — Task 5b.

Sends each annotation sample image to GPT-4o-mini with a structured prompt
derived from the annotation codebook. Outputs Label Studio pre-annotation JSON
so human annotators only need to review and correct AI suggestions.

Cost estimate:  ~$1–2 for 2,000 images (GPT-4o-mini, low image detail)
Runtime:        ~30–60 min for 2,000 images (rate-limited to 60 req/min)

Usage
-----
  python src/annotation/run_gpt4o_annotation.py
  python src/annotation/run_gpt4o_annotation.py --sample 20   # quick test
  python src/annotation/run_gpt4o_annotation.py --model gpt-4o # use full model

Outputs
-------
  data/interim/gpt4o_raw_responses.jsonl        — one JSON per image (resumable)
  data/interim/annotation_label_studio_ai.json  — Label Studio import with pre-fills
  data/interim/gpt4o_annotation_summary.csv     — flat label table
"""
from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
from loguru import logger
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

SAMPLE_CSV   = ROOT / "data/interim/annotation_sample.csv"
RAW_OUT      = ROOT / "data/interim/gpt4o_raw_responses.jsonl"
LS_OUT       = ROOT / "data/interim/annotation_label_studio_ai.json"
SUMMARY_OUT  = ROOT / "data/interim/gpt4o_annotation_summary.csv"

DEFAULT_MODEL   = "gpt-4o-mini"
REQUESTS_PER_MIN = 60        # stay under rate limit
IMAGE_DETAIL     = "low"     # "low" = 85 tokens/image; "high" = 765 tokens/image

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

VULNERABILITY_CLASSES = [
    "low_flood_vulnerability",
    "moderate_flood_vulnerability",
    "high_flood_vulnerability",
    "uncertain_requires_field_check",
]

SYSTEM_PROMPT = """You are an expert urban flood risk analyst reviewing street-level imagery from Accra, Ghana.
Your task is to identify visible flood vulnerability indicators in each image.
Focus strictly on what is visible in the image — do not infer from location or context.
Respond ONLY with valid JSON matching the schema provided."""

def _build_user_prompt() -> str:
    label_descriptions = {
        "visible_drain_present":             "Any drain, gutter or channel is clearly visible.",
        "open_gutter_present":               "An open-top roadside gutter is visible (concrete, earth, or stone-lined).",
        "blocked_drain_present":             "A drain or gutter appears blocked, silted, or filled with debris/waste.",
        "stagnant_water_visible":            "Standing or ponded water is visible on the road or in a drain.",
        "poor_road_condition":               "Potholes, cracking, rutting, erosion or surface failure visible on road.",
        "heavy_impervious_surface":          "Scene dominated by concrete/asphalt with little bare soil or vegetation.",
        "unpaved_shoulder":                  "Road shoulder or verge is unpaved (earth, gravel, or bare soil).",
        "informal_structure_near_drainage":  "Kiosks, containers, sheds or market structures next to a drain.",
        "solid_waste_accumulation":          "Visible heap or scatter of solid waste (bags, plastics, refuse).",
        "visible_waterway_or_stream":        "A natural or semi-natural stream, river or channel is visible.",
        "low_lying_street_form":             "Road sits in a depression below surrounding land.",
        "roadside_erosion":                  "Erosion gullies, scour marks, or undercutting at road edge.",
        "pedestrian_exposure":               "Pedestrians present near flood-risk features (drain edge, flooded road).",
        "culvert_or_bridge_visible":         "A culvert pipe or bridge crossing a drainage channel is visible.",
        "no_visible_drainage":               "No drain, gutter or waterway infrastructure visible.",
    }

    label_lines = "\n".join(
        f'  "{lbl}": true/false  — {desc}'
        for lbl, desc in label_descriptions.items()
    )

    return f"""Analyse this street-level image from Accra, Ghana for flood vulnerability indicators.

For each label below, return true if the feature is CLEARLY visible, false otherwise.
When in doubt, return false.

Labels:
{label_lines}

Also assign ONE overall vulnerability class:
  "low_flood_vulnerability"          — no drainage problems, good road, no waste
  "moderate_flood_vulnerability"     — minor issues (partial blockage, some waste, slight damage)
  "high_flood_vulnerability"         — clear problems (blocked drain, standing water, heavy waste, poor road)
  "uncertain_requires_field_check"   — image unclear, obstructed, or ambiguous

Respond with ONLY this JSON structure (no markdown, no explanation):
{{
  "labels": {{
    "visible_drain_present": false,
    "open_gutter_present": false,
    "blocked_drain_present": false,
    "stagnant_water_visible": false,
    "poor_road_condition": false,
    "heavy_impervious_surface": false,
    "unpaved_shoulder": false,
    "informal_structure_near_drainage": false,
    "solid_waste_accumulation": false,
    "visible_waterway_or_stream": false,
    "low_lying_street_form": false,
    "roadside_erosion": false,
    "pedestrian_exposure": false,
    "culvert_or_bridge_visible": false,
    "no_visible_drainage": false
  }},
  "vulnerability_class": "low_flood_vulnerability",
  "confidence": "high",
  "notes": ""
}}"""


def _encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _call_gpt4o(client, image_path: str, model: str) -> dict:
    b64 = _encode_image(image_path)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url":    f"data:image/jpeg;base64,{b64}",
                            "detail": IMAGE_DETAIL,
                        },
                    },
                    {"type": "text", "text": _build_user_prompt()},
                ],
            },
        ],
        max_tokens=400,
        temperature=0,
    )
    raw_text = response.choices[0].message.content.strip()
    # Strip markdown code fences if model adds them
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
    return json.loads(raw_text)


def _load_already_done() -> set[str]:
    done = set()
    if RAW_OUT.exists():
        with open(RAW_OUT) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done.add(rec["annotation_id"])
                except Exception:
                    pass
    return done


def run(model: str = DEFAULT_MODEL, sample_n: int | None = None) -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Add it to your .env file:\n"
            "  OPENAI_API_KEY=sk-..."
        )

    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Run: .venv/bin/pip install openai")

    client = OpenAI(api_key=api_key)

    sample = pd.read_csv(SAMPLE_CSV)
    if sample_n:
        sample = sample.head(sample_n)
    logger.info(f"Annotation sample: {len(sample):,} images | model: {model}")

    already_done = _load_already_done()
    pending = sample[~sample["annotation_id"].isin(already_done)]
    logger.info(f"Already done: {len(already_done)} | Pending: {len(pending)}")

    if pending.empty:
        logger.info("All images already annotated.")
    else:
        RAW_OUT.parent.mkdir(parents=True, exist_ok=True)
        interval = 60.0 / REQUESTS_PER_MIN   # seconds between requests

        with open(RAW_OUT, "a") as out_f:
            for i, (_, row) in enumerate(pending.iterrows()):
                t0 = time.time()
                try:
                    result = _call_gpt4o(client, row["local_path"], model)
                    record = {
                        "annotation_id": row["annotation_id"],
                        "pano_id":       row["pano_id"],
                        "heading":       int(row["heading"]),
                        "point_id":      int(row["point_id"]),
                        "local_path":    row["local_path"],
                        "flood_stratum": row.get("flood_stratum", ""),
                        "highway":       row.get("highway", ""),
                        **result,
                    }
                    out_f.write(json.dumps(record) + "\n")
                    out_f.flush()

                    vc = result.get("vulnerability_class", "?")
                    conf = result.get("confidence", "?")
                    n_true = sum(1 for v in result.get("labels", {}).values() if v)
                    logger.info(
                        f"[{i+1+len(already_done)}/{len(sample)}] "
                        f"{row['annotation_id']} | {vc} ({conf}) | {n_true} labels"
                    )
                except Exception as e:
                    logger.warning(f"Failed {row['annotation_id']}: {e}")

                # Rate limiting
                elapsed = time.time() - t0
                if elapsed < interval:
                    time.sleep(interval - elapsed)

    _export_results(sample)


def _export_results(sample: pd.DataFrame) -> None:
    """Convert raw JSONL responses to Label Studio JSON and summary CSV."""
    if not RAW_OUT.exists():
        logger.warning("No raw responses found.")
        return

    records = []
    with open(RAW_OUT) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                pass

    if not records:
        return

    logger.info(f"Exporting {len(records)} annotated records …")

    # ── Label Studio pre-annotation JSON ──────────────────────────────────
    ann_map = {r["annotation_id"]: r for r in records}
    tasks   = []
    for _, row in sample.iterrows():
        aid = row["annotation_id"]
        rec = ann_map.get(aid, {})
        labels_dict = rec.get("labels", {})
        vc          = rec.get("vulnerability_class", "")

        # Build Label Studio annotation result list
        ls_results = []

        # Multi-label checkboxes
        checked = [lbl for lbl, val in labels_dict.items() if val]
        if checked:
            ls_results.append({
                "from_name": "image_labels",
                "to_name":   "image",
                "type":      "choices",
                "value":     {"choices": checked},
            })

        # Single-select vulnerability class
        if vc:
            ls_results.append({
                "from_name": "vulnerability_class",
                "to_name":   "image",
                "type":      "choices",
                "value":     {"choices": [vc]},
            })

        task = {
            "id":   int(row.name),
            "data": {
                "image":         f"/data/local-files/?d={row['local_path']}",
                "annotation_id": aid,
                "pano_id":       row["pano_id"],
                "heading":       int(row["heading"]),
                "highway":       str(row.get("highway", "")),
                "flood_stratum": str(row.get("flood_stratum", "")),
                "point_id":      int(row.get("point_id", 0)),
            },
            "annotations": [{
                "result":        ls_results,
                "ground_truth":  False,
                "was_cancelled": False,
                "lead_time":     0,
            }] if ls_results else [],
            "predictions": [{
                "result": ls_results,
                "score":  0.8 if rec.get("confidence") == "high" else 0.5,
                "model_version": f"gpt4o-annotation-v1",
            }] if ls_results else [],
        }
        tasks.append(task)

    LS_OUT.parent.mkdir(parents=True, exist_ok=True)
    LS_OUT.write_text(json.dumps(tasks, indent=2))
    logger.info(f"Label Studio JSON → {LS_OUT}  ({len(tasks)} tasks)")

    # ── Summary CSV ───────────────────────────────────────────────────────
    rows = []
    for r in records:
        row = {
            "annotation_id":   r["annotation_id"],
            "pano_id":         r["pano_id"],
            "heading":         r["heading"],
            "point_id":        r["point_id"],
            "flood_stratum":   r.get("flood_stratum", ""),
            "highway":         r.get("highway", ""),
            "vulnerability_class": r.get("vulnerability_class", ""),
            "confidence":      r.get("confidence", ""),
            "notes":           r.get("notes", ""),
        }
        for lbl in IMAGE_LABELS:
            row[lbl] = int(r.get("labels", {}).get(lbl, False))
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(SUMMARY_OUT, index=False)
    logger.info(f"Summary CSV → {SUMMARY_OUT}")

    # ── Print stats ───────────────────────────────────────────────────────
    print("\n── GPT-4o Annotation Summary ───────────────────────────────")
    print(f"  Images annotated: {len(df):,}")
    print()
    print("  Vulnerability class distribution:")
    for vc, n in df["vulnerability_class"].value_counts().items():
        print(f"    {vc:<40} {n:>5,}  ({100*n/len(df):.1f}%)")
    print()
    print("  Label prevalence (% of images with label = true):")
    label_cols = [c for c in df.columns if c in IMAGE_LABELS]
    for lbl in label_cols:
        pct = 100 * df[lbl].mean()
        bar = "█" * int(pct / 5)
        print(f"    {lbl:<45} {pct:>5.1f}%  {bar}")
    print()
    print(f"  Output files:")
    for p in [RAW_OUT, LS_OUT, SUMMARY_OUT]:
        if p.exists():
            kb = p.stat().st_size / 1024
            print(f"    {p.name:<50} {kb:>7.1f} KB")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model",  default=DEFAULT_MODEL,
                   help="OpenAI model (default: gpt-4o-mini)")
    p.add_argument("--sample", type=int, default=None,
                   help="Annotate only first N images (for testing)")
    args = p.parse_args()
    run(model=args.model, sample_n=args.sample)
