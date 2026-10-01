"""Gemini 1.5 Flash batch annotation for flood vulnerability images.

Uses the free Google AI Studio tier (15 req/min, 1,500 req/day).
2,000 images will complete in ~2.5 hours or across 2 days on the free tier.

The script is fully resumable — safe to stop and restart at any time.

Usage
-----
  python src/annotation/run_gemini_annotation.py --sample 5   # test on 5 images
  python src/annotation/run_gemini_annotation.py               # full 2,000 images

Setup
-----
  1. Get a free API key at https://aistudio.google.com/app/apikey
  2. Add to .env:  GOOGLE_API_KEY=AIza...
  3. pip install google-generativeai python-dotenv

Outputs
-------
  data/interim/gemini_raw_responses.jsonl        — resumable checkpoint
  data/interim/annotation_label_studio_ai.json   — Label Studio pre-annotation import
  data/interim/gemini_annotation_summary.csv     — flat label table
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

SAMPLE_CSV  = ROOT / "data/interim/annotation_sample.csv"
RAW_OUT     = ROOT / "data/interim/gemini_raw_responses.jsonl"
LS_OUT      = ROOT / "data/interim/annotation_label_studio_ai.json"
SUMMARY_OUT = ROOT / "data/interim/gemini_annotation_summary.csv"

MODEL            = "gemini-2.5-flash"
REQUESTS_PER_MIN = 14          # free tier = 15 RPM; stay just under
DAILY_LIMIT      = 1500        # free tier daily cap

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

PROMPT = """You are an expert urban flood risk analyst reviewing a street-level image from Accra, Ghana.

Identify which flood vulnerability indicators are CLEARLY VISIBLE in the image.
Focus only on what you can see — do not infer from location or context.

For each label, return true only if the feature is clearly visible, false otherwise.

Label definitions:
- visible_drain_present: Any drain, gutter or channel clearly visible
- open_gutter_present: Open-top roadside gutter visible (concrete, earth, or stone-lined)
- blocked_drain_present: Drain or gutter blocked, silted, or filled with debris/waste
- stagnant_water_visible: Standing or ponded water on road surface or in drain
- poor_road_condition: Potholes, cracking, rutting, erosion or surface failure on road
- heavy_impervious_surface: Scene dominated by concrete/asphalt with little vegetation
- unpaved_shoulder: Road shoulder or verge is unpaved (earth, gravel, bare soil)
- informal_structure_near_drainage: Kiosks, containers or market stalls next to a drain
- solid_waste_accumulation: Visible heap or scatter of solid waste (bags, plastics, refuse)
- visible_waterway_or_stream: Natural or semi-natural stream, river or channel visible
- low_lying_street_form: Road sits in a depression below surrounding land
- roadside_erosion: Erosion gullies, scour marks or undercutting at road edge
- pedestrian_exposure: Pedestrians near flood-risk features (drain edge, flooded road)
- culvert_or_bridge_visible: Culvert pipe or bridge crossing a drainage channel visible
- no_visible_drainage: No drain, gutter or waterway infrastructure visible

Vulnerability class (choose ONE):
- low_flood_vulnerability: no drainage problems, good road, no waste
- moderate_flood_vulnerability: minor issues — partial blockage, some waste, slight damage
- high_flood_vulnerability: clear problems — blocked drain, standing water, heavy waste, poor road
- uncertain_requires_field_check: image unclear, obstructed or ambiguous

Respond with ONLY valid JSON — no markdown, no explanation:
{
  "labels": {
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
  },
  "vulnerability_class": "low_flood_vulnerability",
  "confidence": "high",
  "notes": ""
}"""


def _load_already_done() -> set[str]:
    done = set()
    if RAW_OUT.exists():
        with open(RAW_OUT) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["annotation_id"])
                except Exception:
                    pass
    return done


def _parse_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def run(sample_n: int | None = None) -> None:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY not set.\n"
            "1. Get a free key at https://aistudio.google.com/app/apikey\n"
            "2. Add to .env:  GOOGLE_API_KEY=AIza..."
        )

    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        raise ImportError("Run: uv pip install google-genai --python .venv/bin/python3")

    client = genai.Client(api_key=api_key)

    sample = pd.read_csv(SAMPLE_CSV)
    if sample_n:
        sample = sample.head(sample_n)

    already_done = _load_already_done()
    pending = sample[~sample["annotation_id"].isin(already_done)]

    logger.info(f"Total images:   {len(sample):,}")
    logger.info(f"Already done:   {len(already_done):,}")
    logger.info(f"Pending:        {len(pending):,}")
    if len(pending) > DAILY_LIMIT:
        logger.warning(
            f"Free tier limit is {DAILY_LIMIT}/day. "
            f"Remaining {len(pending) - DAILY_LIMIT} images will need a second run tomorrow."
        )

    if pending.empty:
        logger.info("All images already annotated — exporting results.")
        _export_results(sample)
        return

    interval   = 60.0 / REQUESTS_PER_MIN
    done_today = 0
    RAW_OUT.parent.mkdir(parents=True, exist_ok=True)

    with open(RAW_OUT, "a") as out_f:
        for i, (_, row) in enumerate(pending.iterrows()):
            if done_today >= DAILY_LIMIT:
                logger.warning(
                    f"Reached daily limit of {DAILY_LIMIT} requests. "
                    "Run again tomorrow to continue."
                )
                break

            t0 = time.time()
            try:
                img_data = Path(row["local_path"]).read_bytes()
                img_part = genai_types.Part.from_bytes(
                    data=img_data, mime_type="image/jpeg"
                )
                response = client.models.generate_content(
                    model=MODEL,
                    contents=[PROMPT, img_part],
                    config=genai_types.GenerateContentConfig(
                        temperature=0,
                        max_output_tokens=2048,
                        thinking_config=genai_types.ThinkingConfig(
                            thinking_budget=0,   # disable thinking to save tokens
                        ),
                    ),
                )
                result = _parse_response(response.text)

                record = {
                    "annotation_id": row["annotation_id"],
                    "pano_id":       row["pano_id"],
                    "heading":       int(row["heading"]),
                    "point_id":      int(row["point_id"]),
                    "local_path":    row["local_path"],
                    "flood_stratum": str(row.get("flood_stratum", "")),
                    "highway":       str(row.get("highway", "")),
                    **result,
                }
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                done_today += 1

                vc     = result.get("vulnerability_class", "?")
                conf   = result.get("confidence", "?")
                n_true = sum(1 for v in result.get("labels", {}).values() if v)
                logger.info(
                    f"[{len(already_done)+i+1}/{len(sample)}] "
                    f"{row['annotation_id']} | {vc} ({conf}) | {n_true} labels true"
                )

            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse failed {row['annotation_id']}: {e}")
                time.sleep(2)
            except Exception as e:
                err = str(e)
                logger.warning(f"Failed {row['annotation_id']}: {err[:120]}")
                if "429" in err or "quota" in err.lower() or "exhausted" in err.lower():
                    logger.info("Rate limit hit — waiting 60 s …")
                    time.sleep(60)
                elif "500" in err or "503" in err:
                    logger.info("Server error — waiting 15 s …")
                    time.sleep(15)
                else:
                    time.sleep(5)

            elapsed = time.time() - t0
            if elapsed < interval:
                time.sleep(interval - elapsed)

    _export_results(sample)


def _export_results(sample: pd.DataFrame) -> None:
    if not RAW_OUT.exists():
        logger.warning("No responses found yet.")
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

    logger.info(f"Exporting {len(records):,} annotated records …")
    ann_map = {r["annotation_id"]: r for r in records}

    # ── Label Studio JSON with pre-annotations ────────────────────────────
    tasks = []
    for _, row in sample.iterrows():
        aid = row["annotation_id"]
        rec = ann_map.get(aid, {})
        labels_dict = rec.get("labels", {})
        vc          = rec.get("vulnerability_class", "")
        conf        = rec.get("confidence", "medium")

        ls_results = []
        checked = [lbl for lbl, val in labels_dict.items() if val]
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

        score = {"high": 0.85, "medium": 0.65, "low": 0.40}.get(conf, 0.60)
        tasks.append({
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
            "predictions": [{
                "result":        ls_results,
                "score":         score,
                "model_version": "gemini-1.5-flash-v1",
            }] if ls_results else [],
        })

    LS_OUT.parent.mkdir(parents=True, exist_ok=True)
    LS_OUT.write_text(json.dumps(tasks, indent=2))
    logger.info(f"Label Studio JSON → {LS_OUT}  ({len(tasks)} tasks)")

    # ── Summary CSV ───────────────────────────────────────────────────────
    rows = []
    for r in records:
        row = {
            "annotation_id":       r["annotation_id"],
            "pano_id":             r["pano_id"],
            "heading":             r["heading"],
            "point_id":            r["point_id"],
            "flood_stratum":       r.get("flood_stratum", ""),
            "highway":             r.get("highway", ""),
            "vulnerability_class": r.get("vulnerability_class", ""),
            "confidence":          r.get("confidence", ""),
            "notes":               r.get("notes", ""),
        }
        for lbl in IMAGE_LABELS:
            row[lbl] = int(r.get("labels", {}).get(lbl, False))
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(SUMMARY_OUT, index=False)
    logger.info(f"Summary CSV → {SUMMARY_OUT}")

    # ── Stats ─────────────────────────────────────────────────────────────
    print("\n── Gemini Annotation Summary ───────────────────────────────")
    print(f"  Images annotated: {len(df):,} / {len(sample):,}")
    remaining = len(sample) - len(df)
    if remaining > 0:
        print(f"  Remaining:        {remaining:,}  (run again to continue)")
    print()
    print("  Vulnerability class distribution:")
    for vc, n in df["vulnerability_class"].value_counts().items():
        print(f"    {vc:<42} {n:>5,}  ({100*n/len(df):.1f}%)")
    print()
    print("  Label prevalence (% images where label = true):")
    for lbl in IMAGE_LABELS:
        if lbl in df.columns:
            pct = 100 * df[lbl].mean()
            bar = "█" * int(pct / 5)
            print(f"    {lbl:<45} {pct:>5.1f}%  {bar}")
    print()
    print("  Output files:")
    for p in [RAW_OUT, LS_OUT, SUMMARY_OUT]:
        if p.exists():
            kb = p.stat().st_size / 1024
            print(f"    {p.name:<50} {kb:>7.1f} KB")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=None,
                   help="Annotate only first N images (for testing)")
    args = p.parse_args()
    run(sample_n=args.sample)
