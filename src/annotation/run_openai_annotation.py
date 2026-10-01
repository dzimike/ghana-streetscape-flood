"""GPT-4o-mini batch annotation for flood vulnerability images.

Fully resumable — safe to stop and restart at any time.
Processes all 2,000 images in one run (~2–3 hours on gpt-4o-mini).

Usage
-----
  python src/annotation/run_openai_annotation.py --sample 5   # test on 5 images
  python src/annotation/run_openai_annotation.py               # full 2,000 images
  python src/annotation/run_openai_annotation.py --model gpt-4o  # higher quality

Cost estimate (gpt-4o-mini)
---------------------------
  ~$0.01–0.02 per image  →  ~$20–40 for 2,000 images

Setup
-----
  Add to .env:  OPENAI_API_KEY=sk-proj-...

Outputs
-------
  data/interim/openai_raw_responses.jsonl        — resumable checkpoint
  data/interim/annotation_label_studio_ai.json   — Label Studio pre-annotation import
  data/interim/openai_annotation_summary.csv     — flat label table
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
RAW_OUT     = ROOT / "data/interim/openai_raw_responses.jsonl"
LS_OUT      = ROOT / "data/interim/annotation_label_studio_ai.json"
SUMMARY_OUT = ROOT / "data/interim/openai_annotation_summary.csv"

DEFAULT_MODEL    = "gpt-4o-mini"
REQUESTS_PER_MIN = 50   # free-tier is 3 RPM; paid tier is 500 RPM — stay conservative
MAX_RETRIES      = 3

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

SYSTEM_PROMPT = (
    "You are an expert urban flood risk analyst. "
    "You review street-level images from Accra, Ghana and identify visible flood "
    "vulnerability indicators. Respond only with valid JSON — no markdown, no explanation."
)

USER_PROMPT = """Identify which flood vulnerability indicators are CLEARLY VISIBLE in this street-level image from Accra, Ghana.

Label definitions:
- visible_drain_present: Any drain, gutter or channel clearly visible
- open_gutter_present: Open-top roadside gutter (concrete, earth, or stone-lined)
- blocked_drain_present: Drain or gutter blocked, silted, or filled with debris/waste
- stagnant_water_visible: Standing or ponded water on road or in drain
- poor_road_condition: Potholes, cracking, rutting, erosion or surface failure on road
- heavy_impervious_surface: Scene dominated by concrete/asphalt with little vegetation
- unpaved_shoulder: Road shoulder or verge is unpaved (earth, gravel, bare soil)
- informal_structure_near_drainage: Kiosks, containers or stalls next to a drain
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

Respond with ONLY this JSON (no markdown):
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


def _encode_image(path: str) -> str:
    return base64.standard_b64encode(Path(path).read_bytes()).decode("utf-8")


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


def run(sample_n: int | None = None, model: str = DEFAULT_MODEL) -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set in .env")

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    sample = pd.read_csv(SAMPLE_CSV)
    if sample_n:
        sample = sample.head(sample_n)

    already_done = _load_already_done()
    pending = sample[~sample["annotation_id"].isin(already_done)]

    logger.info(f"Model:          {model}")
    logger.info(f"Total images:   {len(sample):,}")
    logger.info(f"Already done:   {len(already_done):,}")
    logger.info(f"Pending:        {len(pending):,}")

    if pending.empty:
        logger.info("All images already annotated — exporting results.")
        _export_results(sample, model)
        return

    interval = 60.0 / REQUESTS_PER_MIN
    RAW_OUT.parent.mkdir(parents=True, exist_ok=True)

    with open(RAW_OUT, "a") as out_f:
        for i, (_, row) in enumerate(pending.iterrows()):
            t0 = time.time()
            result = None

            for attempt in range(MAX_RETRIES):
                try:
                    b64 = _encode_image(row["local_path"])
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
                                            "url": f"data:image/jpeg;base64,{b64}",
                                            "detail": "low",  # cheaper; enough for scene-level labels
                                        },
                                    },
                                    {"type": "text", "text": USER_PROMPT},
                                ],
                            },
                        ],
                        max_tokens=512,
                        temperature=0,
                    )
                    result = _parse_response(response.choices[0].message.content)
                    break

                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parse error (attempt {attempt+1}): {e}")
                    time.sleep(2)
                except Exception as e:
                    err = str(e)
                    logger.warning(f"API error (attempt {attempt+1}): {err[:120]}")
                    if "429" in err or "rate" in err.lower():
                        wait = 30 * (attempt + 1)
                        logger.info(f"Rate limit — waiting {wait}s …")
                        time.sleep(wait)
                    elif "insufficient_quota" in err or "billing" in err.lower():
                        logger.error("Billing/quota error — check your OpenAI account.")
                        raise
                    else:
                        time.sleep(5 * (attempt + 1))

            if result is None:
                logger.warning(f"Skipping {row['annotation_id']} after {MAX_RETRIES} failures")
                elapsed = time.time() - t0
                if elapsed < interval:
                    time.sleep(interval - elapsed)
                continue

            record = {
                "annotation_id": row["annotation_id"],
                "pano_id":       row["pano_id"],
                "heading":       int(row["heading"]),
                "point_id":      int(row["point_id"]),
                "local_path":    row["local_path"],
                "flood_stratum": str(row.get("flood_stratum", "")),
                "highway":       str(row.get("highway", "")),
                "model":         model,
                **result,
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            vc     = result.get("vulnerability_class", "?")
            conf   = result.get("confidence", "?")
            n_true = sum(1 for v in result.get("labels", {}).values() if v)
            logger.info(
                f"[{len(already_done)+i+1}/{len(sample)}] "
                f"{row['annotation_id']} | {vc} ({conf}) | {n_true} labels true"
            )

            elapsed = time.time() - t0
            if elapsed < interval:
                time.sleep(interval - elapsed)

    _export_results(sample, model)


def _export_results(sample: pd.DataFrame, model: str = DEFAULT_MODEL) -> None:
    if not RAW_OUT.exists():
        logger.warning("No responses found.")
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

    # ── Label Studio JSON ─────────────────────────────────────────────────────
    tasks = []
    for _, row in sample.iterrows():
        aid = row["annotation_id"]
        rec = ann_map.get(aid, {})
        labels_dict = rec.get("labels", {})
        vc   = rec.get("vulnerability_class", "")
        conf = rec.get("confidence", "medium")

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

        score = {"high": 0.88, "medium": 0.68, "low": 0.45}.get(conf, 0.65)
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
                "model_version": model,
            }] if ls_results else [],
        })

    LS_OUT.parent.mkdir(parents=True, exist_ok=True)
    LS_OUT.write_text(json.dumps(tasks, indent=2))
    logger.info(f"Label Studio JSON → {LS_OUT}  ({len(tasks)} tasks)")

    # ── Summary CSV ───────────────────────────────────────────────────────────
    rows = []
    for r in records:
        row_data = {
            "annotation_id":       r["annotation_id"],
            "pano_id":             r["pano_id"],
            "heading":             r["heading"],
            "point_id":            r["point_id"],
            "flood_stratum":       r.get("flood_stratum", ""),
            "highway":             r.get("highway", ""),
            "vulnerability_class": r.get("vulnerability_class", ""),
            "confidence":          r.get("confidence", ""),
            "model":               r.get("model", model),
            "notes":               r.get("notes", ""),
        }
        for lbl in IMAGE_LABELS:
            row_data[lbl] = int(r.get("labels", {}).get(lbl, False))
        rows.append(row_data)

    df = pd.DataFrame(rows)
    df.to_csv(SUMMARY_OUT, index=False)
    logger.info(f"Summary CSV → {SUMMARY_OUT}")

    # ── Stats ─────────────────────────────────────────────────────────────────
    print("\n── OpenAI Annotation Summary ───────────────────────────────")
    print(f"  Model:            {model}")
    print(f"  Images annotated: {len(df):,} / {len(sample):,}")
    remaining = len(sample) - len(df)
    if remaining > 0:
        print(f"  Remaining:        {remaining:,}  (re-run to continue)")
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
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="OpenAI model (default: gpt-4o-mini)")
    args = p.parse_args()
    run(sample_n=args.sample, model=args.model)
