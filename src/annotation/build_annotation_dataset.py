"""Build stratified annotation dataset — Task 5.

Samples 2,000 images from the 15,344-image manifest, stratified by:
  - flood_stratum       (high_risk 60% / reference 40%)
  - highway group       (arterial / collector / local)
  - heading             (balanced across 0°, 90°, 180°, 270°)
  - capture year        (proportional)

Outputs:
  data/interim/annotation_sample.csv          — 2,000 selected images
  data/interim/annotation_label_studio.json   — Label Studio import
  data/interim/annotation_cvat.xml            — CVAT XML label skeleton
  data/interim/annotation_coco_categories.json
  outputs/reports/annotation_codebook.md      — annotator guide
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from loguru import logger

from src.config.settings import load_config
from src.annotation.schema import (
    IMAGE_LABELS, OBJECT_CATEGORIES, SEGMENT_LABELS
)

cfg = load_config("accra_pilot")

MANIFEST    = ROOT / "data/interim/image_manifest.parquet"
SAMPLE_OUT  = ROOT / "data/interim/annotation_sample.csv"
LS_OUT      = ROOT / "data/interim/annotation_label_studio.json"
CVAT_OUT    = ROOT / "data/interim/annotation_cvat.xml"
COCO_OUT    = ROOT / "data/interim/annotation_coco_categories.json"
CODEBOOK    = ROOT / "outputs/reports/annotation_codebook.md"

TARGET_N    = 2000
RANDOM_SEED = 42

HIGHWAY_GROUPS = {
    "arterial":  ["motorway", "trunk", "primary", "secondary"],
    "collector": ["tertiary", "unclassified"],
    "local":     ["residential", "service"],
}

# Target proportions: high-risk 60%, reference 40%
STRATUM_ALLOC = {"high_risk_proximity": 0.60, "low_risk_reference": 0.40}

# Within each stratum, highway group allocation
HIGHWAY_ALLOC = {"arterial": 0.25, "collector": 0.20, "local": 0.55}

# Headings balanced equally
HEADING_ALLOC = {0: 0.25, 90: 0.25, 180: 0.25, 270: 0.25}


# ── Stratified sampling ───────────────────────────────────────────────────────
def _highway_group(hw: str) -> str:
    for grp, types in HIGHWAY_GROUPS.items():
        if hw in types:
            return grp
    return "local"


def sample_for_annotation(manifest: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"Sampling {TARGET_N} images from {len(manifest):,} records")
    rng    = np.random.default_rng(RANDOM_SEED)
    manifest = manifest[manifest["downloaded"].fillna(False)].copy()
    manifest["highway_group"] = manifest["highway"].apply(_highway_group)
    manifest["year"] = pd.to_datetime(
        manifest["capture_date"], errors="coerce"
    ).dt.year.fillna(2024).astype(int)

    parts = []
    for stratum, s_frac in STRATUM_ALLOC.items():
        s_pool = manifest[manifest["flood_stratum"] == stratum]
        s_n    = int(TARGET_N * s_frac)

        for hw_grp, h_frac in HIGHWAY_ALLOC.items():
            h_pool = s_pool[s_pool["highway_group"] == hw_grp]
            h_n    = int(s_n * h_frac)
            if len(h_pool) == 0:
                continue

            # Balance headings within this cell
            per_heading = max(1, h_n // 4)
            for heading in [0, 90, 180, 270]:
                hd_pool = h_pool[h_pool["heading"] == heading]
                k = min(per_heading, len(hd_pool))
                if k > 0:
                    parts.append(
                        hd_pool.sample(n=k, random_state=int(rng.integers(9999)))
                    )

    sample = pd.concat(parts).drop_duplicates("pano_id").head(TARGET_N)
    # Top up if short due to rounding
    if len(sample) < TARGET_N:
        remaining = manifest[~manifest["pano_id"].isin(sample["pano_id"])]
        shortfall  = TARGET_N - len(sample)
        extra = remaining.sample(
            n=min(shortfall, len(remaining)), random_state=RANDOM_SEED
        )
        sample = pd.concat([sample, extra])

    sample = sample.reset_index(drop=True)
    sample["annotation_id"] = [f"ACC_{i:05d}" for i in range(len(sample))]
    logger.info(f"Selected {len(sample):,} images for annotation")
    return sample


# ── Label Studio export ───────────────────────────────────────────────────────
def export_label_studio(sample: pd.DataFrame, output_path: Path) -> None:
    """Export as Label Studio JSON with image-level and segment-level tasks."""
    tasks = []
    for _, row in sample.iterrows():
        fname   = Path(str(row["local_path"])).name
        rel     = f"images/{fname}"
        tasks.append({
            "id":   int(row.name),
            "data": {
                "image":          f"/data/local-files/?d={row['local_path']}",
                "annotation_id":  row["annotation_id"],
                "pano_id":        row["pano_id"],
                "heading":        int(row["heading"]),
                "capture_date":   str(row.get("capture_date", "")),
                "highway":        str(row.get("highway", "")),
                "highway_group":  str(row.get("highway_group", "")),
                "flood_stratum":  str(row.get("flood_stratum", "")),
                "point_id":       int(row.get("point_id", 0)),
            },
            "meta": {
                "lat": float(row.get("pano_lat", 0)),
                "lon": float(row.get("pano_lon", 0)),
            },
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(tasks, indent=2))
    logger.info(f"Label Studio tasks → {output_path}  ({len(tasks)} tasks)")


# ── CVAT XML export ───────────────────────────────────────────────────────────
def export_cvat_xml(output_path: Path, task_name: str = "Accra Flood Vulnerability") -> None:
    img_labels_xml = "\n".join(
        f'      <label>\n        <name>{lbl}</name>\n        <type>checkbox</type>\n'
        f'        <attributes/>\n      </label>'
        for lbl in IMAGE_LABELS
    )
    seg_labels_xml = "\n".join(
        f'      <label>\n        <name>{lbl}</name>\n        <type>radio</type>\n'
        f'        <attributes/>\n      </label>'
        for lbl in SEGMENT_LABELS
    )
    obj_labels_xml = "\n".join(
        f'      <label>\n        <name>{cat["name"]}</name>\n        <type>rectangle</type>\n'
        f'        <attributes/>\n      </label>'
        for cat in OBJECT_CATEGORIES
    )
    xml = f"""<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta>
    <task>
      <name>{task_name}</name>
      <size>2000</size>
      <labels>
        <!-- Image-level vulnerability indicators (multi-label checkbox) -->
{img_labels_xml}
        <!-- Overall vulnerability class (single-select radio) -->
{seg_labels_xml}
        <!-- Object bounding box categories -->
{obj_labels_xml}
      </labels>
    </task>
  </meta>
</annotations>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(xml)
    logger.info(f"CVAT XML → {output_path}")


# ── COCO categories export ────────────────────────────────────────────────────
def export_coco_categories(output_path: Path) -> None:
    payload = {
        "info": {
            "description": "Accra Flood Vulnerability Object Detection Categories",
            "version": "1.0",
        },
        "categories": OBJECT_CATEGORIES,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    logger.info(f"COCO categories → {output_path}")


# ── Annotation codebook ───────────────────────────────────────────────────────
LABEL_DESCRIPTIONS = {
    "visible_drain_present":           "Any drain, gutter or channel is clearly visible in the scene.",
    "open_gutter_present":             "An open-top roadside gutter is visible (may be concrete, earth, or stone-lined).",
    "blocked_drain_present":           "A drain or gutter appears blocked, silted, or filled with debris/waste.",
    "stagnant_water_visible":          "Standing or ponded water is visible on the road surface or in a drain.",
    "poor_road_condition":             "Potholes, cracking, rutting, erosion, or surface failure is visible on the road.",
    "heavy_impervious_surface":        "The scene is dominated by concrete, asphalt, or compacted surfaces with little bare soil or vegetation.",
    "unpaved_shoulder":                "The road shoulder or verge is unpaved (earth, gravel, or bare soil).",
    "informal_structure_near_drainage":"Kiosks, containers, sheds, or market structures are sited directly adjacent to a drain or waterway.",
    "solid_waste_accumulation":        "Visible heap or scatter of solid waste (bags, plastics, refuse) in the scene.",
    "visible_waterway_or_stream":      "A natural or semi-natural stream, river, or channel is visible.",
    "low_lying_street_form":           "The road appears to sit in a depression, bowl, or low-lying corridor relative to surrounding land.",
    "roadside_erosion":                "Erosion gullies, scour marks, or undercutting are visible at the road edge or verge.",
    "pedestrian_exposure":             "Pedestrians are present near flood-risk features (edge of drain, flooded road, etc.).",
    "culvert_or_bridge_visible":       "A culvert pipe or bridge structure crossing a drain or waterway is visible.",
    "no_visible_drainage":             "No drain, gutter, or waterway infrastructure is visible in the scene.",
}

OBJECT_DESCRIPTIONS = {
    "drain":           "Engineered drainage channel — concrete or masonry lined.",
    "gutter":          "Roadside open gutter, any material.",
    "culvert":         "Pipe or box culvert where road crosses a channel.",
    "water":           "Ponded, flowing, or stagnant water body.",
    "solid_waste":     "Heap or scatter of solid waste or refuse.",
    "road":            "Paved or unpaved road surface.",
    "sidewalk":        "Pedestrian walkway.",
    "vegetation":      "Trees, grass, shrubs, or other vegetation.",
    "building":        "Any building structure.",
    "kiosk_container": "Roadside kiosk, shipping container, or informal market stall.",
    "bridge":          "Bridge structure over a waterway.",
    "stream_channel":  "Natural or semi-natural stream, river, or open channel.",
}


def export_codebook(output_path: Path) -> None:
    lines = [
        "# Annotation Codebook — Accra Flood Vulnerability Streetscape Dataset",
        "",
        "## Overview",
        "",
        "Each Street View image is annotated at three levels:",
        "1. **Image-level labels** — multi-label checkboxes for visible flood indicators",
        "2. **Object-level bounding boxes** — for detecting specific drainage and waste features",
        "3. **Segment vulnerability class** — overall flood-vulnerability rating for the scene",
        "",
        "Annotators should use the Street View heading metadata to understand viewing direction.",
        "Focus on what is visible in *this specific image*, not what you know about the area.",
        "",
        "---",
        "",
        "## Image-Level Labels (multi-label — check all that apply)",
        "",
        "| Label | When to check |",
        "|---|---|",
    ]
    for lbl in IMAGE_LABELS:
        desc = LABEL_DESCRIPTIONS.get(lbl, "")
        lines.append(f"| `{lbl}` | {desc} |")

    lines += [
        "",
        "---",
        "",
        "## Segment Vulnerability Class (single choice)",
        "",
        "Choose the **one best class** that describes the overall flood vulnerability visible in the scene:",
        "",
        "| Class | Description |",
        "|---|---|",
        "| `low_flood_vulnerability` | No drainage problems, road in good condition, no waste, no water |",
        "| `moderate_flood_vulnerability` | Minor issues visible — partial blockage, some waste, slight surface damage |",
        "| `high_flood_vulnerability` | Clear problems — blocked drain, standing water, significant waste, poor road |",
        "| `uncertain_requires_field_check` | Image is unclear, obstructed, or the situation requires physical verification |",
        "",
        "---",
        "",
        "## Object Bounding Box Categories",
        "",
        "Draw tight bounding boxes around visible objects. One box per object instance.",
        "",
        "| Category | What to box |",
        "|---|---|",
    ]
    for cat in OBJECT_CATEGORIES:
        desc = OBJECT_DESCRIPTIONS.get(cat["name"], "")
        lines.append(f"| `{cat['name']}` | {desc} |")

    lines += [
        "",
        "---",
        "",
        "## Annotation Protocol",
        "",
        "1. Open the task in Label Studio.",
        "2. View the image at full resolution.",
        "3. Check all applicable **image-level labels**.",
        "4. Draw **bounding boxes** around all visible drain, water, waste, culvert, and stream objects.",
        "5. Select the **segment vulnerability class**.",
        "6. Add a note in the comments field for any unusual or ambiguous scene.",
        "7. Mark as complete.",
        "",
        "### Quality rules",
        "",
        "- If fewer than 20% of the image shows meaningful content (e.g. camera pointing at sky or wall), mark `uncertain_requires_field_check`.",
        "- Do not guess — if a feature is not clearly visible, do not label it.",
        "- For `blocked_drain_present`, the obstruction must be clearly visible — not just inferred from context.",
        "- At least 15% of images will be double-coded for inter-annotator agreement checking.",
        "",
        "---",
        "",
        "## Ghana-Specific Notes",
        "",
        "- **Open gutters** are common along Accra roads — these are the rectangular concrete channels alongside roads.",
        "- **Drains filled with plastic bags and organic waste** are a major flood driver — always label `blocked_drain_present` AND `solid_waste_accumulation`.",
        "- **Informal kiosks built over drains** should trigger `informal_structure_near_drainage`.",
        "- **Earth shoulders** on paved roads should trigger `unpaved_shoulder`.",
        "- **Low road geometry** — if the road surface appears to be below the surrounding land level, mark `low_lying_street_form`.",
        "",
        "---",
        "*Codebook version 1.0 — Streetscape Flood Vulnerability Project, Accra, Ghana*",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    logger.info(f"Codebook → {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    manifest = pd.read_parquet(MANIFEST)
    sample   = sample_for_annotation(manifest)

    # Save CSV
    SAMPLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(SAMPLE_OUT, index=False)
    logger.info(f"Annotation sample → {SAMPLE_OUT}")

    # Exports
    export_label_studio(sample, LS_OUT)
    export_cvat_xml(CVAT_OUT)
    export_coco_categories(COCO_OUT)
    export_codebook(CODEBOOK)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n── Annotation Dataset Summary ──────────────────────────")
    print(f"  Total images selected:    {len(sample):>6,}")
    print()
    print("  By flood stratum:")
    for s, n in sample["flood_stratum"].value_counts().items():
        print(f"    {s:<30} {n:>5,}")
    print()
    print("  By highway group:")
    for g, n in sample["highway_group"].value_counts().items():
        print(f"    {g:<30} {n:>5,}")
    print()
    print("  By heading (°):")
    for h, n in sample["heading"].value_counts().sort_index().items():
        print(f"    {h:<30} {n:>5,}")
    print()
    print("  By capture year:")
    for y, n in sample["year"].value_counts().sort_index().items():
        print(f"    {y:<30} {n:>5,}")
    print()
    print("  Output files:")
    for p in [SAMPLE_OUT, LS_OUT, CVAT_OUT, COCO_OUT, CODEBOOK]:
        kb = p.stat().st_size / 1024 if p.exists() else 0
        print(f"    {p.name:<45} {kb:>7.1f} KB")


if __name__ == "__main__":
    main()
