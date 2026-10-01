"""Zero-shot CLIP classification for flood vulnerability indicators — Task 6.

Scores each image against 15 text prompts (one per IMAGE_LABEL) using
OpenAI CLIP (ViT-B/32). Produces a probability per label per image.

Output:
  data/processed/clip_predictions.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from loguru import logger

from src.config.settings import load_config
from src.annotation.schema import IMAGE_LABELS

cfg = load_config("accra_pilot")

MANIFEST_IN = ROOT / "data/interim/image_manifest.parquet"
OUT_PATH    = ROOT / "data/processed/clip_predictions.parquet"
BATCH_SIZE  = 32
MODEL_NAME  = "openai/clip-vit-base-patch32"

# Text prompts — one per label, phrased as a natural scene description
LABEL_PROMPTS: dict[str, str] = {
    "visible_drain_present":             "a street with a visible concrete drain or gutter channel",
    "open_gutter_present":               "an open roadside gutter running alongside the road",
    "blocked_drain_present":             "a blocked or clogged drain filled with waste and debris",
    "stagnant_water_visible":            "stagnant or ponded water on the road surface",
    "poor_road_condition":               "a road with potholes, cracks and surface damage",
    "heavy_impervious_surface":          "a street dominated by concrete and asphalt with no vegetation",
    "unpaved_shoulder":                  "an earth or gravel unpaved road shoulder",
    "informal_structure_near_drainage":  "informal market stalls or kiosks built next to a drain",
    "solid_waste_accumulation":          "solid waste and plastic bags dumped along the road",
    "visible_waterway_or_stream":        "a stream or river channel visible from the road",
    "low_lying_street_form":             "a road in a low-lying depression below surrounding land",
    "roadside_erosion":                  "erosion gullies and soil loss at the road edge",
    "pedestrian_exposure":               "pedestrians walking near a flood-risk area or open drain",
    "culvert_or_bridge_visible":         "a culvert pipe or small bridge crossing a drainage channel",
    "no_visible_drainage":               "a street with no visible drains or water infrastructure",
}

# Negative counterpart used to form binary probability
NEGATIVE_PROMPT = "a normal clean urban street with no flood risk features"


def run(sample_only: bool = False) -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Device: {device}")

    logger.info(f"Loading {MODEL_NAME} …")
    model     = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()

    manifest = pd.read_parquet(MANIFEST_IN)
    manifest = manifest[manifest["downloaded"].fillna(False)].copy()
    if sample_only:
        manifest = manifest.sample(200, random_state=42)
    logger.info(f"Running CLIP on {len(manifest):,} images")

    # Build text embeddings once for all labels
    labels  = list(LABEL_PROMPTS.keys())
    prompts = [LABEL_PROMPTS[l] for l in labels] + [NEGATIVE_PROMPT]

    with torch.no_grad():
        txt_inputs  = processor(text=prompts, return_tensors="pt", padding=True).to(device)
        txt_feats   = model.get_text_features(**txt_inputs)
        txt_feats   = txt_feats / txt_feats.norm(dim=-1, keepdim=True)

    rows   = []
    paths  = manifest["local_path"].tolist()
    ids    = manifest["pano_id"].tolist()
    heads  = manifest["heading"].tolist()
    pts    = manifest["point_id"].tolist()

    for batch_start in tqdm(range(0, len(paths), BATCH_SIZE), desc="CLIP inference"):
        batch_paths  = paths[batch_start: batch_start + BATCH_SIZE]
        batch_ids    = ids[batch_start: batch_start + BATCH_SIZE]
        batch_heads  = heads[batch_start: batch_start + BATCH_SIZE]
        batch_pts    = pts[batch_start: batch_start + BATCH_SIZE]

        images = []
        valid  = []
        for i, p in enumerate(batch_paths):
            try:
                images.append(Image.open(p).convert("RGB"))
                valid.append(i)
            except Exception:
                pass

        if not images:
            continue

        with torch.no_grad():
            img_inputs = processor(images=images, return_tensors="pt").to(device)
            img_feats  = model.get_image_features(**img_inputs)
            img_feats  = img_feats / img_feats.norm(dim=-1, keepdim=True)
            # cosine similarity: (n_images, n_prompts)
            sims       = (img_feats @ txt_feats.T).cpu().float()

        for local_i, global_i in enumerate(valid):
            sim_row  = sims[local_i]           # (n_labels + 1,)
            pos_sims = sim_row[:-1]            # label prompts
            neg_sim  = sim_row[-1]             # negative prompt

            # Sigmoid of (positive - negative) sim as a probability proxy
            probs = torch.sigmoid(10 * (pos_sims - neg_sim)).tolist()

            row = {
                "pano_id":  batch_ids[global_i],
                "heading":  batch_heads[global_i],
                "point_id": batch_pts[global_i],
            }
            for lbl, prob in zip(labels, probs):
                row[f"clip_{lbl}"] = round(prob, 4)
            rows.append(row)

    df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    logger.info(f"CLIP predictions → {OUT_PATH}  ({len(df):,} rows)")

    # Quick stats
    print("\n── CLIP Zero-Shot Summary ──────────────────────────────")
    print(f"  Images scored: {len(df):,}")
    score_cols = [c for c in df.columns if c.startswith("clip_")]
    means = df[score_cols].mean().sort_values(ascending=False)
    print("  Mean probability per label (top 10):")
    for col, v in means.head(10).items():
        print(f"    {col.replace('clip_',''):<40} {v:.3f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sample", action="store_true", help="Run on 200 images only (test mode)")
    args = p.parse_args()
    run(sample_only=args.sample)
