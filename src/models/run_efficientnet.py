"""Task 7: EfficientNet-B0 inference over all images.

Loads the gold-label checkpoint and runs batch inference over every image
in data/interim/image_manifest.parquet.  Outputs per-image vulnerability
probability and 15 flood-indicator probabilities.

Output
------
  data/processed/efficientnet_predictions.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

MANIFEST      = ROOT / "data/interim/image_manifest.parquet"
CHECKPOINT    = ROOT / "models/classification/efficientnet_b0_flood_gold.pth"
OUT_PREDS     = ROOT / "data/processed/efficientnet_predictions.parquet"

IMAGE_SIZE = 224
LABEL_COLS = [
    "visible_drain_present", "open_gutter_present", "blocked_drain_present",
    "stagnant_water_visible", "poor_road_condition", "heavy_impervious_surface",
    "unpaved_shoulder", "informal_structure_near_drainage", "solid_waste_accumulation",
    "visible_waterway_or_stream", "low_lying_street_form", "roadside_erosion",
    "pedestrian_exposure", "culvert_or_bridge_visible", "no_visible_drainage",
]
N_LABELS = len(LABEL_COLS)

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class InferenceDataset(Dataset):
    def __init__(self, paths: list[str]):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        try:
            img = Image.open(p).convert("RGB")
            return VAL_TRANSFORM(img), idx, True
        except (FileNotFoundError, UnidentifiedImageError, OSError):
            # Return a blank tensor for missing/corrupt images
            return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE), idx, False


class FloodClassifier(nn.Module):
    def __init__(self, n_labels: int):
        super().__init__()
        backbone = models.efficientnet_b0(weights=None)
        n_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone   = backbone
        self.dropout    = nn.Dropout(0.3)
        self.vuln_head  = nn.Linear(n_features, 2)
        self.label_head = nn.Linear(n_features, n_labels)

    def forward(self, x):
        feat = self.dropout(self.backbone(x))
        return self.vuln_head(feat), self.label_head(feat)


def run(batch_size: int = 32, sample_n: int = 0) -> None:
    OUT_PREDS.parent.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(MANIFEST)
    if sample_n:
        manifest = manifest.head(sample_n)

    paths = manifest["local_path"].tolist()
    logger.info(f"Images to score: {len(paths):,}")

    device = (
        torch.device("mps")  if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    logger.info(f"Device: {device}")

    model = FloodClassifier(N_LABELS).to(device)
    state = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(state)
    model.eval()
    logger.info(f"Loaded checkpoint: {CHECKPOINT.name}")

    dataset = InferenceDataset(paths)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=0, pin_memory=False)

    vuln_probs  = np.full(len(paths), np.nan)
    label_probs = np.full((len(paths), N_LABELS), np.nan)
    valid_mask  = np.zeros(len(paths), dtype=bool)

    n_done = 0
    with torch.no_grad():
        for imgs, idxs, valids in loader:
            imgs = imgs.to(device)
            vuln_logit, label_logit = model(imgs)

            vp  = torch.softmax(vuln_logit, dim=1)[:, 1].cpu().numpy()
            lp  = torch.sigmoid(label_logit).cpu().numpy()
            v   = valids.numpy()

            for i, (idx, ok) in enumerate(zip(idxs.numpy(), v)):
                if ok:
                    vuln_probs[idx]     = vp[i]
                    label_probs[idx, :] = lp[i]
                    valid_mask[idx]     = True

            n_done += len(idxs)
            if n_done % 1000 == 0 or n_done == len(paths):
                logger.info(f"  Scored {n_done:,}/{len(paths):,} images")

    n_valid = int(valid_mask.sum())
    logger.info(f"Valid images scored: {n_valid:,} / {len(paths):,}  "
                f"({100*n_valid/len(paths):.1f}%)")

    result = manifest[["pano_id", "heading", "point_id"]].copy()
    result["enet_vuln_prob"] = vuln_probs
    result["enet_valid"]     = valid_mask
    for i, lbl in enumerate(LABEL_COLS):
        result[f"enet_{lbl}"] = label_probs[:, i]

    result.to_parquet(OUT_PREDS, index=False)
    kb = OUT_PREDS.stat().st_size / 1024
    logger.info(f"Saved → {OUT_PREDS}  ({kb/1024:.1f} MB)")

    print("\n── EfficientNet Prediction Summary ─────────────────────────")
    print(f"  Total images:   {len(result):,}")
    print(f"  Valid scored:   {n_valid:,}  ({100*n_valid/len(result):.1f}%)")
    valid = result[result["enet_valid"]]
    print(f"  Mean vuln prob: {valid['enet_vuln_prob'].mean():.3f}")
    print(f"  High-vuln (>0.6): {(valid['enet_vuln_prob']>0.6).sum():,}  "
          f"({100*(valid['enet_vuln_prob']>0.6).mean():.1f}%)")
    print()
    print(f"  Top labels by mean probability:")
    label_means = {lbl: valid[f"enet_{lbl}"].mean() for lbl in LABEL_COLS}
    for lbl, mean in sorted(label_means.items(), key=lambda x: -x[1])[:8]:
        print(f"    {lbl:<45} {mean:.3f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--batch",  type=int, default=32)
    p.add_argument("--sample", type=int, default=0,
                   help="Score only first N images (0=all)")
    args = p.parse_args()
    run(batch_size=args.batch, sample_n=args.sample)
