"""Semantic segmentation wrapper (SegFormer) for surface-type estimation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

SEGMENTATION_CLASSES = [
    "road", "vegetation", "water", "building",
    "drain", "bare_ground", "waste", "sidewalk", "sky", "other",
]
NUM_SEG_CLASSES = len(SEGMENTATION_CLASSES)


class StreetscapeSegmentor:
    """SegFormer-based semantic segmentor fine-tuned on streetscape classes."""

    def __init__(self, model_name: str = "nvidia/segformer-b2-finetuned-cityscapes-512-1024",
                 device: str = "cpu"):
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.model.eval().to(device)
        self.device = device

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict[str, float]:
        """Return pixel-fraction per class for a single image."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits  # (1, num_classes, H, W)
        pred = logits.argmax(dim=1).squeeze().cpu().numpy()
        total = pred.size
        fractions = {}
        for i, cls_name in enumerate(SEGMENTATION_CLASSES):
            fractions[f"seg_frac_{cls_name}"] = float((pred == i).sum() / total)
        return fractions
