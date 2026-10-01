"""Image-level flood vulnerability classifier (EfficientNet / ConvNeXt backbone)."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

from ..annotation.schema import IMAGE_LABELS


NUM_CLASSES = len(IMAGE_LABELS)

TRANSFORM = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class FloodVulnerabilityClassifier(nn.Module):
    """Multi-label classifier for streetscape flood indicators."""

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_B2_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b2(weights=weights)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )
        self.label_names = IMAGE_LABELS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def save(self, path: Path) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: Path, **kwargs) -> "FloodVulnerabilityClassifier":
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return model
