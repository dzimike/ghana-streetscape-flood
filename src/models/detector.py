"""YOLOv8 object detection wrapper for drain/waste/water detection."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class DrainDetector:
    """Thin wrapper around Ultralytics YOLO for flood-indicator detection."""

    def __init__(self, weights_path: Path | str | None = None, device: str = "cpu"):
        from ultralytics import YOLO  # lazy import — not needed at import time

        if weights_path and Path(weights_path).exists():
            self.model = YOLO(str(weights_path))
        else:
            # Start from YOLOv8n pretrained on COCO; fine-tune on annotated data
            self.model = YOLO("yolov8n.pt")
        self.device = device

    def predict(self, image_paths: list[str | Path], conf: float = 0.25) -> list[dict[str, Any]]:
        """Run inference on a list of image paths.

        Returns:
            List of dicts with keys: path, boxes (xyxy), scores, class_ids, class_names.
        """
        results = self.model.predict(
            [str(p) for p in image_paths],
            conf=conf,
            device=self.device,
            verbose=False,
        )
        output = []
        for r, path in zip(results, image_paths):
            boxes = r.boxes
            output.append({
                "path": str(path),
                "boxes_xyxy": boxes.xyxy.cpu().tolist() if boxes else [],
                "scores": boxes.conf.cpu().tolist() if boxes else [],
                "class_ids": boxes.cls.cpu().tolist() if boxes else [],
                "class_names": [r.names[int(c)] for c in boxes.cls] if boxes else [],
            })
        return output

    def train(
        self,
        data_yaml: Path,
        epochs: int = 50,
        imgsz: int = 640,
        batch: int = 16,
        project: str = "models/detection",
    ) -> None:
        self.model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=project,
            device=self.device,
        )
