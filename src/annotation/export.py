"""Export image manifests to CVAT and Label Studio annotation formats."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .schema import IMAGE_LABELS, OBJECT_CATEGORIES, SEGMENT_LABELS


def export_label_studio(
    manifest: pd.DataFrame,
    output_path: Path,
    image_root_url: str = "http://localhost:8080/data/local-files/?d=",
) -> None:
    """Export manifest as Label Studio JSON import file.

    Args:
        manifest: DataFrame with 'pano_id', 'heading', 'local_path'.
        output_path: Path to write the JSON file.
        image_root_url: Base URL that Label Studio uses to serve images.
    """
    tasks = []
    for _, row in manifest.iterrows():
        image_url = image_root_url + row["local_path"]
        tasks.append({
            "data": {
                "image": image_url,
                "pano_id": row["pano_id"],
                "heading": int(row["heading"]),
                "capture_date": str(row.get("capture_date", "")),
                "point_id": str(row.get("point_id", "")),
            }
        })
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(tasks, indent=2))


def export_cvat_xml_header(
    output_path: Path,
    task_name: str = "Accra Flood Vulnerability",
) -> None:
    """Write a minimal CVAT XML skeleton with label definitions."""
    labels_xml = "\n".join(
        f'    <label><name>{lbl}</name><type>checkbox</type></label>'
        for lbl in IMAGE_LABELS + SEGMENT_LABELS
    )
    xml = f"""<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta>
    <task>
      <name>{task_name}</name>
      <labels>
{labels_xml}
      </labels>
    </task>
  </meta>
</annotations>
"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(xml)


def export_coco_categories(output_path: Path) -> None:
    """Write COCO-format category list for object detection tasks."""
    payload = {"categories": OBJECT_CATEGORIES}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
