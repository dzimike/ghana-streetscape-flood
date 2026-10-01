"""Annotation schema: label definitions for CVAT / Label Studio export."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ── Image-level classification labels ──────────────────────────────────────
IMAGE_LABELS: list[str] = [
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

# ── Object detection categories ──────────────────────────────────────────────
OBJECT_CATEGORIES: list[dict] = [
    {"id": 1,  "name": "drain",          "supercategory": "drainage"},
    {"id": 2,  "name": "gutter",         "supercategory": "drainage"},
    {"id": 3,  "name": "culvert",        "supercategory": "drainage"},
    {"id": 4,  "name": "water",          "supercategory": "water"},
    {"id": 5,  "name": "solid_waste",    "supercategory": "waste"},
    {"id": 6,  "name": "road",           "supercategory": "surface"},
    {"id": 7,  "name": "sidewalk",       "supercategory": "surface"},
    {"id": 8,  "name": "vegetation",     "supercategory": "vegetation"},
    {"id": 9,  "name": "building",       "supercategory": "built"},
    {"id": 10, "name": "kiosk_container","supercategory": "built"},
    {"id": 11, "name": "bridge",         "supercategory": "drainage"},
    {"id": 12, "name": "stream_channel", "supercategory": "water"},
]

# ── Segment-level vulnerability labels ──────────────────────────────────────
SEGMENT_LABELS: list[str] = [
    "low_flood_vulnerability",
    "moderate_flood_vulnerability",
    "high_flood_vulnerability",
    "uncertain_requires_field_check",
]

VulnerabilityClass = Literal[
    "low_flood_vulnerability",
    "moderate_flood_vulnerability",
    "high_flood_vulnerability",
    "uncertain_requires_field_check",
]


@dataclass
class AnnotationRecord:
    """One annotated image record."""
    image_id: str
    pano_id: str
    heading: int
    local_path: str
    image_labels: dict[str, bool] = field(default_factory=dict)
    vulnerability_class: VulnerabilityClass | None = None
    annotator_id: str | None = None
    annotation_timestamp: str | None = None
    notes: str = ""
