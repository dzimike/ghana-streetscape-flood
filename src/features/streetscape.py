"""Aggregate model predictions into per-road-segment streetscape features."""
from __future__ import annotations

import pandas as pd
import numpy as np
from loguru import logger


DRAIN_LABELS = [
    "visible_drain_present", "open_gutter_present", "blocked_drain_present",
    "culvert_or_bridge_visible", "visible_waterway_or_stream",
]
WASTE_LABELS = ["solid_waste_accumulation"]
WATER_LABELS = ["stagnant_water_visible"]
ROAD_LABELS = ["poor_road_condition", "roadside_erosion", "unpaved_shoulder"]
INFORMAL_LABELS = ["informal_structure_near_drainage"]


def aggregate_to_segments(
    predictions: pd.DataFrame,
    segment_col: str = "segment_id",
) -> pd.DataFrame:
    """Aggregate image-level predictions to road segment level.

    Args:
        predictions: DataFrame with 'segment_id', one column per label (0/1 or probability).
        segment_col: Column identifying the road segment.

    Returns:
        Segment-level feature DataFrame.
    """
    logger.info(f"Aggregating {len(predictions):,} image predictions to segments")
    agg = predictions.groupby(segment_col).agg(
        n_images=("pano_id", "count"),
        **{f"mean_{lbl}": (lbl, "mean") for lbl in _available(predictions, DRAIN_LABELS)},
        **{f"mean_{lbl}": (lbl, "mean") for lbl in _available(predictions, WASTE_LABELS)},
        **{f"mean_{lbl}": (lbl, "mean") for lbl in _available(predictions, WATER_LABELS)},
        **{f"mean_{lbl}": (lbl, "mean") for lbl in _available(predictions, ROAD_LABELS)},
        **{f"mean_{lbl}": (lbl, "mean") for lbl in _available(predictions, INFORMAL_LABELS)},
    ).reset_index()

    # Composite scores (0–1)
    drain_cols = [f"mean_{l}" for l in DRAIN_LABELS if f"mean_{l}" in agg.columns]
    waste_cols = [f"mean_{l}" for l in WASTE_LABELS if f"mean_{l}" in agg.columns]
    water_cols = [f"mean_{l}" for l in WATER_LABELS if f"mean_{l}" in agg.columns]
    road_cols  = [f"mean_{l}" for l in ROAD_LABELS  if f"mean_{l}" in agg.columns]

    agg["drain_score"]  = agg[drain_cols].mean(axis=1) if drain_cols else 0.0
    agg["waste_score"]  = agg[waste_cols].mean(axis=1) if waste_cols else 0.0
    agg["water_score"]  = agg[water_cols].mean(axis=1) if water_cols else 0.0
    agg["road_score"]   = agg[road_cols].mean(axis=1)  if road_cols  else 0.0

    # Streetscape sensitivity composite (S in SLFVI)
    agg["streetscape_sensitivity"] = (
        0.40 * agg["drain_score"]
        + 0.25 * agg["waste_score"]
        + 0.20 * agg["water_score"]
        + 0.15 * agg["road_score"]
    )
    logger.info(f"Aggregated to {len(agg):,} segments")
    return agg


def _available(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]
