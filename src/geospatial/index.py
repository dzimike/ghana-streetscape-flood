"""Street-Level Flood Vulnerability Index (SLFVI) computation."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger


SLFVI_THRESHOLDS = {
    "very_low":  0.20,
    "low":       0.40,
    "moderate":  0.60,
    "high":      0.80,
}


def normalise(series: pd.Series) -> pd.Series:
    """Min-max normalise to 0–1, handling constant columns."""
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.0, index=series.index)
    return (series - lo) / (hi - lo)


def compute_slfvi(
    df: pd.DataFrame,
    hazard_col: str = "hazard_score",
    exposure_col: str = "exposure_score",
    sensitivity_col: str = "streetscape_sensitivity",
    adaptive_col: str = "adaptive_capacity_score",
    weights: dict | None = None,
) -> pd.DataFrame:
    """Compute SLFVI for each road segment / grid cell.

    SLFVI = 0.30H + 0.20E + 0.35S + 0.15A

    Args:
        df: Feature DataFrame containing component columns.
        weights: Override default component weights.

    Returns:
        DataFrame with 'slfvi' and 'slfvi_class' columns added.
    """
    w = weights or {"hazard": 0.30, "exposure": 0.20,
                    "sensitivity": 0.35, "adaptive_capacity": 0.15}

    result = df.copy()
    H = normalise(result[hazard_col])
    E = normalise(result[exposure_col])
    S = normalise(result[sensitivity_col])
    A = normalise(result[adaptive_col])

    result["slfvi"] = (
        w["hazard"] * H
        + w["exposure"] * E
        + w["sensitivity"] * S
        + w["adaptive_capacity"] * A
    )

    result["slfvi_class"] = pd.cut(
        result["slfvi"],
        bins=[0, 0.20, 0.40, 0.60, 0.80, 1.01],
        labels=["very_low", "low", "moderate", "high", "very_high"],
        right=True,
    )

    logger.info(
        f"SLFVI computed for {len(result):,} units. "
        f"Mean={result['slfvi'].mean():.3f}, "
        f"High/VeryHigh={(result['slfvi'] > 0.60).sum():,}"
    )
    return result


def rank_maintenance_priorities(
    gdf: gpd.GeoDataFrame,
    top_n: int = 100,
    score_col: str = "slfvi",
) -> gpd.GeoDataFrame:
    """Return top N road segments ranked by SLFVI for drainage maintenance."""
    ranked = gdf.nlargest(top_n, score_col).copy()
    ranked["priority_rank"] = range(1, len(ranked) + 1)
    return ranked
