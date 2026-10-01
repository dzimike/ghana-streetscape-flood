"""Unit tests for SLFVI computation."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.geospatial.index import compute_slfvi, normalise


def _make_df(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "segment_id": [f"seg_{i}" for i in range(n)],
        "hazard_score": rng.uniform(0, 1, n),
        "exposure_score": rng.uniform(0, 1, n),
        "streetscape_sensitivity": rng.uniform(0, 1, n),
        "adaptive_capacity_score": rng.uniform(0, 1, n),
    })


def test_slfvi_range():
    df = compute_slfvi(_make_df())
    assert df["slfvi"].between(0, 1).all(), "SLFVI values must be in [0, 1]"


def test_slfvi_classes():
    df = compute_slfvi(_make_df(100))
    valid = {"very_low", "low", "moderate", "high", "very_high"}
    assert set(df["slfvi_class"].cat.categories).issubset(valid)


def test_normalise_constant():
    s = pd.Series([5.0] * 10)
    out = normalise(s)
    assert (out == 0.0).all()
