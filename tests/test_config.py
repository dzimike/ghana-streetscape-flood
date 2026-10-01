"""Smoke tests for config loading and path resolution."""
from pathlib import Path
import pytest
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.config.settings import load_config, resolve_path


def test_load_accra_pilot():
    cfg = load_config("accra_pilot")
    assert cfg["project"]["name"] == "accra-pilot"
    assert "sampling" in cfg
    assert cfg["sampling"]["target_points"] == 5000


def test_resolve_path():
    cfg = load_config("accra_pilot")
    p = resolve_path(cfg, "data_raw")
    assert p.name == "raw"


def test_slfvi_weights_sum():
    cfg = load_config("accra_pilot")
    w = cfg["slfvi"]["weights"]
    total = sum(w.values())
    assert abs(total - 1.0) < 1e-6, f"Weights must sum to 1.0, got {total}"
