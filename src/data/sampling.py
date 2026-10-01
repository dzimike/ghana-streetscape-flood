"""Road-segment sampling frame builder for Accra.

Phase 2 of the pipeline: downloads OSM roads, cleans, splits into
segments, generates sample points, and stratifies by flood-risk class.
"""
from __future__ import annotations

# Implementation lives in notebooks/01_sampling_design.ipynb
# and src/data/osm.py — this module is the public entry point.

from .osm import download_roads, clean_roads, split_into_segments, generate_sample_points

__all__ = ["download_roads", "clean_roads", "split_into_segments", "generate_sample_points"]
