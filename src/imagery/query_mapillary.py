"""Query Mapillary API for street-level imagery in Street View coverage gaps.

Strategy
--------
1. Find road points where Street View returned ZERO_RESULTS.
2. Query Mapillary Images API for images within 50 m of each gap point.
3. Download images (up to MAX_IMAGES_PER_POINT per point).
4. Store metadata and local paths for EfficientNet inference.

Mapillary API
-------------
- Register at https://www.mapillary.com/developer
- Create an application → copy the Client Token (MLY|...)
- Pass token via --token argument or MAPILLARY_TOKEN env variable

Usage
-----
  python src/imagery/query_mapillary.py --token MLY|... --sample 50
  python src/imagery/query_mapillary.py --token MLY|... --focus-districts
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import requests
from loguru import logger

# ── Config ────────────────────────────────────────────────────────────────────
MAPILLARY_API  = "https://graph.mapillary.com"
SEARCH_RADIUS  = 50       # metres — search for images within 50 m of gap point
MAX_PER_POINT  = 4        # max images to download per gap point
REQUEST_DELAY  = 0.2      # seconds between API calls (rate limit)
IMAGE_SIZE     = "thumb_1024_url"   # 1024px thumbnail

FOCUS_DISTRICTS = [
    "Ablekuma Central Municipal",
    "Ledzokuku Municipal",
    "Ga East",
    "La Dade-kotopon",
]

# ── Paths ─────────────────────────────────────────────────────────────────────
METADATA_PARQ  = ROOT / "data/interim/streetview_metadata.parquet"
SAMPLE_PTS     = ROOT / "data/interim/sample_points.csv"
ADM2           = ROOT / "data/external/GHA_ADM2.geojson"
OUT_META       = ROOT / "data/interim/mapillary_metadata.parquet"
OUT_MANIFEST   = ROOT / "data/interim/mapillary_manifest.csv"
IMG_DIR        = ROOT / "data/raw/images/mapillary"

# ── Mapillary image fields to fetch ──────────────────────────────────────────
IMG_FIELDS = [
    "id", "captured_at", "geometry",
    "compass_angle", "is_pano",
    "thumb_256_url", "thumb_1024_url",
]


def get_token() -> str:
    token = os.environ.get("MAPILLARY_TOKEN", "")
    if not token:
        raise ValueError(
            "Mapillary token not found.\n"
            "  1. Register at https://www.mapillary.com/developer\n"
            "  2. Create an application and copy the Client Token (MLY|...)\n"
            "  3. Pass it via --token MLY|... or set MAPILLARY_TOKEN env var"
        )
    return token


def search_images_near(lat: float, lon: float, radius: int,
                       token: str, max_results: int = 10) -> list[dict]:
    """Search Mapillary for images within radius metres of (lat, lon)."""
    url = f"{MAPILLARY_API}/images"
    params = {
        "access_token": token,
        "fields":       ",".join(IMG_FIELDS),
        "lat":          lat,
        "lng":          lon,
        "radius":       radius,
        "limit":        max_results,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            return resp.json().get("data", [])
        elif resp.status_code == 401:
            raise ValueError("Invalid Mapillary token — check your MLY|... key.")
        else:
            logger.warning(f"Mapillary {resp.status_code} at ({lat:.4f},{lon:.4f})")
            return []
    except requests.RequestException as e:
        logger.warning(f"Request failed: {e}")
        return []


def download_image(url: str, out_path: Path, token: str) -> bool:
    """Download a Mapillary image thumbnail."""
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(resp.content)
            return True
    except Exception as e:
        logger.warning(f"Download failed {url}: {e}")
    return False


def load_gap_points(focus_districts_only: bool = False) -> pd.DataFrame:
    """Return sample points where Street View had no coverage."""
    meta = pd.read_parquet(METADATA_PARQ)
    gaps = meta[meta["status"] != "OK"].copy()
    # normalise column names: query_lat/query_lon → latitude/longitude
    if "query_lat" in gaps.columns and "latitude" not in gaps.columns:
        gaps = gaps.rename(columns={"query_lat": "latitude", "query_lon": "longitude"})
    logger.info(f"Street View gap points: {len(gaps):,} / {len(meta):,}")

    if focus_districts_only:
        try:
            import geopandas as gpd
            adm = gpd.read_file(ADM2)
            focus = adm[adm["shapeName"].isin(FOCUS_DISTRICTS)]
            focus_union = focus.union_all()
            gap_gdf = gpd.GeoDataFrame(
                gaps,
                geometry=gpd.points_from_xy(gaps["longitude"], gaps["latitude"]),
                crs="EPSG:4326",
            )
            gaps = gap_gdf[gap_gdf.within(focus_union)].drop(columns="geometry")
            logger.info(f"Gap points in focus districts: {len(gaps):,}")
        except Exception as e:
            logger.warning(f"District filtering failed: {e} — using all gaps")

    return gaps.reset_index(drop=True)


def run(token: str, sample_n: int = 0,
        focus_districts: bool = False,
        download: bool = True) -> None:

    IMG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load gap points ───────────────────────────────────────────────────────
    gaps = load_gap_points(focus_districts_only=focus_districts)
    if sample_n:
        gaps = gaps.sample(min(sample_n, len(gaps)), random_state=42)
        logger.info(f"Sampled {len(gaps)} gap points")

    # Resume: skip already-queried points
    if OUT_META.exists():
        existing = pd.read_parquet(OUT_META)
        done_pts = set(zip(existing["query_lat"].round(5),
                           existing["query_lon"].round(5)))
        gaps = gaps[~gaps.apply(
            lambda r: (round(float(r["latitude"]), 5), round(float(r["longitude"]), 5))
            in done_pts, axis=1
        )]
        logger.info(f"Resuming — {len(gaps):,} gap points remaining")
    else:
        existing = pd.DataFrame()

    if len(gaps) == 0:
        logger.info("All gap points already queried.")
        return

    # ── Query Mapillary ───────────────────────────────────────────────────────
    records = []
    manifest_rows = []
    n_found = 0
    n_downloaded = 0

    for i, (_, row) in enumerate(gaps.iterrows()):
        lat, lon = float(row["latitude"]), float(row["longitude"])
        images = search_images_near(lat, lon, SEARCH_RADIUS, token, MAX_PER_POINT)

        for img in images[:MAX_PER_POINT]:
            img_id    = img.get("id", "")
            geom      = img.get("geometry", {})
            img_lon   = geom.get("coordinates", [lon, lat])[0]
            img_lat   = geom.get("coordinates", [lon, lat])[1]
            heading   = img.get("compass_angle", 0)
            captured  = img.get("captured_at", "")
            thumb_url = img.get(IMAGE_SIZE) or img.get("thumb_256_url", "")
            is_pano   = img.get("is_pano", False)

            local_path = IMG_DIR / f"{img_id}.jpg"

            records.append({
                "mapillary_id":  img_id,
                "query_lat":     lat,
                "query_lon":     lon,
                "img_lat":       img_lat,
                "img_lon":       img_lon,
                "heading":       heading,
                "captured_at":   captured,
                "is_pano":       is_pano,
                "thumb_url":     thumb_url,
                "local_path":    str(local_path),
                "downloaded":    False,
            })

            if download and thumb_url and not local_path.exists():
                ok = download_image(thumb_url, local_path, token)
                if ok:
                    records[-1]["downloaded"] = True
                    n_downloaded += 1

            n_found += 1

        time.sleep(REQUEST_DELAY)

        if (i + 1) % 50 == 0 or (i + 1) == len(gaps):
            logger.info(f"  Queried {i+1:,}/{len(gaps):,} points | "
                        f"found {n_found:,} images | downloaded {n_downloaded:,}")

    # ── Save results ──────────────────────────────────────────────────────────
    if records:
        new_df = pd.DataFrame(records)
        combined = pd.concat([existing, new_df], ignore_index=True) \
                   if len(existing) else new_df
        combined.to_parquet(OUT_META, index=False)

        manifest = combined[combined["downloaded"]][
            ["mapillary_id", "img_lat", "img_lon", "heading",
             "captured_at", "local_path"]
        ].rename(columns={
            "mapillary_id": "pano_id",
            "img_lat": "pano_lat",
            "img_lon": "pano_lon",
        })
        manifest["source"] = "mapillary"
        manifest.to_csv(OUT_MANIFEST, index=False)

        print("\n── Mapillary Query Results ─────────────────────────────────")
        print(f"  Gap points queried:     {len(gaps):,}")
        print(f"  Images found:           {n_found:,}")
        print(f"  Images downloaded:      {n_downloaded:,}")
        print(f"  Points with coverage:   "
              f"{new_df.groupby(['query_lat','query_lon']).size().gt(0).sum():,}")
        print()
        print("── Output Files ────────────────────────────────────────────")
        for p in [OUT_META, OUT_MANIFEST]:
            if p.exists():
                kb = p.stat().st_size / 1024
                tag = f"{kb/1024:.1f} MB" if kb > 1024 else f"{kb:.1f} KB"
                print(f"  {p.relative_to(ROOT)!s:<55} {tag}")
        print(f"  Images directory:  {IMG_DIR.relative_to(ROOT)}")
        print(f"    ({n_downloaded:,} .jpg files)")
    else:
        print("  No Mapillary images found for the queried gap points.")
        print("  This may mean:")
        print("    - Mapillary has limited coverage in these areas")
        print("    - The search radius (50 m) is too narrow — try --radius 100")
        print("    - The token is valid but the areas are truly uncovered")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Query Mapillary for imagery in Street View coverage gaps"
    )
    p.add_argument("--token",            type=str, default="",
                   help="Mapillary Client Token (MLY|...)")
    p.add_argument("--sample",           type=int, default=0,
                   help="Query only N gap points (0 = all)")
    p.add_argument("--focus-districts",  action="store_true",
                   help="Restrict to 4 focus districts only")
    p.add_argument("--no-download",      action="store_true",
                   help="Query metadata only, do not download images")
    args = p.parse_args()

    if args.token:
        os.environ["MAPILLARY_TOKEN"] = args.token

    try:
        token = get_token()
    except ValueError as e:
        print(f"\nError: {e}")
        sys.exit(1)

    run(
        token=token,
        sample_n=args.sample,
        focus_districts=args.focus_districts,
        download=not args.no_download,
    )
