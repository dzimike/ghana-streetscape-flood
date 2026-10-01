"""Street View Metadata query pipeline — Task 3.

Reads sample_points.csv, queries the Street View Metadata API for each
point, deduplicates panorama IDs, and writes:

  data/interim/streetview_metadata.parquet   — one row per sample point
  data/interim/image_manifest.parquet        — one row per (pano, heading)
  data/interim/image_manifest.csv            — same, for inspection

Designed to be re-run: already-queried points are skipped via a checkpoint
file so a crash or rate-limit pause can be resumed without re-querying.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import httpx
import pandas as pd
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config.settings import load_config, get_api_key

# ── Config ───────────────────────────────────────────────────────────────────
cfg = load_config("accra_pilot")

POINTS_CSV    = ROOT / "data/interim/sample_points.csv"
META_OUT      = ROOT / "data/interim/streetview_metadata.parquet"
MANIFEST_OUT  = ROOT / "data/interim/image_manifest.parquet"
MANIFEST_CSV  = ROOT / "data/interim/image_manifest.csv"
CHECKPOINT    = ROOT / "data/interim/.metadata_checkpoint.parquet"

METADATA_URL  = cfg["streetview"]["metadata_url"]
STATIC_URL    = cfg["streetview"]["static_url"]
HEADINGS      = cfg["sampling"]["headings"]          # [0, 90, 180, 270]
PITCH         = cfg["sampling"]["pitch"]             # 0
FOV           = cfg["sampling"]["fov"]               # 90
IMAGE_SIZE    = cfg["sampling"]["image_size"]        # "640x640"
RPS           = cfg["streetview"]["requests_per_second"]   # 10
RETRY_ATT     = cfg["streetview"]["retry_attempts"]        # 3
RADIUS        = 50   # metres — search radius for nearest panorama


# ── API client ───────────────────────────────────────────────────────────────
class _Client:
    def __init__(self, api_key: str, rps: float):
        self._key = api_key
        self._interval = 1.0 / rps
        self._last = 0.0
        self._http = httpx.Client(timeout=15)

    def _throttle(self):
        wait = self._interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=10))
    def metadata(self, lat: float, lon: float) -> dict:
        self._throttle()
        r = self._http.get(METADATA_URL, params={
            "location": f"{lat},{lon}",
            "radius": RADIUS,
            "source": "outdoor",
            "key": self._key,
        })
        r.raise_for_status()
        return r.json()

    def static_url(self, pano_id: str, heading: int) -> str:
        return (
            f"{STATIC_URL}?pano={pano_id}&heading={heading}"
            f"&pitch={PITCH}&fov={FOV}&size={IMAGE_SIZE}&key={self._key}"
        )

    def close(self):
        self._http.close()


# ── Helpers ──────────────────────────────────────────────────────────────────
def _load_checkpoint() -> tuple[pd.DataFrame, set[str]]:
    """Return (rows_so_far, seen_pano_ids)."""
    if CHECKPOINT.exists():
        df = pd.read_parquet(CHECKPOINT)
        seen = set(df.loc[df["pano_id"].notna(), "pano_id"].astype(str))
        logger.info(f"Resuming from checkpoint: {len(df):,} rows, {len(seen)} pano IDs seen")
        return df, seen
    return pd.DataFrame(), set()


def _save_checkpoint(rows: list[dict]) -> None:
    pd.DataFrame(rows).to_parquet(CHECKPOINT, index=False)


# ── Main query loop ──────────────────────────────────────────────────────────
def run(batch_size: int = 500) -> None:
    api_key = get_api_key("GOOGLE_MAPS_API_KEY")
    points  = pd.read_csv(POINTS_CSV)
    logger.info(f"Loaded {len(points):,} sample points")

    existing, seen_panos = _load_checkpoint()
    done_ids = set(existing["point_id"].tolist()) if len(existing) else set()

    pending = points[~points["point_id"].isin(done_ids)]
    logger.info(f"Pending: {len(pending):,} points to query")

    if len(pending) == 0:
        logger.info("All points already queried — loading from checkpoint")
        meta_rows = existing.to_dict("records")
    else:
        meta_rows = existing.to_dict("records") if len(existing) else []
        client = _Client(api_key, RPS)

        try:
            for i, (_, pt) in enumerate(pending.iterrows(), 1):
                pid    = pt["point_id"]
                lat    = pt["latitude"]
                lon    = pt["longitude"]

                try:
                    resp = client.metadata(lat, lon)
                except Exception as exc:
                    logger.warning(f"[{i}/{len(pending)}] point {pid} FAILED: {exc}")
                    resp = {"status": "ERROR"}

                status   = resp.get("status", "UNKNOWN")
                pano_id  = resp.get("pano_id")
                location = resp.get("location", {})

                meta_rows.append({
                    "point_id":    pid,
                    "query_lat":   lat,
                    "query_lon":   lon,
                    "highway":     pt.get("highway"),
                    "flood_stratum": pt.get("flood_stratum"),
                    "dist_drain_m": pt.get("dist_drain_m"),
                    "status":      status,
                    "pano_id":     pano_id,
                    "pano_lat":    location.get("lat"),
                    "pano_lon":    location.get("lng"),
                    "capture_date": resp.get("date"),
                    "copyright":   resp.get("copyright", ""),
                })

                if pano_id:
                    seen_panos.add(pano_id)

                # Checkpoint every batch_size
                if i % batch_size == 0:
                    _save_checkpoint(meta_rows)
                    ok = sum(1 for r in meta_rows if r["status"] == "OK")
                    logger.info(f"Progress {i}/{len(pending)} — {ok} valid so far")

        finally:
            client.close()
            _save_checkpoint(meta_rows)

    # ── Build metadata DataFrame ─────────────────────────────────────────────
    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_parquet(META_OUT, index=False)
    logger.info(f"Metadata saved → {META_OUT}  ({len(meta_df):,} rows)")

    # ── Build image manifest (deduplicated by pano_id) ───────────────────────
    valid = meta_df[meta_df["status"] == "OK"].drop_duplicates("pano_id")
    image_rows = []

    for _, row in valid.iterrows():
        pano_id = row["pano_id"]
        for heading in HEADINGS:
            url = (
                f"{STATIC_URL}?pano={pano_id}&heading={heading}"
                f"&pitch={PITCH}&fov={FOV}&size={IMAGE_SIZE}&key={api_key}"
            )
            image_rows.append({
                "point_id":     row["point_id"],
                "pano_id":      pano_id,
                "heading":      heading,
                "capture_date": row["capture_date"],
                "pano_lat":     row["pano_lat"],
                "pano_lon":     row["pano_lon"],
                "highway":      row["highway"],
                "flood_stratum": row["flood_stratum"],
                "image_url":    url,
                "downloaded":   False,
                "local_path":   None,
            })

    manifest_df = pd.DataFrame(image_rows)
    manifest_df.to_parquet(MANIFEST_OUT, index=False)
    manifest_df.drop(columns=["image_url"]).to_csv(MANIFEST_CSV, index=False)

    # ── Summary ──────────────────────────────────────────────────────────────
    n_ok      = (meta_df["status"] == "OK").sum()
    n_zero    = (meta_df["status"] == "ZERO_RESULTS").sum()
    n_err     = (~meta_df["status"].isin(["OK", "ZERO_RESULTS"])).sum()
    n_unique  = meta_df.loc[meta_df["status"] == "OK", "pano_id"].nunique()
    n_images  = len(manifest_df)

    print("\n── Street View Metadata Summary ────────────────────────")
    print(f"  Points queried:       {len(meta_df):>7,}")
    print(f"  Status OK:            {n_ok:>7,}  ({n_ok/len(meta_df)*100:.1f}%)")
    print(f"  Zero results:         {n_zero:>7,}  ({n_zero/len(meta_df)*100:.1f}%)")
    print(f"  Errors:               {n_err:>7,}")
    print(f"  Unique panoramas:     {n_unique:>7,}")
    print(f"  Image records (×4):   {n_images:>7,}")
    print(f"\n  Metadata:  {META_OUT}")
    print(f"  Manifest:  {MANIFEST_OUT}")

    if n_ok > 0:
        dates = pd.to_datetime(
            meta_df.loc[meta_df["status"] == "OK", "capture_date"], errors="coerce"
        )
        print(f"\n  Capture date range:   {dates.min().date()} → {dates.max().date()}")

        by_stratum = (
            meta_df[meta_df["status"] == "OK"]
            .groupby("flood_stratum")["pano_id"].count()
        )
        print("\n  Valid panoramas by stratum:")
        for stratum, n in by_stratum.items():
            print(f"    {stratum:<30} {n:,}")


# ── Dry-run / mock mode ──────────────────────────────────────────────────────
def run_dry_run(coverage_rate: float = 0.72) -> None:
    """Generate synthetic metadata for pipeline testing without a live API key.

    Simulates realistic Street View coverage (~72%) over Accra, with lower
    coverage on service roads and higher coverage on primary/trunk roads.
    """
    import numpy as np
    rng = np.random.default_rng(42)

    points = pd.read_csv(POINTS_CSV)
    logger.info(f"[DRY-RUN] Generating synthetic metadata for {len(points):,} points")

    COVERAGE = {
        "motorway": 0.95, "trunk": 0.90, "primary": 0.88,
        "secondary": 0.82, "tertiary": 0.75, "residential": 0.68,
        "unclassified": 0.60, "service": 0.45,
    }

    meta_rows = []
    image_rows = []
    seen: set[str] = set()

    for _, pt in points.iterrows():
        p_hit = COVERAGE.get(str(pt.get("highway", "")), coverage_rate)
        ok = rng.random() < p_hit
        status = "OK" if ok else "ZERO_RESULTS"
        pano_id = f"MOCK_{pt['point_id']:05d}" if ok else None

        # Simulate capture date between 2019 and 2024
        year = rng.integers(2019, 2025)
        month = rng.integers(1, 13)
        date = f"{year}-{month:02d}"

        meta_rows.append({
            "point_id":     pt["point_id"],
            "query_lat":    pt["latitude"],
            "query_lon":    pt["longitude"],
            "highway":      pt.get("highway"),
            "flood_stratum": pt.get("flood_stratum"),
            "dist_drain_m": pt.get("dist_drain_m"),
            "status":       status,
            "pano_id":      pano_id,
            "pano_lat":     pt["latitude"] + rng.uniform(-0.0001, 0.0001),
            "pano_lon":     pt["longitude"] + rng.uniform(-0.0001, 0.0001),
            "capture_date": date if ok else None,
            "copyright":    "© Google" if ok else "",
        })

        if ok and pano_id and pano_id not in seen:
            seen.add(pano_id)
            client_tmp = _Client("MOCK_KEY", RPS)
            for heading in HEADINGS:
                image_rows.append({
                    "point_id":     pt["point_id"],
                    "pano_id":      pano_id,
                    "heading":      heading,
                    "capture_date": date,
                    "pano_lat":     meta_rows[-1]["pano_lat"],
                    "pano_lon":     meta_rows[-1]["pano_lon"],
                    "highway":      pt.get("highway"),
                    "flood_stratum": pt.get("flood_stratum"),
                    "image_url":    f"MOCK_URL/{pano_id}_{heading}.jpg",
                    "downloaded":   False,
                    "local_path":   None,
                })
            client_tmp.close()

    meta_df     = pd.DataFrame(meta_rows)
    manifest_df = pd.DataFrame(image_rows)

    for p in [META_OUT, MANIFEST_OUT, MANIFEST_CSV]:
        p.parent.mkdir(parents=True, exist_ok=True)

    meta_df.to_parquet(META_OUT, index=False)
    manifest_df.to_parquet(MANIFEST_OUT, index=False)
    manifest_df.drop(columns=["image_url"]).to_csv(MANIFEST_CSV, index=False)

    n_ok = (meta_df["status"] == "OK").sum()
    logger.info(f"[DRY-RUN] {n_ok:,}/{len(meta_df):,} points with coverage  →  {len(manifest_df):,} image records")

    print("\n── Street View Metadata Summary (DRY-RUN / MOCK) ───────")
    print(f"  Points queried:       {len(meta_df):>7,}")
    print(f"  Status OK (simulated): {n_ok:>6,}  ({n_ok/len(meta_df)*100:.1f}%)")
    print(f"  Unique panoramas:     {meta_df['pano_id'].nunique():>7,}")
    print(f"  Image records (×4):   {len(manifest_df):>7,}")
    print(f"\n  Metadata:  {META_OUT}")
    print(f"  Manifest:  {MANIFEST_OUT}")
    by_stratum = meta_df[meta_df["status"]=="OK"].groupby("flood_stratum")["pano_id"].count()
    print("\n  Valid panoramas by stratum (simulated):")
    for s, n in by_stratum.items():
        print(f"    {s:<30} {n:,}")
    print("\n  NOTE: run without --dry-run once Street View API is enabled.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Query Street View metadata for Accra sample points")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate synthetic metadata without calling the API")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="Checkpoint every N queries (default 500)")
    args = parser.parse_args()

    if args.dry_run:
        run_dry_run()
    else:
        run(batch_size=args.batch_size)
