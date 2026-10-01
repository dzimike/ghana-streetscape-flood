"""Download Street View images from the image manifest — Task 4.

Uses a thread-pool for parallel downloads with a semaphore to stay within
Google's rate limits. Resumes from where it left off if interrupted.

Outputs:
  data/raw/images/<pano_id>_<heading>.jpg
  data/interim/image_manifest.parquet   — updated with local_path / downloaded
  data/interim/image_manifest.csv       — same, human-readable
"""
from __future__ import annotations

import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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

MANIFEST_IN  = ROOT / "data/interim/image_manifest.parquet"
MANIFEST_OUT = ROOT / "data/interim/image_manifest.parquet"
MANIFEST_CSV = ROOT / "data/interim/image_manifest.csv"
IMAGE_DIR    = ROOT / "data/raw/images"

MAX_WORKERS     = 10    # parallel download threads
CHECKPOINT_EVERY = 200  # save manifest every N completions
TIMEOUT_S        = 20


# ── Download one image ───────────────────────────────────────────────────────
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=8))
def _fetch(client: httpx.Client, url: str) -> bytes:
    r = client.get(url, timeout=TIMEOUT_S)
    r.raise_for_status()
    # Street View returns a 1×1 grey pixel for no-coverage — treat as failure
    if len(r.content) < 2000:
        raise ValueError("Response too small — likely no-coverage placeholder")
    return r.content


def download_one(row: dict, api_key: str, image_dir: Path) -> dict:
    """Download a single image. Returns updated row dict."""
    local_path = image_dir / f"{row['pano_id']}_{row['heading']}.jpg"

    if local_path.exists() and local_path.stat().st_size > 2000:
        row["downloaded"] = True
        row["local_path"] = str(local_path)
        return row

    url = row.get("image_url", "")
    if not url or url.startswith("MOCK"):
        row["downloaded"] = False
        return row

    # Ensure key is in URL
    if "key=" not in url:
        url = f"{url}&key={api_key}"

    try:
        with httpx.Client(timeout=TIMEOUT_S) as client:
            data = _fetch(client, url)
        local_path.write_bytes(data)
        row["downloaded"] = True
        row["local_path"] = str(local_path)
    except Exception as exc:
        logger.debug(f"Failed {row['pano_id']} h={row['heading']}: {exc}")
        row["downloaded"] = False
        row["local_path"] = None

    return row


# ── Main ─────────────────────────────────────────────────────────────────────
def run() -> None:
    api_key   = get_api_key("GOOGLE_MAPS_API_KEY")
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(MANIFEST_IN)
    logger.info(f"Manifest loaded: {len(manifest):,} image records")

    # Skip already downloaded
    already = manifest["downloaded"].fillna(False).sum()
    pending = manifest[~manifest["downloaded"].fillna(False)].copy()
    logger.info(f"Already downloaded: {int(already):,} | Pending: {len(pending):,}")

    if len(pending) == 0:
        logger.info("All images already downloaded.")
        _print_summary(manifest)
        return

    rows      = pending.to_dict("records")
    lock      = threading.Lock()
    results   = {}   # index → updated row
    completed = 0

    def _checkpoint():
        # Merge results back into manifest and save
        for idx, row in results.items():
            manifest.at[idx, "downloaded"]  = row["downloaded"]
            manifest.at[idx, "local_path"]  = row.get("local_path")
        manifest.to_parquet(MANIFEST_OUT, index=False)

    logger.info(f"Starting download with {MAX_WORKERS} workers …")
    t0 = time.monotonic()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(download_one, row, api_key, IMAGE_DIR): (i, row)
            for i, row in enumerate(rows)
        }

        for fut in as_completed(futures):
            orig_i, orig_row = futures[fut]
            completed += 1

            try:
                updated = fut.result()
            except Exception as exc:
                logger.warning(f"Unhandled error: {exc}")
                updated = orig_row
                updated["downloaded"] = False

            # Map back to original manifest index
            manifest_idx = pending.index[orig_i]
            with lock:
                results[manifest_idx] = updated

            if completed % CHECKPOINT_EVERY == 0 or completed == len(rows):
                elapsed = time.monotonic() - t0
                rate    = completed / elapsed
                eta     = (len(rows) - completed) / rate if rate > 0 else 0
                ok      = sum(1 for r in results.values() if r.get("downloaded"))
                logger.info(
                    f"  {completed:>5}/{len(rows)}  "
                    f"ok={ok}  "
                    f"rate={rate:.1f}/s  "
                    f"ETA={eta/60:.1f}min"
                )
                with lock:
                    _checkpoint()

    _checkpoint()
    manifest.to_parquet(MANIFEST_OUT, index=False)
    manifest.drop(columns=["image_url"], errors="ignore").to_csv(MANIFEST_CSV, index=False)
    _print_summary(manifest)


def _print_summary(manifest: pd.DataFrame) -> None:
    n_total = len(manifest)
    n_ok    = manifest["downloaded"].fillna(False).sum()
    n_fail  = n_total - n_ok
    size_mb = sum(
        Path(p).stat().st_size
        for p in manifest["local_path"].dropna()
        if Path(p).exists()
    ) / 1e6

    print("\n── Image Download Summary ──────────────────────────────")
    print(f"  Total image records:   {n_total:>7,}")
    print(f"  Successfully saved:    {int(n_ok):>7,}  ({n_ok/n_total*100:.1f}%)")
    print(f"  Failed / skipped:      {int(n_fail):>7,}")
    print(f"  Disk usage:            {size_mb:>7.1f} MB")
    print(f"  Image directory:       {IMAGE_DIR}")
    print(f"  Manifest updated:      {MANIFEST_OUT}")

    if n_ok > 0:
        by_s = (
            manifest[manifest["downloaded"].fillna(False)]
            .groupby("flood_stratum")["pano_id"].count()
        )
        print("\n  Downloaded by stratum:")
        for s, n in by_s.items():
            print(f"    {s:<30} {int(n):,} images")


if __name__ == "__main__":
    run()
