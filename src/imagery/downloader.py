"""Image downloader for Street View Static API.

Downloads images listed in an image manifest, tracks progress,
and respects Google Maps Platform terms by storing local copies
only for internal research use.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import httpx
import pandas as pd
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=10))
def _fetch_image(client: httpx.Client, url: str) -> bytes:
    resp = client.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content


def download_images(
    manifest: pd.DataFrame,
    output_dir: Path,
    api_key: str,
    requests_per_second: float = 5.0,
    skip_existing: bool = True,
) -> pd.DataFrame:
    """Download images from the image manifest.

    Args:
        manifest: DataFrame with columns 'pano_id', 'heading', 'image_url'.
        output_dir: Directory to write JPEG files.
        api_key: Google Maps API key (appended to URLs if missing).
        requests_per_second: Download rate limit.
        skip_existing: Skip rows where local_path already exists.

    Returns:
        Updated manifest with 'downloaded' and 'local_path' filled in.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    interval = 1.0 / requests_per_second
    manifest = manifest.copy()

    with httpx.Client(timeout=30) as client:
        for idx, row in manifest.iterrows():
            local_path = output_dir / f"{row['pano_id']}_{row['heading']}.jpg"
            if skip_existing and local_path.exists():
                manifest.at[idx, "downloaded"] = True
                manifest.at[idx, "local_path"] = str(local_path)
                continue

            url = row["image_url"]
            if "key=" not in url:
                url = f"{url}&key={api_key}"

            try:
                data = _fetch_image(client, url)
                local_path.write_bytes(data)
                manifest.at[idx, "downloaded"] = True
                manifest.at[idx, "local_path"] = str(local_path)
            except Exception as exc:
                logger.warning(f"Failed to download {row['pano_id']} h={row['heading']}: {exc}")
                manifest.at[idx, "downloaded"] = False

            time.sleep(interval)

    n_ok = manifest["downloaded"].sum()
    logger.info(f"Downloaded {n_ok}/{len(manifest)} images to {output_dir}")
    return manifest
