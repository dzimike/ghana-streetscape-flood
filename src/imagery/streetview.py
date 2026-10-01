"""Google Street View Metadata API client.

Queries the metadata endpoint (not the image endpoint) to check coverage
and retrieve panorama IDs, dates, and attribution — all without incurring
Static API charges for uncovered points.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


BASE_METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
BASE_STATIC_URL = "https://maps.googleapis.com/maps/api/streetview"


class StreetViewClient:
    """Rate-limited Google Street View API client."""

    def __init__(self, api_key: str, requests_per_second: float = 10.0):
        self.api_key = api_key
        self._interval = 1.0 / requests_per_second
        self._last_call = 0.0
        self._client = httpx.Client(timeout=15)

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._interval:
            time.sleep(self._interval - elapsed)
        self._last_call = time.monotonic()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=10))
    def get_metadata(
        self,
        lat: float,
        lon: float,
        radius_m: int = 50,
        source: str = "outdoor",
    ) -> dict[str, Any]:
        """Query metadata for a single point.

        Args:
            lat: Latitude in WGS84.
            lon: Longitude in WGS84.
            radius_m: Search radius for nearest panorama.
            source: "outdoor" restricts to outdoor Street View.

        Returns:
            Parsed API response dict.
        """
        self._throttle()
        params = {
            "location": f"{lat},{lon}",
            "radius": radius_m,
            "source": source,
            "key": self.api_key,
        }
        resp = self._client.get(BASE_METADATA_URL, params=params)
        resp.raise_for_status()
        return resp.json()

    def build_static_url(
        self,
        pano_id: str,
        heading: int,
        pitch: int = 0,
        fov: int = 90,
        size: str = "640x640",
    ) -> str:
        """Build a Street View Static API URL (does not fetch the image).

        Storing the URL rather than the image keeps us compliant with
        Google Maps Platform terms when redistribution is restricted.
        """
        params = (
            f"pano={pano_id}&heading={heading}&pitch={pitch}"
            f"&fov={fov}&size={size}&key={self.api_key}"
        )
        return f"{BASE_STATIC_URL}?{params}"

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def query_metadata_batch(
    sample_points: pd.DataFrame,
    api_key: str,
    headings: list[int] | None = None,
    radius_m: int = 50,
    requests_per_second: float = 10.0,
    seen_pano_ids: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Query Street View metadata for a batch of sample points.

    Args:
        sample_points: DataFrame with 'point_id', 'latitude', 'longitude'.
        api_key: Google Maps Platform API key.
        headings: Image headings to generate URLs for.
        radius_m: Panorama search radius.
        requests_per_second: API rate limit.
        seen_pano_ids: Set of already-queried panorama IDs to deduplicate.

    Returns:
        Tuple of (metadata_df, image_manifest_df).
        metadata_df: One row per sample point with panorama details.
        image_manifest_df: One row per (panorama, heading) image request.
    """
    headings = headings or [0, 90, 180, 270]
    seen = seen_pano_ids or set()
    metadata_rows = []
    image_rows = []

    with StreetViewClient(api_key, requests_per_second) as client:
        for _, pt in sample_points.iterrows():
            point_id = pt["point_id"]
            lat, lon = pt["latitude"], pt["longitude"]

            try:
                meta = client.get_metadata(lat, lon, radius_m=radius_m)
            except Exception as exc:
                logger.warning(f"Metadata query failed for point {point_id}: {exc}")
                meta = {"status": "ERROR"}

            status = meta.get("status", "UNKNOWN")
            pano_id = meta.get("pano_id")
            date = meta.get("date")
            copyright_ = meta.get("copyright", "")
            pano_lat = meta.get("location", {}).get("lat")
            pano_lon = meta.get("location", {}).get("lng")

            metadata_rows.append({
                "point_id": point_id,
                "query_lat": lat,
                "query_lon": lon,
                "status": status,
                "pano_id": pano_id,
                "pano_lat": pano_lat,
                "pano_lon": pano_lon,
                "capture_date": date,
                "copyright": copyright_,
            })

            if status == "OK" and pano_id and pano_id not in seen:
                seen.add(pano_id)
                for heading in headings:
                    url = client.build_static_url(pano_id, heading)
                    image_rows.append({
                        "point_id": point_id,
                        "pano_id": pano_id,
                        "heading": heading,
                        "capture_date": date,
                        "pano_lat": pano_lat,
                        "pano_lon": pano_lon,
                        "image_url": url,
                        "downloaded": False,
                        "local_path": None,
                    })

    metadata_df = pd.DataFrame(metadata_rows)
    image_df = pd.DataFrame(image_rows)
    logger.info(
        f"Queried {len(metadata_df)} points — "
        f"{(metadata_df['status'] == 'OK').sum()} valid, "
        f"{len(image_df)} image records generated"
    )
    return metadata_df, image_df
