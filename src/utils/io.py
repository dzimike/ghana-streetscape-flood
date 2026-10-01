"""I/O helpers: GeoPackage, Parquet, GeoJSON."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd


def save_geodataframe(gdf: gpd.GeoDataFrame, path: Path, layer: str | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".gpkg":
        gdf.to_file(path, layer=layer or path.stem, driver="GPKG")
    elif suffix in (".geojson", ".json"):
        gdf.to_file(path, driver="GeoJSON")
    elif suffix == ".parquet":
        gdf.to_parquet(path)
    else:
        raise ValueError(f"Unsupported format: {suffix}")


def load_geodataframe(path: Path, layer: str | None = None) -> gpd.GeoDataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return gpd.read_parquet(path)
    return gpd.read_file(path, layer=layer)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported format: {suffix}")
