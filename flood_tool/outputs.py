"""
outputs.py
----------
Write final results to disk: risk raster (GeoTIFF) and risk polygons
(GeoJSON), matching the deliverables list in the brief.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes as rio_shapes
from shapely.geometry import shape

from .overlay import ZONE_LABELS


def write_raster(zones: np.ndarray, grid, out_path: str):
    profile = {
        "driver": "GTiff",
        "height": grid.height,
        "width": grid.width,
        "count": 1,
        "dtype": "uint8",
        "crs": grid.crs,
        "transform": grid.transform,
        "nodata": 0,
        "compress": "lzw",
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(zones, 1)


def polygonize_zones(zones: np.ndarray, grid, out_path: str) -> gpd.GeoDataFrame:
    mask = zones > 0
    records = []
    for geom, value in rio_shapes(zones, mask=mask, transform=grid.transform):
        records.append({"geometry": shape(geom), "risk_zone": ZONE_LABELS[int(value)]})

    gdf = gpd.GeoDataFrame(records, crs=grid.crs)
    gdf = gdf.dissolve(by="risk_zone", as_index=False)  # merge adjacent same-zone polygons
    gdf["area_km2"] = gdf.geometry.area / 1_000_000
    gdf.to_file(out_path, driver="GeoJSON")
    return gdf
