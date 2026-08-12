"""
river_distance.py
------------------
Distance-from-river risk scoring via continuous Euclidean distance decay.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt


def rasterize_rivers(rivers: gpd.GeoDataFrame, grid) -> np.ndarray:
    if rivers.empty:
        raise ValueError("Rivers layer has no features to rasterize.")
    shapes = [(geom, 1) for geom in rivers.geometry if geom is not None and not geom.is_empty]
    burned = rasterize(
        shapes,
        out_shape=grid.shape,
        transform=grid.transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    )
    return burned


def distance_to_rivers(river_raster: np.ndarray, pixel_size: float) -> np.ndarray:
    dist_px = distance_transform_edt(river_raster == 0)
    return dist_px * pixel_size


def river_distance_risk_score(distance_m: np.ndarray, mask: np.ndarray,
                               near: float = 500.0, far: float = 1500.0) -> np.ndarray:
    d = distance_m.astype("float64")
    risk = np.where(
        d <= near, 1.0,
        np.where(d >= far, 0.0, 1.0 - (d - near) / (far - near))
    )
    out = np.full(distance_m.shape, np.nan, dtype="float64")
    out[mask] = risk[mask]
    return out
