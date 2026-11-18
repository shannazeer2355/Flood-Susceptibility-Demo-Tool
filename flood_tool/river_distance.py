"""
river_distance.py
------------------
Distance-from-river susceptibility scoring via continuous Euclidean
distance decay.
"""
from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt

logger = logging.getLogger(__name__)


def rasterize_rivers(rivers: gpd.GeoDataFrame, grid) -> np.ndarray:
    shapes = [(geom, 1) for geom in rivers.geometry if geom is not None and not geom.is_empty]
    if not shapes:
        raise ValueError("Rivers layer has no valid features to rasterize.")
    return rasterize(
        shapes,
        out_shape=grid.shape,
        transform=grid.transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    )


def distance_to_rivers(river_raster: np.ndarray, pixel_size) -> np.ndarray:
    """Euclidean distance in metres. pixel_size: float or (size_y, size_x)."""
    if not (river_raster > 0).any():
        logger.warning(
            "No river/water pixels fall inside the DEM grid; river-distance "
            "susceptibility will be 0 everywhere. Check that the rivers layer "
            "overlaps the AOI."
        )
        return np.full(river_raster.shape, np.inf)

    sampling = pixel_size if np.ndim(pixel_size) else (pixel_size, pixel_size)
    return distance_transform_edt(river_raster == 0, sampling=sampling)


def river_distance_risk_score(distance_m: np.ndarray, mask: np.ndarray,
                              near: float = 500.0, far: float = 1500.0) -> np.ndarray:
    """Score 1 within `near` m, linear decay to 0 at `far` m."""
    if far <= near:
        raise ValueError("'far' distance must be greater than 'near' distance.")
    d = distance_m.astype("float64")
    risk = np.clip(1.0 - (d - near) / (far - near), 0.0, 1.0)

    out = np.full(distance_m.shape, np.nan, dtype="float64")
    out[mask] = risk[mask]
    return out
