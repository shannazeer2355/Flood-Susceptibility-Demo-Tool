"""
drainage.py
-----------
Drainage density: length of mapped streams per unit area within a moving
window. Densely drained terrain collects runoff faster.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from scipy.ndimage import uniform_filter


def drainage_density(waterways: gpd.GeoDataFrame, grid, radius_m: float = 1000.0) -> np.ndarray:
    """Stream length per area (km/km²) in a (2·radius)-wide square window."""
    lines = [(g, 1) for g in waterways.geometry if g is not None and not g.is_empty]
    res = grid.resolution
    if not lines:
        return np.zeros(grid.shape, dtype="float32")
    streams = rasterize(lines, out_shape=grid.shape, transform=grid.transform, fill=0, dtype="uint8")
    size = max(3, int(round(2 * radius_m / res)) | 1)
    share = uniform_filter(streams.astype("float32"), size=size, mode="constant")
    return share * (1000.0 / res)
