"""
terrain.py
----------
Elevation and slope scores (0-1, NaN outside the AOI).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


def elevation_risk_score(elevation: np.ndarray, mask: np.ndarray, log_scale: bool = False) -> np.ndarray:
    """
    Lower ground -> higher score (1st-99th percentile normalisation).
    log_scale stretches the low end, so differences of a few metres on a
    coastal plain still count when the AOI also contains high hills.
    """
    elev = elevation.astype("float64")
    valid = elev[mask]
    if valid.size == 0:
        raise ValueError("No valid elevation pixels found inside AOI.")

    e_min, e_max = np.nanpercentile(valid, 1), np.nanpercentile(valid, 99)
    if e_max <= e_min:
        e_max = e_min + 1e-6

    if log_scale:
        norm = np.log1p(np.clip(elev - e_min, 0, None)) / np.log1p(e_max - e_min)
        norm = np.clip(norm, 0, 1)
    else:
        norm = np.clip((elev - e_min) / (e_max - e_min), 0, 1)
    risk = 1.0 - norm

    out = np.full(elevation.shape, np.nan, dtype="float64")
    out[mask] = risk[mask]
    return out


def _fill_nodata_nearest(z: np.ndarray, nan_mask: np.ndarray) -> np.ndarray:
    if not nan_mask.any():
        return z
    if nan_mask.all():
        raise ValueError("DEM contains no valid elevation values.")
    idx = distance_transform_edt(nan_mask, return_distances=False, return_indices=True)
    return z[tuple(idx)]


def compute_slope_degrees(elevation: np.ndarray, pixel_size_x: float, pixel_size_y: float,
                          nodata: float = -9999.0) -> np.ndarray:
    """Slope in degrees, Horn (1981) 3x3 kernel."""
    z = elevation.astype("float64")
    z = np.where(z == nodata, np.nan, z)
    nan_mask = np.isnan(z)
    z_filled = _fill_nodata_nearest(z, nan_mask)

    p = np.pad(z_filled, 1, mode="edge")
    dz_dx = (
        (p[0:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:])
        - (p[0:-2, 0:-2] + 2 * p[1:-1, 0:-2] + p[2:, 0:-2])
    ) / (8 * pixel_size_x)
    dz_dy = (
        (p[2:, 0:-2] + 2 * p[2:, 1:-1] + p[2:, 2:])
        - (p[0:-2, 0:-2] + 2 * p[0:-2, 1:-1] + p[0:-2, 2:])
    ) / (8 * pixel_size_y)

    slope_deg = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    slope_deg[nan_mask] = np.nan
    return slope_deg


def slope_risk_score(slope_deg: np.ndarray, mask: np.ndarray, max_slope_cutoff: float = 15.0) -> np.ndarray:
    """Flatter ground -> higher score, 0 at or above max_slope_cutoff."""
    if max_slope_cutoff <= 0:
        raise ValueError("max_slope_cutoff must be positive.")
    norm = np.clip(slope_deg / max_slope_cutoff, 0, 1)
    risk = 1.0 - norm

    out = np.full(slope_deg.shape, np.nan, dtype="float64")
    out[mask] = risk[mask]
    return out
