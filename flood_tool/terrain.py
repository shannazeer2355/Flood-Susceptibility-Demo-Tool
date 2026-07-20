"""
terrain.py
----------
Elevation and slope based flood-risk scoring.

Both scores are normalised to the 0-1 range, where 1 = highest flood risk,
so they can be combined directly in the weighted overlay (overlay.py).

Elevation risk
    Lower elevation -> higher risk. We normalise elevation within the
    AOI's own min/max (relative risk within the study area), then invert.

Slope risk
    Flatter terrain -> higher risk (water pools / drains slowly).
    Slope is computed with a standard Horn (1981) 3x3 gradient kernel,
    matching what QGIS/GDAL's `gdaldem slope` produces.
"""
from __future__ import annotations

import numpy as np


def elevation_risk_score(elevation: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Normalise elevation to 0-1 then invert (low elevation -> risk near 1).
    Pixels outside `mask` are set to np.nan.
    """
    elev = elevation.astype("float64")
    valid = elev[mask]
    if valid.size == 0:
        raise ValueError("No valid elevation pixels found inside AOI.")

    e_min, e_max = np.nanpercentile(valid, 1), np.nanpercentile(valid, 99)
    if e_max <= e_min:
        e_max = e_min + 1e-6

    norm = (elev - e_min) / (e_max - e_min)
    norm = np.clip(norm, 0, 1)
    risk = 1.0 - norm  # low elevation -> high risk

    out = np.full(elevation.shape, np.nan, dtype="float64")
    out[mask] = risk[mask]
    return out


def compute_slope_degrees(elevation: np.ndarray, pixel_size_x: float, pixel_size_y: float,
                           nodata: float = -9999.0) -> np.ndarray:
    """
    Compute slope (degrees) from a DEM using Horn's 3x3 method.
    Edge pixels are handled by reflect-padding.
    """
    z = elevation.astype("float64")
    z_filled = np.where(z == nodata, np.nan, z)

    # Fill remaining NaNs locally so the kernel doesn't propagate holes
    nan_mask = np.isnan(z_filled)
    if nan_mask.any():
        mean_val = np.nanmean(z_filled)
        z_filled = np.where(nan_mask, mean_val, z_filled)

    padded = np.pad(z_filled, 1, mode="edge")
    dz_dx = (
        (padded[0:-2, 2:] + 2 * padded[1:-1, 2:] + padded[2:, 2:])
        - (padded[0:-2, 0:-2] + 2 * padded[1:-1, 0:-2] + padded[2:, 0:-2])
    ) / (8 * pixel_size_x)
    dz_dy = (
        (padded[2:, 0:-2] + 2 * padded[2:, 1:-1] + padded[2:, 2:])
        - (padded[0:-2, 0:-2] + 2 * padded[0:-2, 1:-1] + padded[0:-2, 2:])
    ) / (8 * pixel_size_y)

    slope_rad = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    slope_deg = np.degrees(slope_rad)
    slope_deg[nan_mask] = np.nan
    return slope_deg


def slope_risk_score(slope_deg: np.ndarray, mask: np.ndarray, max_slope_cutoff: float = 15.0) -> np.ndarray:
    """
    Normalise slope 0-1 then invert (flat -> risk near 1).
    Slopes at/above `max_slope_cutoff` degrees are treated as minimal risk (0).
    """
    norm = np.clip(slope_deg / max_slope_cutoff, 0, 1)
    risk = 1.0 - norm

    out = np.full(slope_deg.shape, np.nan, dtype="float64")
    out[mask] = risk[mask]
    return out
