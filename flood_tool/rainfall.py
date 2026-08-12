"""
rainfall.py
-----------
Rainfall-based flood risk scoring, using long-term monsoon-season
climatology (a multi-year average, not any single year's total).

Rationale: elevation, slope, and river-distance are all structural
properties of the terrain that don't change year to year. Using a
multi-year monsoon-season average keeps rainfall in that same spirit —
a stable indicator of "how much rain does this area typically receive
during the flood-risk season", rather than a one-off event forecast
tied to a specific year.

Higher average monsoon rainfall -> higher risk (unlike elevation/slope,
this is a direct, non-inverted relationship).
"""
from __future__ import annotations

import numpy as np


def rainfall_risk_score(rainfall: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Normalise rainfall to a 0-1 risk score. Higher rainfall -> higher risk.
    Normalised within the AOI's own 1st-99th percentile range, same
    approach as elevation_risk_score, so an unusually wet/dry single pixel
    doesn't dominate the scale.
    """
    r = rainfall.astype("float64")
    valid = r[mask]
    if valid.size == 0:
        raise ValueError("No valid rainfall pixels found inside AOI.")

    r_min, r_max = np.nanpercentile(valid, 1), np.nanpercentile(valid, 99)
    if r_max <= r_min:
        r_max = r_min + 1e-6

    norm = np.clip((r - r_min) / (r_max - r_min), 0, 1)

    out = np.full(rainfall.shape, np.nan, dtype="float64")
    out[mask] = norm[mask]
    return out
