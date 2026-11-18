"""
scoring.py
----------
Shared 0-1 scaling used by the continuous criteria.
"""
from __future__ import annotations

import numpy as np


def percentile_score(values: np.ndarray, mask: np.ndarray, invert: bool = False,
                     low: float = 1, high: float = 99) -> np.ndarray:
    """Scale to 0-1 between the AOI's low and high percentiles. NaN outside mask or where data is missing."""
    v = values.astype("float64")
    ok = mask & ~np.isnan(v)
    out = np.full(values.shape, np.nan, dtype="float32")
    if not ok.any():
        return out
    v_min, v_max = np.percentile(v[ok], low), np.percentile(v[ok], high)
    if v_max <= v_min:
        v_max = v_min + 1e-6
    norm = np.clip((v - v_min) / (v_max - v_min), 0, 1)
    out[ok] = (1 - norm[ok]) if invert else norm[ok]
    return out
