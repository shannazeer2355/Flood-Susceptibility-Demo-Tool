"""
landcover.py
------------
Land-cover score from ESA WorldCover classes. Surfaces that shed water
quickly or are already wet score high; dense vegetation scores low.
"""
from __future__ import annotations

import numpy as np

WORLDCOVER_SCORES = {
    10: ("Tree cover", 0.1),
    20: ("Shrubland", 0.3),
    30: ("Grassland", 0.4),
    40: ("Cropland", 0.6),
    50: ("Built-up", 0.9),
    60: ("Bare / sparse vegetation", 0.7),
    70: ("Snow and ice", 0.2),
    80: ("Permanent water", 1.0),
    90: ("Herbaceous wetland", 0.9),
    95: ("Mangroves", 0.8),
    100: ("Moss and lichen", 0.3),
}


def landcover_score(classes: np.ndarray, mask: np.ndarray, scores: dict | None = None) -> np.ndarray:
    table = {k: v[1] for k, v in WORLDCOVER_SCORES.items()}
    if scores:
        table.update(scores)
    out = np.full(classes.shape, np.nan, dtype="float32")
    for code, value in table.items():
        out[(classes == code) & mask] = value
    return out
