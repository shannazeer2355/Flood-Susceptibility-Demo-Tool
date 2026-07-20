"""
overlay.py
----------
Combine individual 0-1 risk layers into a single weighted flood-risk
score, and classify that score into High / Medium / Low zones.

    Final Risk Score = w_elev * elevation_risk
                      + w_slope * slope_risk
                      + w_river * river_risk

Default weights match the brief: 0.4 / 0.3 / 0.3.
Classification thresholds:
    score > 0.7        -> High
    0.4 <= score <= 0.7 -> Medium
    score < 0.4         -> Low
"""
from __future__ import annotations

import numpy as np

DEFAULT_WEIGHTS = {"elevation": 0.4, "slope": 0.3, "river": 0.3}


def weighted_overlay(elevation_risk: np.ndarray, slope_risk: np.ndarray,
                      river_risk: np.ndarray, weights: dict | None = None) -> np.ndarray:
    weights = weights or DEFAULT_WEIGHTS
    total = weights["elevation"] + weights["slope"] + weights["river"]
    if not np.isclose(total, 1.0):
        # Auto-normalise so the caller can pass arbitrary relative weights
        weights = {k: v / total for k, v in weights.items()}

    score = (
        weights["elevation"] * np.nan_to_num(elevation_risk)
        + weights["slope"] * np.nan_to_num(slope_risk)
        + weights["river"] * np.nan_to_num(river_risk)
    )
    # Preserve NaN (outside-AOI) pixels
    nan_mask = np.isnan(elevation_risk) | np.isnan(slope_risk) | np.isnan(river_risk)
    score[nan_mask] = np.nan
    return score


def classify_risk(score: np.ndarray, high_cut: float = 0.7, medium_cut: float = 0.4) -> np.ndarray:
    """
    Classify continuous score into integer zones:
        1 = Low, 2 = Medium, 3 = High, 0 = NoData
    """
    zones = np.zeros(score.shape, dtype="uint8")
    valid = ~np.isnan(score)

    low = valid & (score < medium_cut)
    medium = valid & (score >= medium_cut) & (score <= high_cut)
    high = valid & (score > high_cut)

    zones[low] = 1
    zones[medium] = 2
    zones[high] = 3
    return zones


ZONE_LABELS = {0: "NoData", 1: "Low", 2: "Medium", 3: "High"}
ZONE_COLORS = {1: "#2ecc71", 2: "#f1c40f", 3: "#e74c3c"}  # green / yellow / red
