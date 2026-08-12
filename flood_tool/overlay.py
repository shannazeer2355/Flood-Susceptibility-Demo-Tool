"""
overlay.py
----------
Combine individual 0-1 risk layers into a single weighted flood-risk
score, and classify that score into High / Medium / Low zones.

Two modes:
  * 3-criteria (default, backward compatible):
      Score = w_elev*elevation + w_slope*slope + w_river*river
      Default weights: 0.4 / 0.3 / 0.3
  * 4-criteria (when a rainfall layer is supplied):
      Score = w_elev*elevation + w_slope*slope + w_river*river + w_rain*rainfall
      Default weights: 0.3 / 0.2 / 0.2 / 0.3

Rainfall is optional so existing 3-criteria runs (and previously
published results, e.g. the Alappuzha case study) remain reproducible
exactly as before when no rainfall layer is provided.

Classification thresholds:
    score > 0.7        -> High
    0.4 <= score <= 0.7 -> Medium
    score < 0.4         -> Low
"""
from __future__ import annotations

import numpy as np

DEFAULT_WEIGHTS_3 = {"elevation": 0.4, "slope": 0.3, "river": 0.3}
DEFAULT_WEIGHTS_4 = {"elevation": 0.3, "slope": 0.2, "river": 0.2, "rainfall": 0.3}


def weighted_overlay(elevation_risk: np.ndarray, slope_risk: np.ndarray,
                      river_risk: np.ndarray, weights: dict | None = None,
                      rainfall_risk: np.ndarray | None = None) -> np.ndarray:
    has_rainfall = rainfall_risk is not None
    default = DEFAULT_WEIGHTS_4 if has_rainfall else DEFAULT_WEIGHTS_3
    weights = weights or default

    keys = ("elevation", "slope", "river", "rainfall") if has_rainfall else ("elevation", "slope", "river")
    total = sum(weights.get(k, default[k]) for k in keys)
    if total <= 0:
        raise ValueError("Overlay weights must sum to a positive number.")
    norm_w = {k: weights.get(k, default[k]) / total for k in keys}

    score = (
        norm_w["elevation"] * np.nan_to_num(elevation_risk)
        + norm_w["slope"] * np.nan_to_num(slope_risk)
        + norm_w["river"] * np.nan_to_num(river_risk)
    )
    nan_mask = np.isnan(elevation_risk) | np.isnan(slope_risk) | np.isnan(river_risk)

    if has_rainfall:
        score = score + norm_w["rainfall"] * np.nan_to_num(rainfall_risk)
        nan_mask = nan_mask | np.isnan(rainfall_risk)

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
ZONE_COLORS = {1: "#2ecc71", 2: "#f1c40f", 3: "#e74c3c"}
