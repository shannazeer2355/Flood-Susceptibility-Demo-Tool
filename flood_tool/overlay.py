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

Per-pixel fallback: where a rainfall value legitimately doesn't exist
for a pixel (outside the rainfall source's coverage), that PIXEL uses
the 3-criteria score instead of being dropped entirely. Weights are
renormalised to sum to 1 in whichever mode applies, at the same
relative ratio the caller specified. No pixel ever receives a
fabricated/interpolated rainfall value — only real data is used,
just at whatever combination of layers is actually available there.

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

    # --- 3-criteria baseline score (always computed) ---------------------------
    base_keys = ("elevation", "slope", "river")
    base_defaults = DEFAULT_WEIGHTS_3
    base_input = weights or (DEFAULT_WEIGHTS_4 if has_rainfall else base_defaults)
    base_total = sum(base_input.get(k, base_defaults[k]) for k in base_keys)
    if base_total <= 0:
        raise ValueError("Elevation/slope/river weights must sum to a positive number.")
    base_w = {k: base_input.get(k, base_defaults[k]) / base_total for k in base_keys}

    score_base = (
        base_w["elevation"] * np.nan_to_num(elevation_risk)
        + base_w["slope"] * np.nan_to_num(slope_risk)
        + base_w["river"] * np.nan_to_num(river_risk)
    )
    # True out-of-AOI pixels (no elevation/slope/river data at all) stay excluded
    nan_mask = np.isnan(elevation_risk) | np.isnan(slope_risk) | np.isnan(river_risk)

    if not has_rainfall:
        score_base[nan_mask] = np.nan
        return score_base

    # --- 4-criteria score (only where rainfall has real data) ------------------
    all_keys = ("elevation", "slope", "river", "rainfall")
    default = DEFAULT_WEIGHTS_4
    w_in = weights or default
    all_total = sum(w_in.get(k, default[k]) for k in all_keys)
    if all_total <= 0:
        raise ValueError("Overlay weights must sum to a positive number.")
    all_w = {k: w_in.get(k, default[k]) / all_total for k in all_keys}

    score_full = (
        all_w["elevation"] * np.nan_to_num(elevation_risk)
        + all_w["slope"] * np.nan_to_num(slope_risk)
        + all_w["river"] * np.nan_to_num(river_risk)
        + all_w["rainfall"] * np.nan_to_num(rainfall_risk)
    )

    # Per-pixel choice: use the 4-criteria score where rainfall has real
    # data, fall back to the 3-criteria score where it doesn't — instead
    # of dropping the pixel entirely.
    rainfall_missing = np.isnan(rainfall_risk)
    score = np.where(rainfall_missing, score_base, score_full)
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
