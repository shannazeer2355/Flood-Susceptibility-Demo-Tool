"""
overlay.py
----------
Combine the individual 0-1 criterion scores into a single weighted
flood-susceptibility score, and classify it into High / Medium / Low zones.

    Score = sum(w_i * s_i) / sum(w_i)

over the criteria that have data at each pixel. With the original three
criteria (elevation, slope, distance to water) the default weights are
0.4 / 0.3 / 0.3; adding rainfall gives 0.3 / 0.2 / 0.2 / 0.3. Weights for
the full open-data set of criteria come from AHP (see ahp.py).

Classification uses fixed cut-offs (score > 0.7 High, 0.4-0.7 Medium,
< 0.4 Low) or natural breaks fitted to the AOI's own score distribution.
"""
from __future__ import annotations

import numpy as np

DEFAULT_WEIGHTS_3 = {"elevation": 0.4, "slope": 0.3, "river": 0.3}
DEFAULT_WEIGHTS_4 = {"elevation": 0.3, "slope": 0.2, "river": 0.2, "rainfall": 0.3}


def combine_scores(scores: dict, weights: dict, required=("elevation", "slope")) -> np.ndarray:
    """
    Weighted sum of the criterion scores. At each pixel the weights of the
    criteria that have data are rescaled to sum to 1, so a gap in one layer
    does not pull the score down. Pixels missing a required criterion are NaN.
    """
    keys = [k for k in scores if weights.get(k, 0) > 0]
    if not keys:
        raise ValueError("At least one criterion needs a positive weight.")
    shape = next(iter(scores.values())).shape
    total = np.zeros(shape)
    weight_sum = np.zeros(shape)
    for k in keys:
        s = scores[k]
        ok = ~np.isnan(s)
        total[ok] += weights[k] * s[ok]
        weight_sum[ok] += weights[k]
    with np.errstate(invalid="ignore", divide="ignore"):
        score = total / weight_sum
    score[weight_sum == 0] = np.nan
    for k in required:
        if k in scores:
            score[np.isnan(scores[k])] = np.nan
    return score


def weighted_overlay(elevation_risk: np.ndarray, slope_risk: np.ndarray,
                     river_risk: np.ndarray, weights: dict | None = None,
                     rainfall_risk: np.ndarray | None = None) -> np.ndarray:
    scores = {"elevation": elevation_risk, "slope": slope_risk, "river": river_risk}
    default = DEFAULT_WEIGHTS_3
    if rainfall_risk is not None:
        scores["rainfall"] = rainfall_risk
        default = DEFAULT_WEIGHTS_4
    w = {k: (weights or default).get(k, default[k]) for k in scores}
    if sum(w.values()) <= 0:
        raise ValueError("Overlay weights must sum to a positive number.")
    return combine_scores(scores, w, required=("elevation", "slope", "river"))


def natural_breaks(score: np.ndarray, n_sample: int = 2000, seed: int = 0) -> tuple[float, float]:
    """
    Jenks natural breaks for three classes, fitted on a random sample of the
    scores. Returns (medium_cut, high_cut) for classify_risk.
    """
    v = score[~np.isnan(score)]
    if v.size < 3:
        return 0.4, 0.7
    rng = np.random.default_rng(seed)
    v = np.sort(rng.choice(v, size=min(n_sample, v.size), replace=False))
    n = v.size
    c1 = np.concatenate([[0.0], np.cumsum(v)])
    c2 = np.concatenate([[0.0], np.cumsum(v * v)])

    def ssd(a, b):
        cnt = b - a
        return c2[b] - c2[a] - (c1[b] - c1[a]) ** 2 / cnt

    best, breaks = np.inf, (0.4, 0.7)
    for i in range(1, n - 1):
        j = np.arange(i + 1, n)
        cost = ssd(0, i) + ssd(i, j) + ssd(j, n)
        k = int(np.argmin(cost))
        if cost[k] < best:
            best, breaks = cost[k], (v[i], v[j[k] - 1])
    return float(breaks[0]), float(breaks[1])


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


def majority_filter(zones: np.ndarray, size: int = 5) -> np.ndarray:
    """Replace each zone pixel with the most common zone in a size x size window (NoData kept)."""
    from scipy.ndimage import uniform_filter

    counts = np.stack([uniform_filter((zones == z).astype("float32"), size=size, mode="nearest")
                       for z in (1, 2, 3)])
    smoothed = (np.argmax(counts, axis=0) + 1).astype("uint8")
    smoothed[zones == 0] = 0
    return smoothed


ZONE_LABELS = {0: "NoData", 1: "Low", 2: "Medium", 3: "High"}
ZONE_COLORS = {1: "#2ecc71", 2: "#f1c40f", 3: "#e74c3c"}
