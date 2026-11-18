"""
ahp.py
------
Analytic Hierarchy Process (Saaty 1980). Criterion weights are the
principal eigenvector of a pairwise comparison matrix; the consistency
ratio (CR) flags contradictory judgements and should stay below 0.1.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

RANDOM_INDEX = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

CRITERIA = {
    "elevation": "Elevation",
    "river": "Distance to water",
    "slope": "Slope",
    "rainfall": "Monsoon rainfall",
    "drainage": "Drainage density",
    "landcover": "Land cover",
    "soil": "Soil clay content",
}

DEFAULT_JUDGEMENTS = {
    ("elevation", "river"): 1,
    ("elevation", "slope"): 2,
    ("elevation", "rainfall"): 3,
    ("elevation", "drainage"): 4,
    ("elevation", "landcover"): 5,
    ("elevation", "soil"): 6,
    ("river", "slope"): 2,
    ("river", "rainfall"): 3,
    ("river", "drainage"): 4,
    ("river", "landcover"): 5,
    ("river", "soil"): 6,
    ("slope", "rainfall"): 2,
    ("slope", "drainage"): 3,
    ("slope", "landcover"): 4,
    ("slope", "soil"): 5,
    ("rainfall", "drainage"): 2,
    ("rainfall", "landcover"): 3,
    ("rainfall", "soil"): 4,
    ("drainage", "landcover"): 2,
    ("drainage", "soil"): 3,
    ("landcover", "soil"): 2,
}


def pairwise_matrix(criteria: list[str], judgements: dict | None = None) -> np.ndarray:
    """
    Build the reciprocal matrix. judgements[(a, b)] = how many times more
    important a is than b on the 1-9 scale (use 1/x when b is more important).
    """
    judgements = DEFAULT_JUDGEMENTS if judgements is None else judgements
    n = len(criteria)
    m = np.ones((n, n))
    for i, j in combinations(range(n), 2):
        a, b = criteria[i], criteria[j]
        if (a, b) in judgements:
            v = float(judgements[(a, b)])
        elif (b, a) in judgements:
            v = 1.0 / float(judgements[(b, a)])
        else:
            v = 1.0
        if v <= 0:
            raise ValueError(f"Judgement for {a} vs {b} must be positive.")
        m[i, j], m[j, i] = v, 1.0 / v
    return m


def ahp_weights(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (weights, consistency ratio)."""
    n = matrix.shape[0]
    if n == 1:
        return np.array([1.0]), 0.0
    eigvals, eigvecs = np.linalg.eig(matrix)
    k = int(np.argmax(eigvals.real))
    w = np.abs(eigvecs[:, k].real)
    w = w / w.sum()
    lambda_max = float(eigvals[k].real)
    ci = (lambda_max - n) / (n - 1)
    ri = RANDOM_INDEX.get(n, 1.49)
    cr = ci / ri if ri else 0.0
    return w, max(cr, 0.0)


def weights_for(criteria: list[str], judgements: dict | None = None) -> tuple[dict, float]:
    w, cr = ahp_weights(pairwise_matrix(criteria, judgements))
    return dict(zip(criteria, w.tolist())), cr
