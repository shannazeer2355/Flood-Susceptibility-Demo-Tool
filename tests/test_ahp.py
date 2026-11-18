import numpy as np
import pytest

from flood_tool import ahp


def test_consistent_matrix_has_zero_cr():
    w_true = np.array([0.5, 0.3, 0.2])
    m = w_true[:, None] / w_true[None, :]
    w, cr = ahp.ahp_weights(m)
    assert np.allclose(w, w_true)
    assert cr == pytest.approx(0, abs=1e-9)


def test_reciprocal_matrix_from_judgements():
    m = ahp.pairwise_matrix(["a", "b", "c"], {("a", "b"): 3, ("c", "a"): 2})
    assert m[0, 1] == 3 and m[1, 0] == pytest.approx(1 / 3)
    assert m[0, 2] == pytest.approx(0.5) and m[2, 0] == 2
    assert m[1, 2] == 1


def test_default_judgements_are_consistent():
    weights, cr = ahp.weights_for(list(ahp.CRITERIA))
    assert cr < 0.1
    assert sum(weights.values()) == pytest.approx(1)
    assert weights["elevation"] > weights["soil"]


def test_inconsistent_judgements_are_flagged():
    _, cr = ahp.weights_for(["a", "b", "c"], {("a", "b"): 9, ("b", "c"): 9, ("c", "a"): 9})
    assert cr > 0.1


def test_subset_of_criteria():
    weights, _ = ahp.weights_for(["elevation", "slope"])
    assert set(weights) == {"elevation", "slope"}
    assert weights["elevation"] == pytest.approx(2 / 3)
