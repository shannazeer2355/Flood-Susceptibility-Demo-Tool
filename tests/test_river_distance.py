import numpy as np
import pytest

from flood_tool import river_distance


def test_distance_uses_pixel_size():
    river = np.zeros((1, 6), dtype="uint8")
    river[0, 0] = 1
    d = river_distance.distance_to_rivers(river, pixel_size=30.0)
    assert np.allclose(d[0], np.arange(6) * 30.0)


def test_distance_with_non_square_pixels():
    river = np.zeros((5, 5), dtype="uint8")
    river[0, 0] = 1
    d = river_distance.distance_to_rivers(river, pixel_size=(20.0, 10.0))
    assert d[4, 0] == pytest.approx(80.0)
    assert d[0, 4] == pytest.approx(40.0)


def test_no_river_pixels_gives_lowest_score():
    river = np.zeros((3, 3), dtype="uint8")
    d = river_distance.distance_to_rivers(river, pixel_size=30.0)
    score = river_distance.river_distance_risk_score(d, np.ones((3, 3), bool))
    assert np.all(score == 0.0)


def test_piecewise_linear_decay():
    d = np.array([[0.0, 500.0, 1000.0, 1500.0, 3000.0]])
    score = river_distance.river_distance_risk_score(d, np.ones_like(d, bool), near=500, far=1500)
    assert np.allclose(score, [[1.0, 1.0, 0.5, 0.0, 0.0]])


def test_far_must_exceed_near():
    with pytest.raises(ValueError):
        river_distance.river_distance_risk_score(np.zeros((1, 1)), np.ones((1, 1), bool), near=500, far=500)
