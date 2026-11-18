import numpy as np

from flood_tool import overlay


def _layers(value):
    return np.full((2, 2), value, dtype="float64")


def test_weights_are_normalised():
    s1 = overlay.weighted_overlay(_layers(1), _layers(0), _layers(0),
                                  weights={"elevation": 2, "slope": 1, "river": 1})
    assert np.allclose(s1, 0.5)


def test_rainfall_used_where_available_and_fallback_elsewhere():
    rain = np.array([[1.0, np.nan], [1.0, np.nan]])
    w = {"elevation": 0.25, "slope": 0.25, "river": 0.25, "rainfall": 0.25}
    s = overlay.weighted_overlay(_layers(0), _layers(0), _layers(0), weights=w, rainfall_risk=rain)
    assert np.allclose(s[:, 0], 0.25)
    assert np.allclose(s[:, 1], 0.0)


def test_out_of_aoi_stays_nan():
    e = _layers(1)
    e[0, 0] = np.nan
    s = overlay.weighted_overlay(e, _layers(1), _layers(1))
    assert np.isnan(s[0, 0]) and not np.isnan(s[1, 1])


def test_classification_thresholds():
    score = np.array([[0.1, 0.4, 0.7, 0.71, np.nan]])
    zones = overlay.classify_risk(score)
    assert zones.tolist() == [[1, 2, 2, 3, 0]]
