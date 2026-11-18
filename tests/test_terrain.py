import numpy as np
import pytest

from flood_tool import terrain


def test_slope_of_known_plane():
    x = np.arange(20) * 10.0
    z = np.tile(x * 0.1, (15, 1))
    slope = terrain.compute_slope_degrees(z, 10.0, 10.0)
    assert np.allclose(slope[1:-1, 1:-1], np.degrees(np.arctan(0.1)))


def test_slope_handles_anisotropic_pixels():
    y = np.arange(12)[:, None] * 30.0
    z = np.tile(y / 30.0, (1, 10))
    slope = terrain.compute_slope_degrees(z, 10.0, 30.0)
    assert np.allclose(slope[1:-1, 1:-1], np.degrees(np.arctan(1 / 30)))


def test_no_artificial_slope_at_nodata_edge():
    z = np.full((10, 10), -9999.0)
    z[2:8, 2:8] = 100.0
    slope = terrain.compute_slope_degrees(z, 30.0, 30.0, nodata=-9999.0)
    assert np.allclose(slope[2:8, 2:8], 0.0)
    assert np.isnan(slope[0, 0])


def test_elevation_score_lower_is_more_susceptible():
    elev = np.array([[0.0, 50.0, 100.0]])
    mask = np.ones_like(elev, dtype=bool)
    score = terrain.elevation_risk_score(elev, mask)
    assert score[0, 0] > score[0, 1] > score[0, 2]
    assert np.nanmin(score) >= 0 and np.nanmax(score) <= 1


def test_slope_score_range_and_cutoff():
    slope = np.array([[0.0, 7.5, 15.0, 40.0]])
    mask = np.ones_like(slope, dtype=bool)
    score = terrain.slope_risk_score(slope, mask, max_slope_cutoff=15.0)
    assert np.allclose(score, [[1.0, 0.5, 0.0, 0.0]])


def test_scores_are_nan_outside_mask():
    elev = np.array([[1.0, 2.0]])
    mask = np.array([[True, False]])
    assert np.isnan(terrain.elevation_risk_score(elev, mask)[0, 1])


def test_empty_mask_raises():
    with pytest.raises(ValueError):
        terrain.elevation_risk_score(np.zeros((2, 2)), np.zeros((2, 2), dtype=bool))
