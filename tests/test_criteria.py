import geopandas as gpd
import numpy as np
import pytest
import rasterio
from shapely.geometry import LineString

from flood_tool import drainage, landcover, overlay, scoring
from flood_tool.io_utils import Grid


def test_percentile_score_direction_and_mask():
    v = np.arange(100, dtype="float64").reshape(10, 10)
    mask = np.ones_like(v, dtype=bool)
    mask[0, 0] = False
    up = scoring.percentile_score(v, mask)
    down = scoring.percentile_score(v, mask, invert=True)
    assert np.isnan(up[0, 0])
    assert up[-1, -1] == pytest.approx(1) and down[-1, -1] == pytest.approx(0)


def test_landcover_scores():
    classes = np.array([[10, 50], [80, 0]], dtype="float32")
    s = landcover.landcover_score(classes, np.ones((2, 2), bool))
    assert s[0, 0] < s[0, 1] < s[1, 0]
    assert np.isnan(s[1, 1])


def test_drainage_density_of_single_stream():
    grid = Grid(transform=rasterio.Affine(30, 0, 0, 0, -30, 6000), width=200, height=200, crs="EPSG:32643")
    line = gpd.GeoDataFrame(geometry=[LineString([(3000, 0), (3000, 6000)])], crs="EPSG:32643")
    d = drainage.drainage_density(line, grid, radius_m=1000)
    assert d[100, 100] == pytest.approx(0.5, rel=0.1)
    assert d[100, 10] == 0


def test_combine_scores_skips_missing_layers_per_pixel():
    ones = np.ones((1, 2))
    rain = np.array([[0.0, np.nan]])
    s = overlay.combine_scores({"elevation": ones, "slope": ones, "rainfall": rain},
                               {"elevation": 1, "slope": 1, "rainfall": 2})
    assert s[0, 0] == pytest.approx(0.5)
    assert s[0, 1] == pytest.approx(1.0)


def test_natural_breaks_separates_clusters():
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.normal(0.2, 0.02, 400), rng.normal(0.5, 0.02, 400), rng.normal(0.8, 0.02, 400)])
    medium, high = overlay.natural_breaks(x)
    assert 0.3 < medium < 0.45 and 0.55 < high < 0.7
