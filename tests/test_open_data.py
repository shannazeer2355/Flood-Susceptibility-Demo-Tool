import geopandas as gpd
import numpy as np
import pytest
import rasterio

from flood_tool import open_data, pipeline, sources


@pytest.fixture
def offline_sources(synthetic_data, monkeypatch):
    """Serve every 'remote' layer from the synthetic dataset instead of the internet."""
    dem_path = str(synthetic_data / "dem.tif")

    def osm(grid):
        g = gpd.read_file(synthetic_data / "rivers.shp").to_crs(grid.crs)
        g["kind"], g["type"] = "waterway", "river"
        return g

    def jrc(grid):
        raise ConnectionError("service down")

    def gradient(grid):
        return np.tile(np.linspace(1500, 2500, grid.width, dtype="float32"), (grid.height, 1))

    def classes(grid):
        lc = np.full(grid.shape, 10, dtype="float32")
        lc[:, : grid.width // 3] = 40
        return lc

    monkeypatch.setattr(sources, "fetch_dem", lambda grid: sources.read_to_grid(dem_path, grid))
    monkeypatch.setattr(sources, "fetch_osm_water", osm)
    monkeypatch.setattr(sources, "fetch_surface_water", jrc)
    monkeypatch.setattr(sources, "fetch_monsoon_rainfall", gradient)
    monkeypatch.setattr(sources, "fetch_landcover", classes)
    monkeypatch.setattr(sources, "fetch_clay", lambda grid: np.full(grid.shape, 30.0, dtype="float32"))
    return synthetic_data


def test_fetch_layers_records_failures(offline_sources):
    stack = open_data.fetch_layers(str(offline_sources / "boundary.shp"))
    assert stack.status["dem"][0] and stack.status["osm"][0]
    assert stack.status["jrc"][0] is False
    assert stack.mask.sum() > 0.9 * 5000 * 4000 / 30 ** 2


def test_read_to_grid_matches_source(offline_sources):
    stack = open_data.fetch_layers(str(offline_sources / "boundary.shp"), layers=())
    with rasterio.open(offline_sources / "dem.tif") as src:
        dem = src.read(1)
    elev = stack.elevation[stack.mask]
    assert elev.min() >= dem.min() - 1 and elev.max() <= dem.max() + 1
    assert elev.mean() == pytest.approx(dem.mean(), rel=0.02)


def test_open_data_pipeline(offline_sources, tmp_path):
    progress = []
    stack = open_data.fetch_layers(str(offline_sources / "boundary.shp"),
                                   progress=lambda i, n, label: progress.append((i, n)))
    assert progress[-1][0] == progress[-1][1]
    result = pipeline.score_open_data(stack, out_dir=str(tmp_path),
                                      settlements_path=str(offline_sources / "villages.shp"))
    assert set(result["weights"]) == {"elevation", "slope", "river", "drainage", "rainfall", "landcover", "soil"}
    assert sum(result["weights"].values()) == pytest.approx(1)
    assert result["consistency_ratio"] < 0.1
    assert result["area_stats"]["percent_of_aoi"].sum() == pytest.approx(100, abs=0.05)
    assert set(np.unique(result["zones"])) == {0, 1, 2, 3}
    assert result["settlement_stats"]["settlements"].sum() == 15
    for key in ("raster", "geojson", "static_map", "interactive_map", "criteria_map"):
        assert (tmp_path / result[key].split("/")[-1]).stat().st_size > 0


def test_criteria_can_be_restricted(offline_sources, tmp_path):
    stack = open_data.fetch_layers(str(offline_sources / "boundary.shp"), layers=("osm",))
    result = pipeline.score_open_data(stack, out_dir=str(tmp_path), criteria=["elevation", "slope", "river"],
                                      classification="fixed")
    assert set(result["weights"]) == {"elevation", "slope", "river"}
    assert result["breaks"] == (0.4, 0.7)


def test_slow_layer_is_dropped(offline_sources, monkeypatch):
    import time

    def slow(grid):
        time.sleep(3)
        return np.full(grid.shape, 30.0, dtype="float32")

    monkeypatch.setattr(sources, "fetch_clay", slow)
    stack = open_data.fetch_layers(str(offline_sources / "boundary.shp"), layers=("osm", "soil"), timeout_s=1)
    assert stack.status["osm"][0]
    assert stack.status["soil"] == (False, "SoilGrids 2.0 clay, 5-15 cm: no response within 1 s")
    assert stack.clay is None
