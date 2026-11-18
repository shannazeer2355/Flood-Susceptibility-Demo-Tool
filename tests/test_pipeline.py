import geopandas as gpd
import pandas as pd
import pytest
import rasterio

from flood_tool.pipeline import run_pipeline


@pytest.fixture(scope="module")
def result(synthetic_data, tmp_path_factory):
    out = tmp_path_factory.mktemp("outputs")
    return run_pipeline(
        aoi_path=str(synthetic_data / "boundary.shp"),
        dem_path=str(synthetic_data / "dem.tif"),
        rivers_path=str(synthetic_data / "rivers.shp"),
        settlements_path=str(synthetic_data / "villages.shp"),
        out_dir=str(out),
    )


def test_all_outputs_written(result):
    for key in ("raster", "geojson", "summary_csv", "static_map", "interactive_map"):
        with open(result[key], "rb") as f:
            assert f.read(1), f"{key} is empty"


def test_raster_contains_only_valid_zones(result):
    with rasterio.open(result["raster"]) as src:
        data = src.read(1)
        assert src.crs is not None
    assert set(pd.unique(data.ravel())) <= {0, 1, 2, 3}
    assert (data > 0).any()


def test_areas_add_up(result):
    area = result["area_stats"]
    assert area["percent_of_aoi"].sum() == pytest.approx(100, abs=0.05)
    assert area["area_km2"].sum() == pytest.approx(20, rel=0.05)


def test_zone_polygons_are_labelled(result):
    zones = gpd.read_file(result["geojson"])
    assert set(zones["susceptibility_zone"]) <= {"Low", "Medium", "High"}
    high = zones.loc[zones["susceptibility_zone"] == "High", "area_km2"].sum()
    assert high > 0


def test_all_settlements_counted(result):
    s = result["settlement_stats"]
    assert s["settlements"].sum() == 15
