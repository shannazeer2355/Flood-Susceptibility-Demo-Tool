import geopandas as gpd
import numpy as np
from rasterio.transform import from_origin
from shapely.geometry import Point

from flood_tool import reporting
from flood_tool.io_utils import Grid

CRS = "EPSG:32643"


def _grid():
    return Grid(transform=from_origin(1000, 2000, 10, 10), width=4, height=3, crs=CRS)


def test_area_statistics():
    zones = np.array([[1, 1, 2, 3], [1, 2, 2, 3], [0, 0, 3, 3]], dtype="uint8")
    df = reporting.area_statistics(zones, _grid()).set_index("susceptibility_zone")
    assert df.loc["Low", "pixel_count"] == 3
    assert df.loc["High", "area_km2"] == round(4 * 100 / 1e6, 3)
    assert df["percent_of_aoi"].sum() == 100


def test_point_just_outside_grid_is_not_counted_inside(tmp_path):
    zones = np.full((3, 4), 3, dtype="uint8")
    pts = gpd.GeoDataFrame(
        geometry=[Point(1005, 1995),
                  Point(996, 1995)],
        crs=CRS,
    )
    path = tmp_path / "pts.geojson"
    pts.to_file(path)
    df = reporting.settlements_by_zone(str(path), zones, _grid()).set_index("susceptibility_zone")
    assert df.loc["High", "settlements"] == 1
    assert df.loc["Outside AOI / NoData", "settlements"] == 1
