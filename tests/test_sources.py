from unittest import mock

import numpy as np
import pytest
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds

from flood_tool import sources
from flood_tool.io_utils import Grid


def _grid():
    return Grid(transform=rasterio.Affine(30, 0, 700000, 0, -30, 960000), width=100, height=100, crs="EPSG:32643")


def _tiff_bytes(value):
    data = np.full((60, 60), value, dtype="int16")
    data[:, 26:] = 0
    profile = dict(driver="GTiff", width=60, height=60, count=1, dtype="int16", crs="EPSG:4326",
                   transform=from_bounds(76.7, 8.4, 77.0, 8.8, 60, 60), nodata=-32768)
    with MemoryFile() as mem:
        with mem.open(**profile) as dst:
            dst.write(data, 1)
        return mem.read()


def test_clay_from_wcs_in_percent():
    response = mock.Mock(content=_tiff_bytes(350), headers={"content-type": "image/tiff"})
    response.raise_for_status.return_value = None
    with mock.patch.object(sources.requests.Session, "get", return_value=response) as get:
        clay = sources.fetch_clay(_grid())
    assert np.nanmin(clay) == pytest.approx(35.0)
    assert np.isnan(clay).any()
    params = dict(get.call_args.kwargs["params"])
    assert params["COVERAGEID"] == "clay_5-15cm_mean"


def test_clay_falls_back_to_vrt_when_wcs_fails():
    with mock.patch.object(sources.requests.Session, "get", side_effect=sources.requests.ConnectionError("down")), \
         mock.patch.object(sources, "read_to_grid", return_value=np.full((100, 100), 250.0, dtype="float32")) as vrt:
        clay = sources.fetch_clay(_grid())
    assert "clay_5-15cm_mean.vrt" in vrt.call_args.args[0]
    assert np.nanmean(clay) == pytest.approx(25.0)
