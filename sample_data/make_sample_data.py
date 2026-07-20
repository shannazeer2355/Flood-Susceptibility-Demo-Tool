"""
make_sample_data.py
--------------------
Generates a small synthetic dataset (AOI boundary, DEM, rivers,
villages) so the tool can be run and demoed without needing to
download real SRTM/OSM data first.
"""
import os

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon, LineString, Point

OUT = os.path.dirname(__file__)
CRS = "EPSG:32643"  # UTM 43N (arbitrary, works fine for a synthetic demo)

# --- AOI boundary: 5km x 4km rectangle -----------------------------------
minx, miny, maxx, maxy = 0, 0, 5000, 4000
aoi = gpd.GeoDataFrame({"name": ["Demo District"]},
                        geometry=[Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])],
                        crs=CRS)
aoi.to_file(os.path.join(OUT, "boundary.shp"))

# --- DEM: synthetic terrain, low near a "river valley" diagonal ---------
res = 30  # 30m pixels, like SRTM
width = int((maxx - minx) / res)
height = int((maxy - miny) / res)
xs = np.linspace(0, maxx, width)
ys = np.linspace(maxy, 0, height)
X, Y = np.meshgrid(xs, ys)

# Base elevation rises away from a diagonal river valley, plus gentle noise
valley_dist = np.abs((Y - (0.6 * X + 800))) / np.sqrt(1 + 0.6 ** 2)
elevation = 20 + 0.02 * valley_dist ** 1.3
elevation += np.random.default_rng(42).normal(0, 1.5, size=elevation.shape)
elevation = elevation.astype("float32")

transform = from_origin(minx, maxy, res, res)
with rasterio.open(
    os.path.join(OUT, "dem.tif"), "w", driver="GTiff",
    height=height, width=width, count=1, dtype="float32",
    crs=CRS, transform=transform, nodata=-9999,
) as dst:
    dst.write(elevation, 1)

# --- Rivers: a line following the same valley --------------------------
river_line = LineString([(x, 0.6 * x + 800) for x in np.linspace(minx, maxx, 50)])
rivers = gpd.GeoDataFrame({"name": ["Demo River"]}, geometry=[river_line], crs=CRS)
rivers.to_file(os.path.join(OUT, "rivers.shp"))

# --- Villages: scattered points, some near river, some far -------------
rng = np.random.default_rng(7)
village_pts = []
for i in range(15):
    x = rng.uniform(200, maxx - 200)
    y = rng.uniform(200, maxy - 200)
    village_pts.append(Point(x, y))
villages = gpd.GeoDataFrame({"name": [f"Village_{i+1}" for i in range(15)]},
                             geometry=village_pts, crs=CRS)
villages.to_file(os.path.join(OUT, "villages.shp"))

print("Sample data written to:", OUT)
