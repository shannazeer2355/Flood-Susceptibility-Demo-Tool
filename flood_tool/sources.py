"""
sources.py
----------
On-demand access to open datasets. Nothing is downloaded in full: rasters
are read as windows from cloud-optimised GeoTIFFs over HTTP, and vector
data is requested for the AOI's bounding box only. None of the services
needs an API key.

    Copernicus DEM GLO-30       Microsoft Planetary Computer (STAC)
    ESA WorldCover 2021         Microsoft Planetary Computer (STAC)
    JRC Global Surface Water    Microsoft Planetary Computer (STAC)
    OpenStreetMap waterways     Overpass API
    SoilGrids 2.0 clay content  ISRIC (WCS)
    CHELSA V2.1 precipitation   CHELSA (1981-2010 monthly climatology)
"""
from __future__ import annotations

import logging
import math
import os
import tempfile
from functools import lru_cache

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.errors
import requests
from rasterio.io import MemoryFile
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window, from_bounds
from shapely.geometry import LineString, Polygon

logger = logging.getLogger(__name__)

USER_AGENT = "Flood-Susceptibility-Demo-Tool (https://github.com/shannazeer2355/Flood-Susceptibility-Demo-Tool)"

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
PC_TOKEN = "https://planetarycomputer.microsoft.com/api/sas/v1/token/{collection}"

OVERPASS_URLS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
)

SOILGRIDS_WCS = "https://maps.isric.org/mapserv?map=/map/{prop}.map"
SOILGRIDS_URL = "/vsicurl/https://files.isric.org/soilgrids/latest/data/{prop}/{prop}_{depth}_mean.vrt"
CHELSA_URL = (
    "/vsicurl/https://os.zhdk.cloud.switch.ch/chelsav2/GLOBAL/climatologies/1981-2010/pr/"
    "CHELSA_pr_{month:02d}_1981-2010_V.2.1.tif"
)

GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff,.vrt",
    "GDAL_HTTP_MAX_RETRY": "4",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "GDAL_HTTP_TIMEOUT": "60",
    "GDAL_HTTP_USERAGENT": USER_AGENT,
    "VSI_CACHE": "TRUE",
    "GDAL_CACHEMAX": 256,
    "GTIFF_SRS_SOURCE": "EPSG",
}


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


def wgs84_bounds(grid, pad_deg: float = 0.0) -> tuple[float, float, float, float]:
    w, s, e, n = transform_bounds(grid.crs, "EPSG:4326", *grid.bounds, densify_pts=21)
    return (w - pad_deg, s - pad_deg, e + pad_deg, n + pad_deg)


# --- Planetary Computer ------------------------------------------------------

@lru_cache(maxsize=16)
def _pc_token(collection: str) -> str:
    r = _session().get(PC_TOKEN.format(collection=collection), timeout=30)
    r.raise_for_status()
    return r.json()["token"]


def stac_hrefs(collection: str, asset: str, bbox, datetime: str | None = None) -> list[str]:
    """Signed URLs of the asset for every item of the collection that intersects bbox."""
    body = {"collections": [collection], "bbox": list(bbox), "limit": 100}
    if datetime:
        body["datetime"] = datetime
    r = _session().post(f"{PC_STAC}/search", json=body, timeout=60)
    r.raise_for_status()
    items = r.json().get("features", [])
    if not items:
        raise LookupError(f"No {collection} data found for this area.")
    token = _pc_token(collection)
    return [f"{item['assets'][asset]['href']}?{token}" for item in items]


# --- Reading remote rasters onto the analysis grid ---------------------------

def read_to_grid(hrefs, grid, resampling: Resampling = Resampling.bilinear,
                 categorical: bool = False) -> np.ndarray:
    """
    Read one or more rasters onto the grid. Only the part of each source that
    covers the grid is fetched; sources much finer than the grid are read
    from their overviews. Returns float32 with NaN where there is no data.
    """
    if isinstance(hrefs, str):
        hrefs = [hrefs]
    out = np.full(grid.shape, np.nan, dtype="float32")
    with rasterio.Env(**GDAL_ENV):
        for href in hrefs:
            part = _read_one(href, grid, resampling, categorical)
            if part is not None:
                fill = np.isnan(out) & ~np.isnan(part)
                out[fill] = part[fill]
    return out


def _read_one(href, grid, resampling, categorical):
    with rasterio.open(href) as src:
        left, bottom, right, top = transform_bounds(grid.crs, src.crs, *grid.bounds, densify_pts=21)
        pad_x, pad_y = 2 * src.res[0], 2 * src.res[1]
        window = from_bounds(left - pad_x, bottom - pad_y, right + pad_x, top + pad_y, src.transform)
        window = window.round_offsets().round_lengths().intersection(Window(0, 0, src.width, src.height))
        if window.width <= 0 or window.height <= 0:
            return None

        target_res = (right - left) / grid.width
        factor = max(1, int(target_res / src.res[0]))
        out_h = max(1, math.ceil(window.height / factor))
        out_w = max(1, math.ceil(window.width / factor))
        data = src.read(
            1, window=window, out_shape=(out_h, out_w), masked=True,
            resampling=Resampling.nearest if categorical else Resampling.average,
        )
        scale, offset = src.scales[0], src.offsets[0]
        arr = data.astype("float32").filled(np.nan) * np.float32(scale) + np.float32(offset)
        t, sx, sy = src.window_transform(window), window.width / out_w, window.height / out_h
        transform = rasterio.Affine(t.a * sx, t.b * sy, t.c, t.d * sx, t.e * sy, t.f)
        src_crs = src.crs

    dst = np.full(grid.shape, np.nan, dtype="float32")
    reproject(
        source=arr, destination=dst,
        src_transform=transform, src_crs=src_crs, src_nodata=np.nan,
        dst_transform=grid.transform, dst_crs=grid.crs, dst_nodata=np.nan,
        resampling=Resampling.nearest if categorical else resampling,
    )
    return dst


# --- Datasets ----------------------------------------------------------------

def fetch_dem(grid) -> np.ndarray:
    hrefs = stac_hrefs("cop-dem-glo-30", "data", wgs84_bounds(grid, 0.01))
    return read_to_grid(hrefs, grid, Resampling.bilinear)


def fetch_landcover(grid) -> np.ndarray:
    hrefs = stac_hrefs("esa-worldcover", "map", wgs84_bounds(grid, 0.01), datetime="2021-01-01/2021-12-31")
    return read_to_grid(hrefs, grid, categorical=True)


def fetch_surface_water(grid, min_occurrence: float = 50.0) -> np.ndarray:
    """Boolean mask of pixels that are water at least min_occurrence % of the time (1984-2021)."""
    hrefs = stac_hrefs("jrc-gsw", "occurrence", wgs84_bounds(grid, 0.01))
    occ = read_to_grid(hrefs, grid, categorical=True)
    occ[occ > 100] = np.nan
    return np.nan_to_num(occ, nan=0.0) >= min_occurrence


def _soilgrids_wcs(prop: str, depth: str, grid) -> np.ndarray:
    w, s, e, n = wgs84_bounds(grid, 0.01)
    params = [
        ("SERVICE", "WCS"), ("VERSION", "2.0.1"), ("REQUEST", "GetCoverage"),
        ("COVERAGEID", f"{prop}_{depth}_mean"), ("FORMAT", "image/tiff"),
        ("SUBSET", f"long({w},{e})"), ("SUBSET", f"lat({s},{n})"),
        ("SUBSETTINGCRS", "http://www.opengis.net/def/crs/EPSG/0/4326"),
        ("OUTPUTCRS", "http://www.opengis.net/def/crs/EPSG/0/4326"),
    ]
    r = _session().get(SOILGRIDS_WCS.format(prop=prop), params=params, timeout=60)
    r.raise_for_status()
    if not r.headers.get("content-type", "").startswith("image/tiff"):
        raise ValueError(f"SoilGrids WCS returned {r.headers.get('content-type')}: {r.text[:200]}")
    with MemoryFile(r.content) as mem, mem.open() as src:
        data = src.read(1).astype("float32")
        profile = src.profile
    data[data <= 0] = np.nan
    profile.update(dtype="float32", nodata=np.nan)
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
        path = f.name
    try:
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data, 1)
        return read_to_grid(path, grid, Resampling.bilinear)
    finally:
        os.remove(path)


def fetch_clay(grid, depth: str = "5-15cm") -> np.ndarray:
    """Clay content (%) from SoilGrids 2.0: one WCS request, or the tiled VRT if that fails."""
    try:
        clay = _soilgrids_wcs("clay", depth, grid)
    except (requests.RequestException, ValueError, rasterio.errors.RasterioError) as exc:
        logger.warning("SoilGrids WCS failed (%s); reading the tiled VRT instead", exc)
        clay = read_to_grid(SOILGRIDS_URL.format(prop="clay", depth=depth), grid, Resampling.bilinear)
    clay[clay <= 0] = np.nan
    return clay / 10.0


def fetch_monsoon_rainfall(grid, months=(6, 7, 8, 9)) -> np.ndarray:
    """Mean June-September precipitation (mm), CHELSA 1981-2010 climatology."""
    total = np.zeros(grid.shape, dtype="float32")
    for m in months:
        total += read_to_grid(CHELSA_URL.format(month=m), grid, Resampling.bilinear)
    return total


OVERPASS_QUERY = """
[out:json][timeout:75];
(
  way["waterway"~"^(river|stream|canal|drain)$"]({s},{w},{n},{e});
  way["natural"="water"]({s},{w},{n},{e});
);
out geom;
"""


def fetch_osm_water(grid) -> gpd.GeoDataFrame:
    """OpenStreetMap waterways (lines) and water bodies (polygons) in the grid's CRS."""
    w, s, e, n = wgs84_bounds(grid)
    query = OVERPASS_QUERY.format(s=s, w=w, n=n, e=e)
    last_error = None
    for url in OVERPASS_URLS:
        try:
            r = _session().post(url, data={"data": query}, timeout=80)
            r.raise_for_status()
            elements = r.json()["elements"]
            break
        except (requests.RequestException, ValueError, KeyError) as exc:
            logger.warning("Overpass request to %s failed: %s", url, exc)
            last_error = exc
    else:
        raise ConnectionError(f"All Overpass servers failed: {last_error}")

    records = []
    for el in elements:
        coords = [(p["lon"], p["lat"]) for p in el.get("geometry", [])]
        if len(coords) < 2:
            continue
        tags = el.get("tags", {})
        if "waterway" in tags:
            records.append({"kind": "waterway", "type": tags["waterway"], "geometry": LineString(coords)})
        elif len(coords) >= 4 and coords[0] == coords[-1]:
            records.append({"kind": "water", "type": tags.get("water", "water"), "geometry": Polygon(coords)})
    if not records:
        return gpd.GeoDataFrame({"kind": [], "type": []}, geometry=[], crs=grid.crs)
    gdf = gpd.GeoDataFrame(records, crs=4326)
    logger.info("OpenStreetMap: %d waterways, %d water bodies",
                int((gdf["kind"] == "waterway").sum()), int((gdf["kind"] == "water").sum()))
    return gdf.to_crs(grid.crs)
