"""
io_utils.py
-----------
Helpers for loading, reprojecting, and clipping raster (DEM) and
vector (AOI boundary, rivers, settlements) data to a common grid/CRS.

All downstream modules assume:
  * A single working CRS (a projected, metre-based CRS — we auto-pick
    a suitable UTM zone from the AOI if the user doesn't supply one).
  * The DEM defines the reference grid (transform, resolution, shape)
    that every other raster layer is resampled/rasterized onto.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling

logger = logging.getLogger(__name__)


@dataclass
class Grid:
    """Reference raster grid that all layers are aligned to."""
    transform: rasterio.Affine
    width: int
    height: int
    crs: object
    nodata: float = -9999.0

    @property
    def shape(self):
        return (self.height, self.width)


def suggest_utm_crs(gdf: gpd.GeoDataFrame) -> str:
    """Pick a metre-based UTM CRS appropriate for the AOI centroid."""
    wgs = gdf.to_crs(4326)
    minx, miny, maxx, maxy = wgs.total_bounds
    lon = (minx + maxx) / 2
    lat = (miny + maxy) / 2
    zone = int((lon + 180) // 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return f"EPSG:{epsg}"


def load_aoi(aoi_path: str, target_crs: str | None = None) -> gpd.GeoDataFrame:
    """Load the AOI boundary and reproject to a metre-based CRS."""
    aoi = gpd.read_file(aoi_path)
    if aoi.empty:
        raise ValueError(f"AOI file '{aoi_path}' contains no features.")
    if target_crs is None:
        target_crs = suggest_utm_crs(aoi)
    aoi = aoi.to_crs(target_crs)
    logger.info("AOI loaded (%d feature(s)), working CRS = %s", len(aoi), target_crs)
    return aoi


def clip_and_reproject_dem(dem_path: str, aoi: gpd.GeoDataFrame, resolution: float | None = None):
    """
    Reproject a DEM to the AOI's CRS, clip it to the AOI geometry, and
    return (elevation_array, Grid). This DEM grid becomes the reference
    grid for every other layer in the pipeline.
    """
    with rasterio.open(dem_path) as src:
        dst_crs = aoi.crs
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds,
            resolution=resolution,
        )
        dst_array = np.full((height, width), -9999.0, dtype="float32")
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            dst_nodata=-9999.0,
            resampling=Resampling.bilinear,
        )

    # Clip to AOI polygon using an in-memory dataset
    from rasterio.io import MemoryFile

    profile = {
        "driver": "GTiff", "height": height, "width": width, "count": 1,
        "dtype": "float32", "crs": dst_crs, "transform": transform, "nodata": -9999.0,
    }
    with MemoryFile() as memfile:
        with memfile.open(**profile) as tmp:
            tmp.write(dst_array, 1)
            geoms = [g.__geo_interface__ for g in aoi.geometry]
            clipped, clip_transform = rio_mask(tmp, geoms, crop=True, nodata=-9999.0)

    elevation = clipped[0]
    grid = Grid(
        transform=clip_transform,
        width=elevation.shape[1],
        height=elevation.shape[0],
        crs=dst_crs,
        nodata=-9999.0,
    )
    logger.info("DEM clipped to AOI: %d x %d px, resolution=%.1fm",
                grid.width, grid.height, abs(grid.transform.a))
    return elevation, grid


def load_vector_on_grid(path: str, grid: Grid) -> gpd.GeoDataFrame:
    """Load a vector layer (rivers, roads, settlements) reprojected to the grid CRS."""
    gdf = gpd.read_file(path)
    gdf = gdf.to_crs(grid.crs)
    return gdf


def valid_mask(elevation: np.ndarray, nodata: float) -> np.ndarray:
    """Boolean mask of valid (in-AOI, non-nodata) pixels."""
    return (elevation != nodata) & ~np.isnan(elevation)
