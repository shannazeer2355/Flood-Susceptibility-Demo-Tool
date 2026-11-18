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

    @property
    def bounds(self):
        left, top = self.transform.c, self.transform.f
        return (left, top + self.transform.e * self.height, left + self.transform.a * self.width, top)

    @property
    def resolution(self):
        return abs(self.transform.a)


def grid_from_aoi(aoi: gpd.GeoDataFrame, resolution: float | None = None,
                  max_pixels: int = 5_000_000) -> Grid:
    """Grid covering the AOI bounds, snapped to the resolution."""
    minx, miny, maxx, maxy = aoi.total_bounds
    if resolution is None:
        area = (maxx - minx) * (maxy - miny)
        resolution = max(30.0, float(np.ceil(np.sqrt(area / max_pixels))))
    left = np.floor(minx / resolution) * resolution
    top = np.ceil(maxy / resolution) * resolution
    width = int(np.ceil((maxx - left) / resolution))
    height = int(np.ceil((top - miny) / resolution))
    transform = rasterio.Affine(resolution, 0, left, 0, -resolution, top)
    return Grid(transform=transform, width=width, height=height, crs=aoi.crs)


def aoi_mask(aoi: gpd.GeoDataFrame, grid: Grid) -> np.ndarray:
    from rasterio.features import rasterize

    shapes = [(g, 1) for g in aoi.geometry if g is not None and not g.is_empty]
    return rasterize(shapes, out_shape=grid.shape, transform=grid.transform, fill=0, dtype="uint8") > 0


def suggest_utm_crs(gdf: gpd.GeoDataFrame) -> str:
    """Pick a metre-based UTM CRS appropriate for the AOI centroid."""
    wgs = gdf.to_crs(4326)
    minx, miny, maxx, maxy = wgs.total_bounds
    lon = (minx + maxx) / 2
    lat = (miny + maxy) / 2
    zone = int((lon + 180) // 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return f"EPSG:{epsg}"


def load_aoi(aoi_path, target_crs: str | None = None) -> gpd.GeoDataFrame:
    """Load the AOI boundary (path or GeoDataFrame) and reproject to a metre-based CRS."""
    aoi = aoi_path.copy() if isinstance(aoi_path, gpd.GeoDataFrame) else gpd.read_file(aoi_path)
    if aoi.empty:
        raise ValueError("The AOI contains no features.")
    if aoi.crs is None:
        aoi = aoi.set_crs(4326)
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

    Memory efficiency: rather than reprojecting the DEM's full extent
    and only clipping afterwards (which can allocate an enormous
    intermediate array if the source DEM tile covers a much larger area
    than the AOI -- a common situation with downloaded SRTM/ASTER tiles),
    this first reads only a small window of the source DEM that overlaps
    the AOI (with a padding buffer for safe resampling at the edges),
    and reprojects only that window. This keeps memory usage tied to the
    AOI's own size, not the source file's total extent.
    """
    from rasterio.warp import transform_bounds as warp_transform_bounds
    from rasterio.windows import from_bounds as window_from_bounds

    with rasterio.open(dem_path) as src:
        dst_crs = aoi.crs

        # Reproject the AOI's bounding box into the source DEM's own CRS,
        # then figure out which window of source pixels overlaps it.
        aoi_bounds_src_crs = warp_transform_bounds(dst_crs, src.crs, *aoi.total_bounds)
        minx, miny, maxx, maxy = aoi_bounds_src_crs

        # Pad by ~5% of the AOI's extent (minimum a few pixels) so
        # resampling/interpolation near the edges has enough surrounding
        # source data, without pulling in the whole tile.
        pad_x = max((maxx - minx) * 0.05, src.res[0] * 4)
        pad_y = max((maxy - miny) * 0.05, src.res[1] * 4)
        window = window_from_bounds(
            minx - pad_x, miny - pad_y, maxx + pad_x, maxy + pad_y,
            transform=src.transform,
        ).round_offsets().round_lengths()
        # Clamp the window to the raster's actual extent
        window = window.intersection(
            rasterio.windows.Window(0, 0, src.width, src.height)
        )
        if window.width <= 0 or window.height <= 0:
            raise ValueError(
                "The AOI does not overlap the supplied DEM at all. "
                "Check that both files cover the same geographic area."
            )

        src_window_data = src.read(1, window=window)
        src_window_transform = src.window_transform(window)

        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, window.width, window.height,
            *rasterio.windows.bounds(window, src.transform),
            resolution=resolution,
        )
        dst_array = np.full((height, width), -9999.0, dtype="float32")
        reproject(
            source=src_window_data,
            destination=dst_array,
            src_transform=src_window_transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            src_nodata=src.nodata,
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
    logger.info("DEM windowed-read (%d x %d px from source) and clipped to AOI: %d x %d px, resolution=%.1fm",
                window.width, window.height, grid.width, grid.height, abs(grid.transform.a))
    return elevation, grid


def resample_raster_to_grid(raster_path: str, grid: Grid, resampling: Resampling = Resampling.bilinear,
                            extend_edges: bool = False) -> np.ndarray:
    """
    Resample a single-band raster (e.g. rainfall) onto the reference grid.
    extend_edges: fill one ring of missing cells around valid data with the
    nearest value before resampling.
    """
    with rasterio.open(raster_path) as src:
        src_data = src.read(1).astype("float64")
        src_transform = src.transform
        src_crs = src.crs
        if src.nodata is not None:
            src_data[src_data == src.nodata] = np.nan

    if extend_edges:
        src_data = np.pad(src_data, 1, mode="constant", constant_values=np.nan)
        src_transform = src_transform * rasterio.Affine.translation(-1, -1)
        missing = np.isnan(src_data)
        if missing.any() and not missing.all():
            from scipy.ndimage import distance_transform_edt
            dist, idx = distance_transform_edt(missing, return_indices=True)
            ring = missing & (dist < 1.5)
            src_data[ring] = src_data[tuple(i[ring] for i in idx)]

    dst_array = np.full(grid.shape, np.nan, dtype="float64")
    reproject(
        source=src_data,
        destination=dst_array,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=np.nan,
        dst_transform=grid.transform,
        dst_crs=grid.crs,
        dst_nodata=np.nan,
        resampling=resampling,
    )
    dst_array[np.isnan(dst_array)] = grid.nodata
    return dst_array.astype("float32")


def load_vector_on_grid(path: str, grid: Grid) -> gpd.GeoDataFrame:
    """Load a vector layer (rivers, roads, settlements) reprojected to the grid CRS."""
    gdf = gpd.read_file(path)
    gdf = gdf.to_crs(grid.crs)
    return gdf


def valid_mask(elevation: np.ndarray, nodata: float) -> np.ndarray:
    """Boolean mask of valid (in-AOI, non-nodata) pixels."""
    return (elevation != nodata) & ~np.isnan(elevation)
