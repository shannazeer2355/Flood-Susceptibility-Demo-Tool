"""
open_data.py
------------
Assemble every input layer for an AOI from open data services, then turn
the layers into 0-1 criterion scores. Fetching is the slow part, so it is
kept separate from scoring: the same LayerStack can be re-scored with new
weights or parameters without downloading anything again.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np

from . import drainage, io_utils, landcover, river_distance, scoring, sources, terrain

logger = logging.getLogger(__name__)

SOURCES = {
    "dem": "Copernicus DEM GLO-30",
    "osm": "OpenStreetMap waterways (Overpass API)",
    "jrc": "JRC Global Surface Water 1984-2021",
    "rainfall": "CHELSA V2.1 Jun-Sep precipitation, 1981-2010",
    "landcover": "ESA WorldCover 2021",
    "soil": "SoilGrids 2.0 clay, 5-15 cm",
}

OPTIONAL_LAYERS = ("osm", "jrc", "rainfall", "landcover", "soil")

MAJOR_WATERWAYS = ("river", "canal")
MIN_WATER_BODY_M2 = 50_000


@dataclass
class LayerStack:
    aoi: gpd.GeoDataFrame
    grid: io_utils.Grid
    mask: np.ndarray
    elevation: np.ndarray
    water: np.ndarray | None = None
    waterways: gpd.GeoDataFrame | None = None
    rainfall: np.ndarray | None = None
    landcover: np.ndarray | None = None
    clay: np.ndarray | None = None
    status: dict = field(default_factory=dict)


def fetch_layers(aoi, resolution: float | None = None, layers=OPTIONAL_LAYERS, progress=None,
                 timeout_s: float = 120.0) -> LayerStack:
    """
    aoi: path or GeoDataFrame. layers: optional layers to fetch besides the DEM.
    The optional layers are fetched in parallel; any that has not arrived
    within timeout_s is left out. progress: optional callback(done, total, label),
    called as each layer arrives.
    """
    aoi = io_utils.load_aoi(aoi)
    grid = io_utils.grid_from_aoi(aoi, resolution)
    mask = io_utils.aoi_mask(aoi, grid)
    logger.info("Analysis grid: %d x %d px at %.0f m", grid.width, grid.height, grid.resolution)

    optional = [k for k in OPTIONAL_LAYERS if k in layers]
    total = 1 + len(optional)
    status = {}

    dem = sources.fetch_dem(grid)
    mask &= ~np.isnan(dem)
    if not mask.any():
        raise ValueError("No elevation data inside the AOI.")
    elevation = np.where(mask, dem, grid.nodata).astype("float32")
    status["dem"] = (True, SOURCES["dem"])
    if progress:
        progress(1, total, SOURCES["dem"])

    stack = LayerStack(aoi=aoi, grid=grid, mask=mask, elevation=elevation, status=status)
    fetchers = {
        "osm": lambda: sources.fetch_osm_water(grid),
        "jrc": lambda: sources.fetch_surface_water(grid),
        "rainfall": lambda: sources.fetch_monsoon_rainfall(grid),
        "landcover": lambda: sources.fetch_landcover(grid),
        "soil": lambda: sources.fetch_clay(grid),
    }
    attrs = {"osm": "waterways", "jrc": "water", "rainfall": "rainfall", "landcover": "landcover", "soil": "clay"}

    if optional:
        pool = ThreadPoolExecutor(max_workers=len(optional))
        futures = {pool.submit(fetchers[k]): k for k in optional}
        pending, done_count = set(futures), 1
        deadline = time.monotonic() + timeout_s
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            done, pending = wait(pending, timeout=remaining, return_when=FIRST_COMPLETED)
            for f in done:
                key = futures[f]
                done_count += 1
                try:
                    setattr(stack, attrs[key], f.result())
                    status[key] = (True, SOURCES[key])
                except Exception as exc:
                    logger.warning("%s unavailable: %s", SOURCES[key], exc)
                    status[key] = (False, f"{SOURCES[key]}: {exc}")
                if progress:
                    progress(done_count, total, SOURCES[key])
        for f in pending:
            key = futures[f]
            logger.warning("%s did not respond within %.0f s", SOURCES[key], timeout_s)
            status[key] = (False, f"{SOURCES[key]}: no response within {timeout_s:.0f} s")
        pool.shutdown(wait=False, cancel_futures=True)
        if pending and progress:
            progress(total, total, "done")
    stack.status = {k: status[k] for k in ["dem", *optional]}
    return stack


def major_water(waterways: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Rivers, canals and water bodies of at least 5 ha; small streams, drains and ponds are left out."""
    lines = (waterways["kind"] == "waterway") & waterways["type"].isin(MAJOR_WATERWAYS)
    bodies = (waterways["kind"] == "water") & (waterways.geometry.area >= MIN_WATER_BODY_M2)
    return waterways[lines | bodies]


def water_raster(stack: LayerStack) -> np.ndarray | None:
    parts = []
    if stack.waterways is not None and len(stack.waterways):
        major = major_water(stack.waterways)
        if len(major):
            parts.append(river_distance.rasterize_rivers(major, stack.grid) > 0)
    if stack.water is not None:
        parts.append(stack.water)
    if not parts:
        return None
    return np.logical_or.reduce(parts)


def criterion_scores(stack: LayerStack, river_near_m: float = 500.0, river_far_m: float = 1500.0,
                     max_slope_cutoff_deg: float = 15.0, drainage_radius_m: float = 1000.0) -> dict:
    grid, mask = stack.grid, stack.mask
    scores = {
        "elevation": terrain.elevation_risk_score(stack.elevation, mask, log_scale=True).astype("float32"),
    }
    slope = terrain.compute_slope_degrees(stack.elevation, grid.resolution, grid.resolution, nodata=grid.nodata)
    scores["slope"] = terrain.slope_risk_score(slope, mask, max_slope_cutoff=max_slope_cutoff_deg).astype("float32")

    water = water_raster(stack)
    if water is not None and water.any():
        dist = river_distance.distance_to_rivers(water.astype("uint8"), pixel_size=grid.resolution)
        scores["river"] = river_distance.river_distance_risk_score(
            dist, mask, near=river_near_m, far=river_far_m).astype("float32")

    if stack.waterways is not None:
        lines = stack.waterways[stack.waterways["kind"] == "waterway"]
        if len(lines):
            density = drainage.drainage_density(lines, grid, radius_m=drainage_radius_m)
            scores["drainage"] = scoring.percentile_score(density, mask)

    if stack.rainfall is not None and (mask & ~np.isnan(stack.rainfall)).any():
        scores["rainfall"] = scoring.percentile_score(stack.rainfall, mask)
    if stack.landcover is not None:
        scores["landcover"] = landcover.landcover_score(stack.landcover, mask)
    if stack.clay is not None and (mask & ~np.isnan(stack.clay)).any():
        scores["soil"] = scoring.percentile_score(stack.clay, mask)
    return scores
