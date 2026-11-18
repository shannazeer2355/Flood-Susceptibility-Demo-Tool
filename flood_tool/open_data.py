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


def fetch_layers(aoi, resolution: float | None = None, layers=OPTIONAL_LAYERS, progress=None) -> LayerStack:
    """
    aoi: path or GeoDataFrame. layers: optional layers to fetch besides the DEM.
    progress: optional callback(done, total, label).
    """
    aoi = io_utils.load_aoi(aoi)
    grid = io_utils.grid_from_aoi(aoi, resolution)
    mask = io_utils.aoi_mask(aoi, grid)
    logger.info("Analysis grid: %d x %d px at %.0f m", grid.width, grid.height, grid.resolution)

    steps = ["dem", *[k for k in OPTIONAL_LAYERS if k in layers]]
    status = {}

    def report(i, key):
        if progress:
            progress(i, len(steps), SOURCES[key])

    report(0, "dem")
    dem = sources.fetch_dem(grid)
    mask &= ~np.isnan(dem)
    if not mask.any():
        raise ValueError("No elevation data inside the AOI.")
    elevation = np.where(mask, dem, grid.nodata).astype("float32")
    status["dem"] = (True, SOURCES["dem"])

    stack = LayerStack(aoi=aoi, grid=grid, mask=mask, elevation=elevation, status=status)
    fetchers = {
        "osm": lambda: sources.fetch_osm_water(grid),
        "jrc": lambda: sources.fetch_surface_water(grid),
        "rainfall": lambda: sources.fetch_monsoon_rainfall(grid),
        "landcover": lambda: sources.fetch_landcover(grid),
        "soil": lambda: sources.fetch_clay(grid),
    }
    for i, key in enumerate(steps[1:], start=1):
        report(i, key)
        try:
            data = fetchers[key]()
        except Exception as exc:
            logger.warning("%s unavailable: %s", SOURCES[key], exc)
            status[key] = (False, f"{SOURCES[key]}: {exc}")
            continue
        status[key] = (True, SOURCES[key])
        if key == "osm":
            stack.waterways = data
        elif key == "jrc":
            stack.water = data
        elif key == "rainfall":
            stack.rainfall = data
        elif key == "landcover":
            stack.landcover = data
        elif key == "soil":
            stack.clay = data
    if progress:
        progress(len(steps), len(steps), "done")
    return stack


def water_raster(stack: LayerStack) -> np.ndarray | None:
    parts = []
    if stack.waterways is not None and len(stack.waterways):
        shapes = [(g, 1) for g in stack.waterways.geometry if g is not None and not g.is_empty]
        if shapes:
            parts.append(river_distance.rasterize_rivers(stack.waterways, stack.grid) > 0)
    if stack.water is not None:
        parts.append(stack.water)
    if not parts:
        return None
    return np.logical_or.reduce(parts)


def criterion_scores(stack: LayerStack, river_near_m: float = 500.0, river_far_m: float = 1500.0,
                     max_slope_cutoff_deg: float = 15.0, drainage_radius_m: float = 1000.0) -> dict:
    grid, mask = stack.grid, stack.mask
    scores = {
        "elevation": terrain.elevation_risk_score(stack.elevation, mask).astype("float32"),
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
