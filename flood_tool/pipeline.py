"""
pipeline.py
-----------
Two ways to run the analysis:

  run_pipeline            from local files (DEM, water layer, optional rainfall)
  run_open_data_pipeline  every layer fetched for the AOI from open data
                          services, weighted with AHP
"""
from __future__ import annotations

import logging
import os

import numpy as np
from rasterio.warp import Resampling

from . import ahp, io_utils, open_data, terrain, river_distance, rainfall as rainfall_mod, overlay, outputs, reporting, visualize

logger = logging.getLogger(__name__)


def run_pipeline(
    aoi_path: str,
    dem_path: str,
    rivers_path: str,
    settlements_path: str | None = None,
    rainfall_path: str | None = None,
    out_dir: str = "outputs",
    weights: dict | None = None,
    river_near_m: float = 500.0,
    river_far_m: float = 1500.0,
    max_slope_cutoff_deg: float = 15.0,
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    total_steps = 7 if rainfall_path else 6

    # --- Step 1: AOI + clip DEM -------------------------------------------------
    logger.info("Step 1/%d: Loading AOI and clipping DEM...", total_steps)
    aoi = io_utils.load_aoi(aoi_path)
    elevation, grid = io_utils.clip_and_reproject_dem(dem_path, aoi)
    mask = io_utils.valid_mask(elevation, grid.nodata)

    # --- Step 2: Elevation risk --------------------------------------------------
    logger.info("Step 2/%d: Scoring elevation...", total_steps)
    elevation_risk = terrain.elevation_risk_score(elevation, mask)

    # --- Step 3: Slope risk -------------------------------------------------------
    logger.info("Step 3/%d: Computing and scoring slope...", total_steps)
    px_x, px_y = abs(grid.transform.a), abs(grid.transform.e)
    slope_deg = terrain.compute_slope_degrees(elevation, px_x, px_y, nodata=grid.nodata)
    slope_risk = terrain.slope_risk_score(slope_deg, mask, max_slope_cutoff=max_slope_cutoff_deg)

    # --- Step 4: River distance risk ---------------------------------------------
    logger.info("Step 4/%d: Rasterizing rivers and scoring distance to water...", total_steps)
    rivers = io_utils.load_vector_on_grid(rivers_path, grid)
    river_raster = river_distance.rasterize_rivers(rivers, grid)
    dist_m = river_distance.distance_to_rivers(river_raster, pixel_size=(px_y, px_x))
    river_risk = river_distance.river_distance_risk_score(dist_m, mask, near=river_near_m, far=river_far_m)

    # --- Step 4b (optional): Rainfall risk -----------------------------------------
    rainfall_risk = None
    step = 5
    if rainfall_path:
        logger.info("Step %d/%d: Resampling and scoring rainfall...", step, total_steps)
        rainfall_arr = io_utils.resample_raster_to_grid(
            rainfall_path, grid, resampling=Resampling.bilinear, extend_edges=True
        )
        rainfall_mask = mask & (rainfall_arr != grid.nodata)

        n_gaps = int((mask & (rainfall_arr == grid.nodata)).sum())
        if n_gaps:
            logger.info(
                "%d pixel(s) inside the AOI have no rainfall data; those "
                "pixels will use the elevation/slope/river score instead.",
                n_gaps,
            )

        rainfall_risk = rainfall_mod.rainfall_risk_score(rainfall_arr, rainfall_mask)
        step += 1

    # --- Step 5/6: Weighted overlay + classification --------------------------------
    logger.info("Step %d/%d: Combining layers (weighted overlay) and classifying...", step, total_steps)
    score = overlay.weighted_overlay(elevation_risk, slope_risk, river_risk, weights=weights, rainfall_risk=rainfall_risk)
    zones = overlay.classify_risk(score)
    step += 1

    # --- Step 6/7: Write outputs and maps ---------------------------------------------
    logger.info("Step %d/%d: Writing outputs and maps...", step, total_steps)
    result = write_outputs(zones, grid, aoi, out_dir, settlements_path)
    result["used_rainfall"] = rainfall_path is not None
    return result


def write_outputs(zones, grid, aoi, out_dir: str, settlements_path: str | None = None) -> dict:
    raster_path = os.path.join(out_dir, "flood_susceptibility.tif")
    geojson_path = os.path.join(out_dir, "flood_susceptibility_zones.geojson")
    summary_path = os.path.join(out_dir, "summary.csv")
    static_map_path = os.path.join(out_dir, "flood_susceptibility_map.png")
    interactive_map_path = os.path.join(out_dir, "interactive_map.html")

    outputs.write_raster(zones, grid, raster_path)
    outputs.polygonize_zones(zones, grid, geojson_path)

    area_df = reporting.area_statistics(zones, grid)
    settlements_df = None
    if settlements_path:
        settlements_df = reporting.settlements_by_zone(settlements_path, zones, grid)
    reporting.write_summary_csv(area_df, summary_path, settlements_df)

    visualize.plot_static_map(zones, grid, static_map_path)
    visualize.build_interactive_map(raster_path, interactive_map_path, aoi=aoi)
    logger.info("Outputs written to '%s'.", out_dir)

    return {
        "raster": raster_path,
        "geojson": geojson_path,
        "summary_csv": summary_path,
        "static_map": static_map_path,
        "interactive_map": interactive_map_path,
        "area_stats": area_df,
        "settlement_stats": settlements_df,
    }


def score_open_data(stack, out_dir: str = "outputs", judgements: dict | None = None,
                    criteria=None, classification: str = "natural_breaks",
                    settlements_path: str | None = None, river_near_m: float = 500.0,
                    river_far_m: float = 1500.0, max_slope_cutoff_deg: float = 15.0,
                    drainage_radius_m: float = 1000.0, smoothing: int = 5) -> dict:
    """
    Score, weight (AHP), classify and map a fetched LayerStack. smoothing is
    the window (pixels) of the majority filter applied to the zones; 0 or 1 disables it.
    """
    os.makedirs(out_dir, exist_ok=True)
    scores = open_data.criterion_scores(stack, river_near_m, river_far_m,
                                        max_slope_cutoff_deg, drainage_radius_m)
    if criteria is not None:
        scores = {k: v for k, v in scores.items() if k in criteria}
    names = [k for k in ahp.CRITERIA if k in scores]
    weights, cr = ahp.weights_for(names, judgements)
    logger.info("AHP weights: %s (CR = %.3f)", {k: round(v, 3) for k, v in weights.items()}, cr)

    score = overlay.combine_scores(scores, weights)
    score[~stack.mask] = np.nan
    if classification == "natural_breaks":
        medium_cut, high_cut = overlay.natural_breaks(score)
    else:
        medium_cut, high_cut = 0.4, 0.7
    zones = overlay.classify_risk(score, high_cut=high_cut, medium_cut=medium_cut)
    if smoothing and smoothing > 1:
        zones = overlay.majority_filter(zones, smoothing)

    result = write_outputs(zones, stack.grid, stack.aoi, out_dir, settlements_path)
    criteria_png = os.path.join(out_dir, "criteria_scores.png")
    visualize.plot_criteria(scores, stack.grid, criteria_png, weights=weights)
    result.update({
        "criteria_map": criteria_png,
        "weights": weights,
        "consistency_ratio": cr,
        "breaks": (medium_cut, high_cut),
        "layer_status": dict(stack.status),
        "score": score,
        "zones": zones,
    })
    return result


def run_open_data_pipeline(aoi, out_dir: str = "outputs", resolution: float | None = None, **kwargs) -> dict:
    """Fetch every layer for the AOI from open data services and run the full analysis."""
    stack = open_data.fetch_layers(aoi, resolution=resolution)
    return score_open_data(stack, out_dir=out_dir, **kwargs)
