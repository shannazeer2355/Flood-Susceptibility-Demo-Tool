"""
pipeline.py
-----------
Orchestrates the full flood-susceptibility workflow. Rainfall is an
optional 4th criterion — omit rainfall_path to get the original
3-criteria (elevation/slope/river) behaviour, exactly as before.
"""
from __future__ import annotations

import logging
import os

from . import io_utils, terrain, river_distance, rainfall as rainfall_mod, overlay, outputs, reporting, visualize

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
    total_steps = 8 if rainfall_path else 7

    # --- Step 1: AOI + clip DEM -------------------------------------------------
    logger.info("Step 1/%d: Loading AOI and clipping DEM...", total_steps)
    aoi = io_utils.load_aoi(aoi_path)
    elevation, grid = io_utils.clip_and_reproject_dem(dem_path, aoi)
    mask = io_utils.valid_mask(elevation, grid.nodata)

    # --- Step 2: Elevation risk --------------------------------------------------
    logger.info("Step 2/%d: Scoring elevation risk...", total_steps)
    elevation_risk = terrain.elevation_risk_score(elevation, mask)

    # --- Step 3: Slope risk -------------------------------------------------------
    logger.info("Step 3/%d: Computing slope and scoring slope risk...", total_steps)
    px_x, px_y = abs(grid.transform.a), abs(grid.transform.e)
    slope_deg = terrain.compute_slope_degrees(elevation, px_x, px_y, nodata=grid.nodata)
    slope_risk = terrain.slope_risk_score(slope_deg, mask, max_slope_cutoff=max_slope_cutoff_deg)

    # --- Step 4: River distance risk ---------------------------------------------
    logger.info("Step 4/%d: Rasterizing rivers and scoring distance risk...", total_steps)
    rivers = io_utils.load_vector_on_grid(rivers_path, grid)
    river_raster = river_distance.rasterize_rivers(rivers, grid)
    dist_m = river_distance.distance_to_rivers(river_raster, pixel_size=(px_x + px_y) / 2)
    river_risk = river_distance.river_distance_risk_score(dist_m, mask, near=river_near_m, far=river_far_m)

    # --- Step 4b (optional): Rainfall risk -----------------------------------------
    rainfall_risk = None
    step = 5
    if rainfall_path:
        logger.info("Step %d/%d: Resampling rainfall and scoring rainfall risk...", step, total_steps)
        # 'nearest' instead of the default 'bilinear': bilinear requires all
        # 4 surrounding source pixels to be valid, so a single nodata
        # neighbour near the source raster's coverage edge produces a
        # nodata destination pixel even when nearby real data exists.
        # 'nearest' uses only the closest source pixel, recovering those
        # edge cells without inventing any new information.
        from rasterio.warp import Resampling
        rainfall_arr = io_utils.resample_raster_to_grid(rainfall_path, grid, resampling=Resampling.nearest)
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

    # --- Step 6/7: Write outputs ------------------------------------------------------
    logger.info("Step %d/%d: Writing GeoTIFF / GeoJSON / summary.csv...", step, total_steps)
    raster_path = os.path.join(out_dir, "flood_risk.tif")
    geojson_path = os.path.join(out_dir, "flood_risk_zones.geojson")
    summary_path = os.path.join(out_dir, "summary.csv")

    outputs.write_raster(zones, grid, raster_path)
    outputs.polygonize_zones(zones, grid, geojson_path)

    area_df = reporting.area_statistics(zones, grid)
    settlements_df = None
    if settlements_path:
        settlements_df = reporting.settlements_by_zone(settlements_path, zones, grid)
    reporting.write_summary_csv(area_df, summary_path, settlements_df)
    step += 1

    # --- Step 7/8: Visualization --------------------------------------------------
    logger.info("Step %d/%d: Rendering static and interactive maps...", step, total_steps)
    static_map_path = os.path.join(out_dir, "flood_risk_map.png")
    interactive_map_path = os.path.join(out_dir, "interactive_map.html")

    visualize.plot_static_map(zones, grid, static_map_path)
    visualize.build_interactive_map(raster_path, interactive_map_path, aoi=aoi)

    logger.info("Pipeline complete. Outputs written to '%s'.", out_dir)

    return {
        "raster": raster_path,
        "geojson": geojson_path,
        "summary_csv": summary_path,
        "static_map": static_map_path,
        "interactive_map": interactive_map_path,
        "area_stats": area_df,
        "settlement_stats": settlements_df,
        "used_rainfall": rainfall_path is not None,
    }
