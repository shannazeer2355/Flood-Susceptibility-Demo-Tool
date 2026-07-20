"""
pipeline.py
-----------
Orchestrates the full flood-risk workflow described in the project
brief, Step 1 -> Step 7, and returns paths to all generated outputs.
"""
from __future__ import annotations

import logging
import os

from . import io_utils, terrain, river_distance, overlay, outputs, reporting, visualize

logger = logging.getLogger(__name__)


def run_pipeline(
    aoi_path: str,
    dem_path: str,
    rivers_path: str,
    settlements_path: str | None = None,
    out_dir: str = "outputs",
    weights: dict | None = None,
    river_near_m: float = 500.0,
    river_far_m: float = 1500.0,
    max_slope_cutoff_deg: float = 15.0,
) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    # --- Step 1: AOI + clip DEM -------------------------------------------------
    logger.info("Step 1/7: Loading AOI and clipping DEM...")
    aoi = io_utils.load_aoi(aoi_path)
    elevation, grid = io_utils.clip_and_reproject_dem(dem_path, aoi)
    mask = io_utils.valid_mask(elevation, grid.nodata)

    # --- Step 2: Elevation risk --------------------------------------------------
    logger.info("Step 2/7: Scoring elevation risk...")
    elevation_risk = terrain.elevation_risk_score(elevation, mask)

    # --- Step 3: Slope risk -------------------------------------------------------
    logger.info("Step 3/7: Computing slope and scoring slope risk...")
    px_x, px_y = abs(grid.transform.a), abs(grid.transform.e)
    slope_deg = terrain.compute_slope_degrees(elevation, px_x, px_y, nodata=grid.nodata)
    slope_risk = terrain.slope_risk_score(slope_deg, mask, max_slope_cutoff=max_slope_cutoff_deg)

    # --- Step 4: River distance risk ---------------------------------------------
    logger.info("Step 4/7: Rasterizing rivers and scoring distance risk...")
    rivers = io_utils.load_vector_on_grid(rivers_path, grid)
    river_raster = river_distance.rasterize_rivers(rivers, grid)
    dist_m = river_distance.distance_to_rivers(river_raster, pixel_size=(px_x + px_y) / 2)
    river_risk = river_distance.river_distance_risk_score(dist_m, mask, near=river_near_m, far=river_far_m)

    # --- Step 5: Weighted overlay + classification --------------------------------
    logger.info("Step 5/7: Combining layers (weighted overlay) and classifying...")
    score = overlay.weighted_overlay(elevation_risk, slope_risk, river_risk, weights=weights)
    zones = overlay.classify_risk(score)

    # --- Step 6: Write outputs ------------------------------------------------------
    logger.info("Step 6/7: Writing GeoTIFF / GeoJSON / summary.csv...")
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

    # --- Step 7: Visualization --------------------------------------------------
    logger.info("Step 7/7: Rendering static and interactive maps...")
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
    }
