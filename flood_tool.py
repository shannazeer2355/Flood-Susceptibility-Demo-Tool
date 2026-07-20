#!/usr/bin/env python3
"""
flood_tool.py
=============
Command-line entry point for the Flood Risk Mapping Tool.

Usage:
    python flood_tool.py --aoi boundary.shp --dem dem.tif --rivers rivers.shp \\
        [--settlements villages.shp] [--out-dir outputs] \\
        [--w-elevation 0.4] [--w-slope 0.3] [--w-river 0.3] \\
        [--river-near 500] [--river-far 1500]

Example:
    python flood_tool.py --aoi sample_data/boundary.shp \\
        --dem sample_data/dem.tif --rivers sample_data/rivers.shp \\
        --settlements sample_data/villages.shp --out-dir outputs
"""
import argparse
import logging
import sys

from flood_tool.pipeline import run_pipeline


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate a Flood Risk Map from DEM, river, and AOI layers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--aoi", required=True, help="Path to AOI boundary (shapefile/GeoJSON)")
    p.add_argument("--dem", required=True, help="Path to DEM raster (GeoTIFF)")
    p.add_argument("--rivers", required=True, help="Path to rivers/water bodies vector layer")
    p.add_argument("--settlements", default=None, help="Optional villages/settlements point layer")
    p.add_argument("--out-dir", default="outputs", help="Directory to write outputs into")

    p.add_argument("--w-elevation", type=float, default=0.4, help="Weight for elevation risk")
    p.add_argument("--w-slope", type=float, default=0.3, help="Weight for slope risk")
    p.add_argument("--w-river", type=float, default=0.3, help="Weight for river-distance risk")

    p.add_argument("--river-near", type=float, default=500.0, help="Distance (m) below which river risk = High")
    p.add_argument("--river-far", type=float, default=1500.0, help="Distance (m) beyond which river risk = Low")
    p.add_argument("--max-slope", type=float, default=15.0, help="Slope (deg) at/above which slope risk = 0")

    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    weights = {"elevation": args.w_elevation, "slope": args.w_slope, "river": args.w_river}

    try:
        result = run_pipeline(
            aoi_path=args.aoi,
            dem_path=args.dem,
            rivers_path=args.rivers,
            settlements_path=args.settlements,
            out_dir=args.out_dir,
            weights=weights,
            river_near_m=args.river_near,
            river_far_m=args.river_far,
            max_slope_cutoff_deg=args.max_slope,
        )
    except Exception as exc:
        logging.error("Pipeline failed: %s", exc)
        sys.exit(1)

    print("\n=== Flood Risk Mapping Complete ===")
    print(f"  Raster GeoTIFF     : {result['raster']}")
    print(f"  Risk zones GeoJSON : {result['geojson']}")
    print(f"  Summary CSV        : {result['summary_csv']}")
    print(f"  Static map (PNG)   : {result['static_map']}")
    print(f"  Interactive map    : {result['interactive_map']}")
    print("\nArea by risk zone:")
    print(result["area_stats"].to_string(index=False))
    if result["settlement_stats"] is not None:
        print("\nSettlements affected by risk zone:")
        print(result["settlement_stats"].to_string(index=False))


if __name__ == "__main__":
    main()
