#!/usr/bin/env python3
"""
flood_tool.py
=============
Command-line entry point for the Flood Susceptibility Demo Tool.

Usage:
    python flood_tool.py --aoi boundary.shp --dem dem.tif --rivers rivers.shp \\
        [--rainfall rainfall.tif] [--settlements villages.shp] [--out-dir outputs] \\
        [--w-elevation 0.4] [--w-slope 0.3] [--w-river 0.3] [--w-rainfall 0.3] \\
        [--river-near 500] [--river-far 1500] [--max-slope 15]

Example (bundled Trivandrum data, with rainfall):
    python flood_tool.py --aoi sample_data/trivandrum/boundary.shp \\
        --dem sample_data/trivandrum/dem.tif \\
        --rivers sample_data/trivandrum/rivers.shp \\
        --rainfall sample_data/trivandrum/rainfall.tif --out-dir outputs
"""
import argparse
import logging
import sys

from flood_tool.overlay import DEFAULT_WEIGHTS_3, DEFAULT_WEIGHTS_4
from flood_tool.pipeline import run_pipeline


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate a flood-susceptibility map from DEM, river, AOI and (optional) rainfall layers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--aoi", required=True, help="Path to AOI boundary (shapefile/GeoJSON)")
    p.add_argument("--dem", required=True, help="Path to DEM raster (GeoTIFF)")
    p.add_argument("--rivers", required=True, help="Path to rivers/water bodies vector layer")
    p.add_argument("--rainfall", default=None, help="Optional rainfall raster (e.g. mean monsoon rainfall)")
    p.add_argument("--settlements", default=None, help="Optional villages/settlements point layer")
    p.add_argument("--out-dir", default="outputs", help="Directory to write outputs into")

    weights = p.add_argument_group(
        "weights",
        "Criterion weights (auto-normalised to sum to 1). Defaults: 0.4/0.3/0.3 without "
        "rainfall, 0.3/0.2/0.2/0.3 with rainfall.",
    )
    weights.add_argument("--w-elevation", type=float, default=None, help="Elevation weight")
    weights.add_argument("--w-slope", type=float, default=None, help="Slope weight")
    weights.add_argument("--w-river", type=float, default=None, help="River-distance weight")
    weights.add_argument("--w-rainfall", type=float, default=None, help="Rainfall weight (needs --rainfall)")

    p.add_argument("--river-near", type=float, default=500.0, help="Distance (m) within which river score = 1")
    p.add_argument("--river-far", type=float, default=1500.0, help="Distance (m) beyond which river score = 0")
    p.add_argument("--max-slope", type=float, default=15.0, help="Slope (deg) at/above which slope score = 0")

    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = p.parse_args(argv)
    if args.w_rainfall is not None and not args.rainfall:
        p.error("--w-rainfall needs a --rainfall raster.")
    return args


def build_weights(args):
    weights = dict(DEFAULT_WEIGHTS_4 if args.rainfall else DEFAULT_WEIGHTS_3)
    overrides = {
        "elevation": args.w_elevation,
        "slope": args.w_slope,
        "river": args.w_river,
        "rainfall": args.w_rainfall,
    }
    for key, value in overrides.items():
        if value is not None and key in weights:
            weights[key] = value
    return weights


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        result = run_pipeline(
            aoi_path=args.aoi,
            dem_path=args.dem,
            rivers_path=args.rivers,
            settlements_path=args.settlements,
            rainfall_path=args.rainfall,
            out_dir=args.out_dir,
            weights=build_weights(args),
            river_near_m=args.river_near,
            river_far_m=args.river_far,
            max_slope_cutoff_deg=args.max_slope,
        )
    except Exception as exc:
        logging.error("Pipeline failed: %s", exc)
        sys.exit(1)

    print("\n=== Flood Susceptibility Mapping Complete ===")
    print(f"  Raster GeoTIFF     : {result['raster']}")
    print(f"  Zones GeoJSON      : {result['geojson']}")
    print(f"  Summary CSV        : {result['summary_csv']}")
    print(f"  Static map (PNG)   : {result['static_map']}")
    print(f"  Interactive map    : {result['interactive_map']}")
    print(f"  Rainfall used      : {'yes' if result['used_rainfall'] else 'no'}")
    print("\nArea by susceptibility zone:")
    print(result["area_stats"].to_string(index=False))
    if result["settlement_stats"] is not None:
        print("\nSettlements by susceptibility zone:")
        print(result["settlement_stats"].to_string(index=False))


if __name__ == "__main__":
    main()
