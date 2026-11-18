#!/usr/bin/env python3
"""
flood_tool.py
=============
Command-line entry point for the Flood Susceptibility Demo Tool.

Open data (every layer fetched for the AOI, AHP weights):
    python flood_tool.py --district Thiruvananthapuram --out-dir outputs
    python flood_tool.py --aoi my_area.geojson --out-dir outputs

Local files:
    python flood_tool.py --aoi boundary.shp --dem dem.tif --rivers rivers.shp \\
        [--rainfall rainfall.tif] [--settlements villages.shp] [--out-dir outputs] \\
        [--w-elevation 0.4] [--w-slope 0.3] [--w-river 0.3] [--w-rainfall 0.3]
"""
import argparse
import logging
import sys

from flood_tool import ahp
from flood_tool.overlay import DEFAULT_WEIGHTS_3, DEFAULT_WEIGHTS_4
from flood_tool.pipeline import run_open_data_pipeline, run_pipeline


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Map flood susceptibility for an area of interest. Without --dem, every layer is "
                    "fetched from open data services and the criteria are weighted with AHP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    area = p.add_mutually_exclusive_group(required=True)
    area.add_argument("--aoi", help="AOI boundary (shapefile/GeoJSON/GeoPackage)")
    area.add_argument("--district", help="Kerala district name, e.g. Thiruvananthapuram")
    p.add_argument("--settlements", default=None, help="Optional villages/settlements point layer")
    p.add_argument("--out-dir", default="outputs", help="Directory to write outputs into")

    od = p.add_argument_group("open data mode")
    od.add_argument("--criteria", nargs="+", choices=list(ahp.CRITERIA), default=None,
                    help="Criteria to use (default: all that are available)")
    od.add_argument("--resolution", type=float, default=None,
                    help="Grid resolution in metres (default: 30 m, coarser for very large AOIs)")
    od.add_argument("--classification", choices=["natural_breaks", "fixed"], default="natural_breaks",
                    help="Class boundaries: Jenks natural breaks or fixed 0.4 / 0.7")

    files = p.add_argument_group("local files mode")
    files.add_argument("--dem", help="DEM raster (GeoTIFF)")
    files.add_argument("--rivers", help="Rivers/water bodies vector layer")
    files.add_argument("--rainfall", default=None, help="Optional rainfall raster (e.g. mean monsoon rainfall)")
    files.add_argument("--w-elevation", type=float, default=None, help="Elevation weight")
    files.add_argument("--w-slope", type=float, default=None, help="Slope weight")
    files.add_argument("--w-river", type=float, default=None, help="River-distance weight")
    files.add_argument("--w-rainfall", type=float, default=None, help="Rainfall weight (needs --rainfall)")

    p.add_argument("--river-near", type=float, default=500.0, help="Distance (m) within which river score = 1")
    p.add_argument("--river-far", type=float, default=1500.0, help="Distance (m) beyond which river score = 0")
    p.add_argument("--max-slope", type=float, default=15.0, help="Slope (deg) at/above which slope score = 0")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = p.parse_args(argv)
    if bool(args.dem) != bool(args.rivers):
        p.error("Local files mode needs both --dem and --rivers.")
    if args.dem and args.district:
        p.error("--district is for open data mode; use --aoi with local files.")
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


def run(args):
    common = dict(
        settlements_path=args.settlements,
        out_dir=args.out_dir,
        river_near_m=args.river_near,
        river_far_m=args.river_far,
        max_slope_cutoff_deg=args.max_slope,
    )
    if args.dem:
        return run_pipeline(aoi_path=args.aoi, dem_path=args.dem, rivers_path=args.rivers,
                            rainfall_path=args.rainfall, weights=build_weights(args), **common)

    if args.district:
        from flood_tool.boundaries import district
        aoi = district(args.district)
    else:
        aoi = args.aoi
    return run_open_data_pipeline(aoi, resolution=args.resolution, criteria=args.criteria,
                                  classification=args.classification, **common)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        result = run(args)
    except Exception as exc:
        logging.error("Pipeline failed: %s", exc)
        sys.exit(1)

    print("\n=== Flood Susceptibility Mapping Complete ===")
    print(f"  Raster GeoTIFF     : {result['raster']}")
    print(f"  Zones GeoJSON      : {result['geojson']}")
    print(f"  Summary CSV        : {result['summary_csv']}")
    print(f"  Static map (PNG)   : {result['static_map']}")
    print(f"  Interactive map    : {result['interactive_map']}")
    if "weights" in result:
        print(f"  Criteria map (PNG) : {result['criteria_map']}")
        print("\nLayers:")
        for key, (ok, msg) in result["layer_status"].items():
            print(f"  [{'ok' if ok else '--'}] {msg}")
        print(f"\nAHP weights (CR = {result['consistency_ratio']:.3f}):")
        for key, w in result["weights"].items():
            print(f"  {ahp.CRITERIA[key]:<20} {w:.3f}")
        print("Class breaks: Medium >= {:.3f}, High > {:.3f}".format(*result["breaks"]))
    else:
        print(f"  Rainfall used      : {'yes' if result['used_rainfall'] else 'no'}")
    print("\nArea by susceptibility zone:")
    print(result["area_stats"].to_string(index=False))
    if result["settlement_stats"] is not None:
        print("\nSettlements by susceptibility zone:")
        print(result["settlement_stats"].to_string(index=False))


if __name__ == "__main__":
    main()
