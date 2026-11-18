"""
Fetch every open-data layer for a Kerala district (or any boundary file),
report timing and basic statistics per layer, and run the full analysis.

    python scripts/check_sources.py --district Thiruvananthapuram --out-dir live_check
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flood_tool import boundaries, open_data, pipeline  # noqa: E402


def stats(name, arr, mask):
    if arr is None:
        return f"{name:<10} not available"
    v = arr[mask & ~np.isnan(arr)] if arr.dtype.kind == "f" else arr[mask]
    if v.size == 0:
        return f"{name:<10} no data inside the AOI"
    cover = 100 * v.size / mask.sum()
    return (f"{name:<10} min {np.min(v):9.2f}  mean {np.mean(v):9.2f}  max {np.max(v):9.2f}  "
            f"coverage {cover:5.1f}%")


def main():
    p = argparse.ArgumentParser()
    area = p.add_mutually_exclusive_group(required=True)
    area.add_argument("--district")
    area.add_argument("--aoi")
    p.add_argument("--resolution", type=float, default=None)
    p.add_argument("--out-dir", default="live_check")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    aoi = boundaries.district(args.district) if args.district else args.aoi
    times, last = {}, [time.time()]

    def progress(done, total, label):
        now = time.time()
        if done:
            times[prev[0]] = now - last[0]
        last[0], prev[0] = now, label

    prev = [None]
    t0 = time.time()
    stack = open_data.fetch_layers(aoi, resolution=args.resolution, progress=progress)
    print(f"\nGrid {stack.grid.width} x {stack.grid.height} at {stack.grid.resolution:.0f} m, "
          f"AOI {stack.mask.sum() * stack.grid.resolution ** 2 / 1e6:.0f} km², fetched in {time.time() - t0:.0f} s")
    for label, sec in times.items():
        print(f"  {sec:6.1f} s  {label}")
    print("\nLayer status:")
    for key, (ok, msg) in stack.status.items():
        print(f"  [{'ok' if ok else 'FAILED'}] {msg}")

    m = stack.mask
    elev = np.where(m, stack.elevation, np.nan)
    print("\nLayer statistics inside the AOI:")
    print(" ", stats("elevation", elev, m))
    print(" ", stats("rainfall", stack.rainfall, m))
    print(" ", stats("clay %", stack.clay, m))
    if stack.water is not None:
        print(f"  {'water':<10} {100 * stack.water[m].mean():.2f}% of AOI (JRC occurrence >= 50%)")
    if stack.waterways is not None:
        counts = stack.waterways["type"].value_counts().to_dict()
        print(f"  {'osm':<10} {counts}")
    if stack.landcover is not None:
        classes, n = np.unique(stack.landcover[m & ~np.isnan(stack.landcover)], return_counts=True)
        share = {int(c): round(100 * k / n.sum(), 1) for c, k in zip(classes, n)}
        print(f"  {'landcover':<10} % by class {share}")

    result = pipeline.score_open_data(stack, out_dir=args.out_dir)
    print(f"\nAHP weights (CR {result['consistency_ratio']:.3f}):",
          {k: round(v, 3) for k, v in result["weights"].items()})
    print("Breaks:", tuple(round(b, 3) for b in result["breaks"]))
    print(result["area_stats"].to_string(index=False))
    failed = [k for k, (ok, _) in stack.status.items() if not ok]
    if failed:
        print(f"\nUnavailable layers: {', '.join(failed)}")
        sys.exit(2)


if __name__ == "__main__":
    main()
