"""
reporting.py
------------
Summary statistics for the flood risk classification:
  * % / km2 of AOI area in each risk zone
  * number of settlement/village points falling in each zone (if given)
  * writes summary.csv
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from .overlay import ZONE_LABELS


def area_statistics(zones: np.ndarray, grid) -> pd.DataFrame:
    px_area_km2 = abs(grid.transform.a * grid.transform.e) / 1_000_000
    total_valid = int((zones > 0).sum())

    rows = []
    for zone_id in (1, 2, 3):
        count = int((zones == zone_id).sum())
        pct = 100 * count / total_valid if total_valid else 0
        rows.append({
            "risk_zone": ZONE_LABELS[zone_id],
            "pixel_count": count,
            "area_km2": round(count * px_area_km2, 3),
            "percent_of_aoi": round(pct, 2),
        })
    return pd.DataFrame(rows)


def settlements_by_zone(settlements_path: str, zones: np.ndarray, grid) -> pd.DataFrame:
    """Count settlement/village point features falling in each risk zone."""
    gdf = gpd.read_file(settlements_path).to_crs(grid.crs)
    # Use point representation (centroid for polygons) for sampling
    pts = gdf.geometry.representative_point()

    inv_transform = ~grid.transform
    counts = {1: 0, 2: 0, 3: 0, "outside_aoi": 0}
    for pt in pts:
        col, row = inv_transform * (pt.x, pt.y)
        col, row = int(col), int(row)
        if 0 <= row < grid.height and 0 <= col < grid.width:
            z = int(zones[row, col])
            if z in (1, 2, 3):
                counts[z] += 1
            else:
                counts["outside_aoi"] += 1
        else:
            counts["outside_aoi"] += 1

    rows = [{"risk_zone": ZONE_LABELS[z], "settlements_affected": counts[z]} for z in (1, 2, 3)]
    rows.append({"risk_zone": "Outside AOI / NoData", "settlements_affected": counts["outside_aoi"]})
    return pd.DataFrame(rows)


def write_summary_csv(area_df: pd.DataFrame, out_path: str, settlements_df: pd.DataFrame | None = None):
    with open(out_path, "w") as f:
        f.write("# Flood Risk Summary Report\n")
        f.write("# --- Area by Risk Zone ---\n")
        area_df.to_csv(f, index=False)
        if settlements_df is not None:
            f.write("\n# --- Settlements Affected by Risk Zone ---\n")
            settlements_df.to_csv(f, index=False)
