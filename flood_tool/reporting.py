"""
reporting.py
------------
Summary statistics for the flood-susceptibility classification.
"""
from __future__ import annotations

import math

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.transform import rowcol

from .overlay import ZONE_LABELS


def area_statistics(zones: np.ndarray, grid) -> pd.DataFrame:
    px_area_km2 = abs(grid.transform.a * grid.transform.e) / 1_000_000
    total_valid = int((zones > 0).sum())

    rows = []
    for zone_id in (1, 2, 3):
        count = int((zones == zone_id).sum())
        pct = 100 * count / total_valid if total_valid else 0
        rows.append({
            "susceptibility_zone": ZONE_LABELS[zone_id],
            "pixel_count": count,
            "area_km2": round(count * px_area_km2, 3),
            "percent_of_aoi": round(pct, 2),
        })
    return pd.DataFrame(rows)


def settlements_by_zone(settlements_path: str, zones: np.ndarray, grid) -> pd.DataFrame:
    gdf = gpd.read_file(settlements_path).to_crs(grid.crs)
    pts = gdf.geometry.representative_point()

    counts = {1: 0, 2: 0, 3: 0, "outside_aoi": 0}
    for pt in pts:
        row, col = rowcol(grid.transform, pt.x, pt.y, op=math.floor)
        if 0 <= row < grid.height and 0 <= col < grid.width and zones[row, col] in (1, 2, 3):
            counts[int(zones[row, col])] += 1
        else:
            counts["outside_aoi"] += 1

    rows = [{"susceptibility_zone": ZONE_LABELS[z], "settlements": counts[z]} for z in (1, 2, 3)]
    rows.append({"susceptibility_zone": "Outside AOI / NoData", "settlements": counts["outside_aoi"]})
    return pd.DataFrame(rows)


def write_summary_csv(area_df: pd.DataFrame, out_path: str, settlements_df: pd.DataFrame | None = None):
    with open(out_path, "w", newline="") as f:
        f.write("# Flood Susceptibility Summary Report\n")
        f.write("# --- Area by Susceptibility Zone ---\n")
        area_df.to_csv(f, index=False)
        if settlements_df is not None:
            f.write("\n# --- Settlements by Susceptibility Zone ---\n")
            settlements_df.to_csv(f, index=False)
