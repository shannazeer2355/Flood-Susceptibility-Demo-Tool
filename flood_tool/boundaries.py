"""
boundaries.py
-------------
Kerala district boundaries bundled with the tool (geoBoundaries, CC BY 4.0;
regenerate with scripts/make_kerala_districts.py).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import geopandas as gpd

DISTRICTS_PATH = Path(__file__).parent / "data" / "kerala_districts.geojson"


@lru_cache(maxsize=1)
def kerala_districts() -> gpd.GeoDataFrame:
    return gpd.read_file(DISTRICTS_PATH).sort_values("district").reset_index(drop=True)


def district_names() -> list[str]:
    return kerala_districts()["district"].tolist()


def district(name: str) -> gpd.GeoDataFrame:
    gdf = kerala_districts()
    match = gdf[gdf["district"].str.lower() == name.strip().lower()]
    if match.empty:
        raise ValueError(f"Unknown district '{name}'. Choose from: {', '.join(gdf['district'])}")
    return match.reset_index(drop=True)
