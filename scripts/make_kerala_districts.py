"""
Build flood_tool/data/kerala_districts.geojson from geoBoundaries
(https://www.geoboundaries.org, gbOpen, CC BY 4.0): the India ADM2 units
whose interior falls inside the Kerala ADM1 boundary.

    python scripts/make_kerala_districts.py
"""
from pathlib import Path

import geopandas as gpd
import requests

API = "https://www.geoboundaries.org/api/current/gbOpen/IND/{level}/"
OUT = Path(__file__).resolve().parents[1] / "flood_tool" / "data" / "kerala_districts.geojson"


def load(level):
    meta = requests.get(API.format(level=level), timeout=60).json()
    return gpd.read_file(meta["simplifiedGeometryGeoJSON"]).to_crs(4326)


def main():
    states = load("ADM1")
    kerala = states[states["shapeName"].str.lower() == "kerala"]
    if kerala.empty:
        raise SystemExit(f"Kerala not found in ADM1: {sorted(states['shapeName'])[:40]}")
    districts = load("ADM2")
    inside = districts.representative_point().within(kerala.geometry.iloc[0])
    out = districts[inside][["shapeName", "geometry"]].rename(columns={"shapeName": "district"})
    out["geometry"] = out.geometry.make_valid()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.sort_values("district").to_file(OUT, driver="GeoJSON", COORDINATE_PRECISION=5)
    print(f"{len(out)} districts -> {OUT}")
    print(", ".join(sorted(out["district"])))


if __name__ == "__main__":
    main()
