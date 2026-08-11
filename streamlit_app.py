#!/usr/bin/env python3
"""
streamlit_app.py
=================
GUI dashboard for the Flood Susceptibility Demo Tool.

Run with:
    streamlit run streamlit_app.py

Lets the user upload AOI / DEM / rivers / settlements files, tune
overlay weights with sliders, run the pipeline, and preview/download
all outputs without touching the command line. Also includes a
one-click "Try Demo Data" option that generates a synthetic dataset
in-memory, so the app can be tried instantly with no files at all.
"""
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import rasterio
import streamlit as st
import streamlit.components.v1 as components

from flood_tool.pipeline import run_pipeline

st.set_page_config(page_title="Flood Susceptibility Demo Tool", layout="wide")
st.title("🌊 Flood Susceptibility Demo Tool by Mohammed Shan")
st.caption("DEM + Slope + River Distance → Weighted Overlay → Flood Susceptibility Zones")

if "demo_region" not in st.session_state:
    st.session_state.demo_region = None

# Bundled real district datasets. Add more districts here as data becomes
# available — each just needs an AOI, DEM, and rivers file checked into
# the sample_data/ folder in the repo.
REAL_DEMO_REGIONS = {
    "Trivandrum, Kerala": {
        "aoi": "sample_data/trivandrum/boundary.shp",
        "dem": "sample_data/trivandrum/dem.tif",
        "rivers": "sample_data/trivandrum/rivers.shp",
        "settlements": None,
    },
}
DEMO_OPTIONS = ["— Select a demo —", "Synthetic example (instant, no real data)"] + list(REAL_DEMO_REGIONS.keys())


def generate_demo_data(workdir: str) -> dict:
    """
    Build a small synthetic AOI + DEM + rivers + villages dataset so the
    app can be tried instantly with no uploads. Mirrors the logic in
    sample_data/make_sample_data.py.
    """
    import geopandas as gpd
    import rasterio
    from rasterio.transform import from_origin
    from shapely.geometry import Polygon, LineString, Point

    crs = "EPSG:32643"
    minx, miny, maxx, maxy = 0, 0, 5000, 4000

    aoi = gpd.GeoDataFrame(
        {"name": ["Demo District"]},
        geometry=[Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])],
        crs=crs,
    )
    aoi_path = os.path.join(workdir, "boundary.shp")
    aoi.to_file(aoi_path)

    res = 30
    width = int((maxx - minx) / res)
    height = int((maxy - miny) / res)
    xs = np.linspace(0, maxx, width)
    ys = np.linspace(maxy, 0, height)
    X, Y = np.meshgrid(xs, ys)

    valley_dist = np.abs((Y - (0.6 * X + 800))) / np.sqrt(1 + 0.6 ** 2)
    elevation = 20 + 0.02 * valley_dist ** 1.3
    elevation += np.random.default_rng(42).normal(0, 1.5, size=elevation.shape)
    elevation = elevation.astype("float32")

    transform = from_origin(minx, maxy, res, res)
    dem_path = os.path.join(workdir, "dem.tif")
    with rasterio.open(
        dem_path, "w", driver="GTiff", height=height, width=width, count=1,
        dtype="float32", crs=crs, transform=transform, nodata=-9999,
    ) as dst:
        dst.write(elevation, 1)

    river_line = LineString([(x, 0.6 * x + 800) for x in np.linspace(minx, maxx, 50)])
    rivers = gpd.GeoDataFrame({"name": ["Demo River"]}, geometry=[river_line], crs=crs)
    rivers_path = os.path.join(workdir, "rivers.shp")
    rivers.to_file(rivers_path)

    rng = np.random.default_rng(7)
    village_pts = [Point(rng.uniform(200, maxx - 200), rng.uniform(200, maxy - 200)) for _ in range(15)]
    villages = gpd.GeoDataFrame({"name": [f"Village_{i+1}" for i in range(15)]}, geometry=village_pts, crs=crs)
    villages_path = os.path.join(workdir, "villages.shp")
    villages.to_file(villages_path)

    return {"aoi": aoi_path, "dem": dem_path, "rivers": rivers_path, "settlements": villages_path}


def _describe_crs(raster_path: str) -> str:
    """Return a short human-readable CRS description for the processed raster."""
    try:
        with rasterio.open(raster_path) as src:
            crs = src.crs
            if crs is None:
                return "Unknown"
            epsg = crs.to_epsg()
            name = crs.to_string()
            if epsg:
                return f"EPSG:{epsg} — {name}"
            return name
    except Exception:
        return "Unavailable"


def _describe_input_crs(aoi_path: str) -> str:
    """Return a short human-readable CRS description for the original, unmodified AOI upload."""
    try:
        import geopandas as gpd
        gdf = gpd.read_file(aoi_path)
        crs = gdf.crs
        if crs is None:
            return "Unspecified in file"
        epsg = crs.to_epsg()
        return f"EPSG:{epsg}" if epsg else crs.to_string()
    except Exception:
        return "Unavailable"


with st.sidebar:
    st.header("Try it instantly")
    demo_choice = st.selectbox("Select a demo region", DEMO_OPTIONS, index=0)

    if demo_choice == "— Select a demo —":
        st.session_state.demo_region = None
    elif demo_choice == "Synthetic example (instant, no real data)":
        st.session_state.demo_region = "synthetic"
        st.success("Synthetic demo selected — a fake district with a river valley.")
    else:
        region = REAL_DEMO_REGIONS[demo_choice]
        missing = [k for k in ("aoi", "dem", "rivers") if region[k] and not os.path.exists(region[k])]
        if missing:
            st.error(f"Demo data for {demo_choice} isn't bundled with this app yet (missing: {', '.join(missing)}).")
            st.session_state.demo_region = None
        else:
            st.session_state.demo_region = demo_choice
            st.success(f"{demo_choice} selected — real SRTM elevation and OpenStreetMap rivers for this district.")

    st.divider()
    st.header("1. Upload Data")
    disabled = st.session_state.demo_region is not None
    aoi_file = st.file_uploader("AOI boundary (.shp needs zip, or .geojson)", type=["geojson", "json", "zip"], disabled=disabled)
    dem_file = st.file_uploader("DEM raster (.tif)", type=["tif", "tiff"], disabled=disabled)
    rivers_file = st.file_uploader("Rivers layer (.geojson or zipped .shp)", type=["geojson", "json", "zip"], disabled=disabled)
    settlements_file = st.file_uploader("Settlements/villages (optional)", type=["geojson", "json", "zip"], disabled=disabled)

    st.header("2. Overlay Weights")
    w_elev = st.slider("Elevation weight", 0.0, 1.0, 0.4, 0.05)
    w_slope = st.slider("Slope weight", 0.0, 1.0, 0.3, 0.05)
    w_river = st.slider("River-distance weight", 0.0, 1.0, 0.3, 0.05)
    st.caption(f"Weights auto-normalise to sum to 1.0 (currently {w_elev + w_slope + w_river:.2f})")

    st.header("3. River Buffer Zones")
    river_near = st.number_input("High susceptibility within (m)", value=500, step=50)
    river_far = st.number_input("Low susceptibility beyond (m)", value=1500, step=50)

    run_btn = st.button("🚀 Generate Flood Susceptibility Map", type="primary", use_container_width=True)


def _save_upload(upload, workdir):
    if upload is None:
        return None
    path = os.path.join(workdir, upload.name)
    with open(path, "wb") as f:
        f.write(upload.getbuffer())
    if path.endswith(".zip"):
        extract_dir = path.replace(".zip", "")
        shutil.unpack_archive(path, extract_dir)
        for fn in os.listdir(extract_dir):
            if fn.endswith(".shp"):
                return os.path.join(extract_dir, fn)
        raise ValueError(f"No .shp found inside {upload.name}")
    return path


if run_btn:
    if st.session_state.demo_region is None and not (aoi_file and dem_file and rivers_file):
        st.error("Please upload AOI, DEM, and Rivers layers, or select a demo region in the sidebar.")
    else:
        with tempfile.TemporaryDirectory() as workdir:
            if st.session_state.demo_region == "synthetic":
                with st.spinner("Generating synthetic demo dataset..."):
                    demo_paths = generate_demo_data(workdir)
                aoi_path = demo_paths["aoi"]
                dem_path = demo_paths["dem"]
                rivers_path = demo_paths["rivers"]
                settlements_path = demo_paths["settlements"]
            elif st.session_state.demo_region in REAL_DEMO_REGIONS:
                region = REAL_DEMO_REGIONS[st.session_state.demo_region]
                aoi_path = region["aoi"]
                dem_path = region["dem"]
                rivers_path = region["rivers"]
                settlements_path = region["settlements"]
            else:
                aoi_path = _save_upload(aoi_file, workdir)
                dem_path = _save_upload(dem_file, workdir)
                rivers_path = _save_upload(rivers_file, workdir)
                settlements_path = _save_upload(settlements_file, workdir)

            out_dir = os.path.join(workdir, "outputs")
            with st.spinner("Running pipeline: clipping DEM, computing slope, river distance, overlay..."):
                result = run_pipeline(
                    aoi_path=aoi_path, dem_path=dem_path, rivers_path=rivers_path,
                    settlements_path=settlements_path, out_dir=out_dir,
                    weights={"elevation": w_elev, "slope": w_slope, "river": w_river},
                    river_near_m=river_near, river_far_m=river_far,
                )

            st.success("Flood susceptibility map generated!")

            input_crs_desc = _describe_input_crs(aoi_path)
            working_crs_desc = _describe_crs(result["raster"])
            st.caption(
                f"📐 **Projection:** your AOI was supplied in **{input_crs_desc}** and was "
                f"automatically reprojected to **{working_crs_desc}** for analysis — the correct "
                "local, metre-based coordinate system for this region, auto-detected from your data's "
                "location. This works the same way for data from anywhere in the world, so distances "
                "and areas are always measured accurately regardless of what projection you upload in."
            )

            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Interactive Map")
                with open(result["interactive_map"], "r", encoding="utf-8") as f:
                    components.html(f.read(), height=550, scrolling=False)

            with col2:
                st.subheader("Area by Susceptibility Zone")
                st.dataframe(result["area_stats"], use_container_width=True, hide_index=True)
                if result["settlement_stats"] is not None:
                    st.subheader("Settlements in Each Susceptibility Zone")
                    st.dataframe(result["settlement_stats"], use_container_width=True, hide_index=True)

            st.subheader("Static Map")
            st.image(result["static_map"], use_container_width=True)

            st.subheader("Downloads")
            dl_cols = st.columns(4)
            labels_paths = [
                ("GeoTIFF", result["raster"]),
                ("GeoJSON zones", result["geojson"]),
                ("Summary CSV", result["summary_csv"]),
                ("Static PNG", result["static_map"]),
            ]
            for col, (label, path) in zip(dl_cols, labels_paths):
                with open(path, "rb") as f:
                    col.download_button(label, f, file_name=os.path.basename(path))
else:
    if st.session_state.demo_region is not None:
        st.info("Demo data is ready — click **Generate Flood Susceptibility Map** in the sidebar to run it.")
    else:
        st.info("Select a demo region above for an instant example, or upload your own AOI, DEM, and Rivers layers in the sidebar.")
    st.markdown("""
    **Expected inputs**
    - **AOI**: district/study-area boundary polygon
    - **DEM**: elevation raster (SRTM 30m or ASTER), GeoTIFF
    - **Rivers**: water body/river lines or polygons (e.g. from OpenStreetMap)
    - **Settlements** *(optional)*: village/settlement points, to count how many fall in each susceptibility zone
    """)