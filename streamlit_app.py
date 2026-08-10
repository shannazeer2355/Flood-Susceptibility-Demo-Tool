#!/usr/bin/env python3
"""
streamlit_app.py
=================
Optional GUI dashboard for the Flood Susceptible Mapping Tool.

Run with:
    streamlit run streamlit_app.py

Lets the user upload AOI / DEM / rivers / settlements files, tune
overlay weights with sliders, run the pipeline, and preview/download
all outputs without touching the command line.
"""
import os
import shutil
import tempfile

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from flood_tool.pipeline import run_pipeline

st.set_page_config(page_title="Flood Susceptibility Tool", layout="wide")
st.title("🌊 Interactive Flood Susceptibility Demo Tool by Mohammed Shan")
st.caption("DEM + Slope + River Distance → Weighted Overlay → Flood Susceptible Zones")

with st.sidebar:
    st.header("1. Upload Data")
    aoi_file = st.file_uploader("AOI boundary (.shp needs zip, or .geojson)", type=["geojson", "json", "zip"])
    dem_file = st.file_uploader("DEM raster (.tif)", type=["tif", "tiff"])
    rivers_file = st.file_uploader("Rivers layer (.geojson or zipped .shp)", type=["geojson", "json", "zip"])
    settlements_file = st.file_uploader("Settlements/villages (optional)", type=["geojson", "json", "zip"])

    st.header("2. Overlay Weights")
    w_elev = st.slider("Elevation weight", 0.0, 1.0, 0.4, 0.05)
    w_slope = st.slider("Slope weight", 0.0, 1.0, 0.3, 0.05)
    w_river = st.slider("River-distance weight", 0.0, 1.0, 0.3, 0.05)
    st.caption(f"Weights auto-normalise to sum to 1.0 (currently {w_elev + w_slope + w_river:.2f})")

    st.header("3. River Buffer Zones")
    river_near = st.number_input("High Susceptible within (m)", value=500, step=50)
    river_far = st.number_input("Low Susceptible beyond (m)", value=1500, step=50)

    run_btn = st.button("🚀 Generate Flood Susceptible Map", type="primary", use_container_width=True)


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
    if not (aoi_file and dem_file and rivers_file):
        st.error("Please upload AOI, DEM, and Rivers layers before running.")
    else:
        with tempfile.TemporaryDirectory() as workdir:
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

            st.success("Flood Susceptible Map generated!")

            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Interactive Map")
                with open(result["interactive_map"], "r", encoding="utf-8") as f:
                    components.html(f.read(), height=550, scrolling=False)

            with col2:
                st.subheader("Area by Risk Zone")
                st.dataframe(result["area_stats"], use_container_width=True, hide_index=True)
                if result["settlement_stats"] is not None:
                    st.subheader("Settlements Affected")
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
    st.info("Upload your AOI, DEM, and Rivers layers in the sidebar, then click **Generate Flood Risk Map**.")
    st.markdown("""
    **Expected inputs**
    - **AOI**: district/study-area boundary polygon
    - **DEM**: elevation raster (SRTM 30m or ASTER), GeoTIFF
    - **Rivers**: water body/river lines or polygons (e.g. from OpenStreetMap)
    - **Settlements** *(optional)*: village/settlement points, to count how many fall in each risk zone
    """)
