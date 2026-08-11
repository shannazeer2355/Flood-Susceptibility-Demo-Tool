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
one-click "Use sample data" option that loads a real, bundled
district dataset, so the app can be tried instantly without any
uploads.
"""
import os
import shutil
import tempfile

import rasterio
import streamlit as st
import streamlit.components.v1 as components

from flood_tool.pipeline import run_pipeline

st.set_page_config(page_title="Flood Susceptibility Demo Tool", layout="wide")
st.title("🌊 Flood Susceptibility Demo Tool by Mohammed Shan")
st.caption("DEM + Slope + River Distance → Weighted Overlay → Flood Susceptibility Zones")

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
    st.header("1. Data Source")

    data_mode = st.radio(
        "How do you want to provide data?",
        options=["Use my own data", "Use sample data (Trivandrum, Kerala)"],
        index=0,
    )
    use_sample_data = data_mode.startswith("Use sample")

    aoi_file = dem_file = rivers_file = settlements_file = None
    sample_region_ready = False

    if use_sample_data:
        region = REAL_DEMO_REGIONS["Trivandrum, Kerala"]
        missing = [k for k in ("aoi", "dem", "rivers") if region[k] and not os.path.exists(region[k])]
        if missing:
            st.error(
                f"Sample data isn't bundled with this app yet (missing: {', '.join(missing)})."
            )
        else:
            sample_region_ready = True
            st.success("Real SRTM elevation and OpenStreetMap rivers for Trivandrum, Kerala will be used.")
    else:
        st.caption("Upload your own AOI, DEM, and river layers below.")
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
    if use_sample_data and not sample_region_ready:
        st.error("Sample data isn't available. Please switch to 'Use my own data' and upload your files.")
    elif not use_sample_data and not (aoi_file and dem_file and rivers_file):
        st.error("Please upload AOI, DEM, and Rivers layers.")
    else:
        with tempfile.TemporaryDirectory() as workdir:
            if use_sample_data:
                region = REAL_DEMO_REGIONS["Trivandrum, Kerala"]
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
    if use_sample_data and sample_region_ready:
        st.info("Sample data is ready — click **Generate Flood Susceptibility Map** in the sidebar to run it.")
    else:
        st.info("Upload your own AOI, DEM, and Rivers layers in the sidebar, or switch to sample data for an instant example.")
    st.markdown("""
    **Expected inputs**
    - **AOI**: district/study-area boundary polygon
    - **DEM**: elevation raster (SRTM 30m or ASTER), GeoTIFF
    - **Rivers**: water body/river lines or polygons (e.g. from OpenStreetMap)
    - **Settlements** *(optional)*: village/settlement points, to count how many fall in each susceptibility zone
    """)
