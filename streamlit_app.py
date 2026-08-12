#!/usr/bin/env python3
"""
streamlit_app.py
=================
GUI dashboard for the Flood Susceptibility Demo Tool.

Run with:
    streamlit run streamlit_app.py
"""
import os
import shutil
import tempfile
import traceback

import pandas as pd
import rasterio
import streamlit as st
import streamlit.components.v1 as components

from flood_tool.pipeline import run_pipeline

st.set_page_config(page_title="Flood Susceptibility Demo Tool", layout="wide")
st.title("🌊 Flood Susceptibility Demo Tool by Mohammed Shan")
st.caption("DEM + Slope + River Distance + Rainfall (optional) → Weighted Overlay → Flood Susceptibility Zones")

# Bundled real district datasets. Add more districts here as data becomes
# available — each just needs an AOI, DEM, and rivers file checked into
# the sample_data/ folder in the repo. "rainfall" is optional per region.
REAL_DEMO_REGIONS = {
    "Trivandrum, Kerala": {
        "aoi": "sample_data/trivandrum/boundary.shp",
        "dem": "sample_data/trivandrum/dem.tif",
        "rivers": "sample_data/trivandrum/rivers.shp",
        "settlements": None,
        "rainfall": "sample_data/trivandrum/rainfall.tif",  # optional; used if present
    },
}

# --- Contact details shown in the sidebar footer -----------------------------
CONTACT_EMAIL = "shangeography@gmail.com"

# This app runs on a free hosting tier with limited memory. A state-wide DEM
# (e.g. all of Kerala at 30m) can exceed available RAM once loaded, clipped,
# and used for slope/distance calculations — causing a crash. District- or
# taluk-sized AOIs work reliably; recommend users stay under this size.
DEM_SIZE_WARNING_MB = 40


def _describe_crs(raster_path: str) -> str:
    """Return a short human-readable CRS description for the processed raster."""
    try:
        with rasterio.open(raster_path) as src:
            crs = src.crs
            if crs is None:
                return "Unknown"
            epsg = crs.to_epsg()
            name = crs.to_string()
            return f"EPSG:{epsg} — {name}" if epsg else name
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


def _save_upload(upload, workdir):
    if upload is None:
        return None
    path = os.path.join(workdir, upload.name)
    with open(path, "wb") as f:
        f.write(upload.getbuffer())
    if path.endswith(".zip"):
        extract_dir = path.replace(".zip", "")
        shutil.unpack_archive(path, extract_dir)
        # Search recursively — some zip tools nest contents inside a
        # subfolder matching the zip's own name, so we can't assume the
        # .shp sits directly at the top level.
        for root, _dirs, files in os.walk(extract_dir):
            for fn in files:
                if fn.endswith(".shp"):
                    return os.path.join(root, fn)
        raise ValueError(
            f"No .shp file found anywhere inside {upload.name}. "
            "Make sure the zip contains the .shp plus its .dbf/.shx/.prj siblings."
        )
    return path


with st.sidebar:
    st.header("1. Choose your data source")
    data_source = st.radio(
        "Data source",
        ["Use my own data", "Use sample data (Trivandrum, Kerala)"],
        label_visibility="collapsed",
    )
    using_sample = data_source == "Use sample data (Trivandrum, Kerala)"

    if using_sample:
        region = REAL_DEMO_REGIONS["Trivandrum, Kerala"]
        missing = [k for k in ("aoi", "dem", "rivers") if not os.path.exists(region[k])]
        if missing:
            st.error(f"Sample data isn't fully bundled with this app yet (missing: {', '.join(missing)}).")
            using_sample = False
        else:
            st.success("Trivandrum, Kerala selected — real SRTM elevation and OpenStreetMap water body data.")
            if region.get("rainfall") and os.path.exists(region["rainfall"]):
                st.caption("✓ Rainfall climatology also available for this region.")

    st.divider()
    st.header("2. Upload Data")
    disabled = using_sample
    aoi_file = st.file_uploader("AOI boundary (.shp needs zip, or .geojson)", type=["geojson", "json", "zip"], disabled=disabled)
    dem_file = st.file_uploader(
        "DEM raster (.tif)", type=["tif", "tiff"], disabled=disabled,
        help=(
            f"Keep this under ~{DEM_SIZE_WARNING_MB}MB. This app runs on limited "
            "memory, and a state-wide DEM (e.g. all of Kerala) can crash the app "
            "when it's clipped and processed. Use a district/taluk-sized AOI "
            "instead — clip your DEM to that extent in QGIS ('Raster \u2192 "
            "Extraction \u2192 Clip Raster by Mask Layer') or in Earth Engine before "
            "exporting, using a smaller boundary than the full state."
        ),
    )
    if dem_file is not None:
        dem_size_mb = dem_file.size / (1024 * 1024)
        if dem_size_mb > DEM_SIZE_WARNING_MB:
            st.warning(
                f"⚠️ This DEM is {dem_size_mb:.1f}MB — larger than the "
                f"{DEM_SIZE_WARNING_MB}MB guideline. Large, state-wide rasters "
                "often crash this app due to memory limits. Consider clipping "
                "to a smaller area (a single district or taluk) before "
                "uploading, or the app may fail when you click Generate."
            )
        else:
            st.caption(f"DEM size: {dem_size_mb:.1f}MB ✓")

    rivers_file = st.file_uploader("Rivers layer (.geojson or zipped .shp)", type=["geojson", "json", "zip"], disabled=disabled)
    settlements_file = st.file_uploader("Settlements/villages (optional)", type=["geojson", "json", "zip"], disabled=disabled)
    rainfall_file = st.file_uploader(
        "Rainfall raster (optional, .tif — e.g. monsoon-season average)",
        type=["tif", "tiff"], disabled=disabled,
        help="Long-term average monsoon (Jun–Sep) rainfall works best — a stable indicator, "
             "not tied to any single year's weather.",
    )

    st.header("3. Overlay Weights")
    rainfall_available_in_sample = bool(using_sample and REAL_DEMO_REGIONS["Trivandrum, Kerala"].get("rainfall") and os.path.exists(REAL_DEMO_REGIONS["Trivandrum, Kerala"]["rainfall"]))
    include_rainfall = rainfall_available_in_sample or (rainfall_file is not None)
    if include_rainfall:
        w_elev = st.slider("Elevation weight", 0.0, 1.0, 0.3, 0.05)
        w_slope = st.slider("Slope weight", 0.0, 1.0, 0.2, 0.05)
        w_river = st.slider("River-distance weight", 0.0, 1.0, 0.2, 0.05)
        w_rain = st.slider("Rainfall weight", 0.0, 1.0, 0.3, 0.05)
        st.caption(f"Weights auto-normalise to sum to 1.0 (currently {w_elev + w_slope + w_river + w_rain:.2f})")
    else:
        w_elev = st.slider("Elevation weight", 0.0, 1.0, 0.4, 0.05)
        w_slope = st.slider("Slope weight", 0.0, 1.0, 0.3, 0.05)
        w_river = st.slider("River-distance weight", 0.0, 1.0, 0.3, 0.05)
        w_rain = 0.0
        st.caption(f"Weights auto-normalise to sum to 1.0 (currently {w_elev + w_slope + w_river:.2f})")
        st.caption("Add a rainfall raster above to unlock a 4th weighted criterion.")

    st.header("4. River Buffer Zones")
    river_near = st.number_input("High susceptibility within (m)", value=500, step=50)
    river_far = st.number_input("Low susceptibility beyond (m)", value=1500, step=50)

    run_btn = st.button("🚀 Generate Flood Susceptibility Map", type="primary", use_container_width=True)

    st.divider()
    st.markdown("### 📩 Questions or Issues?")
    st.markdown(f"**Email:** [{CONTACT_EMAIL}](mailto:{CONTACT_EMAIL})")
    st.caption("Built by Mohammed Shan")


if run_btn:
    if not using_sample and not (aoi_file and dem_file and rivers_file):
        st.error("Please upload AOI, DEM, and Rivers layers, or choose the sample data option in the sidebar.")
    else:
        try:
            with tempfile.TemporaryDirectory() as workdir:
                if using_sample:
                    region = REAL_DEMO_REGIONS["Trivandrum, Kerala"]
                    aoi_path = region["aoi"]
                    dem_path = region["dem"]
                    rivers_path = region["rivers"]
                    settlements_path = region["settlements"]
                    rainfall_path = region["rainfall"] if region.get("rainfall") and os.path.exists(region["rainfall"]) else None
                else:
                    aoi_path = _save_upload(aoi_file, workdir)
                    dem_path = _save_upload(dem_file, workdir)
                    rivers_path = _save_upload(rivers_file, workdir)
                    settlements_path = _save_upload(settlements_file, workdir)
                    rainfall_path = _save_upload(rainfall_file, workdir)

                out_dir = os.path.join(workdir, "outputs")
                weights = {"elevation": w_elev, "slope": w_slope, "river": w_river}
                if rainfall_path:
                    weights["rainfall"] = w_rain

                with st.spinner("Running pipeline: clipping DEM, computing slope, river distance, overlay..."):
                    result = run_pipeline(
                        aoi_path=aoi_path, dem_path=dem_path, rivers_path=rivers_path,
                        settlements_path=settlements_path, rainfall_path=rainfall_path,
                        out_dir=out_dir, weights=weights,
                        river_near_m=river_near, river_far_m=river_far,
                    )

                st.success("Flood susceptibility map generated!")

                if result.get("used_rainfall"):
                    st.info("📊 This result includes rainfall as a 4th weighted criterion.")

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

        except Exception as exc:
            st.error(f"⚠️ Something went wrong while processing your data: {exc}")
            with st.expander("Show full technical details"):
                st.code(traceback.format_exc())
            st.info(
                "Common causes: a zipped shapefile missing its .dbf/.shx/.prj siblings, "
                "a DEM without a defined coordinate system, a rivers/AOI file with no "
                "valid geometry, or a DEM covering too large an area for this app's "
                "memory limit (see the size guidance next to the DEM upload field)."
            )
else:
    st.info("Choose **Use my own data** or **Use sample data (Trivandrum, Kerala)** in the sidebar to get started.")
    st.markdown(f"""
    **Expected inputs**
    - **AOI**: district/study-area boundary polygon
    - **DEM**: elevation raster (SRTM 30m or ASTER), GeoTIFF — keep it under ~{DEM_SIZE_WARNING_MB}MB (a district or taluk works well; a full state DEM is likely too large for this app's memory limit)
    - **Rivers**: water body/river lines or polygons (e.g. from OpenStreetMap)
    - **Rainfall** *(optional)*: a raster of average monsoon-season rainfall — adds a 4th weighted criterion
    - **Settlements** *(optional)*: village/settlement points, to count how many fall in each susceptibility zone
    """)
