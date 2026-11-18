#!/usr/bin/env python3
"""
streamlit_app.py
=================
Web app for the Flood Susceptibility Demo Tool.

Run with:
    streamlit run streamlit_app.py
"""
import io
import os
import shutil
import tempfile
import traceback
import zipfile
from fractions import Fraction
from itertools import combinations
from pathlib import Path

import geopandas as gpd
import pandas as pd
import streamlit as st

from flood_tool import ahp, boundaries, open_data
from flood_tool.pipeline import run_pipeline, score_open_data

st.set_page_config(page_title="Flood Susceptibility Demo Tool", layout="wide")
st.title("🌊 Flood Susceptibility Demo Tool by Mohammed Shan")
st.caption("Pick an area → layers fetched from open data → AHP-weighted overlay → High / Medium / Low susceptibility")

CONTACT_EMAIL = "shangeography@gmail.com"
DEM_SIZE_WARNING_MB = 25
SAATY = ["1/9", "1/8", "1/7", "1/6", "1/5", "1/4", "1/3", "1/2", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
OPEN_LAYERS = {
    "river": ("osm", "jrc"),
    "drainage": ("osm",),
    "rainfall": ("rainfall",),
    "landcover": ("landcover",),
    "soil": ("soil",),
}
LAYER_NAMES = {
    "dem": "Elevation & slope",
    "osm": "Rivers and streams",
    "jrc": "Surface water",
    "rainfall": "Monsoon rainfall",
    "landcover": "Land cover",
    "soil": "Soil",
}


def _save_upload(upload, workdir):
    if upload is None:
        return None
    path = os.path.join(workdir, upload.name)
    with open(path, "wb") as f:
        f.write(upload.getbuffer())
    if path.endswith(".zip"):
        extract_dir = path[:-4]
        shutil.unpack_archive(path, extract_dir)
        for root, _dirs, files in os.walk(extract_dir):
            for fn in files:
                if fn.endswith(".shp"):
                    return os.path.join(root, fn)
        raise ValueError(f"No .shp file found inside {upload.name}; include the .shp with its .dbf/.shx/.prj.")
    return path


def _workdir():
    if "workdir" not in st.session_state:
        st.session_state.workdir = tempfile.mkdtemp(prefix="flood_")
    return st.session_state.workdir


def _zip_outputs(result) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for key in ("raster", "geojson", "summary_csv", "static_map", "interactive_map", "criteria_map"):
            if result.get(key) and os.path.exists(result[key]):
                z.write(result[key], os.path.basename(result[key]))
    return buf.getvalue()


def _judgement_table(criteria):
    rows = []
    for a, b in combinations(criteria, 2):
        if (a, b) in ahp.DEFAULT_JUDGEMENTS:
            v = Fraction(ahp.DEFAULT_JUDGEMENTS[(a, b)])
        elif (b, a) in ahp.DEFAULT_JUDGEMENTS:
            v = 1 / Fraction(ahp.DEFAULT_JUDGEMENTS[(b, a)])
        else:
            v = Fraction(1)
        rows.append({"A": ahp.CRITERIA[a], "B": ahp.CRITERIA[b], "A vs B": str(v), "_a": a, "_b": b})
    return pd.DataFrame(rows)


def _ahp_editor(criteria):
    """Pairwise comparison table; returns (judgements, weights, CR)."""
    key = "ahp_" + "_".join(criteria)
    if key not in st.session_state:
        st.session_state[key] = _judgement_table(criteria)
    table = st.session_state[key]

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Pairwise comparisons** (Saaty 1–9 scale: 3 = A moderately more important than B, "
                    "9 = extremely more important, 1/3 = B moderately more important than A)")
        edited = st.data_editor(
            table[["A", "B", "A vs B"]], hide_index=True, width="stretch", key=key + "_editor",
            disabled=["A", "B"],
            column_config={"A vs B": st.column_config.SelectboxColumn("A vs B", options=SAATY, required=True)},
        )
        if st.button("Reset to default judgements", key=key + "_reset"):
            for k in (key, key + "_editor"):
                st.session_state.pop(k, None)
            st.rerun()
    judgements = {(a, b): float(Fraction(v)) for a, b, v in zip(table["_a"], table["_b"], edited["A vs B"])}
    weights, cr = ahp.weights_for(criteria, judgements)
    with right:
        st.markdown("**Resulting weights**")
        wdf = pd.DataFrame({"Criterion": [ahp.CRITERIA[k] for k in weights],
                            "Weight": [round(w, 3) for w in weights.values()]})
        st.dataframe(wdf, hide_index=True, width="stretch")
        if cr < 0.1:
            st.success(f"Consistency ratio CR = {cr:.3f} (acceptable, < 0.10)")
        else:
            st.warning(f"Consistency ratio CR = {cr:.3f}. Above 0.10 the judgements contradict each other; "
                       "consider revising them.")
    return judgements, weights, cr


def _show_results(result, aoi_crs_note=None):
    st.success("Flood susceptibility map generated.")
    if aoi_crs_note:
        st.caption(aoi_crs_note)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Interactive map")
        st.iframe(Path(result["interactive_map"]), height=550)
    with col2:
        st.subheader("Area by zone")
        st.dataframe(result["area_stats"], width="stretch", hide_index=True)
        if result.get("settlement_stats") is not None:
            st.subheader("Settlements by zone")
            st.dataframe(result["settlement_stats"], width="stretch", hide_index=True)
        if "weights" in result:
            m, h = result["breaks"]
            st.caption(f"Class breaks: Medium ≥ {m:.3f}, High > {h:.3f}")

    if result.get("layer_status"):
        st.subheader("Data layers")
        st.dataframe(pd.DataFrame([
            {"Layer": LAYER_NAMES.get(k, k), "Status": "✓" if ok else "✗ not available", "Source / message": msg}
            for k, (ok, msg) in result["layer_status"].items()
        ]), hide_index=True, width="stretch")

    if result.get("criteria_map"):
        st.subheader("Criterion scores")
        st.image(result["criteria_map"], width="stretch")

    st.subheader("Static map")
    st.image(result["static_map"], width="stretch")

    st.subheader("Downloads")
    cols = st.columns(4)
    cols[0].download_button("All outputs (.zip)", _zip_outputs(result), file_name="flood_susceptibility.zip")
    for col, (label, key) in zip(cols[1:], [("GeoTIFF", "raster"), ("GeoJSON zones", "geojson"),
                                            ("Summary CSV", "summary_csv")]):
        with open(result[key], "rb") as f:
            col.download_button(label, f.read(), file_name=os.path.basename(result[key]))


# --- Sidebar -----------------------------------------------------------------
with st.sidebar:
    st.header("1. Area of interest")
    has_districts = boundaries.DISTRICTS_PATH.exists()
    aoi_options = (["Kerala district"] if has_districts else []) + ["Upload a boundary"]
    aoi_mode = st.radio("Area", aoi_options, label_visibility="collapsed")
    district_name, aoi_file = None, None
    if aoi_mode == "Kerala district":
        names = boundaries.district_names()
        default = names.index("Thiruvananthapuram") if "Thiruvananthapuram" in names else 0
        district_name = st.selectbox("District", names, index=default)
    else:
        aoi_file = st.file_uploader("Boundary polygon (.geojson, .gpkg, .kml, or zipped .shp)",
                                    type=["geojson", "json", "gpkg", "kml", "zip"])

    st.header("2. Input layers")
    data_mode = st.radio("Layers", ["Fetch from open data", "Upload my own"], label_visibility="collapsed",
                         help="Open data: elevation, water, rainfall, land cover and soil are downloaded for "
                              "your area only. Nothing needs to be prepared in advance.")
    own = data_mode == "Upload my own"

    if own:
        dem_file = st.file_uploader("DEM raster (.tif)", type=["tif", "tiff"],
                                    help=f"Keep under ~{DEM_SIZE_WARNING_MB} MB (a district or taluk).")
        rivers_file = st.file_uploader("Rivers / water bodies (.geojson or zipped .shp)", type=["geojson", "json", "zip"])
        rainfall_file = st.file_uploader("Rainfall raster (optional, .tif)", type=["tif", "tiff"])
        st.subheader("Weights")
        w_elev = st.slider("Elevation", 0.0, 1.0, 0.3 if rainfall_file else 0.4, 0.05)
        w_slope = st.slider("Slope", 0.0, 1.0, 0.2 if rainfall_file else 0.3, 0.05)
        w_river = st.slider("Distance to water", 0.0, 1.0, 0.2 if rainfall_file else 0.3, 0.05)
        w_rain = st.slider("Rainfall", 0.0, 1.0, 0.3, 0.05) if rainfall_file else 0.0
        st.caption("Weights are normalised to sum to 1.")
    else:
        optional = [k for k in ahp.CRITERIA if k not in ("elevation", "slope")]
        chosen = st.multiselect("Criteria (elevation and slope are always used)", optional, default=optional,
                                format_func=lambda k: ahp.CRITERIA[k])
        criteria = ["elevation", "slope", *[k for k in optional if k in chosen]]

    settlements_file = st.file_uploader("Settlements / villages (optional points)", type=["geojson", "json", "zip"])

    with st.expander("Parameters"):
        river_near = st.number_input("Distance to water: score 1 within (m)", value=500, step=50)
        river_far = st.number_input("Distance to water: score 0 beyond (m)", value=1500, step=50)
        max_slope = st.number_input("Slope: score 0 at or above (°)", value=15.0, step=1.0)
        if not own:
            classification = st.radio("Zone boundaries", ["natural_breaks", "fixed"],
                                      format_func=lambda v: "Natural breaks (Jenks)" if v == "natural_breaks"
                                      else "Fixed: 0.4 / 0.7")
            res_choice = st.selectbox("Grid resolution", ["Auto (30 m)", 30, 60, 90],
                                      help="Auto uses 30 m and coarsens automatically for very large areas.")

    run_btn = st.button("🚀 Generate flood susceptibility map", type="primary", width="stretch")

    st.divider()
    st.markdown(f"**Questions or issues?** [{CONTACT_EMAIL}](mailto:{CONTACT_EMAIL})")
    st.caption("Built by Mohammed Shan")


# --- Main area -----------------------------------------------------------------
judgements = None
if not own:
    with st.expander("⚖️ Criteria weights (AHP)", expanded=not run_btn and "result" not in st.session_state):
        judgements, weights, cr = _ahp_editor(criteria)

if run_btn:
    try:
        workdir = _workdir()
        settlements_path = _save_upload(settlements_file, workdir)
        if district_name:
            aoi = boundaries.district(district_name)
            aoi_key = ("district", district_name)
        elif aoi_file is not None:
            aoi = gpd.read_file(_save_upload(aoi_file, workdir))
            aoi_key = ("upload", aoi_file.name, aoi_file.size)
        else:
            st.error("Choose a district or upload a boundary first.")
            st.stop()
        crs_note = (f"📐 Analysis CRS: UTM, auto-detected from the AOI location "
                    f"(input was {aoi.crs.to_string() if aoi.crs else 'unspecified, assumed WGS 84'}).")
        out_dir = tempfile.mkdtemp(dir=workdir)

        if own:
            if not (dem_file and rivers_file):
                st.error("Upload a DEM and a rivers layer, or switch to 'Fetch from open data'.")
                st.stop()
            aoi_path = os.path.join(workdir, "aoi.geojson")
            aoi.to_file(aoi_path, driver="GeoJSON")
            weights = {"elevation": w_elev, "slope": w_slope, "river": w_river}
            rainfall_path = _save_upload(rainfall_file, workdir)
            if rainfall_path:
                weights["rainfall"] = w_rain
            with st.spinner("Running the analysis..."):
                result = run_pipeline(
                    aoi_path=aoi_path, dem_path=_save_upload(dem_file, workdir),
                    rivers_path=_save_upload(rivers_file, workdir), settlements_path=settlements_path,
                    rainfall_path=rainfall_path, out_dir=out_dir, weights=weights,
                    river_near_m=river_near, river_far_m=river_far, max_slope_cutoff_deg=max_slope,
                )
        else:
            resolution = None if res_choice == "Auto (30 m)" else float(res_choice)
            layers = tuple(sorted({src for k in criteria for src in OPEN_LAYERS.get(k, ())}))
            stack_key = (aoi_key, resolution, layers)
            if st.session_state.get("stack_key") != stack_key:
                bar = st.progress(0.0, text="Fetching elevation...")

                def progress(done, total, label):
                    bar.progress(done / total, text=f"Layers ready: {done} of {total} (last: {label})")

                st.session_state.stack = open_data.fetch_layers(aoi, resolution=resolution, layers=layers,
                                                                progress=progress)
                st.session_state.stack_key = stack_key
                bar.empty()
            with st.spinner("Scoring criteria and building maps..."):
                result = score_open_data(
                    st.session_state.stack, out_dir=out_dir, judgements=judgements, criteria=criteria,
                    classification=classification, settlements_path=settlements_path,
                    river_near_m=river_near, river_far_m=river_far, max_slope_cutoff_deg=max_slope,
                )
            result.pop("score", None)
            result.pop("zones", None)
        st.session_state.result = result
        st.session_state.crs_note = crs_note
    except Exception as exc:
        st.session_state.pop("result", None)
        st.error(f"⚠️ Something went wrong: {exc}")
        with st.expander("Technical details"):
            st.code(traceback.format_exc())
        st.info("Common causes: a zipped shapefile missing its .dbf/.shx/.prj, a boundary with no valid "
                "polygon, or a data service that is temporarily unreachable (try again in a minute).")

if "result" in st.session_state:
    _show_results(st.session_state.result, st.session_state.get("crs_note"))
elif not run_btn:
    st.info("Pick a **Kerala district** (or upload any boundary) in the sidebar, then press **Generate**.")
    st.markdown("""
**How it works**

All layers are fetched for your area only, straight from open data services — no API keys, nothing stored:

| Criterion | Source |
|---|---|
| Elevation, slope | Copernicus DEM GLO-30 |
| Distance to water | OpenStreetMap waterways + JRC Global Surface Water |
| Drainage density | OpenStreetMap waterways |
| Monsoon rainfall | CHELSA V2.1, June–September, 1981–2010 mean |
| Land cover | ESA WorldCover 2021 (10 m) |
| Soil | SoilGrids 2.0 clay content |

Each criterion is scaled to a 0–1 score, weighted with the **Analytic Hierarchy Process** (edit the pairwise
judgements above), and the combined score is split into **High / Medium / Low** zones.
    """)
