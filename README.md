# 🌊 Flood Risk Mapping Tool

A Python GIS pipeline that generates **Flood Risk Zone maps (High / Medium / Low)**
from a Digital Elevation Model (DEM), slope, and distance-to-river layers, using a
weighted overlay method. Built for the internship brief: elevation + slope + river
proximity → weighted score → classified risk map → statistics → static & interactive
visualisation.

![sample output](outputs/flood_risk_map.png)

## Contents

```
flood-risk-tool/
├── flood_tool.py            # CLI entry point
├── streamlit_app.py         # Optional GUI dashboard
├── flood_tool/               # Core package
│   ├── io_utils.py           # AOI/DEM loading, clipping, reprojection
│   ├── terrain.py             # Elevation + slope risk scoring
│   ├── river_distance.py      # Distance-to-river risk scoring
│   ├── overlay.py             # Weighted overlay + classification
│   ├── outputs.py             # GeoTIFF / GeoJSON writers
│   ├── reporting.py           # Area & settlement statistics
│   └── visualize.py           # Matplotlib + folium maps
├── sample_data/
│   └── make_sample_data.py   # Generates a synthetic demo dataset
├── outputs/                   # Generated results land here
└── requirements.txt
```

## Methodology

The tool follows a standard **multi-criteria weighted overlay** approach, a
well-established GIS method for flood susceptibility mapping:

| Criterion | Rule | Weight |
|---|---|---|
| **Elevation** | Lower elevation → higher risk. Elevation is normalised within the AOI (1st–99th percentile) then inverted. | 40% |
| **Slope** | Flatter terrain drains poorly → higher risk. Slope (degrees) computed with Horn's (1981) 3×3 gradient kernel, same method as `gdaldem slope`, then inverted/normalised against a configurable cutoff (default 15°). | 30% |
| **Distance to river** | Closer to water → higher risk. A continuous Euclidean distance transform is computed from rasterized river geometries, then mapped through a piecewise-linear curve: ≤500m → 1.0 (high), 500–1500m → linear taper, ≥1500m → 0.0 (low). | 30% |

Each criterion is normalised to a **0–1 risk score**, combined as:

```
Final Score = 0.4 × Elevation Risk + 0.3 × Slope Risk + 0.3 × River-Distance Risk
```

and classified into zones:

| Score | Zone |
|---|---|
| > 0.7 | 🔴 High |
| 0.4 – 0.7 | 🟡 Medium |
| < 0.4 | 🟢 Low |

All weights and thresholds are configurable via CLI flags or the Streamlit sliders.

**Why a continuous distance decay instead of hard buffer rings?** Hard rings (0–500m,
500–1500m, >1500m) create blocky, unrealistic discontinuities at the boundary. A
piecewise-linear function anchored exactly at the requested breakpoints preserves the
same risk bands while producing a smoother, more defensible surface — the standard
practice in flood susceptibility literature.

## Installation

```bash
pip install -r requirements.txt
```

Requires GDAL under the hood (via `rasterio`/`geopandas`) — on Linux/Mac,
`pip install rasterio geopandas` typically pulls prebuilt wheels; on Windows,
consider `conda install -c conda-forge rasterio geopandas` if pip wheels fail.

## Usage — Command Line

```bash
python flood_tool.py \
    --aoi sample_data/boundary.shp \
    --dem sample_data/dem.tif \
    --rivers sample_data/rivers.shp \
    --settlements sample_data/villages.shp \
    --out-dir outputs
```

Optional flags:

```
--w-elevation 0.4      Elevation weight
--w-slope 0.3          Slope weight
--w-river 0.3          River-distance weight
--river-near 500       Distance (m) below which river risk = High
--river-far 1500       Distance (m) beyond which river risk = Low
--max-slope 15         Slope (deg) at/above which slope risk = 0
-v / --verbose         Debug logging
```

### No real data yet? Generate a synthetic demo dataset:

```bash
python sample_data/make_sample_data.py
python flood_tool.py --aoi sample_data/boundary.shp --dem sample_data/dem.tif \
    --rivers sample_data/rivers.shp --settlements sample_data/villages.shp
```

## Usage — GUI Dashboard (Streamlit)

```bash
streamlit run streamlit_app.py
```

Upload AOI / DEM / rivers / settlements, tune weights with sliders, and preview
the interactive map, static map, and statistics tables in-browser, with one-click
downloads for every output file.

## Outputs

| File | Description |
|---|---|
| `flood_risk.tif` | Classified risk raster (1=Low, 2=Medium, 3=High), GeoTIFF |
| `flood_risk_zones.geojson` | Risk zones dissolved into polygons, with area (km²) |
| `summary.csv` | % / km² of AOI per risk zone + settlements affected per zone |
| `flood_risk_map.png` | Static map (matplotlib) |
| `interactive_map.html` | Interactive Leaflet/folium map with basemap + legend |

## Free Data Sources

- **DEM**: [SRTM 30m](https://earthexplorer.usgs.gov/) · [OpenTopography](https://opentopography.org/) · ASTER GDEM
- **Rivers / water bodies**: [OpenStreetMap](https://www.openstreetmap.org/) (via [Overpass Turbo](https://overpass-turbo.eu/) or [HOT Export Tool](https://export.hotosm.org/)) · [HydroSHEDS](https://www.hydrosheds.org/)
- **Admin boundaries**: [GADM](https://gadm.org/)
- **Rainfall (bonus)**: [CHIRPS](https://www.chc.ucsb.edu/data/chirps)

## Bonus Extensions (for future work)

- Ingest CHIRPS rainfall rasters as a 4th weighted criterion
- Replace the hand-tuned weighted overlay with a trained classifier (Random Forest /
  XGBoost) using historical flood-extent points as labels
- Validate the risk map against a historical flood-points layer (confusion matrix,
  AUC) to quantify accuracy

## License

MIT — free to use for coursework, portfolios, and internship submissions.
