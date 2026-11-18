# Flood Susceptibility Demo Tool

[![tests](https://github.com/shannazeer2355/Flood-Susceptibility-Demo-Tool/actions/workflows/tests.yml/badge.svg)](https://github.com/shannazeer2355/Flood-Susceptibility-Demo-Tool/actions/workflows/tests.yml)
[![live data](https://github.com/shannazeer2355/Flood-Susceptibility-Demo-Tool/actions/workflows/live-data.yml/badge.svg)](https://github.com/shannazeer2355/Flood-Susceptibility-Demo-Tool/actions/workflows/live-data.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Live app](https://img.shields.io/badge/live%20app-Streamlit-ff4b4b)](https://floodsusceptibility-by-shan.streamlit.app)

A Python GIS tool that maps **flood susceptibility (High / Medium / Low)** for any area of
interest. Pick a Kerala district or upload a boundary, and every input layer (terrain,
rivers and water bodies, monsoon rainfall, land cover and soil) is fetched for that area
from open data services. The seven criteria are weighted with the **Analytic Hierarchy
Process (AHP)** and combined in a weighted overlay. No API keys are needed and no data is
stored in the repository. It runs as a command-line tool or as a
[Streamlit web app](https://floodsusceptibility-by-shan.streamlit.app).

Developed during a GIS internship at Avakara GeoScience Research Technologies (AGSRT),
Thiruvananthapuram, for flood susceptibility assessment in Kerala.

![Example output for Thiruvananthapuram district](docs/example_trivandrum.png)

*Thiruvananthapuram district, all seven criteria fetched from open data, default AHP weights.*

## Method

Each criterion is converted to a 0–1 score, where 1 means most susceptible:

| Criterion | Rule | Default AHP weight |
|---|---|---|
| **Elevation** | Lower ground is more susceptible. Log-scaled between the AOI's 1st and 99th percentiles, so a few metres matter on the coastal plain. | 0.278 |
| **Distance to water** | Score 1 within 500 m of a river, canal or water body (≥ 5 ha, or water ≥ 50 % of the time since 1984), falling linearly to 0 at 1500 m. | 0.278 |
| **Slope** | Flatter ground drains more slowly. Slope (Horn 1981) scores 1 at 0° and falls linearly to 0 at 15°. | 0.176 |
| **Monsoon rainfall** | Mean June–September precipitation, scaled to the AOI's 1st–99th percentile range. | 0.114 |
| **Drainage density** | Length of all mapped streams per km² within 1 km. | 0.073 |
| **Land cover** | Score per ESA WorldCover class: built-up 0.9, wetland 0.9, cropland 0.6, grassland 0.4, trees 0.1, … | 0.048 |
| **Soil** | Clay content of the topsoil (5–15 cm); clay-rich soils let less water infiltrate. | 0.033 |

### AHP weights

The weights are the principal eigenvector of a pairwise comparison matrix on Saaty's
1–9 scale (Saaty, 1980). The default judgements rank elevation and distance to water
first, then slope, rainfall, drainage density, land cover and soil; their consistency
ratio is **CR = 0.018** (acceptable below 0.10). In the web app every pairwise judgement
can be edited, and the weights and CR update as you go. If a layer cannot be fetched,
the weights are recomputed from the same judgements for the criteria that remain.

### Overlay and zones

```
score = Σ wᵢ · sᵢ / Σ wᵢ      (over the criteria with data at each pixel)
```

The score is split into High / Medium / Low with Jenks natural breaks fitted to the
AOI's own scores (or fixed cut-offs of 0.4 and 0.7). A 5 × 5 majority filter then
removes isolated pixels before the zones are converted to polygons.

### Implementation notes

- **Only the AOI is downloaded.** Rasters are read as windows from cloud-optimised
  GeoTIFFs over HTTP, using overviews when the source is finer than the 30 m grid.
  Rivers and water bodies come from a bounding-box query to the Overpass API.
- **Projection.** The AOI's UTM zone is detected automatically and every layer is
  reprojected to a 30 m grid in it (coarser for very large areas), so distances and
  areas are in metres.
- **Graceful degradation.** If a service is down, that criterion is left out, the app
  shows which layer is missing, and the remaining weights are renormalised.
- **Fetch once, score many times.** The app keeps the fetched layers, so changing
  weights or parameters re-runs only the scoring step.

## Data sources

| Layer | Dataset | Access |
|---|---|---|
| Elevation, slope | Copernicus DEM GLO-30 (ESA, 2021) | Microsoft Planetary Computer |
| Water bodies | JRC Global Surface Water 1984–2021 (Pekel et al., 2016) | Microsoft Planetary Computer |
| Rivers, canals, streams | OpenStreetMap (© OpenStreetMap contributors, ODbL) | Overpass API |
| Monsoon rainfall | CHELSA V2.1 monthly climatology 1981–2010 (Karger et al., 2017) | CHELSA, EnviDat |
| Land cover | ESA WorldCover 2021 v200 (Zanaga et al., 2022) | Microsoft Planetary Computer |
| Soil | SoilGrids 2.0 clay content (Poggio et al., 2021) | ISRIC |
| District boundaries | geoBoundaries gbOpen, India ADM2 (Runfola et al., 2020) | bundled, see `scripts/make_kerala_districts.py` |

- Karger, D. N. et al. (2017). Climatologies at high resolution for the earth's land surface areas. *Scientific Data*, 4, 170122.
- Pekel, J.-F. et al. (2016). High-resolution mapping of global surface water and its long-term changes. *Nature*, 540, 418–422.
- Poggio, L. et al. (2021). SoilGrids 2.0: producing soil information for the globe with quantified spatial uncertainty. *SOIL*, 7, 217–240.
- Runfola, D. et al. (2020). geoBoundaries: a global database of political administrative boundaries. *PLoS ONE*, 15(4), e0231866.
- Saaty, T. L. (1980). *The Analytic Hierarchy Process*. McGraw-Hill.
- Zanaga, D. et al. (2022). ESA WorldCover 10 m 2021 v200. Zenodo, doi:10.5281/zenodo.7254221.

## Quick start

```bash
pip install -r requirements.txt
```

`rasterio` and `geopandas` install from pip wheels on Linux and macOS. On Windows, if
pip fails, use `conda install -c conda-forge rasterio geopandas`.

### Command line

Any Kerala district, all layers fetched from open data:

```bash
python flood_tool.py --district Ernakulam --out-dir outputs
```

Any other area, from a boundary file:

```bash
python flood_tool.py --aoi my_area.geojson --out-dir outputs
```

Options for open data mode:

```
--criteria C [C ...]  Criteria to use (default: all): elevation river slope rainfall
                      drainage landcover soil
--resolution M        Grid resolution in metres (default 30, coarser for very large AOIs)
--classification      natural_breaks (default) or fixed (0.4 / 0.7)
--settlements PATH    Optional settlement points, counted per zone
--river-near M        Distance (m) within which the water score is 1   [500]
--river-far M         Distance (m) beyond which the water score is 0   [1500]
--max-slope DEG       Slope (deg) at or above which the slope score is 0 [15]
```

Your own layers can still be used instead (elevation, slope, distance to water and an
optional rainfall raster, with manual weights):

```bash
python flood_tool.py --aoi sample_data/boundary.shp --dem sample_data/dem.tif \
    --rivers sample_data/rivers.shp --settlements sample_data/villages.shp
```

### Web app

```bash
streamlit run streamlit_app.py
```

Choose a district or upload a boundary, select the criteria, adjust the AHP judgements
and press **Generate**. The app shows the zones on an interactive map, the area and
settlements per zone, each criterion's score map and the status of every data layer, and
offers all outputs for download.

## Outputs

| File | Description |
|---|---|
| `flood_susceptibility.tif` | Classified raster (1 = Low, 2 = Medium, 3 = High, 0 = NoData) |
| `flood_susceptibility_zones.geojson` | Zones dissolved into polygons, with area in km² |
| `summary.csv` | Area and share of the AOI per zone, plus settlements per zone |
| `flood_susceptibility_map.png` | Static map with projected coordinates, scale bar and legend |
| `criteria_scores.png` | Score map of each criterion, with its weight |
| `interactive_map.html` | Leaflet map on an OpenStreetMap basemap |

## Project structure

```
├── flood_tool.py              # Command-line entry point
├── streamlit_app.py           # Web app
├── flood_tool/
│   ├── sources.py             # Open data access (STAC, COG windows, Overpass)
│   ├── open_data.py           # Fetches the layer stack for an AOI and scores it
│   ├── ahp.py                 # AHP weights and consistency ratio
│   ├── pipeline.py            # Open data and local-file workflows
│   ├── io_utils.py            # Grids, loading, reprojection, clipping
│   ├── terrain.py             # Elevation and slope scores
│   ├── river_distance.py      # Distance-to-water score
│   ├── drainage.py            # Drainage density
│   ├── landcover.py           # Land-cover score
│   ├── rainfall.py            # Rainfall score (local files)
│   ├── scoring.py             # Percentile scaling
│   ├── overlay.py             # Weighted overlay, natural breaks, smoothing
│   ├── outputs.py             # GeoTIFF and GeoJSON writers
│   ├── reporting.py           # Area and settlement statistics
│   ├── visualize.py           # Static, criteria and interactive maps
│   ├── boundaries.py          # Kerala district lookup
│   └── data/                  # Kerala district boundaries
├── scripts/
│   ├── check_sources.py       # Fetches every layer and reports timing and statistics
│   └── make_kerala_districts.py
├── sample_data/               # Small synthetic dataset for tests
├── tests/                     # pytest suite
└── docs/                      # Figures used in this README
```

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

The tests run offline. They check each scoring function against known values (the slope
of a plane, distances on non-square pixels, the drainage density of a single stream),
AHP weights and consistency ratios, natural breaks, the per-pixel overlay, and the full
open-data pipeline with every remote layer replaced by synthetic data, including a
failed service. A second workflow, **live data**, fetches every layer for
Thiruvananthapuram from the real services each week and fails if any source is
unavailable.

## Limitations

- **Susceptibility, not hazard.** The map ranks where flooding is *more likely* from
  terrain, water, climate, land cover and soil. It does not give flood depth, return
  period or probability.
- **Expert judgements.** The AHP judgements and class boundaries are not yet calibrated
  against observed floods, so results depend on these choices.
- **Not yet validated.** The natural next step is to compare the zones with mapped flood
  extents, such as the August 2018 Kerala floods, and report accuracy (e.g. ROC/AUC).
- **Input data.** Copernicus DEM is a surface model, so tree canopy and buildings raise
  the elevation slightly. OpenStreetMap stream coverage varies between areas, which
  affects drainage density. WorldCover maps most Kerala homestead gardens as tree cover.
  SoilGrids has no values for built-up areas, and the rainfall climatology (~1 km)
  cannot resolve very local differences.

## Possible extensions

- Validate against a flood inventory or Sentinel-1 SAR flood extents.
- Learn the weights with a Random Forest trained on historical flood points and compare
  them with the AHP weights.
- Add taluk boundaries and a height-above-nearest-drainage (HAND) criterion.

## Citation

If you use this tool, please cite it using the metadata in [CITATION.cff](CITATION.cff).

## Author

**Mohammed Shan N**: [ORCID 0009-0007-5645-6313](https://orcid.org/0009-0007-5645-6313) ·
[LinkedIn](https://www.linkedin.com/in/mohammed-shan-n-5562051ab) · shangeography@gmail.com

## License

MIT License. See [LICENSE](LICENSE).
