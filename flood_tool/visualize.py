"""
visualize.py
------------
Static (matplotlib) and interactive (folium) visualisation.
"""
from __future__ import annotations

import folium
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap, BoundaryNorm
from rasterio.warp import transform_bounds

from .overlay import ZONE_COLORS

_CMAP = ListedColormap(["#ffffff00", ZONE_COLORS[1], ZONE_COLORS[2], ZONE_COLORS[3]])
_NORM = BoundaryNorm([0, 1, 2, 3, 4], _CMAP.N)


def plot_static_map(zones: np.ndarray, grid, out_path: str, title: str = "Flood Susceptibility Zones"):
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(zones, cmap=_CMAP, norm=_NORM, interpolation="nearest")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Column"); ax.set_ylabel("Row")

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=ZONE_COLORS[1], label="Low Risk"),
        Patch(facecolor=ZONE_COLORS[2], label="Medium Risk"),
        Patch(facecolor=ZONE_COLORS[3], label="High Risk"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def build_interactive_map(zones_tif_path: str, out_html: str, aoi=None):
    with rasterio.open(zones_tif_path) as src:
        data = src.read(1)
        bounds_4326 = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        west, south, east, north = bounds_4326
        center = [(south + north) / 2, (west + east) / 2]

    rgba = np.zeros((*data.shape, 4), dtype="uint8")
    color_map = {
        1: (46, 204, 113, 160),
        2: (241, 196, 15, 160),
        3: (231, 76, 60, 180),
    }
    for zone_id, rgba_val in color_map.items():
        rgba[data == zone_id] = rgba_val

    m = folium.Map(location=center, zoom_start=11, tiles="OpenStreetMap")
    folium.raster_layers.ImageOverlay(
        image=rgba,
        bounds=[[south, west], [north, east]],
        opacity=0.75,
        name="Flood Susceptibility Zones",
    ).add_to(m)

    if aoi is not None:
        aoi_wgs = aoi.to_crs(4326)
        folium.GeoJson(
            aoi_wgs.__geo_interface__,
            name="AOI Boundary",
            style_function=lambda x: {"color": "black", "weight": 2, "fillOpacity": 0},
        ).add_to(m)

    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index:9999;
                background:white; padding:10px; border:2px solid grey; border-radius:6px;
                font-size:14px;">
      <b>Flood Susceptibility</b><br>
      <span style="color:#e74c3c;">&#9632;</span> High<br>
      <span style="color:#f1c40f;">&#9632;</span> Medium<br>
      <span style="color:#2ecc71;">&#9632;</span> Low
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(m)
    m.save(out_html)
