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
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds

from .overlay import ZONE_COLORS, ZONE_LABELS

_CMAP = ListedColormap(["#ffffff00", ZONE_COLORS[1], ZONE_COLORS[2], ZONE_COLORS[3]])
_NORM = BoundaryNorm([0, 1, 2, 3, 4], _CMAP.N)


def _crs_label(crs) -> str:
    try:
        epsg = rasterio.crs.CRS.from_user_input(crs).to_epsg()
        return f"EPSG:{epsg}" if epsg else str(crs)
    except Exception:
        return str(crs)


def plot_static_map(zones: np.ndarray, grid, out_path: str, title: str = "Flood Susceptibility Zones"):
    """Static map in projected coordinates (km)."""
    t = grid.transform
    left, top = t.c, t.f
    right = left + t.a * grid.width
    bottom = top + t.e * grid.height
    extent_km = [left / 1000, right / 1000, bottom / 1000, top / 1000]

    aspect = (extent_km[3] - extent_km[2]) / max(extent_km[1] - extent_km[0], 1e-9)
    fig, ax = plt.subplots(figsize=(9, float(np.clip(8 * aspect + 1.2, 4, 12))))
    ax.imshow(zones, cmap=_CMAP, norm=_NORM, interpolation="nearest", extent=extent_km)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(f"Easting (km), {_crs_label(grid.crs)}")
    ax.set_ylabel("Northing (km)")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.set_aspect("equal")

    width_km = extent_km[1] - extent_km[0]
    nice = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    bar_km = min(nice, key=lambda v: abs(v - width_km / 5))
    x0 = extent_km[0] + 0.05 * width_km
    y0 = extent_km[2] + 0.05 * (extent_km[3] - extent_km[2])
    ax.plot([x0, x0 + bar_km], [y0, y0], color="black", linewidth=3)
    ax.annotate(f"{bar_km} km", xy=(x0 + bar_km / 2, y0), xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=9)

    ax.annotate("N", xy=(0.95, 0.95), xytext=(0.95, 0.85), xycoords="axes fraction",
                ha="center", va="center", fontsize=12, fontweight="bold",
                arrowprops=dict(facecolor="black", width=3, headwidth=10))

    legend_handles = [Patch(facecolor=ZONE_COLORS[z], label=ZONE_LABELS[z]) for z in (3, 2, 1)]
    ax.legend(handles=legend_handles, title="Susceptibility", loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _reproject_to_web_mercator(src) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Resample zones to EPSG:3857 so the overlay matches web map tiles."""
    dst_crs = "EPSG:3857"
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds
    )
    data = np.zeros((height, width), dtype="uint8")
    reproject(
        source=rasterio.band(src, 1),
        destination=data,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=transform,
        dst_crs=dst_crs,
        src_nodata=0,
        dst_nodata=0,
        resampling=Resampling.nearest,
    )
    left, top = transform.c, transform.f
    right = left + transform.a * width
    bottom = top + transform.e * height
    west, south, east, north = transform_bounds(dst_crs, "EPSG:4326", left, bottom, right, top)
    return data, (west, south, east, north)


def build_interactive_map(zones_tif_path: str, out_html: str, aoi=None):
    with rasterio.open(zones_tif_path) as src:
        data, (west, south, east, north) = _reproject_to_web_mercator(src)
    center = [(south + north) / 2, (west + east) / 2]

    rgba = np.zeros((*data.shape, 4), dtype="uint8")
    color_map = {
        1: (46, 204, 113, 160),
        2: (241, 196, 15, 160),
        3: (231, 76, 60, 180),
    }
    for zone_id, rgba_val in color_map.items():
        rgba[data == zone_id] = rgba_val

    m = folium.Map(location=center, tiles="OpenStreetMap")
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

    m.fit_bounds([[south, west], [north, east]])

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
