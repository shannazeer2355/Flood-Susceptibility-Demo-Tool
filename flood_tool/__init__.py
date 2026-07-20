"""
flood_tool — A lightweight GIS pipeline for flood-risk mapping.

Modules:
    io_utils        Reading/clipping/writing raster & vector data
    terrain         Elevation and slope risk scoring from a DEM
    river_distance  Euclidean distance-to-river risk scoring
    overlay         Weighted overlay + risk classification
    reporting       Summary statistics (area %, villages affected)
    visualize       Static (matplotlib) and interactive (folium) maps
"""

__version__ = "1.0.0"
