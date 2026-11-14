"""Small geospatial helpers: coordinate transforms and patch geometry."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from pyproj import Transformer


@lru_cache(maxsize=32)
def get_transformer(src_crs: str, dst_crs: str) -> Transformer:
    return Transformer.from_crs(src_crs, dst_crs, always_xy=True)


def lonlat_to_xy(lon, lat, crs: str):
    """EPSG:4326 (lon, lat) -> projected (x, y) in `crs`."""
    x, y = get_transformer("EPSG:4326", crs).transform(lon, lat)
    return np.asarray(x), np.asarray(y)


def xy_to_lonlat(x, y, crs: str):
    lon, lat = get_transformer(crs, "EPSG:4326").transform(x, y)
    return np.asarray(lon), np.asarray(lat)


def reproject_xy(x, y, src_crs: str, dst_crs: str):
    xx, yy = get_transformer(src_crs, dst_crs).transform(x, y)
    return np.asarray(xx), np.asarray(yy)


def patch_bounds(x: float, y: float, size_px: int, res: float = 10.0, snap: float = 20.0):
    """Square bounds (left, bottom, right, top) centred on (x, y) in a projected CRS.

    The corners are snapped to a multiple of `snap` metres so the window lines up with
    both the 10 m and 20 m Sentinel-2 pixel grids. The centre error is at most snap/2.
    """
    extent = size_px * res
    half = extent / 2.0
    if snap and (extent % snap) == 0:
        left = round((x - half) / snap) * snap
        bottom = round((y - half) / snap) * snap
    else:
        left, bottom = x - half, y - half
    return (left, bottom, left + extent, bottom + extent)


def bounds_center(bounds):
    l, b, r, t = bounds
    return (l + r) / 2.0, (b + t) / 2.0
