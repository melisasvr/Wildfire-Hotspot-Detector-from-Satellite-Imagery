"""Windowed Sentinel-2 reads and patch quality control."""
from __future__ import annotations

import hashlib

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

from .geo import get_transformer, patch_bounds

SCL_CLOUD = (3, 8, 9, 10)  # cloud shadow, cloud medium, cloud high, thin cirrus
SCL_WATER = 6
GDAL_OPTS = dict(
    AWS_NO_SIGN_REQUEST="YES",
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
    GDAL_HTTP_MAX_RETRY="3",
    GDAL_HTTP_RETRY_DELAY="1",
    VSI_CACHE="TRUE",
)


def make_patch_id(scene_id: str, label: int, x: float, y: float) -> str:
    h = hashlib.sha1(f"{scene_id}|{label}|{round(x)}|{round(y)}".encode()).hexdigest()[:12]
    return f"P{h}"


def to_uint16(refl: np.ndarray) -> np.ndarray:
    """Reflectance (0..~1) -> uint16 storage (reflectance * 10000)."""
    return np.round(np.clip(refl, 0.0, 6.5535) * 10000.0).astype(np.uint16)


def from_uint16(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float32) / 10000.0


def window_inside(ds, bounds) -> bool:
    b = ds.bounds
    return bounds[0] >= b.left and bounds[1] >= b.bottom and bounds[2] <= b.right and bounds[3] <= b.top


def read_window(ds, bounds, size: int, resampling=Resampling.nearest) -> np.ndarray:
    """Read `bounds` from band 1 of `ds`, resampled to (size, size).

    Nearest-neighbour is deliberate: bilinear upsampling of the 20 m SWIR bands blurs a
    single hot pixel and cuts its peak roughly in half, which is exactly the signal we need.
    10 m bands are read at native resolution, so this only affects B11, B12 and SCL.
    """
    win = from_bounds(*bounds, transform=ds.transform)
    return ds.read(1, window=win, out_shape=(size, size), resampling=resampling)


def patch_qc(dn: np.ndarray, scl: np.ndarray) -> dict:
    """Quality stats for one patch. dn is (bands, H, W) raw digital numbers."""
    nodata = (dn == 0).any(axis=0) | (scl == 0)
    return {
        "nodata_frac": float(nodata.mean()),
        "cloud_frac": float(np.isin(scl, SCL_CLOUD).mean()),
        "water_frac": float((scl == SCL_WATER).mean()),
    }


def qc_pass(qc: dict, icfg: dict) -> bool:
    return (
        qc["nodata_frac"] <= icfg["max_nodata_frac"]
        and qc["cloud_frac"] <= icfg["max_patch_cloud_frac"]
        and qc["water_frac"] <= icfg["max_water_frac"]
    )


class SceneReader:
    """Open the bands of one scene once and read many small windows from it."""

    def __init__(self, scene: dict, bands, patch_size: int, res: float = 10.0):
        self.scene, self.bands, self.size, self.res = scene, list(bands), patch_size, res
        self._env, self.ds = None, {}

    def __enter__(self):
        self._env = rasterio.Env(**GDAL_OPTS)
        self._env.__enter__()
        self.ds = {k: rasterio.open(h) for k, h in self.scene["hrefs"].items()}
        self.crs = self.ds[self.bands[0]].crs.to_string()
        return self

    def __exit__(self, *exc):
        for d in self.ds.values():
            d.close()
        self._env.__exit__(*exc)

    def lonlat_to_scene_xy(self, lon, lat):
        x, y = get_transformer("EPSG:4326", self.crs).transform(lon, lat)
        return float(x), float(y)

    def read(self, lon: float, lat: float):
        """Return dict(refl, qc, bounds, crs) or None if the window leaves the scene."""
        x, y = self.lonlat_to_scene_xy(lon, lat)
        bounds = patch_bounds(x, y, self.size, self.res)
        if not all(window_inside(self.ds[k], bounds) for k in self.ds):
            return None
        dn = np.stack([read_window(self.ds[b], bounds, self.size) for b in self.bands]).astype(np.float32)
        scl = read_window(self.ds["SCL"], bounds, self.size, Resampling.nearest)
        qc = patch_qc(dn, scl)
        refl = np.clip(dn * self.scene["scale"] + self.scene["offset"], 0.0, None)
        refl[dn == 0] = 0.0
        return {"refl": refl.astype(np.float32), "qc": qc, "bounds": bounds, "crs": self.crs}
