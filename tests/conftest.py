"""Shared synthetic fixtures. Nothing here touches the network or real imagery."""
import copy

import numpy as np
import pandas as pd
import pytest
import rasterio
import yaml
from pathlib import Path
from rasterio.transform import from_origin

from wildfire.geo import lonlat_to_xy
from wildfire.patches import to_uint16

ROOT = Path(__file__).resolve().parents[1]
BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]


@pytest.fixture()
def cfg(tmp_path):
    c = yaml.safe_load(open(ROOT / "configs" / "default.yaml"))
    c = copy.deepcopy(c)
    for k in c["paths"]:
        c["paths"][k] = str(tmp_path / k)
    c["imagery"]["patch_size"] = 32
    c["train"].update(epochs=2, batch_size=16, patience=2)
    c["model"].update(arch="small_cnn")
    c["evaluate"]["n_boot"] = 100
    return c


def make_patch(label: int, rng, size=32, hard_negative=False):
    refl = np.clip(rng.normal([0.05, 0.08, 0.06, 0.30, 0.20, 0.12], 0.02, (size, size, 6)), 0, 1).transpose(2, 0, 1)
    if hard_negative:  # bright bare soil: B12 ~ B11, no hot spike
        refl[4] += 0.25; refl[5] += 0.22
    if label == 1:
        c = size // 2
        refl[4, c:c + 2, c:c + 2] = 0.55
        refl[5, c:c + 2, c:c + 2] = 0.85
    return refl.astype(np.float32)


@pytest.fixture()
def synthetic_raw(cfg):
    """A raw patch manifest + patches on disk, spread over a 300 x 300 km area."""
    rng = np.random.default_rng(0)
    pdir = Path(cfg["paths"]["patches"]); pdir.mkdir(parents=True, exist_ok=True)
    rows = []
    n = 300
    for i in range(n):
        label = int(i % 3 == 0)
        x, y = rng.uniform(0, 300_000, 2)
        refl = make_patch(label, rng, hard_negative=(label == 0 and rng.random() < 0.3))
        pid = f"P{i:05d}"
        np.save(pdir / f"{pid}.npy", to_uint16(refl))
        t = pd.Timestamp("2024-06-01", tz="UTC") + pd.Timedelta(days=int(rng.integers(0, 150)), hours=18)
        rows.append({
            "patch_id": pid, "patch_file": f"{pid}.npy", "label": label, "scene_id": f"S{i % 40}",
            "scene_datetime": t.isoformat(), "scene_cloud": float(rng.uniform(0, 30)),
            "time_gap_h": float(rng.uniform(1, 5)), "lon": -121 + x / 1e5, "lat": 39 + y / 1e5,
            "x_a": float(x), "y_a": float(y), "scene_crs": "EPSG:32610", "bounds": [0.0, 0.0, 1.0, 1.0],
            "nodata_frac": 0.0, "cloud_frac": float(rng.uniform(0, 0.3)), "water_frac": 0.0,
            "firms_confidence": ("h" if rng.random() < .4 else "n") if label else None,
            "firms_conf_ord": (2.0 if rng.random() < .4 else 1.0) if label else np.nan,
            "frp": float(rng.uniform(1, 80)) if label else np.nan,
            "daynight": "D" if label else None, "satellite": "N" if label else None,
            "firms_datetime": t.isoformat() if label else None, "anchor_id": None,
        })
    return pd.DataFrame(rows)


def write_tif(path, arr, crs, transform):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype=arr.dtype, crs=crs, transform=transform) as dst:
        dst.write(arr, 1)


@pytest.fixture()
def fake_scene(tmp_path):
    """A 6 km x 6 km fake Sentinel-2 scene in UTM 10N (10 m bands, 20 m SWIR/SCL) saved as GeoTIFFs."""
    crs = "EPSG:32610"
    cx, cy = lonlat_to_xy(-121.5, 40.0, crs)
    ox = round(float(cx) / 20) * 20 - 3000
    oy = round(float(cy) / 20) * 20 + 3000
    t10, t20 = from_origin(ox, oy, 10, 10), from_origin(ox, oy, 20, 20)
    hrefs = {}
    for b in BANDS:
        res20 = b in ("B11", "B12")
        arr = np.full((300, 300) if res20 else (600, 600), 1000, dtype=np.uint16)
        p = tmp_path / f"{b}.tif"
        write_tif(p, arr, crs, t20 if res20 else t10)
        hrefs[b] = str(p)
    scl = tmp_path / "SCL.tif"
    write_tif(scl, np.full((300, 300), 4, dtype=np.uint8), crs, t20)
    hrefs["SCL"] = str(scl)
    scene = {"scene_id": "FAKE_SCENE", "datetime": "2024-07-10T18:50:00+00:00", "cloud_cover": 3.0,
             "offset": 0.0, "scale": 0.0001, "hrefs": hrefs}
    return {"scene": scene, "crs": crs, "origin": (ox, oy), "center": (float(cx), float(cy)), "dir": tmp_path}


def add_hot_pixel(scene_info, x, y, b12=8000, b11=5000):
    """Paint a hot 20 m cell (SWIR) at projected coordinate (x, y)."""
    ox, oy = scene_info["origin"]
    col, row = int((x - ox) // 20), int((oy - y) // 20)
    for band, val in (("B12", b12), ("B11", b11)):
        path = scene_info["scene"]["hrefs"][band]
        with rasterio.open(path, "r+") as ds:
            a = ds.read(1); a[row, col] = val; ds.write(a, 1)