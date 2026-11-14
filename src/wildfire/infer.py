"""Inference helpers shared by the Streamlit demo and scripts."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .models import build_model
from .patches import SceneReader, from_uint16
from .preprocess import BANDS, build_channels, normalize
from .stac import ASSET_KEYS, SCL_KEY, open_catalog, scene_record, search_items


def load_cnn(path: str | Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ck["model_cfg"], ck["in_ch"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def to_tensor(refl: np.ndarray, stats: dict) -> torch.Tensor:
    x = normalize(build_channels(refl.astype(np.float32), stats.get("indices", [])), stats)
    return torch.from_numpy(x)[None]


@torch.no_grad()
def predict_patch(model, ck, refl: np.ndarray) -> float:
    return float(torch.sigmoid(model(to_tensor(refl, ck["stats"]))).item())


def saliency_map(model, ck, refl: np.ndarray) -> np.ndarray:
    """Input-gradient magnitude, summed over channels, scaled to [0, 1].

    A rough attribution aid, not proof of what the model "understands".
    """
    x = to_tensor(refl, ck["stats"]).requires_grad_(True)
    model.zero_grad()
    model(x).sum().backward()
    s = x.grad.abs().sum(1)[0].numpy()
    s = s - s.min()
    return s / (s.max() + 1e-9)


def load_patch_file(name: str, data: bytes) -> np.ndarray:
    """Read an uploaded patch as float reflectance (6, H, W) in order B02,B03,B04,B08,B11,B12.

    .npy / .npz: uint16 reflectance*10000 (as stored by this repo) or float reflectance.
    .tif: six bands, reflectance 0..1 or scaled by 10000 (auto-detected).
    """
    import io
    name = name.lower()
    if name.endswith(".npy"):
        arr = np.load(io.BytesIO(data))
    elif name.endswith(".npz"):
        z = np.load(io.BytesIO(data)); arr = z["refl"] if "refl" in z else z[z.files[0]]
    elif name.endswith((".tif", ".tiff")):
        import rioxarray
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".tif") as f:
            f.write(data); f.flush()
            arr = rioxarray.open_rasterio(f.name).values
    else:
        raise ValueError("Use .npy, .npz, or .tif")
    if arr.ndim != 3 or arr.shape[0] != len(BANDS):
        raise ValueError(f"Expected shape (6, H, W) with bands {BANDS}, got {arr.shape}")
    if arr.dtype == np.uint16 or float(np.nanmax(arr)) > 10:
        return from_uint16(arr.astype(np.uint16)) if arr.dtype == np.uint16 else (arr / 10000.0).astype(np.float32)
    return arr.astype(np.float32)


def fetch_patch_live(lat: float, lon: float, date: str, cfg: dict, max_cloud: float = 60.0, window_days: int = 1):
    """Download one Sentinel-2 patch around (lat, lon) near `date` from the STAC catalog."""
    from shapely.geometry import Point, shape
    ic = cfg["imagery"]
    d = pd.Timestamp(date, tz="UTC")
    start, end = d - pd.Timedelta(days=window_days), d + pd.Timedelta(days=window_days + 1)
    cat = open_catalog(ic["stac_url"])
    items = search_items(cat, ic["collection"], (lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01), start, end, max_cloud)
    items = [i for i in items if SCL_KEY in i.assets and all(ASSET_KEYS[b] in i.assets for b in ic["bands"])
             and shape(i.geometry).contains(Point(lon, lat))]
    if not items:
        raise LookupError("No Sentinel-2 scene found for that place and date (try a wider date or higher cloud limit).")
    items.sort(key=lambda i: (abs((pd.Timestamp(i.datetime) - d).total_seconds()), i.properties.get("eo:cloud_cover", 100)))
    scene = scene_record(items[0], ic["bands"])
    with SceneReader(scene, ic["bands"], ic["patch_size"], ic["resolution_m"]) as r:
        read = r.read(lon, lat)
    if read is None:
        raise LookupError("The patch window falls outside the scene footprint.")
    meta = {"scene_id": scene["scene_id"], "datetime": scene["datetime"], "scene_cloud": scene["cloud_cover"], **read["qc"]}
    return read["refl"], meta
