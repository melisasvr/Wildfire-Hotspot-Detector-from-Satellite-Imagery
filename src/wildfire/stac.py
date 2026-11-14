"""Find Sentinel-2 L2A scenes close in time to FIRMS detections via a STAC API.

Default catalog is Earth Search (public COGs of Copernicus Sentinel-2 L2A, no auth),
which makes small windowed reads cheap. Copernicus Data Space also hosts the same
mission data; swap `imagery.stac_url` if you prefer, but its assets need credentials.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pystac_client import Client
from shapely.geometry import Point, shape

# Sentinel-2 band -> Earth Search asset key
ASSET_KEYS = {
    "B02": "blue", "B03": "green", "B04": "red",
    "B08": "nir", "B11": "swir16", "B12": "swir22",
}
SCL_KEY = "scl"


def open_catalog(url: str) -> Client:
    return Client.open(url)


def reflectance_offset(item) -> float:
    """Additive offset to apply after DN * 0.0001.

    Prefer the raster extension on the asset; otherwise infer from processing baseline
    (baseline >= 04.00 uses BOA_ADD_OFFSET = -1000 DN, i.e. -0.1).
    """
    for key in ("red", "nir", "blue"):
        a = item.assets.get(key)
        if a is None:
            continue
        rb = a.extra_fields.get("raster:bands")
        if rb and "offset" in rb[0]:
            return float(rb[0]["offset"])
    baseline = item.properties.get("s2:processing_baseline")
    try:
        return -0.1 if float(baseline) >= 4.0 else 0.0
    except (TypeError, ValueError):
        return -0.1


def scene_record(item, bands) -> dict:
    hrefs = {b: item.assets[ASSET_KEYS[b]].href for b in bands}
    hrefs["SCL"] = item.assets[SCL_KEY].href
    return {
        "scene_id": item.id,
        "datetime": pd.Timestamp(item.datetime).isoformat(),
        "cloud_cover": float(item.properties.get("eo:cloud_cover", np.nan)),
        "offset": reflectance_offset(item),
        "scale": 0.0001,
        "hrefs": hrefs,
    }


def search_items(catalog, collection, bbox, start, end, max_cloud):
    search = catalog.search(
        collections=[collection],
        bbox=list(bbox),
        datetime=[start.to_pydatetime(), end.to_pydatetime()],
        query={"eo:cloud_cover": {"lt": float(max_cloud)}},
        max_items=500,
    )
    return list(search.items())


def match_detections_to_scenes(det: pd.DataFrame, catalog, icfg: dict, progress=None):
    """For each detection, pick the closest-in-time scene that contains it.

    Returns (matches_df, scenes_dict). `time_gap_h` = FIRMS time minus Sentinel-2 time.
    """
    gap = pd.Timedelta(hours=float(icfg["max_time_gap_hours"]))
    bands = icfg["bands"]
    det = det.copy()
    det["day"] = det["acq_datetime"].dt.date
    scenes, rows = {}, []
    groups = list(det.groupby("day"))
    it = progress(groups) if progress else groups
    for _, g in it:
        t0, t1 = g.acq_datetime.min() - gap, g.acq_datetime.max() + gap
        pad = 0.05
        bbox = (g.longitude.min() - pad, g.latitude.min() - pad, g.longitude.max() + pad, g.latitude.max() + pad)
        try:
            items = search_items(catalog, icfg["collection"], bbox, t0, t1, icfg["max_scene_cloud"])
        except Exception as e:  # network hiccup: skip that day, keep going
            print(f"[stac] search failed for {g.iloc[0]['day']}: {e}")
            continue
        items = [i for i in items if SCL_KEY in i.assets and all(ASSET_KEYS[b] in i.assets for b in bands)]
        if not items:
            continue
        geoms = [shape(i.geometry) for i in items]
        times = [pd.Timestamp(i.datetime) for i in items]
        for idx, r in g.iterrows():
            pt = Point(r.longitude, r.latitude)
            best = None
            for item, geom, t in zip(items, geoms, times):
                if not geom.contains(pt):
                    continue
                gap_h = (r.acq_datetime - t).total_seconds() / 3600.0
                if abs(gap_h) > icfg["max_time_gap_hours"]:
                    continue
                key = (abs(gap_h), item.properties.get("eo:cloud_cover", 100.0))
                if best is None or key < best[0]:
                    best = (key, item, gap_h)
            if best is None:
                continue
            _, item, gap_h = best
            if item.id not in scenes:
                scenes[item.id] = scene_record(item, bands)
            rows.append({"det_index": idx, "scene_id": item.id, "time_gap_h": gap_h})
    return pd.DataFrame(rows), scenes
