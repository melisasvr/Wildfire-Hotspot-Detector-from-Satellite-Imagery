import numpy as np
import pandas as pd

from conftest import add_hot_pixel
from wildfire.extract import extract_patches
from wildfire.geo import lonlat_to_xy, xy_to_lonlat


def test_extract_labels_alignment_and_negative_rules(cfg, fake_scene):
    acrs = cfg["region"]["analysis_crs"]
    cfg["imagery"]["patch_size"] = 64
    cfg["sampling"].update(neg_min_dist_m=1000, neg_max_dist_m=2500, neg_exclusion_m=800,
                           min_sample_spacing_m=700, negatives_per_positive=2, max_positives=10)
    cx, cy = fake_scene["center"]
    pts = [(cx - 1200, cy), (cx, cy), (cx + 1200, cy)]
    rows = []
    for x, y in pts:
        add_hot_pixel(fake_scene, x, y)
        lon, lat = xy_to_lonlat(x, y, fake_scene["crs"])
        rows.append({"longitude": float(lon), "latitude": float(lat),
                     "acq_datetime": pd.Timestamp("2024-07-10T21:30:00", tz="UTC"),
                     "confidence": "h", "conf_ord": 2, "frp": 20.0, "daynight": "D", "satellite": "N"})
    det = pd.DataFrame(rows)
    matches = pd.DataFrame({"det_index": [0, 1, 2], "scene_id": "FAKE_SCENE", "time_gap_h": 2.7})
    pdir = cfg["paths"]["patches"]; import os; os.makedirs(pdir, exist_ok=True)

    m = extract_patches(det, det, matches, {"FAKE_SCENE": fake_scene["scene"]}, cfg, pdir)
    pos, neg = m[m.label == 1], m[m.label == 0]
    assert len(pos) == 3 and len(neg) >= 1

    # positives are centred on their detection (<= 10 m in the scene CRS)
    dxy = np.array(lonlat_to_xy(pos.lon.values, pos.lat.values, fake_scene["crs"])).T
    bc = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in pos.bounds])
    assert np.abs(dxy - bc).max() <= 10.0 + 1e-6

    # negatives keep their distance from every FIRMS detection and from each other
    det_xy = np.array(lonlat_to_xy(det.longitude.values, det.latitude.values, acrs)).T
    nxy = neg[["x_a", "y_a"]].values
    d = np.linalg.norm(nxy[:, None, :] - det_xy[None, :, :], axis=2)
    assert d.min() > cfg["sampling"]["neg_exclusion_m"]
    allxy = m[["x_a", "y_a"]].values
    dd = np.linalg.norm(allxy[:, None] - allxy[None], axis=2) + np.eye(len(m)) * 1e9
    assert dd.min() >= cfg["sampling"]["min_sample_spacing_m"] - 1e-6
    assert set(neg.anchor_id) <= set(pos.patch_id)
    assert (pos.firms_confidence == "h").all() and neg.firms_confidence.isna().all()
