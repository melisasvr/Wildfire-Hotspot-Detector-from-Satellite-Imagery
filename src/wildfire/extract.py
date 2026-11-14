"""Turn matched FIRMS detections into positive and negative Sentinel-2 patches."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .geo import reproject_xy, xy_to_lonlat, lonlat_to_xy
from .patches import SceneReader, make_patch_id, qc_pass, to_uint16
from .sampling import far_from, propose_negative_centres, thin_points


def _record(pid, label, scene, read, x_a, y_a, lon, lat, anchor=None, gap_h=np.nan):
    rec = {
        "patch_id": pid, "patch_file": f"{pid}.npy", "label": int(label),
        "scene_id": scene["scene_id"], "scene_datetime": scene["datetime"],
        "scene_cloud": scene["cloud_cover"], "time_gap_h": gap_h,
        "lon": float(lon), "lat": float(lat), "x_a": float(x_a), "y_a": float(y_a),
        "scene_crs": read["crs"], "bounds": list(map(float, read["bounds"])),
        **read["qc"],
        "firms_confidence": None, "firms_conf_ord": np.nan, "frp": np.nan,
        "daynight": None, "satellite": None, "firms_datetime": None, "anchor_id": None,
    }
    if anchor is not None:
        rec.update({
            "firms_confidence": anchor["confidence"], "firms_conf_ord": anchor["conf_ord"],
            "frp": anchor["frp"], "daynight": anchor.get("daynight"),
            "satellite": anchor.get("satellite"), "firms_datetime": anchor["acq_datetime"].isoformat(),
        })
    return rec


def extract_patches(filtered: pd.DataFrame, all_det: pd.DataFrame, matches: pd.DataFrame,
                    scenes: dict, cfg: dict, patches_dir, progress=None) -> pd.DataFrame:
    """Read patches for positives (centred on detections) and matched negatives.

    Negatives come from the SAME scene, are at least `neg_exclusion_m` from ANY FIRMS
    detection (any confidence, +/- 1 day), and never overlap other sampled patches.
    "No detection nearby" is not proof of "no fire", so negatives are weak labels too.
    """
    icfg, scfg, acrs = cfg["imagery"], cfg["sampling"], cfg["region"]["analysis_crs"]
    rng = np.random.default_rng(scfg["seed"])
    size, res = icfg["patch_size"], icfg["resolution_m"]

    f = filtered.copy()
    f["x_a"], f["y_a"] = lonlat_to_xy(f.longitude.values, f.latitude.values, acrs)
    a = all_det.copy()
    a["x_a"], a["y_a"] = lonlat_to_xy(a.longitude.values, a.latitude.values, acrs)
    a["t"] = pd.to_datetime(a["acq_datetime"], utc=True)

    m = matches.merge(f.reset_index().rename(columns={"index": "det_index"}), on="det_index", how="left", suffixes=("", "_det"))
    scene_ids = m["scene_id"].unique().tolist()
    rng.shuffle(scene_ids)
    it = progress(scene_ids) if progress else scene_ids

    records, n_pos = [], 0
    for sid in it:
        if n_pos >= scfg["max_positives"]:
            break
        g = m[m.scene_id == sid].copy()
        scene = scenes[sid]
        keep = thin_points(g[["x_a", "y_a"]].values, scfg["min_sample_spacing_m"],
                           priority=(g["conf_ord"].fillna(0) * 1e6 + g["frp"].fillna(0)).values)
        g = g.iloc[keep].head(scfg["max_positives_per_scene"])

        s_time = pd.Timestamp(scene["datetime"])
        near = a[(a.t - s_time).abs() <= pd.Timedelta(days=1)]
        tree = cKDTree(near[["x_a", "y_a"]].values) if len(near) else None
        taken = []  # centres (analysis CRS) of every patch accepted from this scene

        try:
            reader_ctx = SceneReader(scene, icfg["bands"], size, res)
            reader_ctx.__enter__()
        except Exception as e:
            print(f"[extract] cannot open {sid}: {e}")
            continue
        try:
            for _, r in g.iterrows():
                if n_pos >= scfg["max_positives"]:
                    break
                try:
                    read = reader_ctx.read(r.longitude, r.latitude)
                except Exception as e:
                    print(f"[extract] read failed in {sid}: {e}")
                    continue
                if read is None or not qc_pass(read["qc"], icfg):
                    continue
                pid = make_patch_id(sid, 1, r.x_a, r.y_a)
                np.save(f"{patches_dir}/{pid}.npy", to_uint16(read["refl"]))
                records.append(_record(pid, 1, scene, read, r.x_a, r.y_a, r.longitude, r.latitude,
                                       anchor=r, gap_h=r.time_gap_h))
                taken.append([r.x_a, r.y_a]); n_pos += 1

                got = 0
                for _ in range(scfg["neg_attempts"]):
                    if got >= scfg["negatives_per_positive"]:
                        break
                    c = propose_negative_centres([r.x_a, r.y_a], rng, 1, scfg["neg_min_dist_m"], scfg["neg_max_dist_m"])
                    if not far_from(c, tree, scfg["neg_exclusion_m"])[0]:
                        continue
                    if taken and np.min(np.linalg.norm(np.asarray(taken) - c[0], axis=1)) < scfg["min_sample_spacing_m"]:
                        continue
                    lon, lat = xy_to_lonlat(c[0, 0], c[0, 1], acrs)
                    try:
                        nread = reader_ctx.read(float(lon), float(lat))
                    except Exception:
                        continue
                    if nread is None or not qc_pass(nread["qc"], icfg):
                        continue
                    pid_n = make_patch_id(sid, 0, c[0, 0], c[0, 1])
                    np.save(f"{patches_dir}/{pid_n}.npy", to_uint16(nread["refl"]))
                    rec = _record(pid_n, 0, scene, nread, c[0, 0], c[0, 1], lon, lat, gap_h=r.time_gap_h)
                    rec["anchor_id"] = pid
                    records.append(rec)
                    taken.append([c[0, 0], c[0, 1]]); got += 1
        finally:
            reader_ctx.__exit__(None, None, None)
    return pd.DataFrame(records)
