"""NASA FIRMS VIIRS 375 m active-fire labels: download, clean, filter.

FIRMS detections are thermal anomalies, not verified wildfire ground truth.
Everything here treats them as weak labels and logs every filtering step.
"""
from __future__ import annotations

import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

FIRMS_AREA_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
CONF_ORDER = {"l": 0, "low": 0, "n": 1, "nominal": 1, "h": 2, "high": 2}
MAX_DAY_RANGE = 5  # FIRMS area API allows 1 to 5 days per request


def _fetch_chunk(session, map_key, source, bbox, start_date, day_range, retries=3):
    west, south, east, north = bbox
    url = f"{FIRMS_AREA_URL}/{map_key}/{source}/{west},{south},{east},{north}/{day_range}/{start_date}"
    last = ""
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=120)
            text = r.text.strip()
            if r.status_code == 200 and text.startswith("latitude"):
                return pd.read_csv(io.StringIO(text))
            if r.status_code == 200 and not text:
                return pd.DataFrame()
            last = f"HTTP {r.status_code}: {text[:200]}"
        except requests.RequestException as e:
            last = str(e)
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"FIRMS request failed for {source} {start_date}: {last}")


def download_firms(map_key: str, sources, bbox, start_date: str, end_date: str) -> pd.DataFrame:
    """Download archive detections in 5-day chunks. Returns a raw (unfiltered) DataFrame."""
    session = requests.Session()
    starts = pd.date_range(start_date, end_date, freq=f"{MAX_DAY_RANGE}D")
    end = pd.Timestamp(end_date)
    frames = []
    for source in sources:
        for s in starts:
            day_range = int(min(MAX_DAY_RANGE, (end - s).days + 1))
            df = _fetch_chunk(session, map_key, source, bbox, s.strftime("%Y-%m-%d"), day_range)
            if len(df):
                df["source"] = source
                frames.append(df)
            time.sleep(0.2)
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    return raw.drop_duplicates(subset=["source", "latitude", "longitude", "acq_date", "acq_time"]).reset_index(drop=True)


def parse_detections(raw: pd.DataFrame) -> pd.DataFrame:
    """Add acquisition datetime (UTC) and ordinal confidence."""
    df = raw.copy()
    hhmm = df["acq_time"].astype(int).astype(str).str.zfill(4)
    df["acq_datetime"] = pd.to_datetime(
        df["acq_date"].astype(str) + " " + hhmm.str[:2] + ":" + hhmm.str[2:], utc=True
    )
    df["confidence"] = df["confidence"].astype(str).str.lower()
    df["conf_ord"] = df["confidence"].map(CONF_ORDER)
    return df


def flag_persistent_sources(df: pd.DataFrame, cell_deg: float, min_days: int) -> pd.Series:
    """True for detections in grid cells that fire on many distinct days.

    This is a crude proxy for gas flares, industrial plants, and similar static heat
    sources. It can also catch a wildfire that sits on one pixel for a long time.
    """
    cx = np.floor(df["longitude"] / cell_deg).astype(int).astype(str)
    cy = np.floor(df["latitude"] / cell_deg).astype(int).astype(str)
    key = cx + "_" + cy
    ndays = df.groupby(key)["acq_date"].transform("nunique")
    return ndays >= min_days


def filter_detections(df: pd.DataFrame, fcfg: dict, bbox=None):
    """Apply transparent quality filters. Returns (filtered_df, log_list)."""
    log = [{"step": "start", "n": int(len(df))}]
    out = df.dropna(subset=["latitude", "longitude", "acq_datetime"])
    log.append({"step": "drop_missing_coords_or_time", "n": int(len(out))})

    if bbox is not None:
        w, s, e, n = bbox
        out = out[out.longitude.between(w, e) & out.latitude.between(s, n)]
        log.append({"step": "inside_bbox", "n": int(len(out))})

    min_conf = CONF_ORDER[str(fcfg.get("min_confidence", "nominal")).lower()]
    out = out[out["conf_ord"].fillna(-1) >= min_conf]
    log.append({"step": f"confidence>={fcfg.get('min_confidence', 'nominal')}", "n": int(len(out))})

    out = out[np.isfinite(out["frp"]) & (out["frp"] > float(fcfg.get("min_frp_mw", 0.0)))]
    log.append({"step": "frp_positive_and_finite", "n": int(len(out))})

    out = out.copy()
    if fcfg.get("drop_persistent_sources", True):
        flag = flag_persistent_sources(out, fcfg["persistent_cell_deg"], fcfg["persistent_min_days"])
        out["persistent_source"] = flag.values
        out = out[~out["persistent_source"]]
        log.append({"step": "drop_persistent_sources", "n": int(len(out))})
    return out.reset_index(drop=True), log


def to_geodataframe(df: pd.DataFrame) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")


def save_labels(raw_all: pd.DataFrame, filtered: pd.DataFrame, log, out_dir: Path, meta: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    to_geodataframe(raw_all).to_parquet(out_dir / "firms_all.parquet")
    to_geodataframe(filtered).to_parquet(out_dir / "firms_filtered.parquet")
    meta = dict(meta)
    meta.update(
        {
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "filter_log": log,
            "columns_preserved": list(raw_all.columns),
            "note": "FIRMS detections are thermal anomalies used as WEAK labels, not verified wildfire ground truth.",
        }
    )
    with open(out_dir / "firms_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
