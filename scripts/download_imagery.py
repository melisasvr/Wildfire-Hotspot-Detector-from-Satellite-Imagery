"""Stage 2: match detections to Sentinel-2 scenes (STAC) and extract positive/negative patches.

Only small windows are read from cloud-hosted COGs, so nothing large lands on disk.
"""
import json

import geopandas as gpd
import pandas as pd
from tqdm import tqdm

from _common import parse

from wildfire.config import path_of
from wildfire.extract import extract_patches
from wildfire.stac import match_detections_to_scenes, open_catalog


def main():
    _, cfg = parse(__doc__)
    raw_dir, interim, patches = path_of(cfg, "raw") / "firms", path_of(cfg, "interim"), path_of(cfg, "patches")
    filt = pd.DataFrame(gpd.read_parquet(raw_dir / "firms_filtered.parquet").drop(columns="geometry"))
    allp = pd.DataFrame(gpd.read_parquet(raw_dir / "firms_all.parquet").drop(columns="geometry"))
    print(f"{len(filt)} filtered detections, {len(allp)} total (used for negative exclusion)")

    cat = open_catalog(cfg["imagery"]["stac_url"])
    matches, scenes = match_detections_to_scenes(filt, cat, cfg["imagery"], progress=tqdm)
    print(f"matched {len(matches)} detections to {len(scenes)} scenes "
          f"(|gap| <= {cfg['imagery']['max_time_gap_hours']} h, cloud < {cfg['imagery']['max_scene_cloud']}%)")
    if matches.empty:
        raise SystemExit("No matches. Widen the date range, time gap, or cloud limit in the config.")
    matches.to_parquet(interim / "matches.parquet")
    (interim / "scenes.json").write_text(json.dumps(scenes, indent=2))

    manifest = extract_patches(filt, allp, matches, scenes, cfg, patches, progress=tqdm)
    manifest.to_parquet(interim / "patch_manifest_raw.parquet")
    print(manifest.label.value_counts().rename({1: "positive", 0: "negative"}).to_string())


if __name__ == "__main__":
    main()
