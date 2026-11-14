"""Stage 1: download FIRMS VIIRS 375 m archive detections and filter them transparently.

Needs a free MAP_KEY: https://firms.modaps.eosdis.nasa.gov/api/area/  ->  export FIRMS_MAP_KEY=...
"""
import os
import sys

from _common import parse

from wildfire.config import path_of
from wildfire.firms import download_firms, filter_detections, parse_detections, save_labels


def extra(ap):
    ap.add_argument("--start"); ap.add_argument("--end")


def main():
    args, cfg = parse(__doc__, extra)
    key = os.environ.get("FIRMS_MAP_KEY")
    if not key:
        sys.exit("Set FIRMS_MAP_KEY first (free key from the FIRMS API page).")
    start = args.start or cfg["period"]["start_date"]
    end = args.end or cfg["period"]["end_date"]
    bbox = cfg["region"]["bbox"]
    raw = download_firms(key, cfg["firms"]["sources"], bbox, start, end)
    if raw.empty:
        sys.exit("FIRMS returned no detections for this box and period.")
    allp = parse_detections(raw)
    filt, log = filter_detections(allp, cfg["firms"], bbox)
    out = path_of(cfg, "raw") / "firms"
    save_labels(allp, filt, log, out, {"region": cfg["region"], "period": {"start": start, "end": end},
                                       "sources": cfg["firms"]["sources"], "filters": cfg["firms"]})
    for row in log:
        print(f"{row['step']:<45} {row['n']:>9}")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
