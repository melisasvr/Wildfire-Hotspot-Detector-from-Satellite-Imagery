"""Pre-training report: counts, balance, clouds, splits, example patches, leakage check."""
import json

import pandas as pd

from _common import parse

from wildfire.config import path_of
from wildfire.pipeline import write_dataset_report


def main():
    _, cfg = parse(__doc__)
    proc = path_of(cfg, "processed")
    m = pd.read_parquet(proc / "manifest.parquet")
    leak = json.loads((proc / "leakage_report.json").read_text())
    summ = write_dataset_report(m, leak, cfg, path_of(cfg, "reports") / "dataset")
    print(json.dumps({k: summ[k] for k in ("n_total", "n_pos", "n_neg", "per_split")}, indent=2))
    print("figures in", path_of(cfg, "reports") / "dataset")


if __name__ == "__main__":
    main()
