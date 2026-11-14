"""Fit the spectral-rule and classical-ML baselines (train split only) and save them."""
import pandas as pd

from _common import parse

from wildfire.config import path_of
from wildfire.evaluation import fit_baselines


def main():
    _, cfg = parse(__doc__)
    m = pd.read_parquet(path_of(cfg, "processed") / "manifest.parquet")
    out = path_of(cfg, "runs") / "baselines"
    fit_baselines(cfg, m, out)
    print("saved baselines to", out)


if __name__ == "__main__":
    main()
