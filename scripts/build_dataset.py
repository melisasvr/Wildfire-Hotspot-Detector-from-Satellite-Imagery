"""Stage 3: QC filters, spatial/temporal split, leakage purge, train-only normalisation stats."""
import json

import pandas as pd

from _common import parse

from wildfire.config import path_of
from wildfire.pipeline import build_processed_dataset
from wildfire.preprocess import save_stats


def main():
    _, cfg = parse(__doc__)
    raw = pd.read_parquet(path_of(cfg, "interim") / "patch_manifest_raw.parquet")
    df, rep, stats = build_processed_dataset(raw, cfg)
    out = path_of(cfg, "processed")
    df.to_parquet(out / "manifest.parquet")
    save_stats(stats, out / "norm_stats.json")
    (out / "leakage_report.json").write_text(json.dumps(rep, indent=2))
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0))
    if "matching" in rep:
        print("\nland-cover matching (SMD near 0 = balanced; before -> after):")
        for sp, r in rep["matching"]["splits"].items():
            sb, sa = r.get("smd_before", {}), r.get("smd_after", {})
            print(f"  {sp:<5} negatives {r['n_neg_before']} -> {r['n_neg_after']}; "
                  + "; ".join(f"{f} {sb[f]:+.2f} -> {sa[f]:+.2f}" for f in sb))
    print("leakage check passed:", rep["passed"])
    if not rep["passed"]:
        raise SystemExit("Leakage check failed. See leakage_report.json before training.")


if __name__ == "__main__":
    main()