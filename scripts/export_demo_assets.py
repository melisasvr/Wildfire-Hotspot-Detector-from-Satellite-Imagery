"""Copy the model, stats, rule, and a few example patches into demo_assets/ for the Streamlit app."""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from _common import parse

from wildfire.config import path_of
from wildfire.preprocess import load_patch


def extra(ap):
    ap.add_argument("--run", default=None)
    ap.add_argument("--n", type=int, default=6, help="examples per class")


def main():
    args, cfg = parse(__doc__, extra)
    out = path_of(cfg, "demo_assets")
    run = Path(args.run) if args.run else Path((path_of(cfg, "runs") / "latest.txt").read_text().strip())
    shutil.copy(run / "best.pt", out / "model.pt")
    shutil.copy(path_of(cfg, "runs") / "baselines" / "spectral_rule.json", out / "spectral_rule.json")
    pred = pd.read_parquet(path_of(cfg, "reports") / "predictions_test.parquet")
    pd_ = path_of(cfg, "patches")
    rng = np.random.default_rng(0)
    meta = []
    for lab in (1, 0):
        d = pred[pred.label == lab]
        d = d.iloc[rng.permutation(len(d))[: args.n]]
        for _, r in d.iterrows():
            arr = np.load(pd_ / r.patch_file)
            np.savez_compressed(out / f"{r.patch_id}.npz", refl=arr)
            meta.append({"patch_id": r.patch_id, "label": int(lab), "lat": r.lat, "lon": r.lon,
                         "scene_datetime": r.scene_datetime, "firms_confidence": r.firms_confidence,
                         "frp": None if pd.isna(r.frp) else float(r.frp)})
    (out / "examples.json").write_text(json.dumps(meta, indent=2))
    print("demo assets in", out)


if __name__ == "__main__":
    main()
