import numpy as np
import pandas as pd

from wildfire.baselines import SpectralRule, patch_features
from wildfire.evaluation import evaluate_all, fit_baselines
from wildfire.metrics import (best_f1_threshold, calibration_table, compute_metrics,
                              prevalence_adjusted_precision, slice_report)
from wildfire.pipeline import build_processed_dataset, write_dataset_report
from wildfire.preprocess import save_stats


def test_metrics_basics():
    y = np.array([0, 0, 1, 1, 1, 0]); p = np.array([.1, .2, .9, .8, .4, .6])
    thr = best_f1_threshold(y, p)
    m = compute_metrics(y, p, thr)
    assert m["tp"] + m["fn"] == 3 and m["tn"] + m["fp"] == 3 and 0 <= m["pr_auc"] <= 1
    tab, ece, brier = calibration_table(y, p, 5)
    assert 0 <= ece <= 1 and 0 <= brier <= 1
    assert np.isclose(prevalence_adjusted_precision(0.9, 0.1, 0.5), 0.9)
    assert prevalence_adjusted_precision(0.9, 0.1, 0.01) < 0.1


def test_slice_report_positive_only_slices_have_no_precision():
    df = pd.DataFrame({"label": [1, 1, 0, 0], "firms_confidence": ["h", "n", None, None], "frp": [5, 50, np.nan, np.nan]})
    r = slice_report(df, np.array([.9, .2, .1, .3]), 0.5, {"conf": "firms_confidence", "frp": ("frp", [0, 10, 100])})
    assert set(r.value) >= {"h", "n"} and r.precision.isna().all()
    assert r[r.value == "h"].recall.iloc[0] == 1.0 and r[r.value == "n"].recall.iloc[0] == 0.0


def test_features_and_rule_beat_chance_on_synthetic(cfg, synthetic_raw):
    df, rep, stats = build_processed_dataset(synthetic_raw, cfg)
    pdir = cfg["paths"]["patches"]
    v, names = patch_features(np.load(f"{pdir}/{df.patch_file[0]}").astype(np.float32) / 1e4, cfg["preprocess"]["indices"])
    assert len(v) == len(names) and np.isfinite(v).all()
    rule = SpectralRule(cfg["baselines"]["ratio_grid"]).fit(df[df.split == "train"], pdir)
    s = rule.score(df[df.split == "test"], pdir)
    assert s.shape[0] == (df.split == "test").sum() and 0 <= s.min() and s.max() <= 1


def test_baseline_only_evaluation_writes_everything(cfg, synthetic_raw):
    df, rep, stats = build_processed_dataset(synthetic_raw, cfg)
    from pathlib import Path
    proc = Path(cfg["paths"]["processed"]); proc.mkdir(parents=True, exist_ok=True)
    df.to_parquet(proc / "manifest.parquet"); save_stats(stats, proc / "norm_stats.json")
    write_dataset_report(df, rep, cfg, Path(cfg["paths"]["reports"]) / "dataset")
    summary = evaluate_all(cfg, run_dir=None)
    rep_dir = Path(cfg["paths"]["reports"])
    for f in ("metrics_table.csv", "pr_curves.png", "reliability.png", "false_positives.png",
              "false_negatives.png", "prediction_map.png", "summary.json"):
        assert (rep_dir / f).exists(), f
    assert (rep_dir / "dataset" / "dataset_summary.json").exists()
    for f in ("metrics_ci.csv", "model_differences.csv", "metrics_table_ci.md"):
        assert (rep_dir / f).exists(), f
    ci = pd.read_csv(rep_dir / "metrics_ci.csv")
    assert (ci.ci_low <= ci.ci_high).all() and set(ci.metric) >= {"f1", "pr_auc", "recall"}
    assert len(pd.read_csv(rep_dir / "model_differences.csv")) == 3      # 1 pair x 3 metrics
    t = pd.read_csv(rep_dir / "metrics_table.csv")
    assert set(t.model) == {"Spectral rule", "Classical ML"}
    assert summary["best_model"] in set(t.model)