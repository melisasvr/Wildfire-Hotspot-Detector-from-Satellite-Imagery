import numpy as np
import pandas as pd
from pathlib import Path

from wildfire.config import load_config
from wildfire.matching import add_context_features, match_negatives
from wildfire.metrics import bootstrap_ci, paired_bootstrap_diff
from wildfire.patches import to_uint16
from wildfire.pipeline import build_processed_dataset

ROOT = Path(__file__).resolve().parents[1]


def _const_patch(nir, red, b11, size=16):
    refl = np.zeros((6, size, size), dtype=np.float32)
    refl[3], refl[2], refl[4], refl[5] = nir, red, b11, b11 + 0.05
    return refl


def _context_dataset(tmp_path, n_pos=40, n_wild=40, n_farm=80, seed=0):
    """Positives in green wildland. Negatives: some wildland-like, many bright farmland."""
    rng = np.random.default_rng(seed)
    rows, pdir = [], tmp_path / "patches"; pdir.mkdir()
    def add(i, label, kind, split):
        if kind == "wild":
            refl = _const_patch(rng.normal(.40, .02), rng.normal(.05, .005), rng.normal(.15, .01))
        else:
            refl = _const_patch(rng.normal(.30, .02), rng.normal(.20, .01), rng.normal(.40, .02))
        pid = f"P{i:04d}"; np.save(pdir / f"{pid}.npy", to_uint16(refl))
        rows.append({"patch_id": pid, "patch_file": f"{pid}.npy", "label": label, "kind": kind, "split": split})
    i = 0
    for split in ("train", "val", "test"):
        for _ in range(n_pos): add(i, 1, "wild", split); i += 1
        for _ in range(n_wild): add(i, 0, "wild", split); i += 1
        for _ in range(n_farm): add(i, 0, "farm", split); i += 1
    return pd.DataFrame(rows), pdir


def test_matching_drops_easy_negatives_keeps_all_positives_and_improves_balance(tmp_path):
    df, pdir = _context_dataset(tmp_path)
    df = add_context_features(df, pdir)
    out, rep = match_negatives(df, k=1, caliper_sd=0.75, seed=0)
    assert (out.label == 1).sum() == (df.label == 1).sum()                       # every positive kept
    neg = out[out.label == 0]
    assert (neg.kind == "wild").mean() > 0.95                                   # farmland negatives dropped
    for split, r in rep["splits"].items():
        assert r["n_neg_after"] <= r["n_pos"] * 1 and r["n_neg_after"] < r["n_neg_before"]
        for f in rep["features"]:
            assert abs(r["smd_after"][f]) < abs(r["smd_before"][f])
    out2, _ = match_negatives(df, k=1, caliper_sd=0.75, seed=0)
    assert list(out.patch_id) == list(out2.patch_id)                             # deterministic


def test_pipeline_with_matching_enabled_still_passes_leakage(cfg, synthetic_raw):
    cfg["matching"] = {"enabled": True, "negatives_per_positive": 2, "caliper_sd": 1.5}
    df, rep, stats = build_processed_dataset(synthetic_raw, cfg)
    assert rep["passed"] and "matching" in rep and {"ndvi_median", "b11_median"} <= set(df.columns)
    for s, r in rep["matching"]["splits"].items():
        assert r["n_neg_after"] <= 2 * r["n_pos"]


def test_matched_config_extends_default_and_uses_separate_folders():
    base, matched = load_config(ROOT / "configs/default.yaml"), load_config(ROOT / "configs/matched.yaml")
    assert matched["matching"]["enabled"] and not base["matching"]["enabled"]
    assert matched["paths"]["processed"] != base["paths"]["processed"]
    assert matched["paths"]["patches"] == base["paths"]["patches"]                # patches are shared
    assert matched["imagery"] == base["imagery"]


def test_bootstrap_ci_brackets_estimate_and_respects_clusters():
    rng = np.random.default_rng(0)
    n = 120; y = (np.arange(n) % 3 == 0).astype(int)
    p = np.clip(y * 0.6 + rng.normal(0.2, 0.15, n), 0, 1)
    clusters = np.arange(n) // 10
    ci = bootstrap_ci(y, p, 0.5, clusters, n_boot=300, seed=1).set_index("metric")
    assert ci.loc["pr_auc", "ci_low"] <= ci.loc["pr_auc", "estimate"] <= ci.loc["pr_auc", "ci_high"]
    wide = bootstrap_ci(y, p, 0.5, np.zeros(n, int) + np.arange(n) // 60, n_boot=300, seed=1).set_index("metric")
    assert (wide.n_valid_resamples <= 300).all()                                  # tiny cluster counts can drop resamples
    w_cl = ci.loc["roc_auc", "ci_high"] - ci.loc["roc_auc", "ci_low"]
    w_pt = bootstrap_ci(y, p, 0.5, None, n_boot=300, seed=1).set_index("metric").eval("roc_auc_w = ci_high - ci_low").loc["roc_auc", "roc_auc_w"]
    assert w_cl >= w_pt * 0.8                                                      # clustered resampling is not narrower


def test_paired_difference_detects_a_real_gap_and_zero_for_identical_models():
    rng = np.random.default_rng(2)
    n = 300; y = (rng.random(n) < 0.35).astype(int)
    good = np.clip(y * 0.5 + rng.normal(0.25, 0.12, n), 0, 1)
    bad = rng.random(n)
    d = paired_bootstrap_diff(y, {"good": good, "bad": bad, "same": good.copy()}, {"good": .5, "bad": .5, "same": .5},
                              None, n_boot=200, seed=3)
    g = d[(d.model_a == "good") & (d.model_b == "bad") & (d.metric == "pr_auc")].iloc[0]
    assert g.ci_low > 0 and g.frac_resamples_a_better > 0.99
    s = d[(d.model_a == "good") & (d.model_b == "same")]
    assert (s.ci_low == 0).all() and (s.ci_high == 0).all()