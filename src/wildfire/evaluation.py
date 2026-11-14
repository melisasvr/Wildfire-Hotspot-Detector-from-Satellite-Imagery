"""Honest comparison of heuristic, classical ML, and CNN on the held-out test split.

Thresholds are chosen on validation (max F1) and then frozen for test.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .baselines import SpectralRule, feature_matrix, make_classical
from .config import path_of
from .dataset import spatial_block_ids
from .metrics import (best_f1_threshold, bootstrap_ci, calibration_table, compute_metrics,
                      paired_bootstrap_diff, prevalence_adjusted_precision, slice_report)
from .preprocess import load_stats
from .viz import (plot_confusion, plot_patch_grid, plot_pr_curves, plot_prediction_map,
                  plot_reliability, write_folium_map)

SLICES = {
    "firms_confidence": "firms_confidence",
    "frp_mw": ("frp", [0, 5, 10, 25, 50, 1e5]),
    "abs_time_gap_h": ("abs_gap_h", [0, 1, 2, 3, 4, 6]),
    "patch_cloud_frac": ("cloud_frac", [0, 0.05, 0.2, 0.4]),
    "firms_daynight": "daynight",
    "ndvi_median (land-cover proxy)": ("ndvi_median", [-1, 0.2, 0.4, 0.6, 1.0]),
}


def cnn_scores(ckpt_path, df, patches_dir, batch_size=64):
    import torch
    from .models import build_model
    from .torch_data import make_loader
    from .train import predict_proba
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(ck["model_cfg"], ck["in_ch"])
    model.load_state_dict(ck["state_dict"])
    loader = make_loader(df, patches_dir, ck["stats"], batch_size)
    return predict_proba(model, loader, "cpu")


def fit_baselines(cfg: dict, manifest: pd.DataFrame, out_dir: Path) -> dict:
    """Fit baselines on train only, save them, return val/test scores."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pd_ = path_of(cfg, "patches")
    stats = load_stats(path_of(cfg, "processed") / "norm_stats.json")
    idx = stats["indices"]
    tr, va = manifest[manifest.split == "train"], manifest[manifest.split == "val"]

    rule = SpectralRule(cfg["baselines"]["ratio_grid"]).fit(tr, pd_)
    rule.save(out_dir / "spectral_rule.json")

    Xtr, names = feature_matrix(tr, pd_, idx)
    clf = make_classical(cfg["baselines"]["classical_model"], cfg["train"]["seed"]).fit(Xtr, tr.label.values)
    joblib.dump({"model": clf, "feature_names": names, "indices": idx}, out_dir / "classical.joblib")

    from .baselines import importance_table
    Xva, _ = feature_matrix(va, pd_, idx)
    importance_table(clf, Xva, va.label.values, names, cfg["train"]["seed"]).to_csv(
        out_dir / "classical_permutation_importance.csv", index=False)
    return {"rule": rule, "classical": clf}


def baseline_scores(cfg, manifest, base_dir: Path):
    pd_ = path_of(cfg, "patches")
    rule = SpectralRule.load(base_dir / "spectral_rule.json")
    bundle = joblib.load(base_dir / "classical.joblib")
    out = {}
    for split in ("val", "test"):
        d = manifest[manifest.split == split]
        X, _ = feature_matrix(d, pd_, bundle["indices"])
        out[split] = {"Spectral rule": rule.score(d, pd_),
                      "Classical ML": bundle["model"].predict_proba(X)[:, 1]}
    return out


def evaluate_all(cfg: dict, run_dir: str | Path | None = None) -> dict:
    processed, pd_ = path_of(cfg, "processed"), path_of(cfg, "patches")
    rep = path_of(cfg, "reports")
    manifest = pd.read_parquet(processed / "manifest.parquet")
    manifest["abs_gap_h"] = manifest["time_gap_h"].abs()
    va, te = (manifest[manifest.split == s].reset_index(drop=True) for s in ("val", "test"))
    if te.label.nunique() < 2:
        raise RuntimeError("Test split lacks one class; rebuild the dataset with more data or a different split.")

    base_dir = path_of(cfg, "runs") / "baselines"
    if not (base_dir / "classical.joblib").exists():
        fit_baselines(cfg, manifest, base_dir)
    bs = baseline_scores(cfg, manifest, base_dir)
    scores = {"val": dict(bs["val"]), "test": dict(bs["test"])}

    if run_dir is None:
        latest = path_of(cfg, "runs") / "latest.txt"
        run_dir = Path(latest.read_text().strip()) if latest.exists() else None
    if run_dir is not None and (Path(run_dir) / "best.pt").exists():
        name = f"CNN ({cfg['model']['arch']})"
        scores["val"][name] = cnn_scores(Path(run_dir) / "best.pt", va, pd_)
        scores["test"][name] = cnn_scores(Path(run_dir) / "best.pt", te, pd_)

    rows, per_model, cal = [], {}, {}
    y_te = te.label.values
    for name, p_te in scores["test"].items():
        thr = best_f1_threshold(va.label.values, scores["val"][name])
        m = compute_metrics(y_te, p_te, thr)
        for prev in cfg["evaluate"]["assumed_prevalences"]:
            m[f"precision_at_prev_{prev}"] = prevalence_adjusted_precision(m["recall"], m["fpr"], prev)
        tab, ece, brier = calibration_table(y_te, p_te)
        m["ece"], m["brier"] = ece, brier
        cal[name] = (tab, ece)
        rows.append({"model": name, **m})
        per_model[name] = (thr, m)
        usable = {k: v for k, v in SLICES.items() if (v[0] if isinstance(v, tuple) else v) in te.columns}
        sl = slice_report(te, p_te, thr, usable)
        sl.to_csv(rep / f"slices_{_slug(name)}.csv", index=False)
        plot_confusion(m, rep / f"confusion_{_slug(name)}.png", name)

    table = pd.DataFrame(rows)
    table.to_csv(rep / "metrics_table.csv", index=False)
    cols = ["model", "precision", "recall", "f1", "pr_auc", "roc_auc", "ece", "brier", "n", "n_pos"]
    (rep / "metrics_table.md").write_text(table[cols].round(3).to_markdown(index=False)
                                          if _has_tabulate() else table[cols].round(3).to_string(index=False))
    # --- uncertainty: cluster bootstrap over spatial blocks, and paired model differences
    clusters = spatial_block_ids(te, cfg["split"]["block_size_km"]).values
    n_boot, seed = int(cfg["evaluate"].get("n_boot", 1000)), int(cfg["train"]["seed"])
    ci = pd.concat([bootstrap_ci(y_te, scores["test"][n], per_model[n][0], clusters, n_boot, seed).assign(model=n)
                    for n in per_model], ignore_index=True)[["model", "metric", "estimate", "ci_low", "ci_high", "n_valid_resamples"]]
    ci.to_csv(rep / "metrics_ci.csv", index=False)
    diffs = paired_bootstrap_diff(y_te, scores["test"], {n: per_model[n][0] for n in per_model}, clusters, n_boot, seed)
    diffs.to_csv(rep / "model_differences.csv", index=False)
    pretty = (ci.assign(txt=lambda d: d.apply(lambda r: f"{r.estimate:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}]", axis=1))
                .pivot(index="model", columns="metric", values="txt").reindex(list(per_model))
                [["precision", "recall", "f1", "pr_auc", "roc_auc"]].reset_index())
    (rep / "metrics_table_ci.md").write_text(pretty.to_markdown(index=False) if _has_tabulate() else pretty.to_string(index=False))

    plot_pr_curves(scores["test"], y_te, rep / "pr_curves.png")
    plot_reliability(cal, rep / "reliability.png")

    # error analysis on the strongest model by validation-chosen test PR-AUC
    best = max(per_model, key=lambda k: per_model[k][1]["pr_auc"])
    thr, _ = per_model[best]
    p = scores["test"][best]
    n = cfg["evaluate"]["n_error_examples"]
    te2 = te.assign(score=p)
    fp = te2[(te2.score >= thr) & (te2.label == 0)].sort_values("score", ascending=False)
    fn = te2[(te2.score < thr) & (te2.label == 1)].sort_values("score")
    plot_patch_grid(fp, pd_, rep / "false_positives.png", f"False positives ({best})", scores=fp.score.values, n=n)
    plot_patch_grid(fn, pd_, rep / "false_negatives.png", f"False negatives ({best})", scores=fn.score.values, n=n)
    fp.head(100).to_csv(rep / "false_positives.csv", index=False)
    fn.head(100).to_csv(rep / "false_negatives.csv", index=False)
    plot_prediction_map(te, p, thr, rep / "prediction_map.png")
    write_folium_map(te, p, thr, rep / "prediction_map.html")
    te2.assign(pred=(te2.score >= thr).astype(int)).drop(columns=["bounds"], errors="ignore").to_parquet(rep / "predictions_test.parquet")

    triage = {
        "false_positives_vs_true_negatives_mean": te2[(te2.label == 0)].assign(fp=lambda d: d.score >= thr)
            .groupby("fp")[["cloud_frac", "water_frac", "scene_cloud"]].mean().round(4).to_dict(),
        "false_negatives_vs_true_positives_mean": te2[(te2.label == 1)].assign(fn=lambda d: d.score < thr)
            .groupby("fn")[["frp", "abs_gap_h", "cloud_frac", "firms_conf_ord"]].mean().round(4).to_dict(),
    }
    summary = {"best_model": best, "models": {k: {"threshold": v[0], **{a: b for a, b in v[1].items()}} for k, v in per_model.items()},
               "triage": triage,
               "bootstrap": {"n_boot": n_boot, "resampling_unit": f"{cfg['split']['block_size_km']} km spatial blocks",
                             "n_test_blocks": int(len(set(clusters))), "n_test_patches": int(len(te)),
                             "caveat": "Intervals cover test-set sampling noise only, not training randomness or threshold choice."}, "run_dir": str(run_dir) if run_dir else None,
               "note": "Test labels are FIRMS-derived weak labels. Review the error grids by eye before drawing conclusions."}
    (rep / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False