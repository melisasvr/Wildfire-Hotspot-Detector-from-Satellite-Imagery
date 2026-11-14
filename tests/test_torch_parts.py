"""Needs PyTorch. Skipped automatically when torch is not installed."""
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from wildfire.evaluation import evaluate_all
from wildfire.models import FocalLoss, build_model
from wildfire.pipeline import build_processed_dataset
from wildfire.preprocess import save_stats
from wildfire.torch_data import make_loader
from wildfire.train import train


def _prepare(cfg, raw):
    df, rep, stats = build_processed_dataset(raw, cfg)
    proc = Path(cfg["paths"]["processed"]); proc.mkdir(parents=True, exist_ok=True)
    df.to_parquet(proc / "manifest.parquet"); save_stats(stats, proc / "norm_stats.json")
    return df, stats


def test_dataloader_shapes_and_augmentation_keeps_shape(cfg, synthetic_raw):
    df, stats = _prepare(cfg, synthetic_raw)
    tr = df[df.split == "train"]
    x, y = next(iter(make_loader(tr, cfg["paths"]["patches"], stats, 8, augment=True, shuffle=True)))
    assert x.shape == (8, 8, 32, 32) and y.shape == (8,) and x.dtype == torch.float32


@pytest.mark.parametrize("arch", ["small_cnn", "resnet18"])
def test_models_output_one_logit_per_sample(arch):
    m = build_model({"arch": arch, "dropout": 0.1, "stem": "standard"}, 8)
    assert m(torch.randn(3, 8, 32, 32)).shape == (3,)
    assert FocalLoss()(torch.zeros(3), torch.tensor([0.0, 1.0, 1.0])).item() > 0


def test_train_then_evaluate_end_to_end_plumbing(cfg, synthetic_raw):
    """Checks that the pipeline runs and writes outputs. It says nothing about real-world accuracy."""
    _prepare(cfg, synthetic_raw)
    run = train(cfg)
    assert (run / "best.pt").exists() and (run / "history.csv").exists()
    summary = evaluate_all(cfg, run)
    t = __import__("pandas").read_csv(Path(cfg["paths"]["reports"]) / "metrics_table.csv")
    assert any(m.startswith("CNN") for m in t.model) and summary["run_dir"] == str(run)
