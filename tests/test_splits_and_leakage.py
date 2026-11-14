import numpy as np

from wildfire.dataset import (assign_spatial_block_splits, leakage_report, make_splits,
                              purge_cross_split_neighbors)
from wildfire.pipeline import build_processed_dataset, dataset_summary
from wildfire.preprocess import build_channels, compute_norm_stats, load_patch, normalize


def test_spatial_blocks_never_straddle_splits(cfg, synthetic_raw):
    s = assign_spatial_block_splits(synthetic_raw, 25, cfg["split"]["fractions"], 0)
    blocks = (np.floor(synthetic_raw.x_a / 25000).astype(int).astype(str) + "_" +
              np.floor(synthetic_raw.y_a / 25000).astype(int).astype(str))
    assert (s.groupby(blocks).nunique() == 1).all()
    assert set(s.unique()) == {"train", "val", "test"}


def test_purge_removes_close_cross_split_pairs_and_report_passes(cfg, synthetic_raw):
    df = make_splits(synthetic_raw, cfg)
    rep = leakage_report(df, cfg["split"]["buffer_m"])
    assert rep["passed"] and rep["duplicate_patch_ids"] == 0
    assert all(v >= cfg["split"]["buffer_m"] for v in rep["min_center_distance_m"].values())


def test_leakage_report_detects_a_leak(cfg, synthetic_raw):
    df = make_splits(synthetic_raw, cfg)
    leak = df[df.split == "train"].iloc[[0]].copy()
    leak["split"] = "test"; leak["patch_id"] = "LEAK"            # same place, other split
    bad = __import__("pandas").concat([df, leak], ignore_index=True)
    assert not leakage_report(bad, cfg["split"]["buffer_m"])["passed"]
    assert len(purge_cross_split_neighbors(bad, cfg["split"]["buffer_m"])) < len(bad)


def test_temporal_split_orders_time(cfg, synthetic_raw):
    cfg["split"]["strategy"] = "temporal"
    cfg["split"]["temporal"].update(val_start="2024-08-15", test_start="2024-09-15", buffer_days=0)
    df = make_splits(synthetic_raw, cfg)
    t = __import__("pandas").to_datetime(df.scene_datetime, utc=True)
    assert t[df.split == "train"].max() < t[df.split == "val"].min() <= t[df.split == "val"].max() < t[df.split == "test"].min()


def test_normalisation_uses_train_only_and_shapes(cfg, synthetic_raw):
    df, rep, stats = build_processed_dataset(synthetic_raw, cfg)
    assert stats["n_patches"] == int((df.split == "train").sum())
    pdir = cfg["paths"]["patches"]
    x = build_channels(load_patch(f"{pdir}/{df.patch_file.iloc[0]}"), cfg["preprocess"]["indices"])
    assert x.shape == (8, 32, 32) and len(stats["mean"]) == 8
    train_x = np.concatenate([build_channels(load_patch(f"{pdir}/{f}"), cfg["preprocess"]["indices"]).reshape(8, -1)
                              for f in df[df.split == "train"].patch_file], axis=1)
    z = normalize(x, stats)
    assert np.allclose(train_x.mean(1), stats["mean"], atol=1e-4)
    assert z.shape == x.shape and z.dtype == np.float32
    s = dataset_summary(df)
    assert s["n_total"] == len(df) and set(s["per_split"]) == {"train", "val", "test"}
