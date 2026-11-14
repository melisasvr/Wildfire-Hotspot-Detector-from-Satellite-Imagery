"""PyTorch dataset and loader for stored patches."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .preprocess import build_channels, load_patch, normalize


class PatchDataset(Dataset):
    """Yields (normalised channels, label). Rows follow the manifest order."""

    def __init__(self, manifest: pd.DataFrame, patches_dir, stats: dict, augment: bool = False, seed: int = 0):
        self.df = manifest.reset_index(drop=True)
        self.dir, self.stats, self.augment = patches_dir, stats, augment
        self.indices = stats.get("indices", [])
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        x = normalize(build_channels(load_patch(f"{self.dir}/{row.patch_file}"), self.indices), self.stats)
        x = torch.from_numpy(x)
        if self.augment:  # flips and 90-degree rotations only: spectral values are untouched
            k = int(self.rng.integers(0, 4))
            x = torch.rot90(x, k, dims=(1, 2))
            if self.rng.random() < 0.5:
                x = torch.flip(x, dims=(2,))
        return x, torch.tensor(float(row.label))


def make_loader(df, patches_dir, stats, batch_size, augment=False, shuffle=False, num_workers=0, seed=0):
    ds = PatchDataset(df, patches_dir, stats, augment=augment, seed=seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
