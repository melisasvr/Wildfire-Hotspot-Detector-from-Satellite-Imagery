"""Model zoo: compact CNNs for multispectral patch classification (outputs one logit)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class SmallCNN(nn.Module):
    def __init__(self, in_ch: int, dropout: float = 0.2):
        super().__init__()

        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.MaxPool2d(2))

        self.features = nn.Sequential(block(in_ch, 32), block(32, 64), block(64, 128), block(128, 128))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(256, 1))

    def forward(self, x):
        f = self.features(x)
        # max pooling keeps tiny hot spots visible; avg pooling keeps context
        return self.head(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)).squeeze(1)


class _Squeeze(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x).squeeze(1)


def build_model(mcfg: dict, in_ch: int) -> nn.Module:
    arch = mcfg.get("arch", "resnet18")
    dropout = float(mcfg.get("dropout", 0.2))
    if arch == "small_cnn":
        return SmallCNN(in_ch, dropout)
    if arch == "resnet18":
        pretrained = bool(mcfg.get("pretrained", False))
        stem = mcfg.get("stem", "standard")
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None)
        old = m.conv1
        new = nn.Conv2d(in_ch, 64, 3, 1, 1, bias=False) if stem == "fine" else nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
        if pretrained and stem == "standard":  # reuse RGB filters: average them, spread over all channels
            with torch.no_grad():
                new.weight.copy_(old.weight.mean(1, keepdim=True).repeat(1, in_ch, 1, 1) * 3.0 / in_ch)
        m.conv1 = new
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, 1))
        return _Squeeze(m)
    raise ValueError(f"unknown arch {arch}")


class FocalLoss(nn.Module):
    """Binary focal loss on logits. `alpha` is the weight on the positive class."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.5):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, logits, y):
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        p = torch.sigmoid(logits)
        pt = y * p + (1 - y) * (1 - p)
        a = y * self.alpha + (1 - y) * (1 - self.alpha)
        return (a * (1 - pt) ** self.gamma * bce).mean()
