"""Train the CNN with class weighting, early stopping on validation PR-AUC, and run logging."""
from __future__ import annotations

import json
import platform
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from .config import path_of
from .metrics import best_f1_threshold, compute_metrics
from .models import FocalLoss, build_model
from .preprocess import load_stats
from .torch_data import make_loader


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


@torch.no_grad()
def predict_proba(model, loader, device="cpu") -> np.ndarray:
    model.eval()
    out = []
    for x, _ in loader:
        out.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def _git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def train(cfg: dict) -> Path:
    tc = cfg["train"]
    seed_everything(tc["seed"])
    if tc.get("num_threads"):
        torch.set_num_threads(int(tc["num_threads"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processed, patches_dir = path_of(cfg, "processed"), path_of(cfg, "patches")
    manifest = pd.read_parquet(processed / "manifest.parquet")
    stats = load_stats(processed / "norm_stats.json")
    tr, va = manifest[manifest.split == "train"], manifest[manifest.split == "val"]
    if tr.label.nunique() < 2 or len(va) == 0:
        raise RuntimeError("Train needs both classes and a non-empty validation split.")

    in_ch = len(stats["mean"])
    model = build_model(cfg["model"], in_ch).to(device)
    n_pos = float(tr.label.sum()); n_neg = float(len(tr) - n_pos)
    if tc["loss"] == "focal":
        loss_fn = FocalLoss(tc.get("focal_gamma", 2.0), alpha=n_neg / (n_pos + n_neg))
    else:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_neg / n_pos, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc["epochs"])

    tl = make_loader(tr, patches_dir, stats, tc["batch_size"], augment=True, shuffle=True,
                     num_workers=tc["num_workers"], seed=tc["seed"])
    vl = make_loader(va, patches_dir, stats, tc["batch_size"], num_workers=tc["num_workers"])

    run_dir = path_of(cfg, "runs") / f"{datetime.now():%Y%m%d_%H%M%S}_{cfg['model']['arch']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))

    history, best, bad, t0 = [], -np.inf, 0, time.time()
    for epoch in range(1, tc["epochs"] + 1):
        model.train(); run_loss = 0.0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = loss_fn(model(x), y); loss.backward(); opt.step()
            run_loss += loss.item() * len(x)
        sched.step()
        vp = predict_proba(model, vl, device)
        thr = best_f1_threshold(va.label.values, vp)
        vm = compute_metrics(va.label.values, vp, thr)
        score = vm["pr_auc"] if vm["pr_auc"] == vm["pr_auc"] else -1.0
        history.append({"epoch": epoch, "train_loss": run_loss / len(tr), "val_pr_auc": vm["pr_auc"],
                        "val_f1": vm["f1"], "val_threshold": thr, "seconds": round(time.time() - t0, 1)})
        print(history[-1])
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        if score > best:
            best, bad = score, 0
            torch.save({"state_dict": model.state_dict(), "model_cfg": cfg["model"], "in_ch": in_ch,
                        "stats": stats, "val_threshold": thr, "epoch": epoch, "val_pr_auc": vm["pr_auc"]},
                       run_dir / "best.pt")
        else:
            bad += 1
            if bad >= tc["patience"]:
                print("early stopping"); break

    info = {"best_val_pr_auc": best, "epochs_run": len(history), "device": device, "train_seconds": round(time.time() - t0, 1),
            "n_train": int(len(tr)), "n_val": int(len(va)), "train_pos": int(n_pos), "train_neg": int(n_neg),
            "torch": torch.__version__, "python": platform.python_version(), "git": _git_hash()}
    (run_dir / "run.json").write_text(json.dumps(info, indent=2))
    (path_of(cfg, "runs") / "latest.txt").write_text(str(run_dir))
    return run_dir
