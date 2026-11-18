"""Training of one SpaCCES-GAE seed, a port of the authors' training loop.

Settings of the paper (``DEFAULTS``): AdamW at lr 0.01, StepLR(1000, 0.9) stepped per
epoch, 32 window graphs per batch, at most 2000 epochs, early stopping after 25
epochs without a lower validation loss, and the best-validation state kept. The loss
is ``(1 - alpha) * MSE(structure) + alpha * MSE(attributes)``; the paper uses
``alpha = 1``, where the structure term has zero weight. Training uses every window
of the healthy training runs; the validation loss and the residual statistics use
every window of the healthy validation runs.

The paper's models were trained on an NVIDIA L4 GPU with deterministic algorithms.
Retraining on another device follows the same procedure but not the same floating-
point path, so a retrained seed is a new draw from the same training distribution,
not a bit copy; the released ``models/`` are the trained originals.
"""
from __future__ import annotations

import os
import random
import time

import numpy as np
import torch

from . import data as D
from . import scoring as S
from .model import SpaccesGAE

DEFAULTS = {"lr": 0.01, "batch": 32, "max_epochs": 2000, "patience": 25, "alpha": 1.0,
            "hidden_channels": 8, "out_channels": 4, "embed_dim": 4, "layers": 4}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batches(n: int, size: int, rng: np.random.Generator, shuffle: bool = True):
    order = rng.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, size):
        yield order[start:start + size]


def loss_fn(model, batch, adj_true: torch.Tensor, alpha: float) -> torch.Tensor:
    adj_rec, x_rec = model(batch)
    target = torch.block_diag(*([adj_true] * batch.num_graphs))
    loss_s = torch.nn.functional.mse_loss(adj_rec, target)
    loss_a = torch.nn.functional.mse_loss(x_rec, batch.x)
    return (1.0 - float(alpha)) * loss_s + float(alpha) * loss_a


def fit(ws: D.WindowSet, a_phys: np.ndarray, train_ids, val_ids, seed: int, device: str = "cpu",
        log=print, **overrides) -> tuple[SpaccesGAE, dict]:
    """Train one seed; returns the best-validation model and its training record."""
    cfg = {**DEFAULTS, **overrides}
    dev = torch.device(device)
    if dev.type == "cuda":   # untested in this repository; deterministic cuBLAS needs a fixed workspace
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
    set_seed(seed)
    model = SpaccesGAE(in_channels=2, hidden_channels=cfg["hidden_channels"], out_channels=cfg["out_channels"],
                       n_nodes=D.N_SENSORS, embed_dim=cfg["embed_dim"], layers=cfg["layers"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1000, gamma=0.9)
    adj_true = torch.tensor(np.asarray(a_phys), dtype=torch.float, device=dev)
    rng = np.random.default_rng(seed)
    train_idx = np.flatnonzero(np.isin(ws.run_id, list(train_ids)))
    val_mask = np.isin(ws.run_id, list(val_ids))
    val_idx = np.flatnonzero(val_mask)

    t0, best, bad = time.time(), float("inf"), 0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    val_curve, train_curve = [], []
    for epoch in range(int(cfg["max_epochs"])):
        model.train()
        losses = []
        for pick in batches(train_idx.size, cfg["batch"], rng):
            b = D.batch(ws, a_phys, train_idx[pick]).to(dev)
            opt.zero_grad()
            loss = loss_fn(model, b, adj_true, cfg["alpha"])
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        train_curve.append(float(np.mean(losses)))
        sched.step()
        model.eval()
        total = 0.0
        with torch.no_grad():
            for pick in batches(val_idx.size, cfg["batch"], rng, shuffle=False):
                b = D.batch(ws, a_phys, val_idx[pick]).to(dev)
                total += float(loss_fn(model, b, adj_true, cfg["alpha"]).item()) * pick.size
        val = total / max(val_idx.size, 1)
        val_curve.append(val)
        log(f"seed {seed} epoch {epoch + 1}: train {train_curve[-1]:.6f} val {val:.6f}")
        if not np.isfinite(val):
            break
        if val < best - 1e-12:
            best, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    model.to("cpu").eval()
    err = S.window_errors(model, ws, a_phys, alpha=cfg["alpha"])
    mu, sigma = S.validation_stats(err, val_mask)
    record = {"seed": seed, "alpha": cfg["alpha"], "config": cfg, "train_ids": sorted(int(r) for r in train_ids),
              "val_ids": sorted(int(r) for r in val_ids), "residual_mu": mu.tolist(), "residual_sigma": sigma.tolist(),
              "val_loss_curve": val_curve, "train_loss_curve": train_curve, "best_val_loss": best,
              "n_epochs": len(val_curve), "device": str(dev), "train_time_s": time.time() - t0}
    return model, record
