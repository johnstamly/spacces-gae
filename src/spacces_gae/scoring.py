"""From reconstruction residuals to sensor scores, run scores and the threshold.

Chain (manuscript Section 4.6):

1. ``eps[w, v]``: per-window, per-sensor reconstruction error of ``[x, c]``
   (mean squared error over the two input channels; at alpha < 1 plus the
   structure term against the support of S_sim).
2. ``z[w, v] = (eps - mu_v) / sigma_v`` with ``mu_v, sigma_v`` from the healthy
   validation windows (stored in each model's record).
3. Sensor score ``n_v`` = median over a run's windows of ``|z|``.
4. Run score ``s_run`` = mean of the five largest ``n_v``.
5. Ensemble: seed means of ``s_run`` and ``n_v``; threshold ``tau`` = mean + 3
   population standard deviations of ``s_run`` over the healthy training runs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D

TOP_K = 5
NODE_COLS = [f"n{j}" for j in range(D.N_SENSORS)]


def window_errors(model, ws: D.WindowSet, a_phys: np.ndarray, alpha: float = 1.0,
                  batch_size: int = 256) -> np.ndarray:
    """``[W, 60]`` per-sensor reconstruction error of every window (float64)."""
    import torch
    with torch.no_grad():
        return _window_errors(torch, model, ws, a_phys, alpha, batch_size)


def _window_errors(torch, model, ws, a_phys, alpha, batch_size) -> np.ndarray:
    model.eval()
    out = np.empty((ws.n_windows, D.N_SENSORS), dtype=np.float64)
    for start in range(0, ws.n_windows, batch_size):
        idx = np.arange(start, min(start + batch_size, ws.n_windows))
        b = D.batch(ws, a_phys, idx)
        z = model.encode(b.x, b.edge_index_phys, b.edge_index_sim, b.edge_weight_sim)
        x_rec = model.decode_attributes(z, b.edge_index_phys)
        err_a = torch.mean((b.x - x_rec) ** 2, dim=1).reshape(idx.size, D.N_SENSORS)
        if float(alpha) == 1.0:
            out[idx] = (0.0 * err_a + 1.0 * err_a).numpy()   # the trained score at alpha = 1
            continue
        zr = z.reshape(idx.size, D.N_SENSORS, -1)
        for k, w in enumerate(idx):
            target = (ws.sim_dense(int(w)) > 0).astype(np.float32)
            adj = model.decode_structure(zr[k])
            refined = torch.zeros_like(adj)
            deg = target.sum(axis=1).astype(int)
            for i in range(D.N_SENSORS):
                if 0 < deg[i] < D.N_SENSORS:
                    refined[i, torch.topk(adj[i], int(deg[i])).indices] = 1
            err_s = torch.mean((torch.tensor(target) - refined) ** 2, dim=1)
            out[w] = ((1 - alpha) * err_s + alpha * err_a[k]).numpy()
    return out


def standardize(errors: np.ndarray, mu, sigma) -> np.ndarray:
    return (np.asarray(errors, dtype=np.float64) - np.asarray(mu, dtype=np.float64)) / np.asarray(sigma, dtype=np.float64)


def validation_stats(errors: np.ndarray, val_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(mu_v, sigma_v)`` over the healthy validation windows (population sigma)."""
    v = np.asarray(errors, dtype=np.float64)[val_mask]
    return v.mean(axis=0), v.std(axis=0, ddof=0)


def aggregate_run(z: np.ndarray, top_k: int = TOP_K) -> tuple[float, np.ndarray]:
    """``(s_run, n_v)`` of one run's ``[w, 60]`` z-map."""
    n = np.median(np.abs(np.asarray(z, dtype=np.float64)), axis=0)
    return float(np.sort(n)[-top_k:].mean()), n


def tukey_flags(n) -> np.ndarray:
    """Indices of the sensors above ``Q3 + 1.5 IQR`` of the run's 60 sensor scores."""
    s = np.asarray(n, dtype=float)
    q1, q3 = np.percentile(s, [25, 75])
    return np.flatnonzero(s > q3 + 1.5 * (q3 - q1))


def dsi(n) -> tuple[float, int]:
    """Damage Severity Index: the sum of the Tukey-flagged sensor scores, and their count."""
    s = np.asarray(n, dtype=float)
    f = tukey_flags(s)
    return float(s[f].sum()), int(f.size)


def score_table(z: np.ndarray, run_id: np.ndarray, runs: pd.DataFrame) -> pd.DataFrame:
    """One model's per-run table: ``damage_level, split, s_run, n0..n59, dsi_iqr, n_anom_iqr``."""
    rows = []
    for r in sorted(set(int(v) for v in run_id)):
        s_run, n = aggregate_run(z[run_id == r])
        d, k = dsi(n)
        rows.append({"run_id": r, "damage_level": int(runs.loc[r, "damage_state"]),
                     "split": runs.loc[r, "split"] if runs.loc[r, "split"] != "damaged" else "test",
                     "s_run": s_run, **{c: float(v) for c, v in zip(NODE_COLS, n)},
                     "dsi_iqr": d, "n_anom_iqr": k})
    return pd.DataFrame(rows).set_index("run_id")


def read_score_table(path) -> pd.DataFrame:
    return pd.read_csv(path).set_index("run_id")


def ensemble(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Seed-mean table; DSI and flag counts are recomputed from the mean sensor scores."""
    base = tables[0]
    for t in tables[1:]:
        if not t.index.equals(base.index):
            raise ValueError("score tables cover different runs")
    out = base[["damage_level", "split"]].copy()
    out["s_run"] = np.mean([t["s_run"].to_numpy(float) for t in tables], axis=0)
    nodes = np.mean([t[NODE_COLS].to_numpy(float) for t in tables], axis=0)
    for j, c in enumerate(NODE_COLS):
        out[c] = nodes[:, j]
    ds = [dsi(row) for row in nodes]
    out["dsi_iqr"] = [d for d, _ in ds]
    out["n_anom_iqr"] = [k for _, k in ds]
    return out


def threshold(table: pd.DataFrame) -> float:
    """``tau`` = mean + 3 population SD of ``s_run`` over the healthy training runs."""
    s = table.loc[table["split"] == "train", "s_run"].to_numpy(float)
    return float(s.mean() + 3 * s.std(ddof=0))
