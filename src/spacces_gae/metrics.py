"""Endpoints of the paper: detection, severity, complete diagnosis and their bootstraps.

Conventions (manuscript sections "Hierarchical scoring of detection, localization and
severity" and "The complete-diagnosis rate"):

* Detection is ``s_run > tau`` (strict).
* Validation AUC: 57 damaged runs against the 14 healthy validation runs.
* Severity: Spearman rho between the damage state and the DSI over the 50 D1-D5 runs.
* Complete diagnosis (the CDR equation): a D1-D5 run is detected, localized (a flagged sensor
  in every damaged interval) and severity-consistent (more than half of its
  cross-state pairs ordered correctly by DSI). CDR is the share of such runs.
* Bootstraps use fixed seeds, so every interval is reproducible exactly:
  validation AUC, class-stratified, 2000 draws, seed 0; the two-level (seeds x
  runs) bootstrap, 2000 draws, seed draws from 20261010 and run draws from 20261014.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from . import localization as LZ
from .scoring import NODE_COLS, dsi, ensemble, threshold

N_BOOT = 2000
AUC_BOOT_SEED = 0
RUN_BOOT_SEED = 20261014
SEED_BOOT_SEED = 20261010


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan")
    return float(spearmanr(x, y)[0])


def heldout_arrays(table: pd.DataFrame, exclude_d6: bool = False) -> tuple[np.ndarray, np.ndarray]:
    lev = table["damage_level"].to_numpy(int)
    split = table["split"].astype(str).to_numpy()
    neg = (lev == 0) & (split == "val")
    pos = (lev > 0) & ((lev != 6) if exclude_d6 else True)
    keep = neg | pos
    return pos[keep].astype(int), table["s_run"].to_numpy(float)[keep]


def bootstrap_auc_ci(y, s, n_boot: int = N_BOOT, seed: int = AUC_BOOT_SEED) -> list[float]:
    """Percentile 95 % interval of the AUC, class-stratified resampling."""
    y, s = np.asarray(y), np.asarray(s, float)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    aucs = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, pos.size, replace=True), rng.choice(neg, neg.size, replace=True)])
        aucs.append(roc_auc_score(y[idx], s[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return [round(float(lo), 10), round(float(hi), 10)]


def concordance(levels, dsi_values) -> np.ndarray:
    """Per run: share of its cross-state pairs whose DSI order matches the state order."""
    lev, d = np.asarray(levels, float), np.asarray(dsi_values, float)
    dl, dd = np.sign(lev[:, None] - lev[None, :]), np.sign(d[:, None] - d[None, :])
    cross = dl != 0
    pairs, matches = cross.sum(1), (cross & (dd == dl)).sum(1)
    return np.divide(matches, pairs, out=np.full(lev.size, np.nan), where=pairs > 0)


def matched_threshold(scores, q: float = 0.95) -> float:
    """Largest threshold with a true-positive rate of at least ``q`` (used with ``>=``)."""
    s = np.sort(np.asarray(scores, float))
    return float(s[s.size - math.ceil(q * s.size - 1e-12)])


def run_quantities(table: pd.DataFrame, x_mm) -> dict:
    """Per D1-D5 run: score, state, DSI and localization; and the ensemble threshold."""
    lev_all = table["damage_level"].to_numpy(int)
    d15 = (lev_all >= 1) & (lev_all <= 5)
    nodes = table[NODE_COLS].to_numpy(float)[d15]
    lev = lev_all[d15]
    exact = np.array([LZ.localized(n, x_mm, L) for n, L in zip(nodes, lev)], bool)
    return {"s": table["s_run"].to_numpy(float)[d15], "lev": lev, "dsi": table["dsi_iqr"].to_numpy(float)[d15],
            "exact": exact, "tau": threshold(table), "run_id": table.index.to_numpy()[d15]}


def cdr(q: dict, idx=None) -> dict:
    """Complete-diagnosis rate at the own threshold and at the matched (TPR >= 0.95) threshold."""
    s, lev, exact, d = q["s"], q["lev"], q["exact"], q["dsi"]
    if idx is not None:
        s, lev, exact, d = s[idx], lev[idx], exact[idx], d[idx]
    consistent = concordance(lev, d) > 0.5
    own = (s > q["tau"]) & exact & consistent
    matched = (s >= matched_threshold(s)) & exact & consistent
    return {"cdr_own": float(own.mean()), "cdr_matched": float(matched.mean()),
            "n_complete": int(own.sum()), "n_localized": int(exact.sum()), "n_consistent": int(consistent.sum())}


def strat_draws(strata, n_boot: int, seed: int) -> np.ndarray:
    """``[n_boot, n]`` positions resampled with replacement within each stratum."""
    strata = np.asarray(strata)
    groups = [np.flatnonzero(strata == g) for g in sorted(set(strata.tolist()), key=str)]
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, strata.size), dtype=np.int64)
    for b in range(n_boot):
        out[b] = np.concatenate([rng.choice(g, g.size, replace=True) for g in groups])
    return out


def two_level_bootstrap(tables: list[pd.DataFrame], x_mm, n_boot: int = N_BOOT) -> dict:
    """Seeds and runs resampled together: 95 % intervals of CDR, rho and validation AUC."""
    base = ensemble(tables)
    S = np.stack([t["s_run"].to_numpy(float) for t in tables])
    NODES = np.stack([t[NODE_COLS].to_numpy(float) for t in tables])
    m = len(tables)
    q_full = run_quantities(base, x_mm)
    n_h = int((base["damage_level"].to_numpy(int) == 0).sum())
    rdraws = strat_draws(np.concatenate([np.zeros(n_h, int), q_full["lev"]]), n_boot, RUN_BOOT_SEED)
    lev = base["damage_level"].to_numpy(int)
    split = base["split"].astype(str).to_numpy()
    akeep = ((lev == 0) & (split == "val")) | (lev > 0)
    ay, apos = (lev[akeep] > 0).astype(int), np.flatnonzero(akeep)
    adraws = strat_draws(lev[akeep], n_boot, RUN_BOOT_SEED)
    rng = np.random.default_rng(SEED_BOOT_SEED)
    sdraws = np.stack([rng.integers(0, m, m) for _ in range(n_boot)])

    def stats(seed_idx, ridx, aidx):
        t = base[["damage_level", "split"]].copy()
        t["s_run"] = S[seed_idx].mean(axis=0)
        nodes = NODES[seed_idx].mean(axis=0)
        for j, c in enumerate(NODE_COLS):
            t[c] = nodes[:, j]
        t["dsi_iqr"] = [dsi(r)[0] for r in nodes]
        q = run_quantities(t, x_mm)
        d = None if ridx is None else ridx[ridx >= n_h] - n_h
        c = cdr(q, d)
        lev_d = q["lev"] if d is None else q["lev"][d]
        dsi_d = q["dsi"] if d is None else q["dsi"][d]
        sc = t["s_run"].to_numpy(float)[apos]
        ai = np.arange(ay.size) if aidx is None else aidx
        return {"cdr_own": c["cdr_own"], "cdr_matched": c["cdr_matched"], "rho_sigma": spearman(lev_d, dsi_d),
                "heldout_auc_all": float(roc_auc_score(ay[ai], sc[ai]))}

    full = np.arange(m)
    point = stats(full, None, None)
    rows_run = [stats(full, rdraws[b], adraws[b]) for b in range(n_boot)]
    rows_two = [stats(sdraws[b], rdraws[b], adraws[b]) for b in range(n_boot)]
    out = {"point": point}
    for name, rows in (("run_only", rows_run), ("two_level", rows_two)):
        df = pd.DataFrame(rows)
        out[name] = {k: [float(v) for v in np.percentile(df[k].dropna(), [2.5, 97.5])] for k in point}
    return out


def holm(pvalues: dict, m: int | None = None) -> dict:
    """Holm step-down adjusted p over a family of ``m`` hypotheses (missing members count as p = 1)."""
    m = len(pvalues) if m is None else int(m)
    vals = sorted([(float(p), k) for k, p in pvalues.items()] + [(1.0, None)] * (m - len(pvalues)),
                  key=lambda t: t[0])
    out, running = {}, 0.0
    for i, (p, k) in enumerate(vals):
        running = max(running, min(1.0, (m - i) * p))
        if k is not None:
            out[k] = running
    return out


def paired_bootstrap(stat_a, stat_b, draws: np.ndarray, floor: bool = False) -> dict:
    """``stat_a - stat_b`` on the full sample and on each draw (the same draw for both).

    The two-sided p is ``2 min(P(d <= 0), P(d >= 0))``; with ``floor`` it is not
    reported below ``1 / n_draws`` (the convention of the unseen-condition test)."""
    full = np.arange(draws.shape[1])
    diffs = np.array([stat_a(i) - stat_b(i) for i in draws], float)
    diffs = diffs[np.isfinite(diffs)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = min(1.0, 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    if floor:
        p = max(1.0 / diffs.size, p)
    return {"diff": float(stat_a(full) - stat_b(full)), "ci95": [float(lo), float(hi)], "p_boot": float(p)}


def healthy_forward_bias(table: pd.DataFrame, x_mm) -> int:
    """Healthy runs whose top-ranked sensor lies inside the forward damage interval."""
    lo, hi = LZ.INTERVALS["fwd"]
    healthy = table[table["damage_level"].to_numpy(int) == 0]
    top = healthy[NODE_COLS].to_numpy(float).argmax(axis=1)
    x = np.asarray(x_mm, float)[top]
    return int(((x >= lo) & (x <= hi)).sum())


def headline(tables: list[pd.DataFrame], x_mm) -> dict:
    """Every Table 5 endpoint of an ensemble of per-seed score tables (point values)."""
    t = ensemble(tables)
    lev = t["damage_level"].to_numpy(int)
    split = t["split"].astype(str).to_numpy()
    s = t["s_run"].to_numpy(float)
    tau = threshold(t)
    y, sc = heldout_arrays(t)
    q = run_quantities(t, x_mm)
    two_site = np.isin(q["lev"], (4, 5))
    per_seed_rho = []
    for tab in tables:
        lv = tab["damage_level"].to_numpy(int)
        k = (lv >= 1) & (lv <= 5)
        per_seed_rho.append(spearman(lv[k], tab["dsi_iqr"].to_numpy(float)[k]))
    return {
        "n_seeds": len(tables), "n_runs": int(len(t)), "tau": tau,
        "validation_auc": float(roc_auc_score(y, sc)),
        "detected_d1_d5": int(((lev >= 1) & (lev <= 5) & (s > tau)).sum()),
        "false_alarms_validation": int(((lev == 0) & (split == "val") & (s > tau)).sum()),
        "false_alarms_healthy70": int(((lev == 0) & (s > tau)).sum()),
        "detected_d6": int(((lev == 6) & (s > tau)).sum()),
        "localized_d1_d5": int(q["exact"].sum()),
        "localized_two_site": int(q["exact"][two_site].sum()), "n_two_site": int(two_site.sum()),
        "rho_sigma": spearman(q["lev"], q["dsi"]),
        "rho_min": float(np.nanmin(per_seed_rho)),
        "healthy_forward_bias": healthy_forward_bias(t, x_mm),
        **cdr(q),
    }
