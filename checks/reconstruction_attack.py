"""How much of the restricted measurements can be recovered from the released inputs?

Run by the authors only: the check compares an attack on the released data with the
restricted per-window strain variances, which are not part of this repository. It
writes aggregate numbers only (``results/privacy/reconstruction_check.json``).

What is released per window w is x_w = (v_w - min v_w) / (max v_w - min v_w), the
min-max scaled variance of the 60 strain signals over one 1000-sample window, plus
the context c (computed from x) and the response graph S_sim. S_sim is NOT computed
from x: it is a sparse top-5 positive-cosine graph between sensors whose vectors are
seven response descriptors per sensor (log-RMS, log-IQR, lane-residual log-RMS,
strain-rate log-RMS, three spectral band fractions), standardised by each sensor's
healthy-training median and IQR. Cosine is not invariant to a change of strain scale
after that standardisation, so S_sim is tested separately (level 2b).

1. Raw strain records. Each released value summarises 1000 samples by one variance,
   so the time series cannot be recovered from the release: the map is many-to-one.
2. Absolute strain level. x and c are unchanged when every variance of a window is
   multiplied by any k > 0 and shifted by any constant; this is verified numerically.
   2b. S_sim: does any simple summary of the released graph track the true level of
   the window (mean log-variance), within runs or across runs?
3. Relative variance within a run, by three attacks scored against the truth up to
   one unknown scale and offset per run:
   a. Overlap chaining: consecutive windows share 800 of 1000 samples, so chain the
      unknown per-window scale and offset assuming v changes little between windows.
   b. One noise-floor anchor: if sensor s has a roughly constant variance and the
      window minimum is small against it, span_w is proportional to 1 / x_w[s].
   c. Two anchors: if sensors s1, s2 both have roughly constant variance, span_w is
      proportional to 1 / (x_w[s1] - x_w[s2]) and v_w follows up to one affine map.
   Anchors are chosen blind (lowest median x, which an attacker can do) and by an
   oracle that uses the truth (an upper bound for this family of attacks).
4. Side information is not simulated: any two known absolute variances in a window fix
   that window exactly, and since the anchor estimates share one unknown scale, one
   calibrated window roughly calibrates all of them. The article plots absolute levels
   (strain in µm/m, window variance in (µm/m)^2), so the release together with the
   article allows an approximate calibration; the release alone does not.

    python checks/reconstruction_attack.py --private-root <restricted tree> --layout <layout file>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

import _layout  # noqa: E402
from spacces_gae import data as D  # noqa: E402


def minmax(v: np.ndarray) -> np.ndarray:
    lo, hi = v.min(axis=1, keepdims=True), v.max(axis=1, keepdims=True)
    return (v - lo) / (hi - lo)


def chain_attack(x: np.ndarray) -> np.ndarray:
    """Recover v_w = a_w x_w + b_w up to one affine map per run, assuming v_{w+1} ~ v_w.

    Starting from a_0 = 1, b_0 = 0, each window's (a, b) is the least-squares fit of
    the previous recovered vector on the current released vector."""
    out = np.empty_like(x)
    out[0] = x[0]
    for w in range(1, x.shape[0]):
        A = np.stack([x[w], np.ones(x.shape[1])], axis=1)
        (a, b), *_ = np.linalg.lstsq(A, out[w - 1], rcond=None)
        out[w] = a * x[w] + b
    return out


def r2_after_affine(est: np.ndarray, truth: np.ndarray) -> float:
    """R^2 of the truth on the estimate after the best single scale and offset (the per-run unknowns)."""
    e, t = est.ravel(), truth.ravel()
    A = np.stack([e, np.ones_like(e)], axis=1)
    coef, *_ = np.linalg.lstsq(A, t, rcond=None)
    resid = t - A @ coef
    return float(1.0 - resid.var() / t.var())


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _summ(vals) -> dict:
    vals = np.asarray(vals, float)
    return {"median": float(np.median(vals)), "p10": float(np.percentile(vals, 10)),
            "p90": float(np.percentile(vals, 90))}


def anchor_estimates(x: np.ndarray, s1: int, s2: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Estimated (log span, v up to an affine map) from one or two noise-floor anchors."""
    tiny = 1e-6
    if s2 is None:
        d = np.maximum(x[:, s1], tiny)
        return -np.log(d), x / d[:, None]
    d = np.maximum(np.abs(x[:, s1] - x[:, s2]), tiny)
    return -np.log(d), (x - x[:, [s1]]) / d[:, None]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    _layout.add_arguments(ap)
    args = ap.parse_args(argv)
    ras = _layout.Layout(args.private_root, args.layout).pipeline()  # restricted feature pipeline

    tw = ras.tw
    wd, wf = tw.wd, tw.wf
    metadata = tw._metadata()
    tr, va, dmg = wd.run_splits(metadata)
    ids = sorted(set(tr) | set(va) | set(dmg))
    ws = wd.load_window_set(ids, 1000, 200, 1.0, wf.CACHE_DIR, centred=False, subsample_ids=set(tr))
    v = np.exp(np.asarray(ws.values, float))                       # restricted: per-window variances [W, 60]
    rel = D.load_windows()
    x = rel.x.astype(float)
    if not (np.array_equal(rel.run_id, ws.run_ids) and np.array_equal(rel.window, ws.starts // 200)):
        raise SystemExit("released windows do not match the restricted window set")

    # Level 2: invariance of the release to any per-window scale and offset.
    rng = np.random.default_rng(0)
    k = rng.uniform(0.01, 100.0, size=(v.shape[0], 1))
    shift = rng.uniform(-1.0, 1.0, size=(v.shape[0], 1)) * v.min(axis=1, keepdims=True)
    inv = float(np.abs(minmax(v * k + shift) - minmax(v)).max())
    match = float(np.abs(minmax(v) - x).max())

    # Level 3: chain attack per run, scored against the truth up to one affine map per run.
    r2_chain, r2_naive, scale_err = [], [], []
    for r in np.unique(rel.run_id):
        m = rel.run_id == r
        order = np.argsort(rel.window[m])
        xr, vr = x[m][order], v[m][order]
        est = chain_attack(xr)
        r2_chain.append(r2_after_affine(est, vr))
        r2_naive.append(r2_after_affine(xr, vr))                   # attacker ignores the overlap
        span_true = vr.max(axis=1) - vr.min(axis=1)
        span_est = est.max(axis=1) - est.min(axis=1)
        c = np.corrcoef(np.log(span_est), np.log(span_true))[0, 1]
        scale_err.append(float(c))
    runs = np.unique(rel.run_id)
    masks = {r: np.flatnonzero(rel.run_id == r)[np.argsort(rel.window[rel.run_id == r])] for r in runs}
    log_span = np.log(v.max(axis=1) - v.min(axis=1))
    level = np.log(v).mean(axis=1)                                 # window level: mean log-variance

    # Level 2b: does a simple summary of the released S_sim track the window level?
    n_edges = np.diff(rel.sim_ptr).astype(float)
    wid = np.repeat(np.arange(rel.n_windows), np.diff(rel.sim_ptr))
    tot_w = np.bincount(wid, weights=rel.sim_weight.astype(float), minlength=rel.n_windows)
    deg = np.zeros((rel.n_windows, 60))
    for k in (0, 1):
        np.add.at(deg, (wid, rel.sim_edges[:, k].astype(int)), rel.sim_weight.astype(float))
    sim_feats = {"edge_count": n_edges, "total_weight": tot_w,
                 **{f"weighted_degree_{i}": deg[:, i] for i in range(60)}}
    within = {k: [_corr(f[m], level[m]) for m in masks.values()] for k, f in sim_feats.items()}
    pooled = {k: _corr(f, level) for k, f in sim_feats.items()}
    best_deg = max((k for k in sim_feats if k.startswith("weighted")),
                   key=lambda k: abs(np.median(within[k])))

    # Level 3b, 3c: noise-floor anchor attacks.
    med_x = np.median(x, axis=0)
    low = [int(i) for i in np.argsort(med_x)]
    blind_single = next(i for i in low if med_x[i] > 0)
    blind_pair = (low[0], low[1])

    def score(s1, s2):
        ls_est, v_est = anchor_estimates(x, s1, s2)
        cw = [_corr(ls_est[m], log_span[m]) for m in masks.values()]
        r2 = [r2_after_affine(v_est[m], v[m]) for m in masks.values()]
        return cw, r2, _corr(ls_est, log_span)

    single_med = {s: np.median([_corr(-np.log(np.maximum(x[m, s], 1e-6)), log_span[m])
                                for m in masks.values()]) for s in range(60)}
    oracle_single = max(single_med, key=single_med.get)
    pair_med = {}
    for a in range(60):
        for b in range(a + 1, 60):
            d = -np.log(np.maximum(np.abs(x[:, a] - x[:, b]), 1e-6))
            pair_med[(a, b)] = np.median([_corr(d[m], log_span[m]) for m in masks.values()])
    oracle_pair = max(pair_med, key=pair_med.get)
    anchors = {}
    for name, (s1, s2) in {"one_anchor_blind": (blind_single, None), "one_anchor_oracle": (oracle_single, None),
                           "two_anchor_blind": blind_pair, "two_anchor_oracle": oracle_pair}.items():
        cw, r2, pooled_c = score(s1, s2)
        anchors[name] = {"log_span_correlation_within_run": _summ(cw),
                         "log_span_correlation_pooled_across_runs": pooled_c,
                         "r2_after_one_affine_per_run": _summ(r2)}

    # Reference: what the released run table alone says about the level across runs.
    rt = D.load_runs()                                             # indexed by run_id
    wave_h = rt.loc[rel.run_id, "wave_height_m"].to_numpy(float)
    speed = rt.loc[rel.run_id, "speed_kn"].to_numpy(float)
    period = rt.loc[rel.run_id, "wave_period_s"].to_numpy(float)
    regular = (rt.loc[rel.run_id, "wave_type"] == "regular").to_numpy(float)
    X = np.column_stack([np.ones_like(wave_h), wave_h, period, speed, regular, wave_h * period])
    coef, *_ = np.linalg.lstsq(X, log_span, rcond=None)
    resid = log_span - X @ coef                                    # what the run table does not explain
    reference = {"pooled_correlation_log_span_vs_nominal_wave_height": _corr(wave_h, log_span),
                 "pooled_correlation_log_span_vs_nominal_speed": _corr(speed, log_span),
                 "pooled_r2_log_span_on_run_table_conditions": float(1 - resid.var() / log_span.var())}
    for name, (s1, s2) in {"one_anchor_blind": (blind_single, None), "one_anchor_oracle": (oracle_single, None),
                           "two_anchor_blind": blind_pair, "two_anchor_oracle": oracle_pair}.items():
        anchors[name]["correlation_with_log_span_left_unexplained_by_run_table_pooled"] = \
            _corr(anchor_estimates(x, s1, s2)[0], resid)

    res = {
        "n_windows": int(v.shape[0]), "n_runs": int(len(r2_chain)),
        "reference_run_table_only": reference,
        "released_x_equals_minmax_of_restricted_variance_max_abs_diff": match,
        "level2_x_invariant_to_scale_and_offset_max_abs_diff": inv,
        "level2b_s_sim_vs_window_level_mean_log_variance": {
            "edge_count": {"within_run": _summ(within["edge_count"]), "pooled": pooled["edge_count"]},
            "total_weight": {"within_run": _summ(within["total_weight"]), "pooled": pooled["total_weight"]},
            "best_single_weighted_degree_oracle": {"within_run": _summ(within[best_deg]),
                                                   "pooled": pooled[best_deg]}},
        "level3bc_anchor_attacks": anchors,
        "level3_chain_attack_r2_after_one_affine_per_run": {
            "median": float(np.median(r2_chain)), "p10": float(np.percentile(r2_chain, 10)),
            "p90": float(np.percentile(r2_chain, 90))},
        "level3_no_chaining_r2_after_one_affine_per_run": {
            "median": float(np.median(r2_naive)), "p10": float(np.percentile(r2_naive, 10))},
        "level3_window_scale_trend_correlation_log_span": {
            "median": float(np.median(scale_err)), "p10": float(np.percentile(scale_err, 10))},
        "reading": ("Raw records: not recoverable (one variance per 1000 samples). Absolute level in physical "
                    "units: not recoverable from x and c (they are invariant to it); simple summaries of S_sim "
                    "track the window level only weakly (level 2b). Relative level: the noise-floor anchor "
                    "attacks recover part of how the window spread varies between windows and runs, up to one "
                    "unknown scale (level 3bc: log-span correlations, within runs and pooled), including "
                    "information the run-table conditions do not explain (correlation with their residual). "
                    "Within a window, x shows which sensors vary more than others, by design; no attack "
                    "improves the per-run reconstruction over x itself (compare the R^2 blocks). Two known "
                    "absolute variances in any window fix it exactly, and one calibrated window roughly "
                    "calibrates all (the anchor estimates share one scale); the article plots absolute levels. "
                    "These are a few simple attacks: evidence, not a proof that no stronger attack exists."),
    }
    dest = REPO / "results" / "privacy" / "reconstruction_check.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
