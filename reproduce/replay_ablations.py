"""Claim 7: what each component contributes (Section 6, Fig. 11, Supplementary Tables S-mechanism and S-rc).

Every ablation arm is recomputed from its per-seed score tables in ``data/ablations``
with the scoring chain of the full model. The significance tests are the authors'
paired stratified bootstraps (2000 draws, seed 20261014, units = the 70 healthy runs
and the 50 D1-D5 runs, p not reported below 1/2000) with Holm correction:

* the mechanism family (5-NN graph M1, no identity M2; three primaries each, m = 6),
  recomputed in full;
* the control family (m = 9: no message passing G1b, and two arms not released here),
  where the two unreleased arms enter as p = 1. That bounds every adjusted p from
  above, and it is exact for a test at the bootstrap floor, which the message-passing
  result is.

    python reproduce/replay_ablations.py      # about a minute
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from _common import REPO, Checker, score_tables, sensor_x

from spacces_gae import metrics as MX
from spacces_gae import scoring as S

ABL = REPO / "data" / "ablations"
STORED = REPO / "results" / "stored" / "ablations"
BOOT_SEED, MIN_P = 20261014, 1.0 / MX.N_BOOT

# Supplementary Table S-mechanism (and S-rc for G2) as printed:
# arm, label, seeds, CDR own, CDR eq. sens., AUC, rho, rho_min, fwd, params (registered) or None.
ROWS = [
    ("M1", "3-D 5-NN graph instead", 20, 32, 32, "0.984", "0.878", "0.623", 1, 960),
    ("M2", "sensor identity E removed", 20, 50, 48, "0.977", "0.882", "-0.574", 9, 656),
    ("PHYS1", "response branch removed", 20, 50, 48, "0.969", "0.956", "0.468", 19, 652),
    ("PHYS2", "A_phys in both branches", 20, 38, 38, "0.976", "0.967", "0.296", 13, 960),
    ("SIM1", "Coordinate branch removed", 20, 50, 48, "0.971", "0.759", "0.563", 9, 652),
    ("SIM2", "S_sim in both branches", 20, 50, 48, "0.972", "0.823", "0.510", 9, 960),
    ("DESC", "+7 trace descriptors", 20, 50, 48, "0.974", "0.897", "0.580", 16, 1072),
    ("G1b", "message passing removed", 20, 50, 48, "0.940", "0.932", "-0.747", 0, 960),
    ("alpha_0.50", "alpha = 0.50", 10, 13, 40, "0.912", "0.639", "0.124", 1, None),
    ("alpha_0.70", "alpha = 0.70", 20, 38, 48, "0.954", "0.878", "-0.035", 1, None),
    ("alpha_0.85", "alpha = 0.85", 10, 50, 48, "0.972", "0.949", "0.696", 2, None),
    ("alpha_0.95", "alpha = 0.95", 10, 50, 48, "0.974", "0.903", "0.676", 2, None),
    ("G2", "no message passing, no c (S-rc)", 10, 27, 28, "0.964", "0.432", "-0.413", 2, None),
]
HOLM = [  # family, m, arm, primary, printed p_Holm
    ("mechanism", 6, "M1", "cdr_matched", "0.003"), ("mechanism", 6, "M1", "fwd_top1", "0.045"),
    ("mechanism", 6, "M2", "rho_sigma", "0.003"), ("controls", 9, "G1b", "fwd_top1", "0.0045"),
]
FAMILY_ARMS = {"mechanism": ("M1", "M2"), "controls": ("G1b",)}


def load(arm: str) -> list:
    return [S.read_score_table(p) for p in sorted((ABL / arm).glob("seed_*_scores.csv.gz"))]


def stat_fns(table: pd.DataFrame, x) -> tuple[dict, np.ndarray]:
    """Bootstrap statistics on positions [70 healthy runs, then the 50 D1-D5 runs]."""
    q = MX.run_quantities(table, x)
    healthy = table[table["damage_level"].to_numpy(int) == 0]
    lo, hi = MX.LZ.INTERVALS["fwd"]
    top_x = np.asarray(x, float)[healthy[S.NODE_COLS].to_numpy(float).argmax(axis=1)]
    fwd = ((top_x >= lo) & (top_x <= hi)).astype(float)
    n_h = fwd.size

    def split(idx):
        idx = np.asarray(idx)
        return idx[idx < n_h], idx[idx >= n_h] - n_h

    fns = {"cdr_matched": lambda i: MX.cdr(q, split(i)[1])["cdr_matched"],
           "rho_sigma": lambda i: MX.spearman(q["lev"][split(i)[1]], q["dsi"][split(i)[1]]),
           "fwd_top1": lambda i: float(fwd[split(i)[0]].mean())}
    return fns, np.concatenate([np.zeros(n_h, int), q["lev"]])


def main() -> int:
    x = sensor_x()
    ref_tables = score_tables("stored")
    ref = S.ensemble(ref_tables)
    c = Checker("Ablations, recomputed from per-seed score tables")
    h_ref = MX.headline(ref_tables, x)
    c.eq("reference: healthy forward bias", f"{h_ref['healthy_forward_bias']}/70", "8/70")
    c.eq("reference: CDR at equal sensitivity", f"{round(h_ref['cdr_matched'] * 50)}/50", "48/50")
    stored = pd.read_csv(STORED / "summary_20_seeds.csv")
    out = {"arms": {}, "tests": {}}
    for arm, label, n_seeds, own, eq, auc, rho, rmin, fwd, params in ROWS:
        tabs = load(arm)
        h = MX.headline(tabs, x)
        c.eq(f"{label}: seeds", len(tabs), n_seeds)
        c.eq(f"{label}: CDR own / eq. sens.", (h["n_complete"], round(h["cdr_matched"] * 50)), (own, eq))
        c.rounds(f"{label}: validation AUC", h["validation_auc"], auc)
        c.rounds(f"{label}: rho_Sigma", h["rho_sigma"], rho)
        c.rounds(f"{label}: rho_min", h["rho_min"], rmin)
        c.eq(f"{label}: healthy forward bias", f"{h['healthy_forward_bias']}/70", f"{fwd}/70")
        if params is not None:
            row = stored[(stored["arm"] == ("REF-a1" if arm == "REF" else arm)) & (stored["seed_set"] == "pooled 0-19")]
            c.eq(f"{label}: registered parameters (stored summary)", int(row["n_params"].iloc[0]), params)
        out["arms"][arm] = h
    if "G2" in out["arms"]:
        c.eq("G2: exact localization (S-rc)", out["arms"]["G2"]["localized_d1_d5"], 31)

    ref_fns, strata = stat_fns(ref, x)
    draws = MX.strat_draws(strata, MX.N_BOOT, BOOT_SEED)
    for fam, arms in FAMILY_ARMS.items():
        tests = {}
        for arm in arms:
            fns, st = stat_fns(S.ensemble(load(arm)), x)
            if not np.array_equal(st, strata):
                raise SystemExit(f"{arm}: run set differs from the reference")
            for k in ("cdr_matched", "rho_sigma", "fwd_top1"):
                a, b = (fns[k], ref_fns[k]) if k == "fwd_top1" else (ref_fns[k], fns[k])
                r = MX.paired_bootstrap(a, b, draws)
                r["p_boot"] = max(r["p_boot"], MIN_P)
                tests[(arm, k)] = r
        m = next(mm for f, mm, *_ in HOLM if f == fam)
        adj = MX.holm({k: r["p_boot"] for k, r in tests.items()}, m=m)
        for k, r in tests.items():
            r["p_holm"] = adj[k]
            out["tests"][f"{k[0]}|{k[1]}"] = r
    for fam, m, arm, k, want in HOLM:
        c.rounds(f"{arm} {k}: p_Holm (family {fam}, m = {m})", out["tests"][f"{arm}|{k}"]["p_holm"], want)
    s6 = pd.read_csv(STORED / "holm_mechanism_m6.csv")
    s6 = s6[s6["family_seed_set"] == "pooled 0-19"].set_index("test")
    for t, r in out["tests"].items():
        if t in s6.index:
            c.eq(f"{t}: difference and CI equal the stored test", [f"{r['diff']:.4f}"] + [f"{v:.4f}" for v in r["ci95"]],
                 [f"{s6.loc[t, 'diff']:.4f}", f"{s6.loc[t, 'ci95_lo']:.4f}", f"{s6.loc[t, 'ci95_hi']:.4f}"])
    dest = REPO / "results" / "replay" / "ablations.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8")
    print(f"  wrote {dest.relative_to(REPO)}")
    return c.done()


if __name__ == "__main__":
    raise SystemExit(main())
