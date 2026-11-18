"""Claim 5: comparison with eight graph anomaly detectors and two non-graph baselines (Table 6).

Every Table 6 value except the parameter counts is recomputed here from the per-seed
score tables in ``data/comparators`` with the same ensemble, threshold and diagnosis
rules as SpaCCES-GAE (``spacces_gae.metrics``). The paired CDR differences use the
original bootstrap families: the same 2000 stratified draws for both methods of a
comparison, the original seed of each family, and Holm correction over the
family's primary comparisons (own and equal-sensitivity CDR of each detector).

    python reproduce/replay_comparators.py          # under a minute
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from _common import REPO, Checker, score_tables, sensor_x

from spacces_gae import metrics as MX
from spacces_gae import scoring as S

COMP = REPO / "data" / "comparators"
STORED = REPO / "results" / "stored" / "comparators"

# Bootstrap families as run by the authors: seed, Holm family size, primary comparisons (key, window set).
FAMILIES = {
    "published": {"seed": 20261006, "m": 4, "primary": [("B4_all", "all"), ("B5_aligned", "aligned")],
                  "descriptive": [("B1_all", "all"), ("B3_all", "all")]},
    "published_ext": {"seed": 20261012, "m": 12,
                      "primary": [("B6_all", "all"), ("B9_all", "all"), ("B10_all", "all"),
                                  ("B7_aligned", "aligned"), ("B8_aligned", "aligned"), ("B11_aligned", "aligned")]},
    "tuned": {"seed": 20261013, "m": 16,
              "primary": [(f"{d}_H_{w}", w) for d, w in (("B4", "all"), ("B5", "aligned"), ("B6", "all"),
                                                       ("B7", "aligned"), ("B8", "aligned"), ("B9", "all"),
                                                       ("B10", "all"), ("B11", "aligned"))]},
}

# Table 6 as printed: key, family, row label, params, AUC, D6, loc D1-D5, 2-loc, rho, CDR, delta, CI, significant.
ROWS = [
    ("B4_all", "published", "DOMINANT", 6545, "0.957", 0, 0, 0, "-0.051", 0, "+1.00", ("1.00", "1.00"), True),
    ("B6_all", "published_ext", "GAD-NR", 88003, "0.961", 0, 27, 0, "0.366", 24, "+0.52", ("0.46", "0.60"), True),
    ("B9_all", "published_ext", "GCAD", None, "0.970", 0, 24, 0, "0.712", 24, "+0.52", ("0.46", "0.58"), True),
    ("B10_all", "published_ext", "OCGNN", 4288, "0.951", 0, 0, 0, "0.653", 0, "+1.00", ("1.00", "1.00"), True),
    ("B5_aligned", "published", "GDN", 4801, "0.962", 2, 39, 0, "0.461", 30, "+0.40", ("0.30", "0.52"), True),
    ("B7_aligned", "published_ext", "MTAD-GAT", 415015, "0.991", 3, 42, 3, "0.853", 42, "+0.16", ("0.10", "0.26"), True),
    ("B8_aligned", "published_ext", "MTGFlow", 9952, "0.975", 0, 27, 0, "0.755", 27, "+0.46", ("0.46", "0.50"), True),
    ("B11_aligned", "published_ext", "FuSAGNet", 481297, "0.975", 1, 39, 0, "0.786", 38, "+0.24", ("0.22", "0.28"), True),
    ("B9_H_all", "tuned", "GCAD tuned", None, "0.974", 0, 0, 0, "0.364", 0, "+1.00", ("1.00", "1.00"), True),
    ("B10_H_all", "tuned", "OCGNN tuned [x,c]", 48, "0.961", 0, 50, 11, "0.899", 50, "0.00", ("0.00", "0.00"), False),
    ("B5_H_aligned", "tuned", "GDN tuned [x,c]", 2561, "0.952", 1, 39, 0, "0.715", 39, "+0.22", ("0.22", "0.28"), True),
    ("B11_H_aligned", "tuned", "FuSAGNet tuned [x,c]", 1565217, "0.972", 0, 39, 0, "0.611", 34, "+0.32", ("0.26", "0.48"), True),
    ("B1_all", "published", "PCA-95", 420, "1.000", 4, 50, 11, "0.964", 50, "0.00", ("0.00", "0.00"), False),
    ("B3_all", "published", "Dense AE (10 seeds)", 977, "1.000", 3, 50, 11, "0.942", 50, "0.00", ("0.00", "0.00"), False),
]
STORED_KEY = {"B1_all": ("published", "all", "B1"), "B3_all": ("published", "all", "B3")}


def load(key: str) -> list:
    paths = sorted((COMP / key).glob("seed_*_scores.csv.gz"))
    if not paths:
        raise SystemExit(f"no score tables in {COMP / key}")
    return [S.read_score_table(p) for p in paths]


def signed(v: float) -> str:
    """The table's format: a sign on non-zero differences, none on 0.00."""
    s = f"{abs(v):.2f}"
    return s if s == "0.00" else ("+" if v > 0 else "-") + s


def stored_params(key: str) -> int | None:
    fam, wset, meth = STORED_KEY.get(key, (None, None, None))
    if fam is None:
        fam = "tuned" if "_H_" in key else ("published" if key in ("B4_all", "B5_aligned") else "published_ext")
        meth, wset = key.rsplit("_", 1)
    df = pd.read_csv(STORED / f"{fam}_summary.csv")
    v = df[(df["method"] == meth) & (df["window_set"] == wset)]["n_params"].iloc[0]
    return None if pd.isna(v) or int(v) == 0 else int(v)     # GCAD has no trainable parameters: n/a in the table


def main() -> int:
    x = sensor_x()
    ref = {"all": score_tables("stored"), "aligned": load("E2_aligned")}
    q_ref = {w: MX.run_quantities(S.ensemble(t), x) for w, t in ref.items()}
    if not np.array_equal(q_ref["all"]["lev"], q_ref["aligned"]["lev"]):
        raise SystemExit("D1-D5 run sets differ between window sets")
    c = Checker("Table 6, comparators recomputed from their per-seed score tables")

    # SpaCCES-GAE rows and the parameter count shown (registered minus the 40 inert structure maps).
    h_all, h_al = MX.headline(ref["all"], x), MX.headline(ref["aligned"], x)
    c.rounds("SpaCCES-GAE aligned windows AUC", h_al["validation_auc"], "0.976")
    c.rounds("SpaCCES-GAE aligned windows rho_Sigma", h_al["rho_sigma"], "0.942")
    # Table 6 shows 920 = 960 registered minus the 40 inert structure-decoder parameters (counted in tests/).
    e2 = pd.read_csv(STORED / "published_summary.csv").query("method == 'E2'")
    c.eq("SpaCCES-GAE registered parameters (stored summary)", int(e2["n_params"].iloc[0]), 960)

    verdicts, out = {}, {"rows": {}, "families": {}}
    for fam, spec in FAMILIES.items():
        draws = MX.strat_draws(q_ref["all"]["lev"], MX.N_BOOT, spec["seed"])
        res, pvals = {}, {}
        for key, wset in spec["primary"] + spec.get("descriptive", []):
            q = MX.run_quantities(S.ensemble(load(key)), x)
            for var in ("own", "matched"):
                r = MX.paired_bootstrap(lambda i, v=var, w=wset: MX.cdr(q_ref[w], i)[f"cdr_{v}"],
                                        lambda i, v=var: MX.cdr(q, i)[f"cdr_{v}"], draws)
                res[(key, var)] = r
                if (key, wset) in spec["primary"]:
                    pvals[(key, var)] = r["p_boot"]
        adj = MX.holm(pvals, m=spec["m"])
        for k, r in res.items():
            r["p_holm"] = adj.get(k)
        verdicts[fam] = res
        out["families"][fam] = {"seed": spec["seed"], "holm_m": spec["m"],
                                "comparisons": {f"{k[0]}|{k[1]}": v for k, v in res.items()}}
        print(f"  bootstrap family {fam}: {len(res)} comparisons, Holm over {spec['m']}", flush=True)

    for key, fam, label, params, auc, d6, loc, loc2, rho, n_cdr, delta, ci, sig in ROWS:
        h = MX.headline(load(key), x)
        r = verdicts[fam][(key, "own")]
        c.rounds(f"{label}: validation AUC", h["validation_auc"], auc)
        c.eq(f"{label}: D6 detected", f"{h['detected_d6']}/7", f"{d6}/7")
        c.eq(f"{label}: localized D1-D5", f"{h['localized_d1_d5']}/50", f"{loc}/50")
        c.eq(f"{label}: both positions D4/D5", f"{h['localized_two_site']}/11", f"{loc2}/11")
        c.rounds(f"{label}: rho_Sigma", h["rho_sigma"], rho)
        c.eq(f"{label}: CDR own", f"{h['n_complete']}/50", f"{n_cdr}/50")
        c.eq(f"{label}: false alarms on 14 validation runs", h["false_alarms_validation"], 0)
        c.eq(f"{label}: delta CDR", signed(r["diff"]), delta)
        c.eq(f"{label}: delta CDR 95 % CI", (f"{r['ci95'][0]:.2f}", f"{r['ci95'][1]:.2f}"), ci)
        c.eq(f"{label}: significant after Holm", r["p_holm"] is not None and r["p_holm"] < 0.05, sig)
        c.eq(f"{label}: params (stored summary)", stored_params(key), params)
        out["rows"][key] = {"label": label, "family": fam, "headline": h, "delta_cdr_own": r}
    tuned_p = [verdicts["tuned"][(k, "own")]["p_holm"] for k, f, *_ in ROWS if f == "tuned"]
    c.eq("tuned detectors with p_Holm < 0.01, of the 8 (text)",
         sum(verdicts["tuned"][(k, "own")]["p_holm"] < 0.01 for k, _ in FAMILIES["tuned"]["primary"]), 7)
    dest = REPO / "results" / "replay" / "comparators.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8")
    print(f"  wrote {dest.relative_to(REPO)} (tuned rows p_Holm: {[round(p, 4) for p in tuned_p]})")
    return c.done()


if __name__ == "__main__":
    raise SystemExit(main())
