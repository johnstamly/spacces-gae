"""Claim 6: false alarms under unseen operating conditions (Table 7, Section 5, Fig. 10).

Each of nine operating conditions was withheld in turn from the healthy training and
validation runs, every method was retrained on the remaining healthy runs (neural
models with three seeds) and its threshold re-estimated from them. ``data/unseen``
holds, per withheld condition and method, the score of every run from the retrained
ensemble, and per condition the split and the per-condition severity statistics.

Recomputed here from the per-run scores: the thresholds, every false-alarm count, the
equal-sensitivity counts, exact localization, the pooled-rate test with its Holm
adjustment, the run-2 sensitivity analysis and the position of the threshold. The
severity correlation and worst seed need sensor scores per seed, which the per-run
file does not hold; they are read from ``conditions.json`` (the authors' per-condition
output) and averaged here.

    python reproduce/replay_unseen.py      # seconds
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from _common import REPO, Checker

from spacces_gae import metrics as MX

U = REPO / "data" / "unseen"
GROUPS = {"Five speeds": ["speed_0kn", "speed_13kn", "speed_18kn", "speed_23kn", "speed_27kn"],
          "Height <= 1.5 m": ["h_le1.5"], "Height 2.25-3 m": ["h_2.25-3.0"], "Regular waves": ["reg"],
          "Post-repair": ["postrepair"]}
METHODS = {"E2": "SpaCCES-GAE", "OC": "Tuned [x,c]-OCGNN", "B1": "PCA-95", "B3": "Dense AE"}
T4_SEED, HOLM_M = 20260928, 6        # bootstrap seed of the pre-specified test; Holm over its six primaries

# Table 7 as printed: per method the five group counts, all nine, equal sensitivity, exact, severity, worst seed.
TABLE = {
    "E2": ([2, 2, 0, 1, 1], 6, 2, "0.97", "0.87", "0.60"),
    "OC": ([2, 17, 0, 0, 0], 19, 1, "0.96", "0.90", "0.34"),
    "B1": ([2, 12, 29, 4, 15], 62, 5, "1.00", "0.95", None),
    "B3": ([5, 14, 30, 4, 14], 67, 2, "0.99", "0.84", "0.54"),
}
TEXT = {"rate": {"E2": "3.6", "B1": "37.6", "B3": "40.6", "OC": "11.5"},
        "no_run2": {"E2": 20, "OC": 29, "B1": 62, "B3": 65}, "no_run2_rate_E2": "12.3",
        "median_z": {"E2": "0.07", "B1": "2.8", "B3": "2.6"}}


def tau_of(s) -> float:
    s = np.asarray(s, float)
    return float(s.mean() + 3 * s.std(ddof=0))


def main() -> int:
    runs = pd.read_csv(U / "t4_runs.csv")
    cond = json.loads((U / "conditions.json").read_text(encoding="utf-8"))
    rows = cond["rows"]
    c = Checker("Table 7, unseen operating conditions")
    held = {r: sorted(cond["conditions"][r]["test_ids"]) for r in rows}
    c.eq("held-out healthy runs per group",
         [sum(len(held[r]) for r in g) for g in GROUPS.values()] + [sum(len(v) for v in held.values())],
         [70, 27, 39, 14, 15, 165])

    res, exceed = {}, {}
    for m, label in METHODS.items():
        fa, fa2, n2, eq, det, det2, mismatch, zmed, exact = {}, 0, 0, 0, 0, 0, 0, [], []
        units = []
        for r in rows:
            q = runs[(runs["method"] == m) & (runs["row"] == r)].set_index("run_id")
            train = q[(q["split"] == "train") & (q["damage_level"] == 0)]["s_run"]
            test = q.loc[held[r]]
            if not (test["damage_level"] == 0).all():
                raise SystemExit(f"{m} {r}: held-out runs are not all healthy")
            dmg = q[(q["damage_level"] >= 1) & (q["damage_level"] <= 5)]
            tau = tau_of(train)
            mismatch += int(((q["s_run"] > tau).astype(int) != q["exceeds_tau"]).sum())
            mismatch += int(abs(tau - cond["conditions"][r]["methods"][m]["tau"]) > 1e-4)
            fa[r] = int((test["s_run"] > tau).sum())
            units += [float(v) for v in (test["s_run"] > tau)]
            det += int((dmg["s_run"] > tau).sum())
            tau2 = tau_of(train.drop(index=2, errors="ignore"))          # run 2 left out, no retraining
            test2 = test.drop(index=2, errors="ignore")
            fa2 += int((test2["s_run"] > tau2).sum())
            n2 += len(test2)
            det2 += int((dmg["s_run"] > tau2).sum())
            eq += int((test["s_run"] >= MX.matched_threshold(dmg["s_run"])).sum())
            zmed.append(float(np.median((test["s_run"] - train.mean()) / train.std(ddof=0))))
            exact.append(float(dmg["exact"].mean()))
        exceed[m] = np.array(units)
        cm = [cond["conditions"][r]["methods"][m] for r in rows]
        groups, total, n_eq, ex, sev, worst = TABLE[m]
        c.eq(f"{label}: thresholds recomputed from training runs (mismatches)", mismatch, 0)
        c.eq(f"{label}: false alarms by group", [sum(fa[r] for r in g) for g in GROUPS.values()], groups)
        c.eq(f"{label}: false alarms, all nine", sum(fa.values()), total)
        c.rounds(f"{label}: false-alarm rate % (text)", 100 * sum(fa.values()) / 165, TEXT["rate"][m])
        c.eq(f"{label}: D1-D5 detected, all conditions", f"{det}/450", "450/450")
        c.eq(f"{label}: false alarms at equal sensitivity", eq, n_eq)
        c.rounds(f"{label}: exact localization, mean over conditions", float(np.mean(exact)), ex)
        c.rounds(f"{label}: severity rho_Sigma, mean over conditions (stored)",
                 float(np.mean([v["rho_sigma"] for v in cm])), sev)
        if worst is not None:
            c.rounds(f"{label}: worst seed rho_min over conditions (stored)",
                     float(np.min([v["rho_min"] for v in cm])), worst)
        c.eq(f"{label}: false alarms without run 2 (text)", f"{fa2}/{n2}", f"{TEXT['no_run2'][m]}/163")
        c.eq(f"{label}: D1-D5 missed without run 2", 450 - det2, 0)
        if m in TEXT["median_z"]:
            c.rounds(f"{label}: median held-out healthy z, mean over conditions", float(np.mean(zmed)),
                     TEXT["median_z"][m])
        if m == "E2":
            c.rounds("SpaCCES-GAE: rate % without run 2 (text)", 100 * fa2 / n2, TEXT["no_run2_rate_E2"])
        res[m] = {"false_alarms": fa, "equal_sensitivity": eq, "no_run2": [fa2, n2],
                  "median_z_mean": float(np.mean(zmed)), "exact_mean": float(np.mean(exact))}

    # Pre-specified pooled-rate test: units = (condition, held-out run), resampled within condition.
    strata = np.concatenate([[i] * len(held[r]) for i, r in enumerate(rows)])
    draws = MX.strat_draws(strata, MX.N_BOOT, T4_SEED)
    stored = json.loads((REPO / "results" / "stored" / "unseen" / "t4_tests.json").read_text())["comparisons"]
    for b in ("B1", "B3"):
        r = MX.paired_bootstrap(lambda i: exceed["E2"][i].mean(), lambda i, b=b: exceed[b][i].mean(), draws,
                                floor=True)
        p_holm = MX.holm({"T4": r["p_boot"]}, m=HOLM_M)["T4"]
        c.rounds(f"SpaCCES-GAE vs {METHODS[b]}: pooled-rate test p_Holm", p_holm, "0.003")
        s = stored[f"E2 vs {b}"]["T4"]
        c.eq(f"SpaCCES-GAE vs {METHODS[b]}: difference and CI equal the stored test",
             [round(r["diff"], 10)] + [round(v, 10) for v in r["ci95"]],
             [round(s["diff_A_minus_B"], 10)] + [round(v, 10) for v in s["ci95"]])
        res[f"test_E2_vs_{b}"] = {**r, "p_holm": p_holm}
    dest = REPO / "results" / "replay" / "unseen.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(res, indent=1, default=float) + "\n", encoding="utf-8")
    print(f"  wrote {dest.relative_to(REPO)}")
    return c.done()


if __name__ == "__main__":
    raise SystemExit(main())
