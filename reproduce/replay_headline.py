"""Claims 1-4: detection, localization, severity and complete diagnosis (Table 5 and Section 4.1).

Replays the twenty-seed ensemble from its per-seed score tables and checks every
Table 5 value against the manuscript, including the bootstrap intervals.

    python reproduce/replay_headline.py                  # stored score tables (seconds; intervals ~10 min)
    python reproduce/replay_headline.py --source rescored  # tables recomputed from data/inputs + models/
    python reproduce/replay_headline.py --no-intervals     # point values only
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.metrics import roc_auc_score

from _common import REPO, Checker, score_tables, sensor_x

from spacces_gae import metrics as MX
from spacces_gae import scoring as S

# Manuscript values (Table 5 and Section 4.1), as printed.
EXPECTED = {
    "tau": "1.74", "validation_auc": "0.975", "validation_auc_ci": ("0.937", "1.000"),
    "detected_d1_d5": 50, "false_alarms_validation": 0, "detected_d6": 0,
    "localized_d1_d5": 50, "localized_two_site": 11, "n_two_site": 11,
    "rho_sigma": "0.962", "rho_ci": ("0.843", "0.967"), "rho_min": "0.784", "n_complete": 50,
    "run66_s_run": "5.84", "run66_dsi": "31.2", "run66_flags": 8,
}


def auc_between(pos, neg) -> float:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return float(roc_auc_score(np.r_[np.ones(pos.size), np.zeros(neg.size)], np.r_[pos, neg]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("stored", "rescored"), default="stored")
    ap.add_argument("--no-intervals", action="store_true", help="skip the bootstrap intervals")
    args = ap.parse_args(argv)
    tables, x = score_tables(args.source), sensor_x()
    h = MX.headline(tables, x)
    ens = S.ensemble(tables)
    c = Checker(f"Table 5, twenty-seed ensemble ({args.source} score tables)")
    c.rounds("threshold tau", h["tau"], EXPECTED["tau"])
    c.rounds("validation AUC (57 damaged vs 14 healthy validation)", h["validation_auc"], EXPECTED["validation_auc"])
    c.eq("D1-D5 runs detected", f"{h['detected_d1_d5']}/50", f"{EXPECTED['detected_d1_d5']}/50")
    c.eq("false alarms, healthy validation runs", f"{h['false_alarms_validation']}/14", "0/14")
    c.eq("D6 runs detected", f"{h['detected_d6']}/7", "0/7")
    c.eq("flagged sensor in every damaged interval", f"{h['localized_d1_d5']}/50", "50/50")
    c.eq("both positions, two-location runs", f"{h['localized_two_site']}/{h['n_two_site']}", "11/11")
    c.rounds("severity rho_Sigma", h["rho_sigma"], EXPECTED["rho_sigma"])
    c.rounds("worst single seed rho_min", h["rho_min"], EXPECTED["rho_min"])
    c.eq("complete diagnosis at own threshold", f"{h['n_complete']}/50", "50/50")
    # Section 4.1 text, Fig. 7 and Supplementary Table S-halves-full.
    lev, split, s = ens["damage_level"].to_numpy(int), ens["split"].astype(str).to_numpy(), ens["s_run"].to_numpy(float)
    healthy, d15 = lev == 0, (lev >= 1) & (lev <= 5)
    c.rounds("highest healthy validation s_run", float(s[healthy & (split == "val")].max()), "0.82")
    c.rounds("lowest D1-D5 s_run", float(s[d15].min()), "2.41")
    y, sc = MX.heldout_arrays(ens, exclude_d6=True)
    c.rounds("validation AUC, D1-D5 only", float(roc_auc_score(y, sc)), "1.000")
    c.rounds("AUC all 127 runs, 70 healthy negatives", float(roc_auc_score((lev > 0).astype(int), s)), "0.967")
    fa_runs = ens.index[healthy & (s > h["tau"])].tolist()
    c.eq("false alarms across all 70 healthy runs (run ids)", fa_runs, [2])
    per_seed = [MX.headline([t], x) for t in tables]
    c.eq("validation AUC range over the 20 seeds",
         (f"{min(p['validation_auc'] for p in per_seed):.3f}", f"{max(p['validation_auc'] for p in per_seed):.3f}"),
         ("0.950", "0.994"))
    c.eq("rho_Sigma range over the 20 seeds",
         (f"{min(p['rho_sigma'] for p in per_seed):.3f}", f"{max(p['rho_sigma'] for p in per_seed):.3f}"),
         ("0.784", "0.965"))
    c.eq("individual seeds completing all 50 runs alone", sum(p["n_complete"] == 50 for p in per_seed), 12)
    c.eq("CDR at equal sensitivity", f"{round(h['cdr_matched'] * 50)}/50", "48/50")
    c.eq("healthy runs with top-1 sensor in the forward position", f"{h['healthy_forward_bias']}/70", "8/70")
    pre, post = healthy & (ens.index <= 55), healthy & (ens.index >= 113)
    c.rounds("AUC post-repair vs pre-damage healthy runs", auc_between(s[post], s[pre]), "0.82")
    d6 = lev == 6
    c.rounds("AUC D6 vs pre-damage healthy runs", auc_between(s[d6], s[pre]), "0.82")
    c.rounds("AUC D6 vs post-repair healthy runs", auc_between(s[d6], s[post]), "0.52")
    r66 = ens.loc[66]
    c.rounds("Fig. 1 run 66 s_run", float(r66["s_run"]), EXPECTED["run66_s_run"])
    c.rounds("Fig. 1 run 66 DSI", float(r66["dsi_iqr"]), EXPECTED["run66_dsi"])
    c.eq("Fig. 1 run 66 flagged sensors |A|", int(r66["n_anom_iqr"]), EXPECTED["run66_flags"])
    out = {"source": args.source, "point": h}
    if not args.no_intervals:
        y, s = MX.heldout_arrays(ens)
        ci = MX.bootstrap_auc_ci(y, s)
        c.rounds("validation AUC 95 % CI, lower", ci[0], EXPECTED["validation_auc_ci"][0])
        c.rounds("validation AUC 95 % CI, upper", ci[1], EXPECTED["validation_auc_ci"][1])
        print("  two-level bootstrap (2000 draws; a few minutes) ...", flush=True)
        b = MX.two_level_bootstrap(tables, x)
        c.rounds("rho_Sigma 95 % CI (seeds x runs), lower", b["two_level"]["rho_sigma"][0], EXPECTED["rho_ci"][0])
        c.rounds("rho_Sigma 95 % CI (seeds x runs), upper", b["two_level"]["rho_sigma"][1], EXPECTED["rho_ci"][1])
        out.update({"validation_auc_ci95": ci, "two_level_bootstrap": b})
    dest = REPO / "results" / "replay" / f"headline_{args.source}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8")
    print(f"  wrote {dest.relative_to(REPO)}")
    return c.done()


if __name__ == "__main__":
    raise SystemExit(main())
