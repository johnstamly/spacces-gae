"""Plain redraws of the data behind the five results figures (Figs. 7-11 of the manuscript).

These are data views for checking, drawn from the released files with default
matplotlib styling; they are not the typeset manuscript figures. Run the replays
first (they write ``results/replay/*.json``).

    python figures/plot_results.py        # writes results/figures/*.png

Fig. 9 (seed-subset curve) is drawn from the authors' exhaustive subset table
(``results/stored/headline/seed_curve.csv``); this script recomputes its M = 1, 2
and 20 rows exhaustively and checks them against the table before drawing.
"""
from __future__ import annotations

import itertools
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "reproduce"))

from _common import score_tables, sensor_x  # noqa: E402
from spacces_gae import data as D  # noqa: E402
from spacces_gae import metrics as MX  # noqa: E402
from spacces_gae import scoring as S  # noqa: E402

OUT = REPO / "results" / "figures"


def fig_detect(ens: pd.DataFrame, tau: float) -> None:
    lev, split, s = ens["damage_level"].to_numpy(int), ens["split"].astype(str).to_numpy(), ens["s_run"].to_numpy()
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [2.2, 1]})
    groups = {"healthy, fitting": (lev == 0) & (split == "train"), "healthy, validation": (lev == 0) & (split == "val"),
              "D1-D5": (lev >= 1) & (lev <= 5), "D6": lev == 6}
    for name, m in groups.items():
        a.scatter(ens.index[m], s[m], s=12, label=name)
    a.axhline(tau, ls="--", c="k", lw=1, label=f"tau = {tau:.2f}")
    a.set(xlabel="run id", ylabel="s_run", yscale="log", title="(a) detection statistic")
    a.legend(fontsize=8)
    y_all = (lev > 0).astype(int)
    fpr, tpr, _ = roc_curve(y_all, s)
    b.plot(fpr, tpr, label=f"all 127 runs, AUC {roc_auc_score(y_all, s):.3f}")
    keep = ((lev == 0) & (split == "val")) | ((lev >= 1) & (lev <= 5))
    fpr, tpr, _ = roc_curve(lev[keep] > 0, s[keep])
    b.plot(fpr, tpr, label=f"validation, D1-D5, AUC {roc_auc_score(lev[keep] > 0, s[keep]):.3f}")
    b.set(xlabel="false-positive rate", ylabel="true-positive rate", title="(b) ROC")
    b.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig07_detect.png", dpi=150)


def fig_heatmap(ens: pd.DataFrame) -> None:
    sens = D.load_sensors()
    order = sens.sort_values(["lane", "x_mm"]).index.to_numpy()
    lev = ens["damage_level"].to_numpy(int)
    mat = np.stack([ens.loc[lev == L, S.NODE_COLS].to_numpy(float).mean(axis=0)[order] for L in range(7)])
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [2, 1]})
    im = a.imshow(mat, aspect="auto", cmap="Greys")
    a.set(yticks=range(7), yticklabels=["H"] + [f"D{L}" for L in range(1, 7)], xlabel="sensor (by lane, then x)",
          title="(a) mean sensor score n_v by state")
    fig.colorbar(im, ax=a)
    for L in range(1, 7):
        d = ens.loc[lev == L, "dsi_iqr"].to_numpy(float)
        b.scatter(np.full(d.size, L), d, s=12)
        b.hlines(np.median(d), L - 0.3, L + 0.3, colors="k")
    b.set(xlabel="damage state", ylabel="DSI", title="(b) severity index")
    fig.tight_layout()
    fig.savefig(OUT / "fig08_heatmap_severity.png", dpi=150)


def subset_values(tables, M: int, x) -> dict:
    vals = {"auc_all": [], "rho_d1_d5": []}
    for sub in itertools.combinations(range(len(tables)), M):
        ens = S.ensemble([tables[i] for i in sub])
        lev = ens["damage_level"].to_numpy(int)
        vals["auc_all"].append(roc_auc_score((lev > 0).astype(int), ens["s_run"].to_numpy(float)))
        d = (lev >= 1) & (lev <= 5)
        vals["rho_d1_d5"].append(MX.spearman(lev[d], ens["dsi_iqr"].to_numpy(float)[d]))
    return {k: np.asarray(v) for k, v in vals.items()}


def fig_seedcurve(tables, x) -> None:
    curve = pd.read_csv(REPO / "results" / "stored" / "headline" / "seed_curve.csv")
    for M in (1, 2, 20):
        got = subset_values(tables, M, x)
        for metric, v in got.items():
            row = curve[(curve["M"] == M) & (curve["metric"] == metric)].iloc[0]
            if int(row["n_subsets"]) != v.size or not np.isclose(np.median(v), row["median"], atol=1e-12):
                raise SystemExit(f"seed curve M={M} {metric}: recomputed median {np.median(v)} != stored {row['median']}")
    print("  seed curve: M = 1, 2, 20 recomputed exhaustively and equal to the stored table")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, metric, title in zip(axes, ("auc_all", "rho_d1_d5"), ("AUC, 57 damaged vs 70 healthy", "rho_Sigma (D1-D5)")):
        c = curve[curve["metric"] == metric].sort_values("M")
        ax.plot(c["M"], c["median"], c="k")
        ax.fill_between(c["M"], c["q05"], c["q95"], color="0.8")
        ax.set(xlabel="ensemble size M", title=title)
    fig.tight_layout()
    fig.savefig(OUT / "fig09_seedcurve.png", dpi=150)


def fig_unseen() -> None:
    runs = pd.read_csv(REPO / "data" / "unseen" / "t4_runs.csv")
    cond = json.loads((REPO / "data" / "unseen" / "conditions.json").read_text(encoding="utf-8"))
    groups = {"speeds": [r for r in cond["rows"] if r.startswith("speed")], "H <= 1.5 m": ["h_le1.5"],
              "H 2.25-3 m": ["h_2.25-3.0"], "regular": ["reg"], "post-repair": ["postrepair"]}
    methods = {"E2": "SpaCCES-GAE", "OC": "tuned OCGNN", "B1": "PCA-95", "B3": "dense AE"}
    fig, ax = plt.subplots(figsize=(11, 4))
    for j, (m, label) in enumerate(methods.items()):
        for g, (gname, rows) in enumerate(groups.items()):
            z = []
            for r in rows:
                q = runs[(runs["method"] == m) & (runs["row"] == r)].set_index("run_id")
                tr = q[(q["split"] == "train") & (q["damage_level"] == 0)]["s_run"]
                z += list((q.loc[cond["conditions"][r]["test_ids"], "s_run"] - tr.mean()) / tr.std(ddof=0))
            xs = g * 5 + j + np.random.default_rng(g).uniform(-0.25, 0.25, len(z))
            ax.scatter(xs, z, s=8, label=label if g == 0 else None, color=f"C{j}")
            ax.hlines(np.median(z), g * 5 + j - 0.35, g * 5 + j + 0.35, colors="k")
    ax.axhline(3, ls="--", c="k", lw=1)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xticks([g * 5 + 1.5 for g in range(len(groups))], list(groups))
    ax.set(ylabel="z (training SDs); tau = 3")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig10_unseen.png", dpi=150)


def fig_ablation() -> None:
    abl = json.loads((REPO / "results" / "replay" / "ablations.json").read_text(encoding="utf-8"))["arms"]
    ref = json.loads((REPO / "results" / "replay" / "headline_stored.json").read_text(encoding="utf-8"))["point"]
    rows = {"full model": ref, **abl}
    cols = [("n_complete", "CDR own"), ("validation_auc", "AUC val."), ("rho_sigma", "rho_Sigma"),
            ("rho_min", "rho_min"), ("healthy_forward_bias", "fwd top-1")]
    vals = np.array([[float(r[k]) for k, _ in cols] for r in rows.values()])
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(rows) + 1))
    ax.imshow(np.zeros_like(vals), cmap="Greys", vmin=0, vmax=1)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            ax.text(j, i, f"{v:.0f}" if cols[j][0] in ("n_complete", "healthy_forward_bias") else f"{v:.3f}",
                    ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(cols)), [c for _, c in cols])
    ax.set_yticks(range(len(rows)), list(rows))
    ax.set_title("ablation outcomes (values of Supplementary Table S-mechanism)")
    fig.tight_layout()
    fig.savefig(OUT / "fig11_ablation_values.png", dpi=150)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tables, x = score_tables("stored"), sensor_x()
    ens = S.ensemble(tables)
    fig_detect(ens, S.threshold(ens))
    fig_heatmap(ens)
    fig_seedcurve(tables, x)
    fig_unseen()
    fig_ablation()
    print(f"  wrote {len(list(OUT.glob('*.png')))} figures to {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
