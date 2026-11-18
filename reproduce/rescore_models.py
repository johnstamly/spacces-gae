"""Recompute the 20 per-seed score tables from the released inputs and models.

Each released model scores every window of ``data/inputs/windows.npz``; the
residuals are standardized with the model's stored validation statistics and
aggregated to sensor and run scores. The result is compared with the stored
table of that seed in ``data/scores``.

The stored tables were computed on the training machine (Linux, NVIDIA L4 host).
On another CPU, float32 rounding differs in the last bits, so sensor scores agree
to about 1e-4 rather than bit for bit; every discrete outcome (Tukey flags, the
run split) must be identical. The headline replay on the rescored tables
(``replay_headline.py --source rescored``) then checks that the manuscript values
are unchanged.

    python reproduce/rescore_models.py            # all 20 seeds, about 3 minutes on a laptop CPU
    python reproduce/rescore_models.py --seeds 0 1
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from _common import REPO, SEEDS

from spacces_gae import data as D
from spacces_gae import model as M
from spacces_gae import scoring as S

TOL_SCORE = 1e-3          # largest allowed |rescored - stored| of any s_run or n_v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    args = ap.parse_args(argv)
    ws, runs, a = D.load_windows(), D.load_runs(), D.load_a_phys()
    out = REPO / "results" / "rescored"
    out.mkdir(parents=True, exist_ok=True)
    worst, failed = 0.0, []
    for seed in args.seeds:
        t0 = time.time()
        model, rec = M.load_model(REPO / "models" / f"seed_{seed:02d}.pt")
        err = S.window_errors(model, ws, a)
        z = S.standardize(err, rec["residual_mu"], rec["residual_sigma"])
        tab = S.score_table(z, ws.run_id, runs)
        tab.to_csv(out / f"seed_{seed:02d}_scores.csv")
        ref = S.read_score_table(REPO / "data" / "scores" / f"seed_{seed:02d}_scores.csv")
        cols = ["s_run"] + S.NODE_COLS
        diff = float(np.abs(tab[cols].to_numpy(float) - ref[cols].to_numpy(float)).max())
        same_flags = np.array_equal(tab["n_anom_iqr"].to_numpy(), ref["n_anom_iqr"].to_numpy())
        same_split = list(tab["split"]) == list(ref["split"])
        ok = diff <= TOL_SCORE and same_flags and same_split and M.n_params(model) == 960
        worst = max(worst, diff)
        if not ok:
            failed.append(seed)
        print(f"seed {seed:2d}: max |rescored - stored| = {diff:.2e}, Tukey flag counts identical: {same_flags}, "
              f"{'ok' if ok else 'FAIL'} ({time.time() - t0:.0f} s)", flush=True)
    print(f"rescored {len(args.seeds)} seeds into {out.relative_to(REPO)}; worst difference {worst:.2e}; "
          f"{'ALL CHECKS PASS' if not failed else 'FAILED seeds ' + str(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
