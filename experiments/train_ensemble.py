"""Retrain SpaCCES-GAE seeds from the released inputs and score them.

Uses the released split (``data/inputs/runs.csv``: 56 healthy training runs, 14
healthy validation runs). Each seed is written to ``results/retrained/`` as a model,
its training record and its score table, in the formats of ``models/`` and
``data/scores/``. The headline replay can then run on the retrained ensemble:

    python experiments/train_ensemble.py --seeds 0 1 2            # CPU: about 30 s per epoch, ~70 epochs per seed
    python experiments/train_ensemble.py --seeds 0 --device cuda
    python experiments/train_ensemble.py --seeds 0 --max-epochs 2  # smoke test, minutes

A retrained seed is a new draw, not a copy of the released one (see
``spacces_gae.train``); compare ensembles, not single seeds.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from spacces_gae import data as D  # noqa: E402
from spacces_gae import scoring as S  # noqa: E402
from spacces_gae import train as T  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-epochs", type=int, default=T.DEFAULTS["max_epochs"])
    ap.add_argument("--out", type=pathlib.Path, default=REPO / "results" / "retrained")
    args = ap.parse_args(argv)
    ws, runs, a = D.load_windows(), D.load_runs(), D.load_a_phys()
    train_ids = runs.index[runs["split"] == "train"].tolist()
    val_ids = runs.index[runs["split"] == "val"].tolist()
    args.out.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        model, rec = T.fit(ws, a, train_ids, val_ids, seed, device=args.device, max_epochs=args.max_epochs)
        z = S.standardize(S.window_errors(model, ws, a), rec["residual_mu"], rec["residual_sigma"])
        S.score_table(z, ws.run_id, runs).to_csv(args.out / f"seed_{seed:02d}_scores.csv")
        c = rec["config"]
        arch = {"in_channels": 2, "hidden_channels": c["hidden_channels"], "out_channels": c["out_channels"],
                "n_nodes": D.N_SENSORS, "embed_dim": c["embed_dim"], "layers": c["layers"]}
        torch.save({"state_dict": model.state_dict(), "config": arch}, args.out / f"seed_{seed:02d}.pt")
        (args.out / f"seed_{seed:02d}.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
        print(f"seed {seed}: {rec['n_epochs']} epochs, best validation loss {rec['best_val_loss']:.6f}, "
              f"{rec['train_time_s']:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
