"""Derive the released inputs and models from the authors' restricted pipeline.

This script is run by the authors only. It reads the restricted window cache and
the original training outputs, which are not part of this repository, and writes
the files under ``data/`` and ``models/``. It is published so that a reader can
see exactly how each released file was derived and what was left out.

What it writes:

* ``data/inputs/windows.npz``: for every 4 s window of every run, the per-window
  min-max scaled variance ``x`` [W, 60], the subgraph context ``c`` [W, 60] (the
  model's ``[x, c]`` input, float32 exactly as the model consumed it), the run id,
  the window index within the run, and the per-window response graph S_sim as an
  upper-triangle edge list with weights.
* ``data/inputs/runs.csv``: run id, damage state, split and nominal test condition.
* ``data/geometry/sensors.csv`` and ``a_phys_edges.csv``: node order, lane, nominal
  longitudinal position and the Coordinate graph.
* ``data/scores/seed_XX_scores.csv``: the stored per-seed score tables, byte copies.
* ``models/seed_XX.pt`` and ``.json``: the 20 trained models (weights only) and
  their training record (validation residual statistics, split, loss curves).

What it never writes: the log-variance window cache (absolute strain scale), the
21-value per-sensor response fingerprints, the dense similarity matrices, sample
offsets, per-window minima and maxima, raw records, dates.

The locations of the restricted files are read from a private layout file (see
``tools/_layout.py``).

Usage (authors):
    python tools/export_release.py --private-root <restricted tree> --layout <layout file>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil

import numpy as np
import pandas as pd
import torch

import _layout

REPO = pathlib.Path(__file__).resolve().parents[1]
SEEDS = tuple(range(20))
ALPHA = 1.0


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stem_file(lay: _layout.Layout, seed: int, suffix: str) -> pathlib.Path:
    """A training output of one seed: ``<stem><suffix>``."""
    stem = lay.model_stem(seed)
    return stem.with_name(stem.name + suffix)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    _layout.add_arguments(ap)
    ap.add_argument("--release-root", type=pathlib.Path, default=REPO,
                    help="where to write the release (default: this repository)")
    args = ap.parse_args(argv)
    lay = _layout.Layout(args.private_root, args.layout)
    root = args.release_root.resolve()

    ras = lay.pipeline()  # restricted feature pipeline
    tw, rv, dm = ras.tw, ras.rv, ras.dm
    wd, wf, rd = tw.wd, tw.wf, tw.rd

    # ---- window set exactly as training built it (seed-independent) ----------------
    meta0 = json.loads(stem_file(lay, 0, ".json").read_text(encoding="utf-8"))
    L, S = int(meta0["args"]["L"]), int(meta0["args"]["S"])
    metadata = tw._metadata()
    train_ids, val_ids, damaged_ids = wd.run_splits(metadata)
    if sorted(train_ids) != sorted(meta0["train_ids"]) or sorted(val_ids) != sorted(meta0["val_ids"]):
        raise AssertionError("run split differs from the stored training record")
    all_ids = sorted(set(train_ids) | set(val_ids) | set(damaged_ids))
    ws = wd.load_window_set(all_ids, L, S, 1.0, wf.CACHE_DIR, centred=False, subsample_ids=set(train_ids))
    x_raw = tw.window_inputs(ws.values, "minmax")
    coords = rd.load_coords()
    adj, order = dm.build_lanewise_physical_graph(coords, k_intra_lane=3, k_inter_lane=2)
    if order != wf.load_node_order():
        raise AssertionError("node order mismatch")
    subs = dm.ego_subgraphs(adj, k_hop=int(meta0["args"]["k_hop"]))
    x_enh, _canonical = tw._d2_views(x_raw, subs)
    # The response-graph cache is bound by the SHA-256 that every training checkpoint recorded.
    # (The cache's JSON index is not used: its hash depends on the checkout's line endings.)
    npz = lay.file("similarity_cache")
    for seed in SEEDS:
        ck = torch.load(stem_file(lay, seed, "_best.pt"), map_location="cpu", weights_only=False)
        if ck["cache"]["npz_sha256"] != sha256(npz):
            raise AssertionError(f"seed {seed}: response-graph cache differs from the one it was trained on")
    with np.load(npz, allow_pickle=False) as packed:
        cache = {k: np.asarray(packed[k]) for k in ("matrices", "run_ids", "starts", "node_order")}
    if not np.array_equal(cache["node_order"], np.asarray(order)):
        raise AssertionError("response-graph cache node order mismatch")
    sim = rv.align_matrices(cache, ws)

    for seed in SEEDS:
        z = np.load(stem_file(lay, seed, "_windows.npz"))
        if not (np.array_equal(z["run_ids"], ws.run_ids) and np.array_equal(z["starts"], ws.starts)):
            raise AssertionError(f"seed {seed}: window keys differ from the rebuilt window set")
    if np.any(ws.starts % S):
        raise AssertionError("window starts are not multiples of the stride")

    # ---- inputs ----------------------------------------------------------------------
    W = x_enh.shape[0]
    iu, ju = np.triu_indices(60, k=1)
    ptr = np.zeros(W + 1, dtype=np.int64)
    edges, weights = [], []
    for w in range(W):
        m = sim[w]
        if not np.array_equal(m, m.T) or np.any(np.diag(m) != 0):
            raise AssertionError(f"window {w}: S_sim is not symmetric with an empty diagonal")
        keep = m[iu, ju] > 0
        edges.append(np.stack([iu[keep], ju[keep]], axis=1).astype(np.uint8))
        weights.append(m[iu, ju][keep].astype(np.float32))
        ptr[w + 1] = ptr[w] + int(keep.sum())
    edges = np.concatenate(edges)
    weights = np.concatenate(weights)
    # round trip: the dense matrix the model received is rebuilt exactly
    for w in range(0, W, 997):
        dense = np.zeros((60, 60), dtype=np.float32)
        e = edges[ptr[w]:ptr[w + 1]].astype(int)
        dense[e[:, 0], e[:, 1]] = weights[ptr[w]:ptr[w + 1]]
        dense[e[:, 1], e[:, 0]] = weights[ptr[w]:ptr[w + 1]]
        if not np.array_equal(dense, sim[w]):
            raise AssertionError(f"window {w}: S_sim round trip failed")

    out = root / "data" / "inputs"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "windows.npz",
        run_id=ws.run_ids.astype(np.int16),
        window=(ws.starts // S).astype(np.int16),
        x=np.ascontiguousarray(x_enh[:, :, 0]),
        c=np.ascontiguousarray(x_enh[:, :, 1]),
        sim_ptr=ptr, sim_edges=edges, sim_weight=weights)

    cond = pd.read_csv(lay.file("conditions_table")).set_index("run_id")
    split = {r: "train" for r in train_ids} | {r: "val" for r in val_ids}
    rows = []
    for r in all_ids:
        lev = int(cond.loc[r, "damage_level"])
        period = ("post-repair" if lev == 0 and r >= 113 else
                  "pre-damage" if lev == 0 else "damaged")
        rows.append({"run_id": r, "damage_state": lev, "split": split.get(r, "damaged"), "period": period,
                     "wave_type": "irregular" if cond.loc[r, "reg_irreg"] == "IRREG" else "regular",
                     "speed_kn": int(cond.loc[r, "vkts"]), "wave_height_m": float(cond.loc[r, "h_m"]),
                     "wave_period_s": float(cond.loc[r, "t_s"]),
                     "n_windows": int((ws.run_ids == r).sum())})
    pd.DataFrame(rows).to_csv(out / "runs.csv", index=False)

    # ---- geometry --------------------------------------------------------------------
    geo = root / "data" / "geometry"
    geo.mkdir(parents=True, exist_ok=True)
    frame = coords.reindex(order)
    lane = np.where(frame["z"].to_numpy(float) > float(frame["z"].median()), "top", "lateral")
    pd.DataFrame({"node": range(60), "sensor": order, "lane": lane,
                  "x_mm": frame["x"].to_numpy(float)}).to_csv(geo / "sensors.csv", index=False)
    a = np.asarray(adj)
    i, j = np.nonzero(np.triu(a, k=1))
    pd.DataFrame({"i": i, "j": j}).to_csv(geo / "a_phys_edges.csv", index=False)

    # ---- scores and models -----------------------------------------------------------
    sc = root / "data" / "scores"
    md = root / "models"
    sc.mkdir(parents=True, exist_ok=True)
    md.mkdir(parents=True, exist_ok=True)
    provenance = []
    for seed in SEEDS:
        f_scores, f_meta, f_ck = (stem_file(lay, seed, s) for s in ("_scores.csv", ".json", "_best.pt"))
        shutil.copyfile(f_scores, sc / f"seed_{seed:02d}_scores.csv")
        meta = json.loads(f_meta.read_text(encoding="utf-8"))
        ck = torch.load(f_ck, map_location="cpu", weights_only=False)
        a_ = meta["args"]
        config = {"in_channels": 2, "hidden_channels": int(a_["d2_hidden"]), "out_channels": int(a_["d2_latent"]),
                  "n_nodes": 60, "embed_dim": int(a_["d2_embed_dim"]), "layers": int(a_["d2_layers"])}
        torch.save({"state_dict": ck["state_dict"], "config": config}, md / f"seed_{seed:02d}.pt")
        record = {
            "seed": seed, "alpha": float(a_["alpha"]), "k_hop": int(a_["k_hop"]),
            "window_length": L, "window_stride": S, "model": config,
            "training": {"optimizer": "AdamW", "lr": float(a_["d2_lr"]), "batch_windows": int(a_["d2_batch"]),
                         "max_epochs": int(a_["epochs"]), "patience": int(a_["patience"]),
                         "deterministic": a_["deterministic"] in (True, "True"),
                         "hardware": "NVIDIA L4, PyTorch 2.11.0+cu128, CUDA 12.8"},
            "train_ids": meta["train_ids"], "val_ids": meta["val_ids"],
            "residual_mu": meta["residual_mu"], "residual_sigma": meta["residual_sigma"],
            "best_val_loss": meta["best_val_loss"], "n_epochs": meta["n_epochs"], "n_params": meta["n_params"],
            "val_loss_curve": meta["val_loss_curve"], "train_loss_curve": meta["train_loss_curve"]}
        (md / f"seed_{seed:02d}.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
        provenance.append({"seed": seed, "source_checkpoint_sha256": sha256(f_ck),
                           "source_scores_sha256": sha256(f_scores),
                           "source_metadata_sha256": sha256(f_meta)})
    (md / "provenance.json").write_text(json.dumps(provenance, indent=1) + "\n", encoding="utf-8")
    print(f"exported {W} windows, {edges.shape[0]} S_sim edges, {len(all_ids)} runs, {len(SEEDS)} models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
