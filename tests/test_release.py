"""Checks of the released data, models and replays against the manuscript.

    pytest -q                 # about 3 minutes; the bootstrap intervals of Table 5 are skipped
    pytest -q -m "not slow"   # under a minute
"""
from __future__ import annotations

import gzip
import hashlib
import json
import pathlib
import re

import numpy as np
import pandas as pd
import pytest

from spacces_gae import data as D
from spacces_gae import scoring as S

needs_torch = pytest.mark.skipif(not __import__("importlib").util.find_spec("torch_geometric"),
                                 reason="PyTorch and PyG are not installed (replay-only environment)")
if not needs_torch.args[0]:
    from spacces_gae import model as M

REPO = pathlib.Path(__file__).resolve().parents[1]
needs_data = pytest.mark.skipif(not (REPO / "data").is_dir(),
                                reason="data/ is not released yet (version 0.1.0: code, models, stored results)")


# ---- released inputs and models -------------------------------------------------------
@needs_data
def test_inputs_shape_and_normalization():
    ws = D.load_windows()
    assert ws.n_windows == 21014
    assert ws.x.shape == ws.c.shape == (21014, 60)
    assert np.allclose(ws.x.min(axis=1), 0.0) and np.allclose(ws.x.max(axis=1), 1.0)
    runs = D.load_runs()
    assert len(runs) == 127
    assert int(runs["n_windows"].sum()) == ws.n_windows
    assert sorted(runs["split"].unique()) == ["damaged", "train", "val"]


@needs_data
def test_coordinate_graph():
    a = D.load_a_phys()
    assert a.shape == (60, 60) and np.array_equal(a, a.T) and not a.diagonal().any()
    assert int(np.triu(a, 1).sum()) == 174


@needs_data
def test_context_is_computed_from_x():
    """c_v = mean of x_u - x_v over the 3-hop ego subgraph of v (v itself excluded), so c is
    a linear function of x and inherits its invariance to a per-window scale and offset."""
    ws, a = D.load_windows(), D.load_a_phys().astype(int)
    eye = np.eye(60, dtype=int)
    reach = np.linalg.matrix_power(eye + a, 3) > 0
    np.fill_diagonal(reach, False)
    members = reach / reach.sum(axis=1, keepdims=True)
    c = ws.x.astype(np.float64) @ members.T - ws.x
    assert np.abs(c - ws.c).max() < 1e-6


@needs_data
def test_sensor_geometry():
    s = D.load_sensors()
    assert len(s) == 60 and set(s["lane"]) == {"top", "lateral"}


@needs_torch
@pytest.mark.parametrize("seed", [0, 19])
def test_model_loads_with_960_parameters(seed):
    model, rec = M.load_model(REPO / "models" / f"seed_{seed:02d}.pt")
    assert M.n_params(model) == 960
    inert = sum(p.numel() for n, p in model.named_parameters() if n.startswith("gsl_"))
    assert inert == 40          # the structure decoder is unused at alpha = 1: Table 6 shows 960 - 40 = 920
    assert rec["seed"] == seed and rec["alpha"] == 1.0
    assert len(rec["residual_mu"]) == len(rec["residual_sigma"]) == 60


@needs_data
@needs_torch
@pytest.mark.slow
def test_rescore_one_seed_matches_stored_table():
    ws, runs, a = D.load_windows(), D.load_runs(), D.load_a_phys()
    model, rec = M.load_model(REPO / "models" / "seed_00.pt")
    z = S.standardize(S.window_errors(model, ws, a), rec["residual_mu"], rec["residual_sigma"])
    tab = S.score_table(z, ws.run_id, runs)
    ref = S.read_score_table(REPO / "data" / "scores" / "seed_00_scores.csv")
    cols = ["s_run"] + S.NODE_COLS
    assert np.abs(tab[cols].to_numpy(float) - ref[cols].to_numpy(float)).max() <= 1e-3
    assert np.array_equal(tab["n_anom_iqr"].to_numpy(), ref["n_anom_iqr"].to_numpy())
    assert list(tab["split"]) == list(ref["split"])


# ---- replays: every printed value -------------------------------------------------------
@needs_data
def test_replay_headline_point_values():
    import replay_headline
    assert replay_headline.main(["--no-intervals"]) == 0


@needs_data
@pytest.mark.slow
def test_replay_comparators():
    import replay_comparators
    assert replay_comparators.main() == 0


@needs_data
def test_replay_unseen():
    import replay_unseen
    assert replay_unseen.main() == 0


@needs_data
@pytest.mark.slow
def test_replay_ablations():
    import replay_ablations
    assert replay_ablations.main() == 0


# ---- provenance -----------------------------------------------------------------------
def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@needs_data
def test_manifest_hashes():
    man = pd.read_csv(REPO / "data" / "MANIFEST.csv")
    assert len(man) > 0
    for path, sha in zip(man["path"], man["sha256_original"]):
        raw = (REPO / path).read_bytes()
        if path.endswith(".gz"):
            raw = gzip.decompress(raw)
        assert _sha(raw) == sha, path


@needs_data
def test_stored_scores_match_provenance():
    prov = json.loads((REPO / "models" / "provenance.json").read_text(encoding="utf-8"))
    for rec in prov:
        raw = (REPO / "data" / "scores" / f"seed_{rec['seed']:02d}_scores.csv").read_bytes()
        assert _sha(raw) == rec["source_scores_sha256"]


# ---- nothing private in the release ------------------------------------------------------
PRIVATE = [r"[A-Za-z]:[\\/](?!/)", r"\bUsers[\\/]", r"/root/", r"/home/", r"/repo/", r"/data/windows", r'"cache_dir"',
           r'"runs_dir"',r"giannis", r"stamatelatos",
           r"johnstamly", r"@gmail", r"paper[\\/]review", r"CLAUDE", r"handoff", r"api[_-]?key", r"token\s*="]
TEXT_SUFFIXES = {".py", ".md", ".csv", ".json", ".txt", ".toml", ".yml", ".yaml", ".cff", ".cfg", ".ini", ""}


def test_no_private_strings():
    pat = re.compile("|".join(PRIVATE), re.IGNORECASE)
    hits = []
    for p in REPO.rglob("*"):
        if ".git" in p.parts or not p.is_file() or p.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if p.name in ("test_release.py", "CITATION.cff", "LICENSE"):     # these carry the author's name
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if pat.search(line):
                hits.append(f"{p.relative_to(REPO)}:{i}: {line.strip()[:120]}")
    assert not hits, "\n".join(hits[:20])


@needs_data
def test_no_calendar_dates_in_run_table():
    runs = D.load_runs()
    assert not any("date" in c.lower() for c in runs.columns)
    text = (REPO / "data" / "inputs" / "runs.csv").read_text(encoding="utf-8")
    assert not re.search(r"\b20\d\d-\d\d-\d\d\b", text)
