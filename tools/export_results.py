"""Copy the stored outputs behind Tables 6 and 7 and the ablations into the release.

Run by the authors only, after ``export_release.py``. It reads the authors' result
files, which are not part of this repository, at the locations a private layout file
gives (see ``tools/_layout.py``), and writes:

* ``data/comparators/<key>/seed_XX_scores.csv.gz``: per-seed run and sensor scores of
  the eight published graph anomaly detectors (as published and after label-free
  tuning), PCA-95, the dense autoencoder, and SpaCCES-GAE on the GDN-aligned windows.
* ``data/ablations/<arm>/seed_XX_scores.csv.gz``: the same for every ablation arm.
* ``data/unseen/t4_runs.csv`` and ``conditions.json``: the unseen-condition test
  (Section 5), per held-out run and per withheld condition.
* ``results/stored/``: the authors' own summary files for these experiments, trimmed
  to the rows and columns the replays compare against.

Score tables are byte-identical to the originals after decompression (gzip with a
fixed timestamp, so the archive bytes are reproducible too). ``data/MANIFEST.csv``
records next to each released path the SHA-256 of the original file (of the released
file itself where it was trimmed or derived).

    python tools/export_results.py --private-root <restricted tree> --layout <layout file>
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import pathlib
import shutil

import pandas as pd

import _layout

REPO = pathlib.Path(__file__).resolve().parents[1]
SCORE_COLS = ["run_id", "damage_level", "split", "s_run"] + [f"n{i}" for i in range(60)] + ["dsi_iqr", "n_anom_iqr"]

# Released comparator tables: key -> seeds. Families are in reproduce/replay_comparators.py.
# The source of every table is looked up in the private layout file (tools/_layout.py).
TUNED_KEYS = [f"{d}_H_{w}" for d, w in (("B4", "all"), ("B5", "aligned"), ("B6", "all"), ("B7", "aligned"),
                                         ("B8", "aligned"), ("B9", "all"), ("B10", "all"), ("B11", "aligned"))]
COMPARATORS = {
    "E2_aligned": range(20), "B4_all": range(10), "B5_aligned": range(10), "B6_all": range(10),
    "B9_all": range(1), "B10_all": range(10), "B7_aligned": range(10), "B8_aligned": range(10),
    "B11_aligned": range(10),
    **{k: range(1 if k.startswith("B9_") else 10) for k in TUNED_KEYS},
    "B1_all": range(1), "B3_all": range(10),
}
# Released ablation arms: arm -> seeds.
ABLATIONS = {
    "M1": range(20), "M2": range(20),
    **{a: range(20) for a in ("PHYS1", "PHYS2", "SIM1", "SIM2", "DESC")},
    "G1b": range(20), "G2": range(10), "alpha_0.70": range(20),
    **{f"alpha_0.{v}": range(10) for v in ("50", "85", "95")},
}

SUMMARY_COLS = ["window_set", "method", "n_params"]     # the only stored column the Table 6 replay reads


def _cols(cols, query=None):
    """Keep only the columns (and rows) a replay compares against."""
    def f(src: pathlib.Path) -> bytes:
        df = pd.read_csv(src)
        if query:
            df = df.query(query)
        return df[cols].to_csv(index=False, lineterminator="\n").encode()
    return f


def _rows(*pairs):
    """Keep only the (method, window_set) rows of Table 6 and the SpaCCES-GAE reference rows."""
    keep = set(pairs) | {("E2", "all"), ("E2", "aligned")}
    def f(src: pathlib.Path) -> bytes:
        df = pd.read_csv(src)
        df = df[[(m, w) in keep for m, w in zip(df["method"], df["window_set"])]][SUMMARY_COLS]
        return df.to_csv(index=False, lineterminator="\n").encode()
    return f


def _t4_tests(src: pathlib.Path) -> bytes:
    """The pre-specified pooled-rate test of Table 7 only (SpaCCES-GAE vs PCA-95 and dense AE)."""
    d = json.loads(src.read_text(encoding="utf-8"))
    out = {"holm_note": d["holm_note"],
           "comparisons": {k: {"T4": d["comparisons"][k]["T4"]} for k in ("E2 vs B1", "E2 vs B3")}}
    return (json.dumps(out, indent=1) + "\n").encode()


# Released path under results/stored -> trimming function, or None for a verbatim copy.
# Only what the replays compare against is released. Sources come from the layout file.
TUNED = [(f"{d}_H", w) for d, w in (("B4", "all"), ("B5", "aligned"), ("B6", "all"), ("B7", "aligned"),
                                     ("B8", "aligned"), ("B9", "all"), ("B10", "all"), ("B11", "aligned"))]
STORED = {
    "headline/seed_curve.csv": None,
    "comparators/published_summary.csv": _rows(("B1", "all"), ("B3", "all"), ("B4", "all"), ("B5", "aligned")),
    "comparators/published_ext_summary.csv": _rows(("B6", "all"), ("B9", "all"), ("B10", "all"),
                                                   ("B7", "aligned"), ("B8", "aligned"), ("B11", "aligned")),
    "comparators/tuned_summary.csv": _rows(*TUNED),
    "unseen/t4_tests.json": _t4_tests,
    "ablations/summary_20_seeds.csv": _cols(["arm", "seed_set", "n_params"], "seed_set == 'pooled 0-19'"),
    "ablations/holm_mechanism_m6.csv": _cols(["family_seed_set", "test", "diff", "ci95_lo", "ci95_hi"],
                                             "family_seed_set == 'pooled 0-19'"),
}

UNSEEN_METHODS = ("E2", "OC", "B1", "B3")      # SpaCCES-GAE, tuned [x,c]-OCGNN, PCA-95, dense AE


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_gz(src: pathlib.Path, dest: pathlib.Path) -> str:
    raw = src.read_bytes()
    head = pd.read_csv(io.BytesIO(raw), nrows=1)
    if list(head.columns) != SCORE_COLS:
        raise ValueError(f"{src.name}: unexpected columns")
    dest.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, compresslevel=9, mtime=0) as gz:
        gz.write(raw)
    dest.write_bytes(buf.getvalue())
    return sha256_bytes(raw)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    _layout.add_arguments(ap)
    ap.add_argument("--release-root", type=pathlib.Path, default=REPO,
                    help="where to write the release (default: this repository)")
    args = ap.parse_args(argv)
    lay = _layout.Layout(args.private_root, args.layout)
    root = args.release_root.resolve()
    manifest = []

    for sub in ("data/comparators", "data/ablations", "data/unseen", "results/stored"):
        shutil.rmtree(root / sub, ignore_errors=True)

    for group, table in (("comparators", COMPARATORS), ("ablations", ABLATIONS)):
        for key, seeds in table.items():
            for s in seeds:
                released = f"data/{group}/{key}/seed_{s:02d}_scores.csv.gz"
                manifest.append((released, write_gz(lay.score_source(released), root / released)))

    # Unseen operating conditions: the nine withheld conditions only.
    t4 = json.loads(lay.unseen("tests").read_text(encoding="utf-8"))
    oc = json.loads(lay.unseen("oc_summary").read_text(encoding="utf-8"))
    rows = [r for r, v in t4["rows"].items() if v["neural_row"]]
    runs = pd.concat([pd.read_csv(lay.unseen("runs")), pd.read_csv(lay.unseen("oc_runs"))], ignore_index=True)
    runs = runs[runs["row"].isin(rows) & runs["method"].isin(UNSEEN_METHODS)]
    runs = runs[["row", "method", "run_id", "damage_level", "split", "s_run", "exceeds_tau", "exact"]]
    order = {r: i for i, r in enumerate(rows)}
    runs = runs.sort_values(["row", "method", "run_id"], key=lambda c: c.map(order) if c.name == "row" else c)
    out = root / "data" / "unseen"
    out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out / "t4_runs.csv", index=False, lineterminator="\n")
    keep = ("tau", "rho_sigma", "rho_min", "exact_d15", "tpr_d1_5", "n_train", "n_test")
    cond = {"rows": rows, "conditions": {}}
    for r in rows:
        v = t4["rows"][r]
        m = {k: {f: v["methods"][k].get(f) for f in keep} for k in ("E2", "B1", "B3")}
        m["OC"] = {f: oc["rows"][r]["ocgnn"].get(f) for f in keep}
        cond["conditions"][r] = {"train_ids": v["train_ids"], "val_ids": v["val_ids"], "test_ids": v["test_ids"],
                                 "methods": m}
    (out / "conditions.json").write_text(json.dumps(cond, indent=1) + "\n", encoding="utf-8")
    for name in ("t4_runs.csv", "conditions.json"):
        manifest.append((f"data/unseen/{name}", sha256_bytes((out / name).read_bytes())))

    for rel, trim in STORED.items():
        src = lay.stored_source(rel)
        dest = root / "results" / "stored" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if trim is None:
            shutil.copyfile(src, dest)
        else:
            dest.write_bytes(trim(src))
        manifest.append((f"results/stored/{rel}", sha256_bytes(dest.read_bytes())))

    cols = ["path", "sha256_original"]
    old = root / "data" / "MANIFEST.csv"
    prev = pd.read_csv(old)[cols] if old.is_file() else pd.DataFrame(columns=cols)
    prev = prev[~prev["path"].str.match(r"^(data/(comparators|ablations|unseen)|results/stored)/")]
    df = pd.concat([prev, pd.DataFrame(manifest, columns=cols)], ignore_index=True)
    df.to_csv(old, index=False, lineterminator="\n")
    n_c = sum(len(v) for v in COMPARATORS.values())
    n_a = sum(len(v) for v in ABLATIONS.values())
    print(f"exported {n_c} comparator tables, {n_a} ablation tables, {len(runs)} unseen-condition rows, "
          f"{len(STORED)} stored result files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
