"""Where the authors' restricted files are: read from a private layout file.

The export scripts and the reconstruction check run only on the authors' machine.
Every location in the authors' analysis tree (folders, file names, the module of the
restricted pipeline) is read from a JSON layout file kept with the restricted data,
so the public scripts carry neutral keys only. Keys used:

* ``pipeline``: ``{"paths": [...], "module": "..."}``, the restricted feature pipeline;
* ``models``: seed -> path stem of that seed's training outputs
  (``<stem>.json``, ``<stem>_best.pt``, ``<stem>_scores.csv``, ``<stem>_windows.npz``);
* ``similarity_cache``, ``conditions_table``: single files;
* ``scores``: released score-table path -> source table;
* ``stored``: released path under ``results/stored`` -> source file;
* ``unseen``: ``tests``, ``runs``, ``oc_summary``, ``oc_runs``.

All paths are relative to ``--private-root``.
"""
from __future__ import annotations

import importlib
import json
import pathlib
import sys


class Layout:
    def __init__(self, private_root: pathlib.Path, layout_file: pathlib.Path):
        self.root = private_root.resolve()
        self.spec = json.loads(layout_file.read_text(encoding="utf-8"))

    def path(self, rel: str) -> pathlib.Path:
        return self.root / rel

    def file(self, key: str) -> pathlib.Path:
        return self.path(self.spec[key])

    def model_stem(self, seed: int) -> pathlib.Path:
        return self.path(self.spec["models"][str(seed)])

    def score_source(self, released: str) -> pathlib.Path:
        return self.path(self.spec["scores"][released])

    def stored_source(self, released: str) -> pathlib.Path:
        return self.path(self.spec["stored"][released])

    def unseen(self, key: str) -> pathlib.Path:
        return self.path(self.spec["unseen"][key])

    def pipeline(self):
        """Import and return the restricted feature pipeline."""
        for p in self.spec["pipeline"]["paths"]:
            sys.path.insert(0, str(self.path(p)))
        return importlib.import_module(self.spec["pipeline"]["module"])


def add_arguments(ap) -> None:
    ap.add_argument("--private-root", required=True, type=pathlib.Path,
                    help="root of the authors' restricted analysis tree")
    ap.add_argument("--layout", required=True, type=pathlib.Path,
                    help="the private layout file (JSON) that maps the keys used here to that tree")
