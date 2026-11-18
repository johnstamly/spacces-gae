"""Shared helpers for the replay scripts: paths, score-table loading, and the check printer."""
from __future__ import annotations

import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from spacces_gae import data as D  # noqa: E402
from spacces_gae import scoring as S  # noqa: E402

if not (REPO / "data").is_dir():
    raise SystemExit(D.DATA_MISSING)

SEEDS = tuple(range(20))


def score_tables(source: str = "stored") -> list:
    """The 20 per-seed tables: ``stored`` (as computed after training) or ``rescored``
    (recomputed from ``data/inputs`` and ``models/`` by ``reproduce/rescore_models.py``)."""
    folder = {"stored": REPO / "data" / "scores", "rescored": REPO / "results" / "rescored"}[source]
    paths = [folder / f"seed_{s:02d}_scores.csv" for s in SEEDS]
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        hint = " (run reproduce/rescore_models.py first)" if source == "rescored" else ""
        raise SystemExit(f"missing {len(missing)} score tables in {folder}{hint}")
    return [S.read_score_table(p) for p in paths]


def sensor_x():
    return D.load_sensors()["x_mm"].to_numpy(float)


class Checker:
    def __init__(self, title: str):
        self.title, self.failed = title, []
        print(f"== {title}")

    def eq(self, label: str, got, want, fmt: str = "{}") -> None:
        ok = got == want
        print(f"  {'ok  ' if ok else 'FAIL'} {label}: {fmt.format(got)} (manuscript {fmt.format(want)})")
        if not ok:
            self.failed.append(label)

    def rounds(self, label: str, got: float, want: str) -> None:
        """``got`` rounded to the manuscript's printed decimals equals the printed value."""
        dec = len(want.split(".")[1]) if "." in want else 0
        shown = f"{got:.{dec}f}"
        ok = shown == want
        print(f"  {'ok  ' if ok else 'FAIL'} {label}: {got:.6f} -> {shown} (manuscript {want})")
        if not ok:
            self.failed.append(label)

    def done(self) -> int:
        print(f"  {self.title}: {'ALL CHECKS PASS' if not self.failed else 'FAILED: ' + ', '.join(self.failed)}")
        return 1 if self.failed else 0
