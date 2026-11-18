"""Localization against the removed-plate intervals (manuscript Section 4.6, Table 2).

A run is *localized* when, for every damaged interval of its state, at least one
Tukey-flagged sensor lies inside the interval (distance 0 along the backbone).
D4 and D5 have two intervals, so a localized D4/D5 run recovers both positions.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from .scoring import tukey_flags

PITCH_MM = 94.0
#: Removed-plate intervals, mm from the stern (manuscript Table 2).
INTERVALS = {"aft": (1160.0, 1350.0), "fwd": (2230.0, 2410.0)}
#: Damaged intervals of each state. D1-D5 are cumulative; D6 is the aft side plate alone.
STATE_INTERVALS = {1: ("fwd",), 2: ("fwd",), 3: ("fwd",), 4: ("fwd", "aft"), 5: ("fwd", "aft"), 6: ("aft",)}


def distance(x: float, interval) -> float:
    lo, hi = interval
    return float(max(lo - x, 0.0, x - hi))


def intervals(state: int) -> list:
    return [INTERVALS[k] for k in STATE_INTERVALS[int(state)]]


def localized(n, x_mm, state: int, radius_mm: float = 0.0) -> bool:
    """True when every damaged interval of ``state`` holds a flagged sensor (within ``radius_mm``)."""
    flagged = tukey_flags(n)
    return all(any(distance(float(x_mm[j]), iv) <= radius_mm for j in flagged) for iv in intervals(state))


def chance_all_sites(x_mm, state: int, n_flag: int, radius_mm: float = 0.0) -> float:
    """Exact probability that ``n_flag`` sensors drawn at random cover every interval (inclusion-exclusion)."""
    x = np.asarray(x_mm, dtype=float)
    ivs = intervals(state)
    total = math.comb(x.size, int(n_flag))
    near = [np.array([distance(v, iv) <= radius_mm for v in x]) for iv in ivs]
    count = 0
    for r in range(len(near) + 1):
        for sub in itertools.combinations(range(len(near)), r):
            banned = np.zeros(x.size, bool)
            for s in sub:
                banned |= near[s]
            avail = int((~banned).sum())
            count += (-1) ** r * (math.comb(avail, int(n_flag)) if avail >= n_flag else 0)
    return float(count / total) if total else 0.0
