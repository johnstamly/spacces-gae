"""Run every replay in order and report one verdict.

    python reproduce/replay_all.py                 # everything: ~15 min on a laptop CPU
    python reproduce/replay_all.py --no-rescore    # stored score tables only, no PyTorch needed
    python reproduce/replay_all.py --quick         # also skip the bootstrap intervals of Table 5
"""
from __future__ import annotations

import argparse
import time

import replay_ablations
import replay_comparators
import replay_headline
import replay_unseen


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-rescore", action="store_true", help="skip rescoring with the released models")
    ap.add_argument("--quick", action="store_true", help="skip the bootstrap intervals of Table 5")
    args = ap.parse_args(argv)
    steps = [("Table 5 (stored tables)", lambda: replay_headline.main(["--no-intervals"] if args.quick else []))]
    if not args.no_rescore:
        import rescore_models
        steps += [("rescore the 20 released models", lambda: rescore_models.main([])),
                  ("Table 5 (rescored tables)", lambda: replay_headline.main(["--source", "rescored", "--no-intervals"]))]
    steps += [("Table 6", replay_comparators.main), ("Table 7", replay_unseen.main), ("Ablations", replay_ablations.main)]
    failed = []
    for name, fn in steps:
        t0 = time.time()
        rc = fn()
        print(f"-- {name}: {'pass' if rc == 0 else 'FAIL'} ({time.time() - t0:.0f} s)\n", flush=True)
        if rc:
            failed.append(name)
    print("ALL REPLAYS PASS" if not failed else "FAILED: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
