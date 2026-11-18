import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "reproduce"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
