"""Released inputs: loading and graph assembly.

Everything the model consumes comes from ``data/``:

* ``inputs/windows.npz``: per window, ``x`` and ``c`` (60 values each), the run id,
  the window index, and the response graph S_sim as weighted upper-triangle edges;
* ``inputs/runs.csv``: per run, damage state, split and nominal test condition;
* ``geometry/sensors.csv`` and ``geometry/a_phys_edges.csv``: sensors and A_phys.
"""
from __future__ import annotations

import dataclasses
import pathlib

import numpy as np
import pandas as pd

# PyTorch and PyG are imported only where graphs are built, so the replay of the
# stored tables runs with numpy, pandas, scipy and scikit-learn alone.

N_SENSORS = 60
DATA_DIR = pathlib.Path(__file__).resolve().parents[2] / "data"
DATA_MISSING = ("data/ is not part of this release. Version 0.1.0 holds the code, the 20 trained models and the stored summary results; the processed inputs and score tables will follow in a later release (see README.md, section Data).")


def require_data(data_dir: pathlib.Path = DATA_DIR) -> pathlib.Path:
    """Return ``data_dir``, or stop with a clear message when the data are not released yet."""
    data_dir = pathlib.Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"{data_dir}: {DATA_MISSING}")
    return data_dir


@dataclasses.dataclass
class WindowSet:
    run_id: np.ndarray        # [W] int
    window: np.ndarray        # [W] int, index within the run
    x: np.ndarray             # [W, 60] float32, per-window min-max scaled variance
    c: np.ndarray             # [W, 60] float32, subgraph context
    sim_ptr: np.ndarray       # [W + 1] offsets into sim_edges / sim_weight
    sim_edges: np.ndarray     # [E, 2] uint8, i < j
    sim_weight: np.ndarray    # [E] float32

    @property
    def n_windows(self) -> int:
        return int(self.run_id.size)

    def x_enh(self, w: int) -> np.ndarray:
        """``[60, 2]`` model input ``[x, c]`` of window ``w``."""
        return np.stack([self.x[w], self.c[w]], axis=1)

    def sim_dense(self, w: int) -> np.ndarray:
        """``[60, 60]`` symmetric weighted S_sim of window ``w``."""
        m = np.zeros((N_SENSORS, N_SENSORS), dtype=np.float32)
        e = self.sim_edges[self.sim_ptr[w]:self.sim_ptr[w + 1]].astype(np.int64)
        v = self.sim_weight[self.sim_ptr[w]:self.sim_ptr[w + 1]]
        m[e[:, 0], e[:, 1]] = v
        m[e[:, 1], e[:, 0]] = v
        return m


def load_windows(data_dir: pathlib.Path = DATA_DIR) -> WindowSet:
    with np.load(require_data(data_dir) / "inputs" / "windows.npz", allow_pickle=False) as z:
        return WindowSet(run_id=z["run_id"].astype(np.int64), window=z["window"].astype(np.int64),
                         x=z["x"], c=z["c"], sim_ptr=z["sim_ptr"], sim_edges=z["sim_edges"],
                         sim_weight=z["sim_weight"])


def load_runs(data_dir: pathlib.Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(require_data(data_dir) / "inputs" / "runs.csv").set_index("run_id")


def load_sensors(data_dir: pathlib.Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(require_data(data_dir) / "geometry" / "sensors.csv").set_index("node")


def load_a_phys(data_dir: pathlib.Path = DATA_DIR) -> np.ndarray:
    """``[60, 60]`` symmetric 0/1 adjacency of the Coordinate graph (174 edges)."""
    e = pd.read_csv(require_data(data_dir) / "geometry" / "a_phys_edges.csv")
    a = np.zeros((N_SENSORS, N_SENSORS), dtype=np.int64)
    a[e["i"], e["j"]] = 1
    a[e["j"], e["i"]] = 1
    return a


def graph_data(x_enh: np.ndarray, a_phys: np.ndarray, s_sim: np.ndarray):
    """One window's two-view graph. Edges are listed row-major, as the models were trained."""
    import torch
    from torch_geometric.data import Data
    from torch_geometric.utils import dense_to_sparse
    ei_phys, _ = dense_to_sparse(torch.tensor(a_phys, dtype=torch.float))
    ei_sim, w_sim = dense_to_sparse(torch.tensor(s_sim, dtype=torch.float))
    return Data(x=torch.tensor(x_enh, dtype=torch.float), edge_index_phys=ei_phys,
                edge_index_sim=ei_sim, edge_weight_sim=w_sim)


def batch(ws: WindowSet, a_phys: np.ndarray, index):
    from torch_geometric.data import Batch
    return Batch.from_data_list([graph_data(ws.x_enh(int(w)), a_phys, ws.sim_dense(int(w))) for w in index])
