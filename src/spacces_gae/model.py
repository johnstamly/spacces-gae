"""The SpaCCES-GAE network: a dual-branch graph convolutional autoencoder.

One branch convolves over the fixed Coordinate graph A_phys, the other over the
per-window response graph S_sim (weighted). Both branches read ``[x, c, e]``,
where ``e`` is a learned 4-value identity per sensor. Their codes are fused into
a 4-value latent per sensor, and a two-layer graph decoder on A_phys
reconstructs ``[x, c]``. A structure decoder (``gsl_1``, ``gsl_2``) is kept for
the alpha < 1 variants; at the published alpha = 1 it receives no gradient and
does not enter the score.

Parameter names follow the trained checkpoints, so ``load_state_dict`` works on
the released models without any key mapping.
"""
from __future__ import annotations

import json
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class SpaccesGAE(nn.Module):
    """Dual-encoder graph autoencoder.

    Args:
        in_channels: node input width (2 for ``[x, c]``).
        hidden_channels: GCN width (8).
        out_channels: latent width (4).
        n_nodes: sensors per graph (60); needed for the identity embedding.
        embed_dim: width of the learned sensor identity (4); 0 disables it.
        layers: GCN layers per encoder branch (4).
    """

    def __init__(self, in_channels: int = 2, hidden_channels: int = 8, out_channels: int = 4,
                 n_nodes: int = 60, embed_dim: int = 4, layers: int = 4):
        super().__init__()
        if layers < 2:
            raise ValueError(f"layers must be >= 2, got {layers}")
        self.n_nodes = int(n_nodes)
        self.embedding = nn.Embedding(self.n_nodes, int(embed_dim)) if int(embed_dim) > 0 else None
        enc_in = int(in_channels) + (int(embed_dim) if self.embedding is not None else 0)
        self.gcn1_phys = GCNConv(enc_in, hidden_channels)
        self.gcn2_phys = GCNConv(hidden_channels, hidden_channels)
        self.gcn1_sim = GCNConv(enc_in, hidden_channels)
        self.gcn2_sim = GCNConv(hidden_channels, hidden_channels)
        self.fc_combine = nn.Linear(hidden_channels * 2, out_channels)
        self.prelu_phys1 = nn.PReLU()
        self.prelu_phys2 = nn.PReLU()
        self.prelu_sim1 = nn.PReLU()
        self.prelu_sim2 = nn.PReLU()
        self.prelu_combine = nn.PReLU()
        self.prelu_dec1 = nn.PReLU()
        self.attr_decoder1 = GCNConv(out_channels, hidden_channels)
        self.attr_decoder2 = GCNConv(hidden_channels, in_channels)
        self.gsl_1 = nn.Linear(out_channels, out_channels)
        self.gsl_2 = nn.Linear(out_channels, out_channels)
        extra = int(layers) - 2
        self.extra_phys = nn.ModuleList([GCNConv(hidden_channels, hidden_channels) for _ in range(extra)])
        self.extra_sim = nn.ModuleList([GCNConv(hidden_channels, hidden_channels) for _ in range(extra)])
        self.extra_prelu_phys = nn.ModuleList([nn.PReLU() for _ in range(extra)])
        self.extra_prelu_sim = nn.ModuleList([nn.PReLU() for _ in range(extra)])

    def encode(self, x, edge_index_phys, edge_index_sim, edge_weight_sim):
        """``[N, in] -> [N, out]``. Node ``k`` of a batch is sensor ``k % n_nodes``."""
        if self.embedding is not None:
            index = torch.arange(x.shape[0], device=x.device) % self.n_nodes
            x = torch.cat([x, self.embedding(index)], dim=1)
        h = self.prelu_phys1(self.gcn1_phys(x, edge_index_phys))
        z_phys = self.prelu_phys2(self.gcn2_phys(h, edge_index_phys))
        for conv, act in zip(self.extra_phys, self.extra_prelu_phys):
            z_phys = act(conv(z_phys, edge_index_phys))
        h = self.prelu_sim1(self.gcn1_sim(x, edge_index_sim, edge_weight_sim))
        z_sim = self.prelu_sim2(self.gcn2_sim(h, edge_index_sim, edge_weight_sim))
        for conv, act in zip(self.extra_sim, self.extra_prelu_sim):
            z_sim = act(conv(z_sim, edge_index_sim, edge_weight_sim))
        return self.prelu_combine(self.fc_combine(torch.cat([z_phys, z_sim], dim=1)))

    def decode_attributes(self, z, edge_index_phys):
        """``[N, out] -> [N, in]``: the reconstruction of ``[x, c]``."""
        return self.attr_decoder2(self.prelu_dec1(self.attr_decoder1(z, edge_index_phys)), edge_index_phys)

    def decode_structure(self, z):
        """``[N, out] -> [N, N]`` symmetric reconstructed adjacency (used only when alpha < 1)."""
        a = torch.sigmoid(F.relu(self.gsl_1(z)) @ F.relu(self.gsl_2(z)).T)
        return (a + a.T) / 2

    def forward(self, data):
        """Returns ``(adjacency reconstruction, attribute reconstruction)``."""
        z = self.encode(data.x, data.edge_index_phys, data.edge_index_sim, data.edge_weight_sim)
        return self.decode_structure(z), self.decode_attributes(z, data.edge_index_phys)


def load_model(path: str | pathlib.Path) -> tuple[SpaccesGAE, dict]:
    """Load one released model (``models/seed_XX.pt``) and its record (``seed_XX.json``), in eval mode."""
    path = pathlib.Path(path)
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = SpaccesGAE(**ck["config"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    record = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    return model, record


def n_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))
