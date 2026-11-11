"""Temporal splitting, feature scaling and per-split graph construction.

The split is by TIME, not at random:

    train = time steps  1..29     fit model weights
    val   = time steps 30..34     choose hyperparameters, early-stopping epoch, threshold
    test  = time steps 35..49     final score, looked at exactly once per model

Why: in production you train on the past and score the future. A random split
would let the model train on transactions from the same fortnight (and the
same connected component!) as the ones it is tested on, which leaks
information and inflates the scores.

Because no edge crosses time steps, "the graph of a split" is simply the nodes
of its time steps plus the edges among them. Nothing has to be cut.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, to_undirected

from .data import FEATURE_SETS, UNKNOWN

DIRECTIONS = ("undirected", "directed", "bidirectional")


# --------------------------------------------------------------------------- #
# Which nodes belong to which split
# --------------------------------------------------------------------------- #
def step_mask(time_step: torch.Tensor, first: int, last: int) -> torch.Tensor:
    """Boolean mask of the nodes whose time step is in [first, last]."""
    return (time_step >= first) & (time_step <= last)


def split_masks(bundle: dict, cfg: dict) -> dict[str, torch.Tensor]:
    """Boolean node masks for train / val / test, plus trainval = train + val.

    `trainval` is what the FINAL models are fit on: once the settings have been
    chosen on validation, we retrain on steps 1..34 so the model that gets
    tested has seen the most recent data available before the test period.
    """
    t = bundle["time_step"]
    s = cfg["split"]
    masks = {name: step_mask(t, *s[name]) for name in ("train", "val", "test")}
    masks["trainval"] = masks["train"] | masks["val"]
    return masks


# --------------------------------------------------------------------------- #
# Feature scaling
# --------------------------------------------------------------------------- #
@dataclass
class Scaler:
    """Standardisation: z = (x - mean) / std, with statistics from training nodes only."""

    mean: torch.Tensor
    std: torch.Tensor

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


def fit_scaler(x: torch.Tensor, fit_mask: torch.Tensor) -> Scaler:
    """Compute mean/std on the nodes selected by `fit_mask` and nothing else.

    `fit_mask` is always the mask of the time steps the model is trained on.
    Statistics of validation/test time steps never enter the scaler; using
    them would leak information about the future into training.

    Unlabelled nodes of the training time steps ARE included. That is fair:
    their features are available at training time and no label is involved.
    """
    mean = x[fit_mask].mean(dim=0, keepdim=True)
    std = x[fit_mask].std(dim=0, keepdim=True)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)  # avoid dividing by ~0
    return Scaler(mean=mean, std=std)


def select_features(x: torch.Tensor, feature_set: str) -> torch.Tensor:
    """Pick the LF (93 local) or AF (all 165) columns."""
    return x[:, FEATURE_SETS[feature_set]]


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #
def prepare_edges(edge_index: torch.Tensor, direction: str) -> torch.Tensor:
    """Apply the edge-direction treatment used by the direction ablation.

    undirected     add the reverse of every edge, so information flows both
                   ways through one shared set of weights (the usual default).
    directed       keep edges as given (payer -> payee). A node then only
                   hears from the transactions that funded it.
    bidirectional  keep edges as given here; the model wraps each layer in
                   DirGNNConv, which runs one convolution along the edges and
                   a second, separately-weighted one against them.
    """
    if direction == "undirected":
        return to_undirected(edge_index)
    if direction in ("directed", "bidirectional"):
        return edge_index
    raise ValueError(f"unknown direction '{direction}', expected one of {DIRECTIONS}")


def make_graph(
    bundle: dict,
    node_mask: torch.Tensor,
    feature_set: str,
    scaler: Scaler,
    direction: str = "undirected",
) -> Data:
    """Build the PyG `Data` object for the nodes selected by `node_mask`.

    This is the induced subgraph: the chosen nodes and every edge whose two
    endpoints are both chosen. Nodes are renumbered 0..n-1 inside the subgraph;
    `data.node_idx` remembers each node's index in the full dataset.

    Unknown-label nodes are KEPT. They carry features and connect labelled
    nodes, so the GNN needs them for message passing. They are simply left out
    of the loss and the metrics via `data.labeled`.
    """
    num_nodes = bundle["x"].shape[0]
    edge_index, _ = subgraph(
        node_mask, bundle["edge_index"], relabel_nodes=True, num_nodes=num_nodes
    )
    # Scale all 165 columns with the training statistics, then keep LF or AF.
    x = select_features(scaler.transform(bundle["x"][node_mask]), feature_set)
    y = bundle["y"][node_mask]
    return Data(
        x=x,
        edge_index=prepare_edges(edge_index, direction),
        y=y,
        time_step=bundle["time_step"][node_mask],
        node_idx=torch.nonzero(node_mask, as_tuple=False).squeeze(1),
        labeled=y != UNKNOWN,
    )


def tabular_arrays(
    bundle: dict,
    node_mask: torch.Tensor,
    feature_set: str,
    scaler: Scaler,
    labeled_only: bool = True,
):
    """Tabular view for the non-graph models: (X, y, time_step, node_idx).

    With `labeled_only=True` (the default) only labelled nodes are returned,
    because a table model cannot compute a loss or a metric on an unlabelled
    row. The demo app asks for all nodes so it can also show scores for
    unlabelled transactions.
    """
    keep = node_mask & (bundle["y"] != UNKNOWN) if labeled_only else node_mask
    x = select_features(scaler.transform(bundle["x"][keep]), feature_set)
    return (
        x.numpy(),
        bundle["y"][keep].numpy(),
        bundle["time_step"][keep].numpy(),
        torch.nonzero(keep, as_tuple=False).squeeze(1).numpy(),
    )


# --------------------------------------------------------------------------- #
# Everything a training script needs, built once
# --------------------------------------------------------------------------- #
def build_graph_set(bundle: dict, cfg: dict, feature_set: str, direction: str = "undirected") -> dict:
    """The four graphs used by the two-phase protocol (see protocol.py).

    Selection phase (choose settings):  fit on `train`, measure on `val`.
        Both are scaled with statistics of the train time steps (1-29).
    Final phase (the reported number):  fit on `trainval`, measure on `test`.
        Both are scaled with statistics of the trainval time steps (1-34).
    """
    masks = split_masks(bundle, cfg)
    select_scaler = fit_scaler(bundle["x"], masks["train"])
    final_scaler = fit_scaler(bundle["x"], masks["trainval"])
    make = lambda name, scaler: make_graph(bundle, masks[name], feature_set, scaler, direction)
    return {
        "train": make("train", select_scaler),
        "val": make("val", select_scaler),
        "trainval": make("trainval", final_scaler),
        "test": make("test", final_scaler),
        "select_scaler": select_scaler,
        "final_scaler": final_scaler,
    }


def build_table_set(bundle: dict, cfg: dict, feature_set: str) -> dict:
    """Same four pieces as build_graph_set, as (X, y, time_step, node_idx) tables.

    Training tables hold labelled rows only. The val/test tables hold every
    node of their time steps (unknown labels included, y = -1), so models can
    also score unlabelled transactions for the demo app; the metric code drops
    the unknown rows before computing anything.
    """
    masks = split_masks(bundle, cfg)
    select_scaler = fit_scaler(bundle["x"], masks["train"])
    final_scaler = fit_scaler(bundle["x"], masks["trainval"])
    return {
        "train": tabular_arrays(bundle, masks["train"], feature_set, select_scaler),
        "val": tabular_arrays(bundle, masks["val"], feature_set, select_scaler, labeled_only=False),
        "trainval": tabular_arrays(bundle, masks["trainval"], feature_set, final_scaler),
        "test": tabular_arrays(bundle, masks["test"], feature_set, final_scaler, labeled_only=False),
    }
