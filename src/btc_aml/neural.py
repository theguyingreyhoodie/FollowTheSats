"""The training loop shared by the MLP and all GNNs.

Full-batch training: every step uses the whole training graph at once. The
graph is small (about 136k nodes, 165 features), so there is no need for
neighbour sampling or mini-batches, which keeps the loop short enough to read:

    logits = model(x, edge_index)                 # forward pass over ALL nodes
    loss   = weighted cross-entropy on LABELLED nodes only
    loss.backward(); optimizer.step()

Unknown-label nodes take part in the forward pass (their features flow to
their neighbours) but contribute nothing to the loss.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch import nn
from torch_geometric.data import Data

from .data import ILLICIT, LICIT


@dataclass
class FitResult:
    epochs_run: int          # how many epochs were actually executed
    best_epoch: int          # epoch with the best validation PR-AUC (= epochs_run without validation)
    best_val_pr_auc: float   # nan when no validation graph was given


def class_weights(y_labeled: torch.Tensor) -> torch.Tensor:
    """"Balanced" class weights: weight_c = n_samples / (2 * n_c).

    With ~11% illicit in training, an unweighted loss can be made small by
    mostly ignoring the illicit class. Weighting each class inversely to its
    frequency makes the two classes contribute equally to the loss in total,
    so one illicit example counts as much as about eight licit ones.
    """
    n = y_labeled.numel()
    n_licit = int((y_labeled == LICIT).sum())
    n_illicit = int((y_labeled == ILLICIT).sum())
    # Index 0 = licit, index 1 = illicit (matches the label encoding).
    return torch.tensor([n / (2 * n_licit), n / (2 * n_illicit)], dtype=torch.float32)


@torch.no_grad()
def predict_scores(model: nn.Module, graph: Data, device: torch.device) -> np.ndarray:
    """Illicit probability for EVERY node of `graph` (labelled or not)."""
    model.eval()  # switches dropout off
    logits = model(graph.x.to(device), graph.edge_index.to(device))
    return F.softmax(logits, dim=1)[:, ILLICIT].cpu().numpy()


@torch.no_grad()
def node_embeddings(model: nn.Module, graph: Data, device: torch.device) -> np.ndarray:
    """Penultimate-layer vector of every node (input to the hybrid tree models)."""
    model.eval()
    return model.embed(graph.x.to(device), graph.edge_index.to(device)).cpu().numpy()


def fit(
    model: nn.Module,
    graph: Data,
    *,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    device: torch.device,
    val_graph: Data | None = None,
    patience: int | None = None,
) -> FitResult:
    """Train `model` on `graph`.

    With `val_graph`: after every epoch the model scores the validation graph;
    we remember the weights with the best validation PR-AUC and stop once
    `patience` epochs pass without improvement (early stopping). The best
    weights are restored at the end.

    Without `val_graph`: train for exactly `max_epochs` epochs. This is used
    for the final models, whose epoch count was already chosen on validation.
    """
    model.to(device)
    x, edge_index = graph.x.to(device), graph.edge_index.to(device)
    y, labeled = graph.y.to(device), graph.labeled.to(device)
    weight = class_weights(graph.y[graph.labeled]).to(device)

    # Adam: gradient descent with a per-parameter adaptive step size.
    # weight_decay adds an L2 penalty that discourages large weights.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if val_graph is not None:
        val_known = val_graph.labeled.numpy()
        val_y = val_graph.y.numpy()[val_known]

    best_score, best_epoch, best_state = float("-inf"), 0, None
    epoch = 0
    for epoch in range(1, max_epochs + 1):
        model.train()  # switches dropout on
        optimizer.zero_grad()
        logits = model(x, edge_index)
        # Loss on labelled nodes only; unknown nodes are masked out here.
        loss = F.cross_entropy(logits[labeled], y[labeled], weight=weight)
        loss.backward()
        optimizer.step()

        if val_graph is None:
            continue
        score = average_precision_score(val_y, predict_scores(model, val_graph, device)[val_known])
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
        elif patience is not None and epoch - best_epoch >= patience:
            break

    if val_graph is None:
        return FitResult(epochs_run=epoch, best_epoch=epoch, best_val_pr_auc=float("nan"))
    model.load_state_dict(best_state)
    return FitResult(epochs_run=epoch, best_epoch=best_epoch, best_val_pr_auc=float(best_score))
