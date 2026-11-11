"""Neural network models: an MLP baseline and four graph neural networks.

All models share one interface so the training code does not care which is which:

    model(x, edge_index)        -> logits, shape [num_nodes, 2]  (licit, illicit)
    model.embed(x, edge_index)  -> the hidden vector of each node just before
                                   the final classifier ("penultimate layer").
                                   The hybrid experiment feeds these to trees.

The idea behind every GNN layer here is MESSAGE PASSING:

    1. each node collects ("aggregates") the vectors of its neighbours,
    2. combines that summary with its own vector,
    3. applies a learned linear map and a non-linearity.

After one layer a node knows about its direct neighbours; after two layers it
knows about neighbours of neighbours (its 2-hop neighbourhood). The four
architectures differ only in HOW step 1 weighs the neighbours:

    GCN        fixed weights from node degrees: 1 / sqrt(deg_i * deg_j)
    Skip-GCN   GCN, plus a direct shortcut from the raw input features
    GraphSAGE  plain mean of the neighbours, with separate weights for "self"
    GAT        learned attention weights: the network decides per edge
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import DirGNNConv, GATConv, GCNConv, SAGEConv

GNN_KINDS = ("gcn", "skip_gcn", "sage", "gat")


# --------------------------------------------------------------------------- #
# MLP baseline (no graph)
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    """A plain feed-forward network on the node's own features.

    It is the "GNN without the graph": same loss, optimiser and training loop,
    but no message passing. Comparing it with the GNNs isolates what the
    graph structure adds. `edge_index` is accepted and ignored so the MLP can
    be dropped into the same training code.
    """

    def __init__(self, in_dim: int, hidden: int, dropout: float, num_layers: int = 2):
        super().__init__()
        dims = [in_dim] + [hidden] * num_layers
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)]
        )
        self.classifier = nn.Linear(hidden, 2)
        self.dropout = dropout

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = F.relu(layer(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        return self.classifier(self.embed(x, edge_index))


# --------------------------------------------------------------------------- #
# GNN building blocks
# --------------------------------------------------------------------------- #
def make_conv(kind: str, in_dim: int, out_dim: int, heads: int, dropout: float) -> nn.Module:
    """Create one message-passing layer of the requested type.

    All four return a vector of size `out_dim` per node.
    """
    if kind in ("gcn", "skip_gcn"):
        # h_i' = W * sum_{j in N(i) or i}  h_j / sqrt(deg_i * deg_j)
        return GCNConv(in_dim, out_dim)
    if kind == "sage":
        # h_i' = W_self * h_i + W_neigh * mean_{j in N(i)} h_j
        return SAGEConv(in_dim, out_dim, aggr="mean")
    if kind == "gat":
        # h_i' = sum_j alpha_ij * W h_j, with alpha_ij a learned attention weight.
        # `heads` independent attention mechanisms run in parallel and their
        # outputs are concatenated, so each head gets out_dim / heads channels.
        if out_dim % heads != 0:
            raise ValueError("hidden size must be divisible by gat_heads")
        return GATConv(in_dim, out_dim // heads, heads=heads, dropout=dropout)
    raise ValueError(f"unknown GNN kind '{kind}', expected one of {GNN_KINDS}")


class GNN(nn.Module):
    """A stack of message-passing layers followed by a linear classifier.

    Args:
        kind:        "gcn", "skip_gcn", "sage" or "gat"
        in_dim:      number of input features (93 for LF, 165 for AF)
        hidden:      size of every hidden layer and of the node embedding
        num_layers:  number of message-passing layers = how many hops away a
                     node can see (2 by default)
        dropout:     probability of zeroing a hidden unit during training
        heads:       number of attention heads (GAT only)
        direction:   "undirected" / "directed" use the edge list as prepared by
                     splits.prepare_edges. "bidirectional" additionally wraps
                     every layer in DirGNNConv: one convolution along the edge
                     direction, one against it, each with its OWN weights.
    """

    def __init__(
        self,
        kind: str,
        in_dim: int,
        hidden: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        heads: int = 4,
        direction: str = "undirected",
    ):
        super().__init__()
        self.kind = kind
        self.direction = direction
        self.dropout = dropout

        self.convs = nn.ModuleList()
        # Extra "self" weights, used only by the bidirectional variant (see below).
        self.roots = nn.ModuleList()
        # Skip-GCN only: linear maps from the RAW input features into each layer.
        self.skips = nn.ModuleList()

        for layer in range(num_layers):
            layer_in = in_dim if layer == 0 else hidden
            conv = make_conv(kind, layer_in, hidden, heads, dropout)
            if direction == "bidirectional":
                # DirGNNConv keeps two copies of `conv`: conv_in is applied to
                # the edges as given (messages from payers), conv_out to the
                # flipped edges (messages from payees). It switches off the
                # inner layers' self-loops, so we add the node's own vector
                # back explicitly through a separate linear map.
                conv = DirGNNConv(conv, alpha=0.5, root_weight=False)
                self.roots.append(nn.Linear(layer_in, hidden))
            self.convs.append(conv)
            if kind == "skip_gcn":
                self.skips.append(nn.Linear(in_dim, hidden, bias=False))

        self.classifier = nn.Linear(hidden, 2)

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Run the message-passing layers and return each node's hidden vector."""
        x_input = x  # kept for the Skip-GCN shortcut
        h = x
        for layer, conv in enumerate(self.convs):
            h_in = h
            h = conv(h_in, edge_index)                 # 1. aggregate neighbours
            if self.direction == "bidirectional":
                h = h + self.roots[layer](h_in)        #    + the node's own vector
            if self.kind == "skip_gcn":
                # Skip connection from the input: the layer output is
                #   GCN(h) + W_skip * x
                # so the node's own raw features cannot be washed out by
                # averaging with its neighbours.
                h = h + self.skips[layer](x_input)
            h = F.relu(h)                              # 2. non-linearity
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Logits for the two classes. Softmax is applied later, in the loss."""
        return self.classifier(self.embed(x, edge_index))


def build_model(kind: str, in_dim: int, hp: dict, direction: str = "undirected") -> nn.Module:
    """Create a model from a hyperparameter dictionary (see configs/*.yaml)."""
    if kind == "mlp":
        return MLP(in_dim, hidden=hp["hidden"], dropout=hp["dropout"])
    return GNN(
        kind,
        in_dim,
        hidden=hp["hidden"],
        num_layers=hp.get("num_layers", 2),
        dropout=hp["dropout"],
        heads=hp.get("gat_heads", 4),
        direction=direction,
    )
