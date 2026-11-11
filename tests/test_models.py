"""Model code: output shapes for every architecture and edge-direction mode."""

import pytest
import torch

from btc_aml.models import GNN_KINDS, build_model
from btc_aml.splits import DIRECTIONS, prepare_edges

HP = {"hidden": 16, "dropout": 0.3, "num_layers": 2, "gat_heads": 4}


def toy_graph(num_nodes=30, in_dim=7, num_edges=60):
    torch.manual_seed(0)
    x = torch.randn(num_nodes, in_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    return x, edge_index


@pytest.mark.parametrize("kind", GNN_KINDS)
@pytest.mark.parametrize("direction", DIRECTIONS)
def test_gnn_shapes(kind, direction):
    x, edge_index = toy_graph()
    edge_index = prepare_edges(edge_index, direction)
    model = build_model(kind, x.shape[1], HP, direction).eval()
    assert model(x, edge_index).shape == (30, 2)
    assert model.embed(x, edge_index).shape == (30, 16)


def test_mlp_ignores_edges():
    x, edge_index = toy_graph()
    model = build_model("mlp", x.shape[1], HP).eval()
    assert torch.equal(model(x, edge_index), model(x, None))


def test_gnn_uses_edges():
    """Changing the edges changes a GNN's output (and not the MLP's)."""
    x, edge_index = toy_graph()
    model = build_model("gcn", x.shape[1], HP).eval()
    other_edges = edge_index.flip(1)[:, :20]
    assert not torch.allclose(model(x, edge_index), model(x, other_edges))


def test_two_layers_see_exactly_two_hops():
    """On a path graph 0-1-2-3-4, node 0 is affected by node 2 but not by node 3."""
    edge_index = prepare_edges(torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]), "undirected")
    x = torch.randn(5, 7)
    model = build_model("sage", 7, HP).eval()
    base = model(x, edge_index)[0]

    x_far = x.clone()
    x_far[3] += 10.0          # 3 hops from node 0
    assert torch.allclose(model(x_far, edge_index)[0], base)

    x_near = x.clone()
    x_near[2] += 10.0         # 2 hops from node 0
    assert not torch.allclose(model(x_near, edge_index)[0], base)
