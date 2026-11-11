"""Split integrity: temporal boundaries, no overlap, no leakage through edges or scaling."""

import torch

from btc_aml import splits
from btc_aml.data import UNKNOWN


def test_split_boundaries(bundle, cfg):
    masks = splits.split_masks(bundle, cfg)
    t = bundle["time_step"]
    assert sorted(t[masks["train"]].unique().tolist()) == list(range(1, 30))
    assert sorted(t[masks["val"]].unique().tolist()) == list(range(30, 35))
    assert sorted(t[masks["test"]].unique().tolist()) == list(range(35, 50))
    assert sorted(t[masks["trainval"]].unique().tolist()) == list(range(1, 35))


def test_no_overlap_and_full_coverage(bundle, cfg):
    masks = splits.split_masks(bundle, cfg)
    train, val, test = masks["train"], masks["val"], masks["test"]
    assert not (train & val).any()
    assert not (train & test).any()
    assert not (val & test).any()
    assert bool((train | val | test).all())  # every node is in exactly one split
    assert int(train.sum() + val.sum() + test.sum()) == bundle["x"].shape[0]
    # Training always strictly precedes validation, which precedes test.
    t = bundle["time_step"]
    assert t[train].max() < t[val].min() <= t[val].max() < t[test].min()


def test_graphs_have_no_cross_split_edges(bundle, cfg):
    masks = splits.split_masks(bundle, cfg)
    scaler = splits.fit_scaler(bundle["x"], masks["train"])
    total_edges = 0
    seen_nodes = set()
    for name in ("train", "val", "test"):
        graph = splits.make_graph(bundle, masks[name], "AF", scaler, direction="directed")
        lo, hi = cfg["split"][name]
        # Every node and both endpoints of every edge are inside the split's time steps.
        assert int(graph.time_step.min()) >= lo and int(graph.time_step.max()) <= hi
        src, dst = graph.edge_index
        assert bool((graph.time_step[src] == graph.time_step[dst]).all())
        assert int(graph.edge_index.max()) < graph.num_nodes
        # node_idx maps back to the right rows of the full dataset.
        assert torch.equal(bundle["y"][graph.node_idx], graph.y)
        nodes = set(graph.node_idx.tolist())
        assert not (nodes & seen_nodes)  # node sets of different splits are disjoint
        seen_nodes |= nodes
        total_edges += graph.edge_index.shape[1]
    # No edge was lost by splitting, because none crosses time steps.
    assert total_edges == bundle["edge_index"].shape[1]
    assert len(seen_nodes) == bundle["x"].shape[0]


def test_unknown_nodes_stay_in_graph_but_not_in_labeled_mask(bundle, cfg):
    masks = splits.split_masks(bundle, cfg)
    scaler = splits.fit_scaler(bundle["x"], masks["train"])
    graph = splits.make_graph(bundle, masks["train"], "LF", scaler)
    assert graph.num_nodes == int(masks["train"].sum())      # unknown nodes kept
    assert int((graph.y == UNKNOWN).sum()) > 0
    assert not (graph.y[graph.labeled] == UNKNOWN).any()     # but masked out of the loss
    assert int(graph.labeled.sum()) == int((graph.y != UNKNOWN).sum())
    assert graph.x.shape[1] == 93


def test_direction_modes(bundle, cfg):
    masks = splits.split_masks(bundle, cfg)
    scaler = splits.fit_scaler(bundle["x"], masks["val"])
    directed = splits.make_graph(bundle, masks["val"], "LF", scaler, "directed")
    undirected = splits.make_graph(bundle, masks["val"], "LF", scaler, "undirected")
    bidirectional = splits.make_graph(bundle, masks["val"], "LF", scaler, "bidirectional")
    n = directed.num_nodes
    as_set = lambda ei: set((ei[0] * n + ei[1]).tolist())
    forward = as_set(directed.edge_index)
    backward = as_set(directed.edge_index.flip(0))
    assert as_set(undirected.edge_index) == forward | backward
    assert torch.equal(bidirectional.edge_index, directed.edge_index)


def test_scaler_uses_training_nodes_only(bundle, cfg):
    masks = splits.split_masks(bundle, cfg)
    x = bundle["x"]
    scaler = splits.fit_scaler(x, masks["train"])

    # Standardised training features have mean 0 and std 1...
    z = scaler.transform(x[masks["train"]])
    assert torch.allclose(z.mean(0), torch.zeros(x.shape[1]), atol=1e-3)
    assert torch.allclose(z.std(0), torch.ones(x.shape[1]), atol=1e-3)

    # ...and the statistics do not change when validation/test rows change.
    tampered = x.clone()
    tampered[~masks["train"]] += 1000.0
    scaler_tampered = splits.fit_scaler(tampered, masks["train"])
    assert torch.equal(scaler.mean, scaler_tampered.mean)
    assert torch.equal(scaler.std, scaler_tampered.std)


def test_tabular_arrays_are_labeled_only(bundle, cfg):
    masks = splits.split_masks(bundle, cfg)
    scaler = splits.fit_scaler(bundle["x"], masks["train"])
    X, y, t, idx = splits.tabular_arrays(bundle, masks["test"], "AF", scaler)
    assert X.shape == (len(y), 165)
    assert set(y.tolist()) == {0, 1}
    assert t.min() >= 35 and t.max() <= 49
    assert (bundle["y"][idx].numpy() == y).all()
