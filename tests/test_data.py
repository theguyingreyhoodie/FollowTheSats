"""Dataset facts. Every statement in the project brief about the data is asserted here."""

import numpy as np
import pandas as pd
import pytest
import torch

from btc_aml import data
from btc_aml.utils import data_dir


def test_node_and_edge_counts(bundle):
    assert bundle["x"].shape[0] == 203_769
    assert bundle["edge_index"].shape == (2, 234_355)
    assert bundle["tx_id"].unique().numel() == 203_769  # ids are unique


def test_label_counts(bundle):
    y = bundle["y"]
    assert int((y == data.ILLICIT).sum()) == 4_545
    assert int((y == data.LICIT).sum()) == 42_019
    assert int((y == data.UNKNOWN).sum()) == 157_205
    assert set(y.unique().tolist()) == {data.ILLICIT, data.LICIT, data.UNKNOWN}


def test_feature_layout(bundle):
    # 167 raw columns = txId + time step + 93 local + 72 aggregated.
    assert data.NUM_RAW_COLUMNS == 1 + 1 + 93 + 72 == 167
    assert bundle["x"].shape[1] == 165
    assert len(data.LOCAL_NAMES) == 93 and data.LOCAL_NAMES[0] == "local_01"
    assert data.LOCAL_NAMES[-1] == "local_93"
    assert len(data.AGG_NAMES) == 72 and data.AGG_NAMES[0] == "agg_01"
    assert data.AGG_NAMES[-1] == "agg_72"
    assert data.FEATURE_NAMES == data.LOCAL_NAMES + data.AGG_NAMES
    assert not torch.isnan(bundle["x"]).any()


def test_feature_sets_exclude_id_and_time(bundle):
    x = bundle["x"]
    assert x[:, data.FEATURE_SETS["LF"]].shape[1] == 93
    assert x[:, data.FEATURE_SETS["AF"]].shape[1] == 165
    # No feature column is a copy of the transaction id or the time step.
    tx_id = bundle["tx_id"].double()
    time_step = bundle["time_step"].double()
    for column in range(x.shape[1]):
        col = x[:, column].double()
        assert not torch.equal(col, tx_id)
        assert not torch.equal(col, time_step)


def test_time_steps_are_1_to_49(bundle):
    assert sorted(bundle["time_step"].unique().tolist()) == list(range(1, 50))


def test_edges_reference_known_nodes(bundle):
    ei = bundle["edge_index"]
    assert int(ei.min()) >= 0
    assert int(ei.max()) < bundle["x"].shape[0]


def test_no_edge_crosses_time_steps(bundle):
    t, ei = bundle["time_step"], bundle["edge_index"]
    assert bool((t[ei[0]] == t[ei[1]]).all())


def test_each_time_step_is_one_connected_component(bundle):
    components = data.count_components_per_time_step(bundle)
    assert len(components) == 49
    assert all(count == 1 for count in components.values())


def test_validate_bundle_passes(bundle):
    data.validate_bundle(bundle)  # raises AssertionError on any violated fact


def test_raw_csv_files(bundle, cfg):
    """Check the files on disk: three CSVs, features file has 167 columns and no header."""
    raw_dir = data.find_raw_dir(data_dir(cfg))
    if raw_dir is None:
        pytest.skip("raw CSVs not on disk (only the processed cache is)")

    head = pd.read_csv(raw_dir / "elliptic_txs_features.csv", header=None, nrows=5)
    assert head.shape[1] == 167
    # No header: the very first row is already a transaction of the dataset,
    # i.e. numeric, with its id in column 0 and its time step in column 1.
    assert int(head.iloc[0, 0]) == int(bundle["tx_id"][0])
    assert int(head.iloc[0, 1]) == int(bundle["time_step"][0])
    assert np.allclose(head.iloc[0, 2:].to_numpy(dtype=np.float32), bundle["x"][0].numpy())

    classes = pd.read_csv(raw_dir / "elliptic_txs_classes.csv", dtype=str)
    assert list(classes.columns) == ["txId", "class"]
    assert set(classes["class"].unique()) == {"1", "2", "unknown"}
    assert (classes["class"] == "1").sum() == 4_545
    assert (classes["class"] == "2").sum() == 42_019
    assert (classes["class"] == "unknown").sum() == 157_205

    edges = pd.read_csv(raw_dir / "elliptic_txs_edgelist.csv")
    assert list(edges.columns) == ["txId1", "txId2"]
    assert len(edges) == 234_355
    # The edge list is directed as given: row 0 of edge_index is txId1 (the
    # transaction whose output is spent), row 1 is txId2 (the spender).
    first = edges.iloc[0]
    tx_id = bundle["tx_id"]
    assert int(tx_id[bundle["edge_index"][0, 0]]) == int(first["txId1"])
    assert int(tx_id[bundle["edge_index"][1, 0]]) == int(first["txId2"])
