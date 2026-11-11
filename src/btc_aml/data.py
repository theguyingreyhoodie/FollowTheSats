"""Download, validate and cache the Elliptic Bitcoin dataset.

The dataset is three CSV files:

    elliptic_txs_features.csv   one row per transaction, 167 columns, NO header
                                col 0      = txId
                                col 1      = time step (1..49)
                                col 2..94  = 93 "local" features of the transaction
                                col 95..166 = 72 "aggregated" features: statistics
                                             of the transaction's one-hop neighbours
    elliptic_txs_classes.csv    txId, class   ("1" illicit, "2" licit, "unknown")
    elliptic_txs_edgelist.csv   txId1, txId2  (bitcoin flows from txId1 to txId2)

This module turns them into one dictionary of tensors (the "bundle") and
caches it, because parsing the 690 MB features file takes a while.

Run it directly to download + build the cache:

    python -m btc_aml.data --config configs/quick.yaml
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .utils import config_arg_parser, data_dir, get_logger, load_config

log = get_logger()

# --------------------------------------------------------------------------- #
# Facts about the dataset. validate_bundle() and the tests assert all of them.
# --------------------------------------------------------------------------- #
NUM_NODES = 203_769
NUM_EDGES = 234_355
NUM_ILLICIT = 4_545
NUM_LICIT = 42_019
NUM_UNKNOWN = 157_205
NUM_TIME_STEPS = 49
NUM_RAW_COLUMNS = 167
NUM_LOCAL = 93
NUM_AGG = 72
NUM_FEATURES = NUM_LOCAL + NUM_AGG  # 165

# Our label encoding. The CSV uses "1"/"2"/"unknown"; we use the usual ML
# convention that the positive (rare, interesting) class is 1.
ILLICIT, LICIT, UNKNOWN = 1, 0, -1

# The features are anonymised, so we can only give them positional names.
LOCAL_NAMES = [f"local_{i:02d}" for i in range(1, NUM_LOCAL + 1)]
AGG_NAMES = [f"agg_{i:02d}" for i in range(1, NUM_AGG + 1)]
FEATURE_NAMES = LOCAL_NAMES + AGG_NAMES

# Feature sets used throughout the project (same names as the original paper):
#   LF = the 93 local features
#   AF = all 165 features (local + aggregated)
# txId and the time step are never part of either set.
FEATURE_SETS = {
    "LF": slice(0, NUM_LOCAL),
    "AF": slice(0, NUM_FEATURES),
}

RAW_FILES = (
    "elliptic_txs_features.csv",
    "elliptic_txs_classes.csv",
    "elliptic_txs_edgelist.csv",
)
KAGGLE_DATASET = "ellipticco/elliptic-data-set"


# --------------------------------------------------------------------------- #
# Step 1: get the raw CSVs
# --------------------------------------------------------------------------- #
def find_raw_dir(base: Path) -> Path | None:
    """Return the folder under `base` that contains all three CSVs, if any.

    Kaggle unzips into data/raw/elliptic_bitcoin_dataset/, PyG downloads into
    data/pyg/raw/, and a manual download could be anywhere under data/, so we
    simply search for the features file.
    """
    if not base.exists():
        return None
    for features_file in sorted(base.rglob(RAW_FILES[0])):
        folder = features_file.parent
        if all((folder / name).exists() for name in RAW_FILES):
            return folder
    return None


def _download_with_kaggle(base: Path) -> bool:
    """Attempt 1: the Kaggle API (needs a Kaggle token on this machine)."""
    try:
        # Importing `kaggle` authenticates immediately and fails (sometimes
        # with SystemExit) when no token is configured, hence BaseException.
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        target = base / "raw"
        target.mkdir(parents=True, exist_ok=True)
        log.info("Downloading %s with the Kaggle API (about 150 MB zipped)...", KAGGLE_DATASET)
        api.dataset_download_files(KAGGLE_DATASET, path=str(target), unzip=True, quiet=True)
        return find_raw_dir(base) is not None
    except BaseException as err:  # noqa: BLE001 - we want to fall through on anything
        log.info("Kaggle download not available (%s: %s).", type(err).__name__, err)
        return False


def _download_with_pyg(base: Path) -> bool:
    """Attempt 2: let PyTorch Geometric download the same CSVs.

    We use the PyG dataset class ONLY as a downloader. Its processed `Data`
    object drops the time-step column, which we need for the temporal split,
    so we always rebuild the graph ourselves from the raw CSVs.
    """
    try:
        from torch_geometric.datasets import EllipticBitcoinDataset

        log.info("Downloading via torch_geometric.datasets.EllipticBitcoinDataset...")
        EllipticBitcoinDataset(root=str(base / "pyg"))
        return find_raw_dir(base) is not None
    except Exception as err:  # noqa: BLE001
        log.info("PyG download failed (%s: %s).", type(err).__name__, err)
        return False


def ensure_raw_data(base: Path) -> Path:
    """Make sure the three CSVs exist locally and return their folder."""
    found = find_raw_dir(base)
    if found is not None:
        return found
    if _download_with_kaggle(base) or _download_with_pyg(base):
        return find_raw_dir(base)
    raise FileNotFoundError(
        "Could not download the Elliptic dataset automatically.\n"
        "Download it by hand from https://www.kaggle.com/datasets/ellipticco/elliptic-data-set\n"
        f"and put these three files into {base / 'raw'}:\n  - "
        + "\n  - ".join(RAW_FILES)
    )


# --------------------------------------------------------------------------- #
# Step 2: CSVs -> tensors
# --------------------------------------------------------------------------- #
def build_bundle(raw_dir: Path) -> dict:
    """Parse the three CSVs into one dictionary of aligned tensors.

    "Aligned" means row i of every tensor describes the same transaction.
    The row number i is the node index used everywhere else in the project.

    Returns a dict with:
        tx_id       int64 [N]       original transaction ids
        time_step   int64 [N]       1..49
        x           float32 [N,165] features (93 local then 72 aggregated)
        y           int64 [N]       1 illicit, 0 licit, -1 unknown
        edge_index  int64 [2,E]     row 0 = source node index, row 1 = target
    """
    # --- features: no header row, so pandas must not treat row 0 as one ---
    log.info("Reading %s (690 MB, this takes a little while)...", RAW_FILES[0])
    feats = pd.read_csv(raw_dir / RAW_FILES[0], header=None)
    if feats.shape[1] != NUM_RAW_COLUMNS:
        raise ValueError(f"expected {NUM_RAW_COLUMNS} columns, found {feats.shape[1]}")

    # copy=True gives us our own writable arrays (pandas may hand out read-only views).
    tx_id = feats.iloc[:, 0].to_numpy(dtype=np.int64, copy=True)
    time_step = feats.iloc[:, 1].to_numpy(dtype=np.int64, copy=True)
    x = feats.iloc[:, 2:].to_numpy(dtype=np.float32, copy=True)

    # --- labels: map the CSV strings onto our integer encoding ---
    classes = pd.read_csv(raw_dir / RAW_FILES[1], dtype=str)
    classes.columns = ["txId", "class"]
    label_map = {"1": ILLICIT, "2": LICIT, "unknown": UNKNOWN}
    if not set(classes["class"].unique()) <= set(label_map):
        raise ValueError(f"unexpected class values: {classes['class'].unique()}")
    label_of = dict(
        zip(classes["txId"].astype(np.int64), classes["class"].map(label_map))
    )
    # Look every transaction up in feature-file order, so labels line up with x.
    y = np.array([label_of[t] for t in tx_id], dtype=np.int64)

    # --- edges: translate transaction ids into node indices 0..N-1 ---
    # A GNN library addresses nodes by position, not by an arbitrary id.
    index_of = {t: i for i, t in enumerate(tx_id)}
    edges = pd.read_csv(raw_dir / RAW_FILES[2])
    src = edges.iloc[:, 0].map(index_of).to_numpy(dtype=np.int64)
    dst = edges.iloc[:, 1].map(index_of).to_numpy(dtype=np.int64)
    edge_index = np.stack([src, dst])

    return {
        "tx_id": torch.from_numpy(tx_id),
        "time_step": torch.from_numpy(time_step),
        "x": torch.from_numpy(x),
        "y": torch.from_numpy(y),
        "edge_index": torch.from_numpy(edge_index),
    }


# --------------------------------------------------------------------------- #
# Step 3: validate
# --------------------------------------------------------------------------- #
def count_components_per_time_step(bundle: dict) -> dict[int, int]:
    """Number of connected components inside each time step.

    Direction is ignored ("weakly connected"): two transactions are in the same
    component if you can walk between them along edges in either direction.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = bundle["x"].shape[0]
    src, dst = bundle["edge_index"].numpy()
    adjacency = coo_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
    _, component_of = connected_components(adjacency, directed=False)
    time_step = bundle["time_step"].numpy()
    return {
        int(t): int(len(np.unique(component_of[time_step == t])))
        for t in np.unique(time_step)
    }


def validate_bundle(bundle: dict) -> None:
    """Fail loudly if the data does not match the published dataset facts.

    Cheap insurance: a truncated download or a silently changed file would
    otherwise produce plausible-looking but wrong results.
    """
    x, y, t, ei = bundle["x"], bundle["y"], bundle["time_step"], bundle["edge_index"]

    assert x.shape == (NUM_NODES, NUM_FEATURES), f"feature matrix shape {tuple(x.shape)}"
    assert ei.shape == (2, NUM_EDGES), f"edge_index shape {tuple(ei.shape)}"
    assert bundle["tx_id"].unique().numel() == NUM_NODES, "transaction ids are not unique"
    assert not torch.isnan(x).any(), "NaN in features"

    assert int((y == ILLICIT).sum()) == NUM_ILLICIT
    assert int((y == LICIT).sum()) == NUM_LICIT
    assert int((y == UNKNOWN).sum()) == NUM_UNKNOWN

    assert sorted(t.unique().tolist()) == list(range(1, NUM_TIME_STEPS + 1))

    # Every edge endpoint was found in the features file...
    assert int(ei.min()) >= 0 and int(ei.max()) < NUM_NODES
    # ...and no edge links two different time steps. This is what makes a
    # clean temporal split possible: the graph is 49 separate pieces.
    assert bool((t[ei[0]] == t[ei[1]]).all()), "an edge crosses time steps"

    components = count_components_per_time_step(bundle)
    assert all(c == 1 for c in components.values()), (
        f"time steps that are not a single connected component: "
        f"{ {k: v for k, v in components.items() if v != 1} }"
    )


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def load_bundle(cfg: dict, rebuild: bool = False) -> dict:
    """Return the tensor bundle, downloading and building it on first use."""
    base = data_dir(cfg)
    cache = base / "processed" / "elliptic_bundle.pt"
    if cache.exists() and not rebuild:
        return torch.load(cache, weights_only=True)

    raw_dir = ensure_raw_data(base)
    bundle = build_bundle(raw_dir)
    validate_bundle(bundle)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, cache)
    log.info("Validated and cached the dataset at %s", cache)
    return bundle


def main() -> None:
    args = config_arg_parser("Download, validate and cache the Elliptic dataset.").parse_args()
    cfg = load_config(args.config)
    bundle = load_bundle(cfg)
    y = bundle["y"]
    log.info(
        "nodes=%d edges=%d illicit=%d licit=%d unknown=%d time_steps=%d",
        bundle["x"].shape[0],
        bundle["edge_index"].shape[1],
        int((y == ILLICIT).sum()),
        int((y == LICIT).sum()),
        int((y == UNKNOWN).sum()),
        bundle["time_step"].unique().numel(),
    )


if __name__ == "__main__":
    main()
