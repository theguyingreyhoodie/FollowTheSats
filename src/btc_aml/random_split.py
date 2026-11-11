"""Optional: how much does a RANDOM split inflate the scores?

    python -m btc_aml.random_split --config configs/quick.yaml

!!! These numbers are NOT results of this project. They exist only to show
!!! why the temporal split matters. They are stored with
!!! experiment = "random_split" and never appear in the main results table.

With a random split, labelled transactions from all 49 time steps are
shuffled and divided 60 / 10 / 30 into train / validation / test. The model
is then tested on transactions from the same fortnights, and the same
connected components, that it trained on. In deployment that never happens:
the transactions to be scored are always newer than the training data.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

from .data import ILLICIT, UNKNOWN, load_bundle
from .evaluate import best_f1_threshold, binary_metrics, chosen_settings, record_result
from .models import build_model
from .neural import fit, predict_scores
from .splits import fit_scaler, prepare_edges, select_features
from .train_baselines import FACTORIES
from .train_gnn import best_gnn
from .utils import config_arg_parser, get_device, get_logger, load_config, set_seed

log = get_logger()


def run(cfg: dict) -> None:
    section = cfg.get("random_split", {})
    if not section.get("enabled", False):
        log.info("random-split comparison disabled in config")
        return
    bundle = load_bundle(cfg)
    seed = cfg["seeds"][0]
    features = "AF"
    y_all = bundle["y"]
    n = y_all.numel()

    # Shuffle the labelled nodes, stratified so each part keeps the class ratio.
    labeled = torch.nonzero(y_all != UNKNOWN).squeeze(1).numpy()
    y_labeled = y_all[labeled].numpy()
    rest, test = train_test_split(
        labeled, test_size=section["test_fraction"], stratify=y_labeled, random_state=seed
    )
    train, val = train_test_split(
        rest, test_size=0.15, stratify=y_all[rest].numpy(), random_state=seed
    )

    # No time order exists in this setting, so all nodes' features are "available".
    scaler = fit_scaler(bundle["x"], torch.ones(n, dtype=torch.bool))
    x = select_features(scaler.transform(bundle["x"]), features)
    y_np = y_all.numpy()

    # ---- Random Forest ----
    hp, n_iter, _ = chosen_settings(cfg, "rf", features, seed)
    start = time.perf_counter()
    model = FACTORIES["rf"](hp, seed)
    model.fit(x[train].numpy(), y_np[train])
    threshold = best_f1_threshold(y_np[val], model.predict_proba(x[val].numpy())[:, ILLICIT])
    metrics = binary_metrics(y_np[test], model.predict_proba(x[test].numpy())[:, ILLICIT], threshold)
    record_result(cfg, experiment="random_split", model="rf", features=features, seed=seed,
                  split="test", metrics=metrics,
                  extra={"hparams": hp, "train_seconds": time.perf_counter() - start})
    log.info("  [RANDOM SPLIT rf/%s] test F1 %.3f P %.3f R %.3f", features,
             metrics["f1"], metrics["precision"], metrics["recall"])

    # ---- best GNN, transductive: one graph, three label masks ----
    kind = best_gnn(cfg, features)
    direction = cfg["gnn"]["direction"]
    hp, _, _ = chosen_settings(cfg, kind, features, seed, variant=direction)
    edge_index = prepare_edges(bundle["edge_index"], direction)

    def view(nodes: np.ndarray) -> Data:
        """The full graph, with only `nodes` marked as labelled."""
        mask = torch.zeros(n, dtype=torch.bool)
        mask[torch.from_numpy(nodes)] = True
        return Data(x=x, edge_index=edge_index, y=y_all, labeled=mask)

    start = time.perf_counter()
    set_seed(seed)
    device = get_device(cfg)
    gnn = build_model(kind, x.shape[1], hp, direction)
    fit(gnn, view(train), lr=hp["lr"], weight_decay=hp["weight_decay"],
        max_epochs=cfg["gnn"]["max_epochs"], patience=cfg["gnn"]["patience"],
        val_graph=view(val), device=device)
    scores = predict_scores(gnn, view(test), device)
    threshold = best_f1_threshold(y_np[val], scores[val])
    metrics = binary_metrics(y_np[test], scores[test], threshold)
    record_result(cfg, experiment="random_split", model=kind, features=features, seed=seed,
                  split="test", metrics=metrics, variant=direction,
                  extra={"hparams": hp, "train_seconds": time.perf_counter() - start})
    log.info("  [RANDOM SPLIT %s/%s] test F1 %.3f P %.3f R %.3f", kind, features,
             metrics["f1"], metrics["precision"], metrics["recall"])


def main() -> None:
    args = config_arg_parser("Random-split comparison (illustration only).").parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
