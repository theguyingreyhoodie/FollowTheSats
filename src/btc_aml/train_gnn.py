"""Train the GNNs (GCN, Skip-GCN, GraphSAGE, GAT) and run the edge-direction ablation.

    python -m btc_aml.train_gnn --config configs/quick.yaml              # main comparison
    python -m btc_aml.train_gnn --config configs/quick.yaml --ablation   # direction ablation

Training is INDUCTIVE: the model is trained on the graph of the training time
steps and then applied to the graph of later time steps it has never seen.
That is possible because a GNN layer's weights do not depend on which nodes
exist; they only describe how to combine a node with its neighbours.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .data import load_bundle
from .evaluate import load_metrics, run_tag
from .models import GNN_KINDS, build_model
from .neural import fit, predict_scores
from .protocol import run_protocol
from .splits import Scaler, build_graph_set
from .utils import config_arg_parser, get_device, get_logger, load_config, models_dir, set_seed

log = get_logger()


class NeuralLearner:
    """Plugs a neural model (MLP or GNN) into the shared protocol.

    fit_select trains on the `train` graph with early stopping on `val`;
    fit_final retrains on `trainval` for the epoch count found by fit_select.
    Both save a checkpoint so later stages (hybrid, explainability, the app)
    can reload the exact model instead of retraining it.
    """

    def __init__(self, cfg: dict, kind: str, features: str, graphs: dict, section: dict,
                 experiment: str, direction: str = "undirected", variant: str = "-"):
        self.cfg = cfg
        self.kind = kind
        self.features = features
        self.graphs = graphs
        self.section = section          # the config block with max_epochs, patience, ...
        self.experiment = experiment
        self.direction = direction
        self.variant = variant
        self.device = get_device(cfg)
        self.in_dim = graphs["train"].x.shape[1]

    def _new_model(self, hp: dict, seed: int):
        # Seeding right before construction fixes the random initial weights.
        set_seed(seed)
        return build_model(self.kind, self.in_dim, hp, self.direction)

    def _save(self, model, hp: dict, seed: int, phase: str, scaler: Scaler, info: dict) -> None:
        tag = run_tag(self.experiment, self.kind, self.features, self.variant, seed)
        torch.save(
            {
                "kind": self.kind,
                "features": self.features,
                "direction": self.direction,
                "in_dim": self.in_dim,
                "hp": hp,
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "scaler_mean": scaler.mean,
                "scaler_std": scaler.std,
                "n_iter": info.get("n_iter"),
                "threshold": info.get("threshold"),
                "phase": phase,
            },
            models_dir(self.cfg) / f"{tag}__{phase}.pt",
        )

    def fit_select(self, hp: dict, seed: int):
        model = self._new_model(hp, seed)
        result = fit(
            model,
            self.graphs["train"],
            lr=hp["lr"],
            weight_decay=hp["weight_decay"],
            max_epochs=self.section["max_epochs"],
            patience=self.section["patience"],
            val_graph=self.graphs["val"],
            device=self.device,
        )
        info = {"n_iter": result.best_epoch}
        self._save(model, hp, seed, "select", self.graphs["select_scaler"], info)
        return predict_scores(model, self.graphs["val"], self.device), info

    def fit_final(self, hp: dict, seed: int, info: dict):
        model = self._new_model(hp, seed)
        fit(
            model,
            self.graphs["trainval"],
            lr=hp["lr"],
            weight_decay=hp["weight_decay"],
            max_epochs=info["n_iter"],      # chosen on validation, no early stopping here
            device=self.device,
        )
        self._save(model, hp, seed, "final", self.graphs["final_scaler"], info)
        return predict_scores(model, self.graphs["test"], self.device)


def load_checkpoint(path: Path):
    """Rebuild a saved model. Returns (model in eval mode, checkpoint dict)."""
    ckpt = torch.load(path, weights_only=False)
    model = build_model(ckpt["kind"], ckpt["in_dim"], ckpt["hp"], ckpt["direction"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt


def checkpoint_path(cfg: dict, experiment: str, kind: str, features: str, variant: str,
                    seed: int, phase: str) -> Path:
    return models_dir(cfg) / f"{run_tag(experiment, kind, features, variant, seed)}__{phase}.pt"


def refs(graphs: dict):
    """(y, time_step) for val and (y, time_step, node_idx) for test, as NumPy arrays."""
    val, test = graphs["val"], graphs["test"]
    return (
        (val.y.numpy(), val.time_step.numpy()),
        (test.y.numpy(), test.time_step.numpy(), test.node_idx.numpy()),
    )


def base_hparams(section: dict) -> dict:
    """Fixed hyperparameters from the config; grid values are merged on top by the protocol."""
    keys = ("hidden", "num_layers", "gat_heads", "lr", "weight_decay", "dropout")
    return {k: section[k] for k in keys if k in section}


def best_gnn(cfg: dict, features: str) -> str:
    """The GNN architecture with the highest mean VALIDATION PR-AUC on `features`.

    Used to decide which GNN the ablation and hybrid experiments build on.
    The choice is made on validation results only; picking "the best on test"
    would be selecting a model using the test set.
    """
    df = load_metrics(cfg)
    rows = df[
        (df["experiment"] == "main") & (df["split"] == "val")
        & (df["features"] == features) & (df["model"].isin(GNN_KINDS))
    ]
    if rows.empty:
        raise RuntimeError("No GNN validation results yet. Run btc_aml.train_gnn first.")
    return rows.groupby("model")["pr_auc"].mean().idxmax()


def run(cfg: dict) -> None:
    """Main comparison: every GNN on every feature set, default edge direction."""
    bundle = load_bundle(cfg)
    section = cfg["gnn"]
    direction = section["direction"]
    for features in cfg["feature_sets"]:
        graphs = build_graph_set(bundle, cfg, features, direction)
        val_ref, test_ref = refs(graphs)
        for kind in section["models"]:
            log.info("GNN %s on %s (%s edges)", kind, features, direction)
            learner = NeuralLearner(cfg, kind, features, graphs, section, "main", direction, direction)
            run_protocol(
                cfg, learner, experiment="main", model=kind, features=features, variant=direction,
                base_hp=base_hparams(section), grid=section.get("grid"),
                val_ref=val_ref, test_ref=test_ref,
            )


def run_ablation(cfg: dict) -> None:
    """Edge-direction ablation on one architecture.

    The default direction was already trained in the main comparison, so only
    the other directions are trained here; the analysis joins the two.
    """
    bundle = load_bundle(cfg)
    section, ab = cfg["gnn"], cfg["ablation"]
    features = ab["features"]
    kind = best_gnn(cfg, features) if ab["model"] == "best" else ab["model"]
    log.info("Direction ablation on %s / %s", kind, features)
    for direction in ab["directions"]:
        if direction == section["direction"]:
            continue  # reuse the main-comparison run
        graphs = build_graph_set(bundle, cfg, features, direction)
        val_ref, test_ref = refs(graphs)
        learner = NeuralLearner(cfg, kind, features, graphs, section, "ablation", direction, direction)
        run_protocol(
            cfg, learner, experiment="ablation", model=kind, features=features, variant=direction,
            base_hp=base_hparams(section), grid=section.get("grid"),
            val_ref=val_ref, test_ref=test_ref,
        )


def main() -> None:
    parser = config_arg_parser("Train GNNs on the Elliptic dataset.")
    parser.add_argument("--ablation", action="store_true", help="run the edge-direction ablation")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_ablation(cfg) if args.ablation else run(cfg)


if __name__ == "__main__":
    main()
