"""Non-graph baselines: Logistic Regression, Random Forest, LightGBM and an MLP.

    python -m btc_aml.train_baselines --config configs/quick.yaml

These models see each transaction as one row of a table. On the LF feature
set they know nothing about the graph. On AF they see the 72 aggregated
columns, which are hand-made summaries of each transaction's neighbours, so
"non-graph model on AF" is really "graph information through feature
engineering". That is the bar the GNNs have to clear.

Class imbalance is handled with class weights in every model.
"""

from __future__ import annotations

import joblib
import numpy as np
from lightgbm import LGBMClassifier, early_stopping
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .data import ILLICIT, UNKNOWN, load_bundle
from .evaluate import run_tag
from .protocol import run_protocol
from .splits import build_graph_set, build_table_set
from .train_gnn import NeuralLearner, base_hparams, refs
from .utils import config_arg_parser, get_logger, load_config, models_dir

log = get_logger()


# --------------------------------------------------------------------------- #
# Model factories: (hyperparameters, seed) -> unfitted scikit-learn style model
# --------------------------------------------------------------------------- #
def make_logreg(hp: dict, seed: int, n_iter: int | None = None):
    """Linear model: score = sigmoid(w . x + b). The simplest possible baseline.

    C is the inverse regularisation strength (smaller C = stronger L2 penalty).
    class_weight="balanced" up-weights the rare illicit class in the loss.
    """
    return LogisticRegression(
        C=hp["C"], class_weight="balanced", max_iter=hp["max_iter"], random_state=seed
    )


def make_rf(hp: dict, seed: int, n_iter: int | None = None):
    """Random Forest: many decision trees, each grown on a bootstrap sample of
    the rows and considering a random subset of features at each split; the
    forest averages their votes.

    "balanced_subsample" recomputes the class weights inside each tree's
    bootstrap sample.
    """
    return RandomForestClassifier(
        n_estimators=hp["n_estimators"],
        max_features=hp["max_features"],
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )


def make_lgbm(hp: dict, seed: int, n_iter: int | None = None):
    """LightGBM: gradient boosting. Trees are added one after another, each
    one fitted to the errors the previous trees still make.

    n_estimators is the maximum number of trees while selecting (early stopping
    on validation decides the real number) and exactly `n_iter` for the final fit.
    """
    return LGBMClassifier(
        n_estimators=n_iter or hp["max_rounds"],
        learning_rate=hp["learning_rate"],
        num_leaves=hp["num_leaves"],
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,  # all cores (capped to 1 on macOS, see btc_aml/__init__.py)
        verbose=-1,
    )


FACTORIES = {"logreg": make_logreg, "rf": make_rf, "lgbm": make_lgbm}


class TableLearner:
    """Plugs a scikit-learn style model into the shared protocol.

    `tables` holds (X, y, time_step, node_idx) for train / val / trainval / test
    (see splits.build_table_set). The hybrid experiment reuses this class with
    wider tables that also contain GNN embeddings.
    """

    def __init__(self, cfg: dict, name: str, tables: dict, save_tag=None):
        self.cfg = cfg
        self.name = name
        self.factory = FACTORIES[name]
        self.tables = tables
        self.save_tag = save_tag  # function seed -> file tag; None = do not save models

    def fit_select(self, hp: dict, seed: int):
        X_train, y_train = self.tables["train"][:2]
        X_val, y_val = self.tables["val"][:2]
        model = self.factory(hp, seed)
        info = {}
        if self.name == "lgbm":
            # Early stopping: keep adding trees while validation PR-AUC improves.
            known = y_val != UNKNOWN
            model.fit(
                X_train, y_train,
                eval_set=[(X_val[known], y_val[known])],
                eval_metric="average_precision",
                callbacks=[early_stopping(hp["early_stopping_rounds"], verbose=False)],
            )
            info["n_iter"] = int(model.best_iteration_)
        else:
            model.fit(X_train, y_train)
        return model.predict_proba(X_val)[:, ILLICIT], info

    def fit_final(self, hp: dict, seed: int, info: dict):
        X, y = self.tables["trainval"][:2]
        model = self.factory(hp, seed, n_iter=info.get("n_iter"))
        model.fit(X, y)
        # Keep one copy of each final model (first seed) for the app and SHAP.
        if self.save_tag is not None and seed == self.cfg["seeds"][0]:
            joblib.dump(
                {"model": model, "threshold": info["threshold"], "hp": hp},
                models_dir(self.cfg) / f"{self.save_tag(seed)}.joblib",
                compress=3,
            )
        return model.predict_proba(self.tables["test"][0])[:, ILLICIT]


def table_refs(tables: dict):
    """(y, time_step) for val and (y, time_step, node_idx) for test."""
    _, y_val, t_val, _ = tables["val"]
    _, y_test, t_test, idx_test = tables["test"]
    return (y_val, t_val), (y_test, t_test, idx_test)


def run(cfg: dict) -> None:
    bundle = load_bundle(cfg)
    section = cfg["baselines"]
    for features in cfg["feature_sets"]:
        tables = build_table_set(bundle, cfg, features)
        val_ref, test_ref = table_refs(tables)

        for name in ("logreg", "rf", "lgbm"):
            log.info("Baseline %s on %s", name, features)
            settings = dict(section[name])
            grid = settings.pop("grid", None)
            learner = TableLearner(
                cfg, name, tables,
                save_tag=lambda seed, n=name, f=features: run_tag("main", n, f, "-", seed),
            )
            run_protocol(
                cfg, learner, experiment="main", model=name, features=features,
                base_hp=settings, grid=grid, val_ref=val_ref, test_ref=test_ref,
            )

        # The MLP goes through the same training loop as the GNNs. It receives
        # graph objects only because that loop expects them; it ignores edges.
        log.info("Baseline mlp on %s", features)
        graphs = build_graph_set(bundle, cfg, features)
        mlp_val_ref, mlp_test_ref = refs(graphs)
        learner = NeuralLearner(cfg, "mlp", features, graphs, section["mlp"], "main")
        run_protocol(
            cfg, learner, experiment="main", model="mlp", features=features,
            base_hp=base_hparams(section["mlp"]), grid=section["mlp"].get("grid"),
            val_ref=mlp_val_ref, test_ref=mlp_test_ref,
        )


def main() -> None:
    args = config_arg_parser("Train the non-graph baselines.").parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
