"""The evaluation protocol, written once and used by every model.

Each model family (sklearn models, neural nets, hybrids) implements a small
"learner" with two methods. This module drives them through the same steps,
so no model can accidentally get a look at the test set that others do not.

    PHASE 1 - selection (uses train = steps 1-29 and val = steps 30-34)
        a. hyperparameters: for every candidate in the config grid, fit on
           train with the first seed and keep the candidate with the best
           validation PR-AUC.
        b. for each seed: fit on train; early-stopped models report how many
           iterations/epochs were best on val; choose the decision threshold
           that maximises illicit F1 on val.

    PHASE 2 - final (uses trainval = steps 1-34 and test = steps 35-49)
        c. for each seed: refit on trainval with the chosen hyperparameters
           and the chosen number of iterations (no validation data is left to
           early-stop on, so the count from phase 1 is reused).
        d. score the test set ONCE with the threshold from step b.

The test labels are touched only inside `evaluate_scores` in step d.
"""

from __future__ import annotations

import itertools
import json
import time
from typing import Protocol

import numpy as np
from sklearn.metrics import average_precision_score

from .data import UNKNOWN
from .evaluate import (
    best_f1_threshold, evaluate_scores, record_result, record_search, run_tag, save_predictions,
)
from .utils import get_logger

log = get_logger()


class Learner(Protocol):
    """What a model family must provide to take part in the protocol."""

    def fit_select(self, hp: dict, seed: int) -> tuple[np.ndarray, dict]:
        """Fit on the train steps. Return (scores for every val row, info).

        `info` may contain "n_iter": the early-stopped number of boosting
        rounds / epochs, which fit_final will reuse.
        """

    def fit_final(self, hp: dict, seed: int, info: dict) -> np.ndarray:
        """Fit on the trainval steps. Return scores for every test row."""


def expand_grid(grid: dict | None) -> list[dict]:
    """{"lr": [0.01, 0.003], "dropout": [0.3]} -> [{"lr": 0.01, "dropout": 0.3}, ...]."""
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def _val_pr_auc(y_val: np.ndarray, scores: np.ndarray) -> float:
    known = y_val != UNKNOWN
    return float(average_precision_score(y_val[known], scores[known]))


def run_protocol(
    cfg: dict,
    learner: Learner,
    *,
    experiment: str,
    model: str,
    features: str,
    base_hp: dict,
    grid: dict | None,
    val_ref: tuple[np.ndarray, np.ndarray],
    test_ref: tuple[np.ndarray, np.ndarray, np.ndarray],
    variant: str = "-",
) -> dict:
    """Run phases 1 and 2 for one model on one feature set and record the results.

    val_ref  = (y, time_step) for the validation rows, aligned with fit_select's scores
    test_ref = (y, time_step, node_idx) for the test rows, aligned with fit_final's scores
    Returns the chosen hyperparameters.
    """
    y_val, t_val = val_ref
    y_test, t_test, idx_test = test_ref
    seeds = cfg["seeds"]
    k = cfg["alert_k"]
    name = f"{model}/{features}" + ("" if variant == "-" else f"/{variant}")

    # ---- step a: hyperparameter search on validation (first seed only) ----
    candidates = expand_grid(grid)
    best_hp = {**base_hp, **candidates[0]}
    if len(candidates) > 1:
        best_score = -1.0
        for candidate in candidates:
            hp = {**base_hp, **candidate}
            start = time.perf_counter()
            scores, info = learner.fit_select(hp, seeds[0])
            score = _val_pr_auc(y_val, scores)
            log.info("  [%s] search %s -> val PR-AUC %.4f", name, candidate, score)
            # Keep every candidate's score, so the choice can be audited later.
            record_search(cfg, {
                "experiment": experiment, "model": model, "features": features, "variant": variant,
                "candidate": json.dumps(candidate, sort_keys=True), "seed": seeds[0],
                "val_pr_auc": score, "n_iter": info.get("n_iter"),
                "seconds": time.perf_counter() - start,
            })
            if score > best_score:
                best_score, best_hp = score, hp
        log.info("  [%s] chosen hyperparameters: %s", name, best_hp)

    for seed in seeds:
        # ---- step b: fit on train, pick iterations + threshold on validation ----
        start = time.perf_counter()
        val_scores, info = learner.fit_select(best_hp, seed)
        select_seconds = time.perf_counter() - start

        known = y_val != UNKNOWN
        threshold = best_f1_threshold(y_val[known], val_scores[known])
        info = {**info, "threshold": threshold}
        val_metrics, _ = evaluate_scores(y_val, val_scores, t_val, threshold, k)
        extra = {"hparams": best_hp, "n_iter": info.get("n_iter"), "train_seconds": select_seconds}
        record_result(
            cfg, experiment=experiment, model=model, features=features, variant=variant,
            seed=seed, split="val", metrics=val_metrics, extra=extra,
        )

        # ---- steps c + d: refit on trainval, evaluate on test exactly once ----
        start = time.perf_counter()
        test_scores = learner.fit_final(best_hp, seed, info)
        final_seconds = time.perf_counter() - start

        test_metrics, per_step = evaluate_scores(y_test, test_scores, t_test, threshold, k)
        extra = {"hparams": best_hp, "n_iter": info.get("n_iter"), "train_seconds": final_seconds}
        record_result(
            cfg, experiment=experiment, model=model, features=features, variant=variant,
            seed=seed, split="test", metrics=test_metrics, per_step=per_step, extra=extra,
        )
        save_predictions(
            cfg, run_tag(experiment, model, features, variant, seed), idx_test, test_scores, threshold
        )
        log.info(
            "  [%s] seed %d | val F1 %.3f PR-AUC %.3f | test F1 %.3f P %.3f R %.3f PR-AUC %.3f | thr %.3f%s | %.1fs",
            name, seed, val_metrics["f1"], val_metrics["pr_auc"],
            test_metrics["f1"], test_metrics["precision"], test_metrics["recall"],
            test_metrics["pr_auc"], threshold,
            f" | n_iter {info['n_iter']}" if info.get("n_iter") else "",
            select_seconds + final_seconds,
        )
    return best_hp
