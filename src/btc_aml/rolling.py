"""Stretch goal: rolling retraining.

    python -m btc_aml.rolling --config configs/quick.yaml

The main protocol trains once on steps 1-34 and then predicts fifteen time
steps (about seven months) into the future without ever updating. A real
system would retrain as new labels arrive. This experiment simulates that:

    for each test step t = 35..49:
        train on all labelled transactions from steps 1..t-1
        predict step t

Question: does keeping the model up to date repair the collapse after the
dark-market shutdown at t = 43?

To stay comparable with the main results (and to avoid tuning on test data),
the hyperparameters, the number of boosting rounds and the decision threshold
are the ones the main protocol chose on validation. Only the training window
changes. Note the optimistic assumption: labels for step t-1 are available
before step t is scored, which real investigations rarely deliver that fast.
"""

from __future__ import annotations

import time

import numpy as np

from .data import ILLICIT, load_bundle
from .evaluate import chosen_settings, evaluate_scores, record_result, run_tag, save_predictions
from .splits import fit_scaler, step_mask, tabular_arrays
from .train_baselines import FACTORIES
from .utils import config_arg_parser, get_logger, load_config

log = get_logger()


def run(cfg: dict) -> None:
    section = cfg.get("rolling", {})
    if not section.get("enabled", False):
        log.info("rolling retraining disabled in config")
        return
    bundle = load_bundle(cfg)
    t_all = bundle["time_step"]
    first_test, last_test = cfg["split"]["test"]
    features = section["features"]
    seeds = cfg["seeds"][: section.get("num_seeds", len(cfg["seeds"]))]

    for name in section["models"]:
        for seed in seeds:
            hp, n_iter, threshold = chosen_settings(cfg, name, features, seed)
            start = time.perf_counter()
            parts = []
            for t in range(first_test, last_test + 1):
                past = step_mask(t_all, 1, t - 1)          # everything before step t
                now = step_mask(t_all, t, t)               # step t itself
                scaler = fit_scaler(bundle["x"], past)     # statistics from the past only
                X_past, y_past, _, _ = tabular_arrays(bundle, past, features, scaler)
                X_now, y_now, t_now, idx_now = tabular_arrays(
                    bundle, now, features, scaler, labeled_only=False
                )
                model = FACTORIES[name](hp, seed, n_iter=n_iter)
                model.fit(X_past, y_past)
                parts.append((y_now, model.predict_proba(X_now)[:, ILLICIT], t_now, idx_now))

            y, scores, steps, idx = (np.concatenate(p) for p in zip(*parts))
            metrics, per_step = evaluate_scores(y, scores, steps, threshold, cfg["alert_k"])
            record_result(
                cfg, experiment="rolling", model=name, features=features, seed=seed, split="test",
                metrics=metrics, per_step=per_step,
                extra={"hparams": hp, "n_iter": n_iter, "train_seconds": time.perf_counter() - start},
            )
            save_predictions(cfg, run_tag("rolling", name, features, "-", seed), idx, scores, threshold)
            log.info("  [rolling %s/%s] seed %d | test F1 %.3f P %.3f R %.3f | %.1fs",
                     name, features, seed, metrics["f1"], metrics["precision"], metrics["recall"],
                     time.perf_counter() - start)


def main() -> None:
    args = config_arg_parser("Rolling retraining over the test time steps.").parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
