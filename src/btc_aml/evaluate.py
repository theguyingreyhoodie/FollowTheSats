"""Metrics, threshold tuning and result files.

Every model in the project produces one number per transaction: a SCORE, the
estimated probability that the transaction is illicit. This module turns
scores into the numbers we report, and it is the only place that writes
results/metrics.csv, so all models are measured in exactly the same way.

Why not accuracy? About 90% of labelled transactions are licit, so a model
that says "licit" for everything is ~90% accurate and catches zero criminals.
We therefore report metrics of the ILLICIT class:

    precision  of the transactions we flagged, what fraction is really illicit
    recall     of the really illicit transactions, what fraction we flagged
    F1         harmonic mean of the two (high only when both are high)
    PR-AUC     area under the precision-recall curve: quality of the ranking
               over all thresholds, sensitive to the rare class
    ROC-AUC    probability an illicit tx is scored above a licit one
    micro-F1   equals accuracy for this task; kept only to compare with the paper
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .data import ILLICIT, UNKNOWN
from .utils import results_dir

# Columns that identify one row of metrics.csv. Re-running an experiment
# replaces its old rows instead of appending duplicates.
KEY_COLUMNS = ["experiment", "model", "features", "variant", "seed", "split"]


# --------------------------------------------------------------------------- #
# Threshold tuning
# --------------------------------------------------------------------------- #
def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Return the score cut-off that maximises illicit-class F1.

    A classifier outputs a probability; "flag it" means probability >= some
    threshold. The default 0.5 is arbitrary, and with a rare positive class
    and class-weighted training it is usually not the best cut-off. We try
    every distinct score as a threshold and keep the best one.

    This function is only ever called with VALIDATION data. Choosing the
    threshold on the test set would be cheating: the reported score would be
    the best of many tries on the very data it is supposed to be judged on.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # precision/recall have one more entry than thresholds (the point
    # recall=0, precision=1 at the end has no threshold), so drop the last.
    precision, recall = precision[:-1], recall[:-1]
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[int(np.argmax(f1))])


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def binary_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """All headline metrics for one model on one set of LABELLED transactions."""
    y_pred = (scores >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "precision": precision_score(y_true, y_pred, pos_label=ILLICIT, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=ILLICIT, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=ILLICIT, zero_division=0),
        "pr_auc": average_precision_score(y_true, scores),
        "roc_auc": roc_auc_score(y_true, scores),
        "micro_f1": f1_score(y_true, y_pred, average="micro"),
        "n_illicit": int((y_true == ILLICIT).sum()),
        "n_labeled": int(len(y_true)),
    }


def precision_recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> tuple[float, float]:
    """The analyst "alert budget" metric.

    A compliance team can only review so many alerts. If they look at the K
    highest-scored transactions of a time step:
        precision@K = fraction of those K that are illicit (is their time well spent?)
        recall@K    = fraction of all illicit transactions that made it into the K
    If fewer than K labelled transactions exist, all of them are "reviewed".
    """
    k = min(k, len(y_true))
    if k == 0:
        return float("nan"), float("nan")
    top = np.argsort(-scores, kind="stable")[:k]
    hits = int((y_true[top] == ILLICIT).sum())
    total_illicit = int((y_true == ILLICIT).sum())
    recall = hits / total_illicit if total_illicit else float("nan")
    return hits / k, recall


def per_time_step_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    time_step: np.ndarray,
    threshold: float,
    k: int,
) -> pd.DataFrame:
    """Illicit precision/recall/F1 and the alert-budget metrics for each time step.

    The overall test score hides WHEN a model fails. Splitting by time step
    shows the collapse after step 43, when a large dark market was shut down
    and the illicit activity the models had learned to recognise disappeared.
    """
    rows = []
    for t in np.unique(time_step):
        in_step = time_step == t
        y_t, s_t = y_true[in_step], scores[in_step]
        pred_t = (s_t >= threshold).astype(int)
        p_at_k, r_at_k = precision_recall_at_k(y_t, s_t, k)
        rows.append(
            {
                "time_step": int(t),
                "precision": precision_score(y_t, pred_t, pos_label=ILLICIT, zero_division=0),
                "recall": recall_score(y_t, pred_t, pos_label=ILLICIT, zero_division=0),
                "f1": f1_score(y_t, pred_t, pos_label=ILLICIT, zero_division=0),
                "precision_at_k": p_at_k,
                "recall_at_k": r_at_k,
                "k": int(k),
                "n_illicit": int((y_t == ILLICIT).sum()),
                "n_labeled": int(in_step.sum()),
                "n_flagged": int(pred_t.sum()),
            }
        )
    return pd.DataFrame(rows)


def evaluate_scores(
    y: np.ndarray, scores: np.ndarray, time_step: np.ndarray, threshold: float, k: int
) -> tuple[dict, pd.DataFrame]:
    """Drop unknown-label nodes, then compute overall and per-time-step metrics.

    `y` may still contain UNKNOWN (-1) entries, e.g. when a GNN scores every
    node of a graph. Unknown nodes have no ground truth, so they cannot count
    as right or wrong and are excluded here.
    """
    known = y != UNKNOWN
    y, scores, time_step = y[known], scores[known], time_step[known]
    return (
        binary_metrics(y, scores, threshold),
        per_time_step_metrics(y, scores, time_step, threshold, k),
    )


# --------------------------------------------------------------------------- #
# Result files
# --------------------------------------------------------------------------- #
def _upsert(path: Path, new_rows: pd.DataFrame, keys: list[str]) -> None:
    """Append rows to a CSV, replacing existing rows that have the same keys."""
    if path.exists():
        old = pd.read_csv(path)
        marker = new_rows[keys].astype(str).agg("|".join, axis=1).unique()
        old = old[~old[keys].astype(str).agg("|".join, axis=1).isin(marker)]
        new_rows = pd.concat([old, new_rows], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_rows.to_csv(path, index=False)


def record_result(
    cfg: dict,
    *,
    experiment: str,
    model: str,
    features: str,
    seed: int,
    split: str,
    metrics: dict,
    per_step: pd.DataFrame | None = None,
    variant: str = "-",
    extra: dict | None = None,
) -> None:
    """Write one evaluation into results/metrics.csv (+ results/per_timestep.csv).

    experiment  which part of the study the row belongs to:
                main | ablation | hybrid | rolling | random_split
    model       logreg, rf, lgbm, mlp, gcn, skip_gcn, sage, gat, rf+emb, ...
    features    LF or AF (hybrid rows use e.g. "AF+emb")
    variant     extra label, e.g. the edge direction for GNNs ("-" when not applicable)
    split       "val" (model trained on steps 1-29) or "test" (final model)
    extra       anything else worth keeping: hyperparameters, timings, best epoch
    """
    ident = {
        "experiment": experiment,
        "model": model,
        "features": features,
        "variant": variant,
        "seed": seed,
        "split": split,
    }
    extra = dict(extra or {})
    if "hparams" in extra:
        extra["hparams"] = json.dumps(extra["hparams"], sort_keys=True)
    row = {**ident, **metrics, **extra, "config": cfg.get("run_name", "")}
    _upsert(results_dir(cfg) / "metrics.csv", pd.DataFrame([row]), KEY_COLUMNS)

    if per_step is not None:
        per_step = per_step.copy()
        for key, value in reversed(ident.items()):
            per_step.insert(0, key, value)
        _upsert(results_dir(cfg) / "per_timestep.csv", per_step, KEY_COLUMNS)


def record_search(cfg: dict, row: dict) -> None:
    """Log one hyperparameter candidate and its validation score to results/hparam_search.csv."""
    keys = ["experiment", "model", "features", "variant", "candidate", "seed"]
    _upsert(results_dir(cfg) / "hparam_search.csv", pd.DataFrame([row]), keys)


def save_predictions(
    cfg: dict,
    name: str,
    node_idx: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> None:
    """Save per-node test scores so plots and the demo app never need to retrain.

    Stored as results/predictions/<name>.npz with the global node indices, the
    illicit probability of each node, and the decision threshold.
    """
    folder = results_dir(cfg) / "predictions"
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        folder / f"{name}.npz",
        node_idx=node_idx.astype(np.int64),
        score=scores.astype(np.float32),
        threshold=np.float32(threshold),
    )


def load_metrics(cfg: dict) -> pd.DataFrame:
    """Read results/metrics.csv."""
    return pd.read_csv(results_dir(cfg) / "metrics.csv")


def run_tag(experiment: str, model: str, features: str, variant: str, seed: int) -> str:
    """File-name-safe identifier of one run, e.g. main__gcn__AF__undirected__seed0."""
    parts = [experiment, model, features, variant, f"seed{seed}"]
    return "__".join(p.replace("+", "_") for p in parts)


def chosen_settings(cfg: dict, model: str, features: str, seed: int,
                    experiment: str = "main", variant: str = "-") -> tuple[dict, int | None, float]:
    """Look up what the protocol chose for a run: (hyperparameters, n_iter, threshold).

    Later experiments (rolling retraining) reuse these instead of tuning
    again, so they stay comparable with the main results and never tune on
    test data.
    """
    df = load_metrics(cfg)
    row = df[
        (df["experiment"] == experiment) & (df["model"] == model) & (df["features"] == features)
        & (df["variant"] == variant) & (df["seed"] == seed) & (df["split"] == "test")
    ]
    if row.empty:
        raise RuntimeError(f"no main result for {model}/{features}/seed {seed}; train it first")
    row = row.iloc[0]
    n_iter = None if pd.isna(row.get("n_iter")) else int(row["n_iter"])
    return json.loads(row["hparams"]), n_iter, float(row["threshold"])
