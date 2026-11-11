"""Metric code: checked on tiny hand-computable examples."""

import numpy as np

from btc_aml import evaluate


def test_best_threshold_separable_case():
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    threshold = evaluate.best_f1_threshold(y, scores)
    assert 0.3 < threshold <= 0.8
    assert evaluate.binary_metrics(y, scores, threshold)["f1"] == 1.0


def test_predicting_all_licit_scores_zero_f1_but_high_accuracy():
    # 1 illicit in 10: "everything is licit" is 90% accurate and useless.
    y = np.array([1] + [0] * 9)
    scores = np.zeros(10)
    m = evaluate.binary_metrics(y, scores, threshold=0.5)
    assert m["micro_f1"] == 0.9
    assert m["f1"] == 0.0 and m["recall"] == 0.0


def test_unknown_labels_are_excluded():
    y = np.array([1, 0, -1, -1, 0, 1])
    scores = np.array([0.9, 0.1, 0.99, 0.01, 0.2, 0.8])
    t = np.array([35, 35, 35, 36, 36, 36])
    metrics, per_step = evaluate.evaluate_scores(y, scores, t, threshold=0.5, k=50)
    assert metrics["n_labeled"] == 4 and metrics["n_illicit"] == 2
    assert metrics["f1"] == 1.0
    assert per_step["n_labeled"].tolist() == [2, 2]


def test_precision_recall_at_k():
    y = np.array([1, 0, 1, 0, 0, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    p, r = evaluate.precision_recall_at_k(y, scores, k=3)
    assert p == 2 / 3      # 2 of the top 3 are illicit
    assert r == 2 / 3      # 2 of the 3 illicit ones are in the top 3
    p, r = evaluate.precision_recall_at_k(y, scores, k=50)  # budget larger than the data
    assert p == 3 / 6 and r == 1.0


def test_per_time_step_metrics():
    y = np.array([1, 0, 0, 1, 0, 0])
    scores = np.array([0.9, 0.1, 0.2, 0.3, 0.8, 0.1])
    t = np.array([40, 40, 40, 44, 44, 44])
    df = evaluate.per_time_step_metrics(y, scores, t, threshold=0.5, k=1)
    assert df["time_step"].tolist() == [40, 44]
    assert df["f1"].tolist() == [1.0, 0.0]          # step 44: missed one, flagged a licit one
    assert df["precision_at_k"].tolist() == [1.0, 0.0]
