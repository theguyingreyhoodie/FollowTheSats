"""btc_aml: illicit Bitcoin transaction detection on the Elliptic dataset.

Module map (read them in this order to follow the pipeline):

    utils            config loading, seeding, logging, device choice, timers
    data             download + validate the CSVs, build the cached graph tensors
    splits           temporal train/val/test split, per-split PyG graphs, scaling
    evaluate         metrics, threshold tuning, per-time-step + alert-budget metrics
    models           the MLP and the four GNNs (GCN, Skip-GCN, GraphSAGE, GAT)
    train_baselines  Logistic Regression, Random Forest, LightGBM, MLP
    train_gnn        GNN training (also runs the edge-direction ablation)
    hybrid           GNN embeddings + features -> tree models
    rolling          stretch goal: retrain before every test time step
    eda / analysis   figures and tables
    explain          SHAP for trees, GNNExplainer for GNN predictions
"""

import sys

# --------------------------------------------------------------------------- #
# macOS workaround: PyTorch + LightGBM in one process
# --------------------------------------------------------------------------- #
# Both libraries use OpenMP for their worker threads, and on macOS each loads
# its OWN copy of the OpenMP runtime (PyTorch ships one inside its wheel,
# LightGBM uses Homebrew's libomp). Two runtimes in one process do not
# cooperate: as soon as LightGBM starts a multi-threaded section (training,
# predicting, even un-pickling a model) the process dies with a segmentation
# fault or hangs.
#
# The fix has two parts, both applied here because Python runs this file
# before any other module of the package:
#   1. import torch first, so its runtime is initialised first;
#   2. cap LightGBM at one thread on macOS, so it never starts a second
#      thread pool. Linux and Windows wheels do not have the problem and keep
#      using all cores.
import torch  # noqa: E402,F401

if sys.platform == "darwin":
    try:
        import ctypes

        import lightgbm.basic

        lightgbm.basic._LIB.LGBM_SetMaxThreads(ctypes.c_int(1))
    except Exception:  # noqa: BLE001 - LightGBM missing or too old: nothing to cap
        pass

__version__ = "0.1.0"
