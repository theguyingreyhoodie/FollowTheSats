"""Explainability: why did a model flag a transaction?

    python -m btc_aml.explain --config configs/quick.yaml

Two families of explanation:

1. Tree models (global view: which features matter overall)
     * SHAP values for LightGBM. For one prediction, SHAP splits the score
       into one contribution per feature. Averaging the absolute
       contributions over many transactions ranks the features.
     * Permutation importance for Random Forest: shuffle one column and
       measure how much the score drops. A big drop means the model relied
       on that column.

2. GNN (local view: one prediction at a time)
     * GNNExplainer learns a soft mask over the EDGES and over the FEATURES
       so that the masked input still reproduces the model's prediction with
       as few edges/features as possible. Edges with a high mask value are
       the ones the prediction depended on. We draw them on the
       transaction's 2-hop neighbourhood (a 2-layer GNN cannot see further).

A LIMIT THAT NO METHOD HERE CAN REMOVE: the features are anonymised. We can
say "local_53 pushed this score up" but not what local_53 measures (a fee? an
output count?). The explanations are useful for debugging the model and for
pointing an analyst at the right neighbouring transactions, not for telling a
human-readable story about why a transaction is suspicious.
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import subgraph

from .data import FEATURE_NAMES, FEATURE_SETS, ILLICIT, LICIT, NUM_LOCAL, UNKNOWN, load_bundle
from .evaluate import run_tag
from .neural import predict_scores
from .plotting import (
    CLASS_COLORS, INK, INK_MUTED, INK_SECONDARY, MODEL_LABELS, SERIES_COLORS, SURFACE,
    apply_style, save,
)
from .splits import build_graph_set, build_table_set
from .train_gnn import best_gnn, checkpoint_path, load_checkpoint
from .utils import (
    config_arg_parser, figures_dir, get_logger, load_config, models_dir, results_dir, save_json,
)

log = get_logger()

GROUP_COLORS = {"local": SERIES_COLORS[0], "aggregated": SERIES_COLORS[1]}


def feature_group(name: str) -> str:
    return "local" if name.startswith("local") else "aggregated"


# --------------------------------------------------------------------------- #
# 1. Tree models
# --------------------------------------------------------------------------- #
def illicit_shap_values(explainer, X: np.ndarray) -> np.ndarray:
    """SHAP values towards the ILLICIT class, shape [rows, features].

    Units are log-odds: for each row, base value + sum of its SHAP values =
    the model's raw score, and sigmoid(raw score) = P(illicit). Depending on
    the shap version, a binary classifier returns one array, a list with one
    array per class, or a 3-D array; all are reduced to the illicit class.
    """
    values = explainer.shap_values(X)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    return values


def tree_importance(cfg: dict, bundle: dict) -> pd.DataFrame | None:
    """SHAP (LightGBM) and permutation importance (Random Forest) on a test sample.

    The sample comes from the TEST period on purpose: we want to know what the
    final models rely on when scoring new data. This is interpretation after
    the fact; nothing is tuned with it.
    """
    import shap
    from sklearn.inspection import permutation_importance

    seed = cfg["seeds"][0]
    lgbm_path = models_dir(cfg) / f"{run_tag('main', 'lgbm', 'AF', '-', seed)}.joblib"
    rf_path = models_dir(cfg) / f"{run_tag('main', 'rf', 'AF', '-', seed)}.joblib"
    if not lgbm_path.exists() or not rf_path.exists():
        log.info("tree models not found; run btc_aml.train_baselines first")
        return None

    X, y, _, _ = build_table_set(bundle, cfg, "AF")["test"]
    labeled = np.flatnonzero(y != UNKNOWN)
    rng = np.random.default_rng(seed)
    sample = rng.choice(labeled, size=min(cfg["explain"]["shap_sample"], len(labeled)), replace=False)
    X_s, y_s = X[sample], y[sample]
    names = FEATURE_NAMES[FEATURE_SETS["AF"]]

    # ---- SHAP for LightGBM ----
    lgbm = joblib.load(lgbm_path)["model"]
    values = illicit_shap_values(shap.TreeExplainer(lgbm), X_s)
    mean_abs_shap = np.abs(values).mean(axis=0)

    # ---- permutation importance for Random Forest ----
    rf = joblib.load(rf_path)["model"]
    perm = permutation_importance(
        rf, X_s, y_s, scoring="average_precision", n_repeats=5, random_state=seed, n_jobs=1
    )

    table = pd.DataFrame(
        {
            "feature": names,
            "group": [feature_group(n) for n in names],
            "lgbm_mean_abs_shap": mean_abs_shap,
            "rf_permutation_importance": perm.importances_mean,
            "rf_permutation_std": perm.importances_std,
        }
    )
    table.to_csv(results_dir(cfg) / "feature_importance.csv", index=False)
    return table


def plot_tree_importance(cfg: dict, table: pd.DataFrame, top: int = 15):
    """Top features by SHAP (LightGBM) and by permutation importance (Random Forest)."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    panels = (
        ("lgbm_mean_abs_shap", "LightGBM: mean |SHAP value|", "average push on the score (log-odds)"),
        ("rf_permutation_importance", "Random Forest: permutation importance",
         "drop in PR-AUC when the column is shuffled"),
    )
    for ax, (column, title, xlabel) in zip(axes, panels):
        best = table.nlargest(top, column).iloc[::-1]
        ax.barh(best["feature"], best[column], height=0.62,
                color=[GROUP_COLORS[g] for g in best["group"]])
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUP_COLORS.values()]
    axes[1].legend(handles, [f"{g} feature" for g in GROUP_COLORS], loc="lower right")
    n = cfg["explain"]["shap_sample"]
    fig.suptitle(f"What the tree models rely on (AF, sample of up to {n} labelled test transactions)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return save(fig, figures_dir(cfg) / "tree_feature_importance.png")


# --------------------------------------------------------------------------- #
# 2. GNN explanations
# --------------------------------------------------------------------------- #
@dataclass
class NodeExplanation:
    """Everything needed to describe and draw one explained transaction."""

    node: int                    # global node index of the explained transaction
    time_step: int
    score: float                 # the GNN's illicit probability for it
    nodes: np.ndarray            # global indices of the neighbourhood nodes drawn
    hop: np.ndarray              # distance of each of those nodes from `node` (0, 1 or 2)
    scores: np.ndarray           # GNN illicit probability of each neighbourhood node
    edges: np.ndarray            # [2, E] directed edges (global indices) inside the neighbourhood
    edge_importance: np.ndarray  # [E] GNNExplainer edge mask value in [0, 1] (not rescaled)
    top_features: list           # [(feature name, mask value), ...] highest first
    truncated: bool              # True if the neighbourhood was larger than the cap
    # Faithfulness checks: re-run the model on a modified graph and see what happens.
    score_no_edges: float        # score from the transaction's own features (ALL edges removed)
    score_kept_edges: float      # score when only the edges with mask > 0.5 are kept
    edges_in_scope: int          # directed edges that can influence the prediction (2 hops)
    edges_kept: int              # how many of those have mask > 0.5


def step_graph(graph: Data, time_step: int) -> Data:
    """The part of `graph` that belongs to one time step.

    Each time step is its own connected component, so this subgraph contains
    everything that can influence a prediction in that step. Explaining on it
    (rather than on the whole 60k-node split) is exact and much faster.
    """
    keep = graph.time_step == time_step
    edge_index, _ = subgraph(keep, graph.edge_index, relabel_nodes=True, num_nodes=graph.num_nodes)
    return Data(x=graph.x[keep], edge_index=edge_index, y=graph.y[keep], node_idx=graph.node_idx[keep])


def explain_node(model, graph: Data, bundle: dict, node: int, feature_names: list[str],
                 epochs: int = 100, hops: int = 2, cap: int = 200, seed: int = 0) -> NodeExplanation:
    """Run GNNExplainer for one transaction and collect its neighbourhood.

    graph  the split graph the transaction lives in (e.g. the test graph)
    node   GLOBAL node index of the transaction
    cap    maximum number of neighbourhood nodes to keep for drawing
    """
    model = model.cpu().eval()
    t = int(bundle["time_step"][node])
    sg = step_graph(graph, t)
    global_ids = sg.node_idx.numpy()
    target = int(np.searchsorted(global_ids, node))
    assert global_ids[target] == node, "transaction is not part of this graph"

    explainer = Explainer(
        model=model,
        # GNNExplainer optimises the masks by gradient descent for `epochs` steps.
        # A higher learning rate than PyG's default (0.01) lets the masks move
        # clearly away from their random start (about 0.5) within 200 steps.
        algorithm=GNNExplainer(epochs=epochs, lr=0.05),
        explanation_type="model",              # explain what the model predicted
        edge_mask_type="object",               # one importance value per edge
        node_mask_type="common_attributes",    # one importance value per feature
        model_config=dict(mode="multiclass_classification", task_level="node", return_type="raw"),
    )
    torch.manual_seed(seed)  # the masks start from random values
    explanation = explainer(sg.x, sg.edge_index, index=target)
    edge_mask = explanation.edge_mask.detach().numpy()
    feature_mask = explanation.node_mask.detach().numpy().ravel()
    scores = predict_scores(model, sg, torch.device("cpu"))

    # ---- faithfulness checks -------------------------------------------------
    # An explanation is only worth showing if it is true to the model. Two
    # direct tests, each one extra forward pass:
    #   * remove ALL edges: what is left is the score from the node's own
    #     features. If it barely changes, no edge matters, whatever a mask says.
    #   * keep ONLY the edges GNNExplainer marked important (mask > 0.5): the
    #     score should stay close to the original one.
    def score_with(edge_index: torch.Tensor) -> float:
        graph = Data(x=sg.x, edge_index=edge_index)
        return float(predict_scores(model, graph, torch.device("cpu"))[target])

    kept = torch.from_numpy(edge_mask > 0.5)
    score_no_edges = score_with(sg.edge_index[:, :0])
    score_kept_edges = score_with(sg.edge_index[:, kept])

    # ---- neighbourhood: breadth-first search, ignoring edge direction ----
    src, dst = sg.edge_index.numpy()
    neighbours: dict[int, set] = {}
    for u, v in zip(src, dst):
        neighbours.setdefault(int(u), set()).add(int(v))
        neighbours.setdefault(int(v), set()).add(int(u))
    hop_of = {target: 0}
    frontier = [target]
    for depth in range(1, hops + 1):
        nxt = []
        for u in frontier:
            for v in neighbours.get(u, ()):
                if v not in hop_of:
                    hop_of[v] = depth
                    nxt.append(v)
        frontier = nxt

    # Importance of an undirected pair = the larger of its two directed mask values.
    pair_importance: dict[tuple[int, int], float] = {}
    for (u, v), m in zip(zip(src.tolist(), dst.tolist()), edge_mask.tolist()):
        key = (min(u, v), max(u, v))
        pair_importance[key] = max(pair_importance.get(key, 0.0), m)

    # ---- cap the neighbourhood: nearest hops first, then most important edges ----
    truncated = len(hop_of) > cap
    if truncated:
        best_edge = {u: 0.0 for u in hop_of}
        for (u, v), m in pair_importance.items():
            if u in best_edge and v in best_edge:
                best_edge[u] = max(best_edge[u], m)
                best_edge[v] = max(best_edge[v], m)
        ranked = sorted(hop_of, key=lambda u: (hop_of[u], -best_edge[u]))[:cap]
        hop_of = {u: hop_of[u] for u in ranked}
    local_nodes = np.array(sorted(hop_of))

    # ---- edges to draw: the ORIGINAL directed edges among the kept nodes ----
    kept_global = global_ids[local_nodes]
    e_src, e_dst = bundle["edge_index"].numpy()
    inside = np.isin(e_src, kept_global) & np.isin(e_dst, kept_global)
    edges = np.stack([e_src[inside], e_dst[inside]])
    to_local = {int(g): int(l) for g, l in zip(global_ids[local_nodes], local_nodes)}
    importance = np.array(
        [pair_importance.get((min(to_local[int(a)], to_local[int(b)]),
                              max(to_local[int(a)], to_local[int(b)])), 0.0)
         for a, b in zip(edges[0], edges[1])]
    )

    order = np.argsort(-feature_mask)[:10]
    return NodeExplanation(
        node=int(node),
        time_step=t,
        score=float(scores[target]),
        nodes=kept_global,
        hop=np.array([hop_of[int(l)] for l in local_nodes]),
        scores=scores[local_nodes],
        edges=edges,
        edge_importance=importance,
        top_features=[(feature_names[i], float(feature_mask[i])) for i in order],
        truncated=truncated,
        score_no_edges=score_no_edges,
        score_kept_edges=score_kept_edges,
        edges_in_scope=int((edge_mask > 0).sum()),
        edges_kept=int(kept.sum()),
    )


def draw_explanation(ax, exp: NodeExplanation, bundle: dict, threshold: float,
                     scores: np.ndarray | None = None, legend: bool = True) -> None:
    """Draw the neighbourhood: fill = true label, ring = flagged, edge weight = importance.

    scores  optional illicit probabilities for `exp.nodes` from another model
            (the app uses this to show any model's flags); default: the GNN's.
    """
    import networkx as nx

    scores = exp.scores if scores is None else scores
    y = bundle["y"].numpy()[exp.nodes]
    g = nx.DiGraph()
    g.add_nodes_from(exp.nodes.tolist())
    g.add_edges_from(zip(exp.edges[0].tolist(), exp.edges[1].tolist()))
    # Kamada-Kawai places nodes so that drawn distance follows graph distance,
    # which keeps the 1-hop ring visibly closer to the target than the 2-hop ring.
    pos = nx.kamada_kawai_layout(g.to_undirected()) if g.number_of_edges() else nx.spring_layout(g, seed=0)

    # Edges: importance drives both width and darkness, on a continuous scale
    # (pale thin line = the prediction did not depend on it; dark thick = it did).
    # The most important edges are drawn last so they end up on top.
    pale, dark = np.array([0.76, 0.76, 0.72]), np.array([0.04, 0.04, 0.04])
    pairs = list(zip(exp.edges[0].tolist(), exp.edges[1].tolist()))
    for (u, v), m in sorted(zip(pairs, exp.edge_importance.tolist()), key=lambda e: e[1]):
        nx.draw_networkx_edges(
            g, pos, edgelist=[(u, v)], ax=ax, width=0.5 + 2.2 * m ** 2,
            edge_color=[tuple(pale + (dark - pale) * m ** 2)], alpha=0.9,
            arrows=True, arrowsize=6 + 4 * m, node_size=120,
        )

    names = {ILLICIT: "illicit", LICIT: "licit", UNKNOWN: "unknown"}
    flagged = scores >= threshold
    for i, node in enumerate(exp.nodes.tolist()):
        is_target = node == exp.node
        ax.scatter(
            *pos[node], s=330 if is_target else 110, color=CLASS_COLORS[names[int(y[i])]],
            edgecolor=INK if flagged[i] else SURFACE, linewidth=2.2 if flagged[i] else 1.0,
            marker="*" if is_target else "o", zorder=4 if is_target else 3,
        )
    ax.set_axis_off()
    if not legend:  # the app prints its own legend under the figure
        return
    handles = [
        Line2D([], [], marker="o", linestyle="", color=CLASS_COLORS[n], markeredgecolor=SURFACE,
               markersize=8, label=f"true label: {n}") for n in ("illicit", "licit", "unknown")
    ] + [
        Line2D([], [], marker="o", linestyle="", color=SURFACE, markeredgecolor=INK,
               markeredgewidth=2, markersize=8, label="dark ring: flagged by the model"),
        Line2D([], [], marker="*", linestyle="", color=INK_MUTED, markeredgecolor=SURFACE,
               markersize=13, label="star: the inspected transaction"),
        Line2D([], [], color=INK, linewidth=3, label="thick dark edge: important"),
        Line2D([], [], color="#c3c2b7", linewidth=1, label="thin pale edge: unimportant"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)


def gnn_examples(cfg: dict, bundle: dict) -> list[dict]:
    """Explain a few correctly-flagged illicit test transactions and save the drawings."""
    seed = cfg["seeds"][0]
    direction = cfg["gnn"]["direction"]
    kind = best_gnn(cfg, "AF")
    path = checkpoint_path(cfg, "main", kind, "AF", direction, seed, "final")
    if not path.exists():
        log.info("GNN checkpoint not found; run btc_aml.train_gnn first")
        return []
    model, ckpt = load_checkpoint(path)
    graph = build_graph_set(bundle, cfg, "AF", direction)["test"]
    scores = predict_scores(model, graph, torch.device("cpu"))
    threshold = float(ckpt["threshold"])
    names = FEATURE_NAMES[FEATURE_SETS["AF"]]

    # Which examples? Correctly flagged illicit transactions with at least three
    # neighbours, preferring those whose score DROPS most when all edges are
    # removed: only there does the neighbourhood matter, so only there is there
    # something for an edge explanation to show. One example per time step.
    y, t = graph.y.numpy(), graph.time_step.numpy()
    no_edges = Data(x=graph.x, edge_index=graph.edge_index[:, :0])
    scores_no_edges = predict_scores(model, no_edges, torch.device("cpu"))
    degree = np.bincount(graph.edge_index[0].numpy(), minlength=graph.num_nodes)
    candidates = np.flatnonzero((y == ILLICIT) & (scores >= threshold) & (degree >= 3))
    drop = scores[candidates] - scores_no_edges[candidates]
    chosen, used_steps = [], set()
    for i in candidates[np.argsort(-drop)]:
        if int(t[i]) not in used_steps:
            chosen.append(int(i))
            used_steps.add(int(t[i]))
        if len(chosen) == cfg["explain"]["gnn_examples"]:
            break

    apply_style()
    records = []
    for rank, local in enumerate(chosen, start=1):
        node = int(graph.node_idx[local])
        exp = explain_node(model, graph, bundle, node, names, epochs=cfg["explain"]["explainer_epochs"],
                           seed=seed)
        tx_id = int(bundle["tx_id"][node])
        fig, ax = plt.subplots(figsize=(9.5, 6.2))
        draw_explanation(ax, exp, bundle, threshold)
        shown = f"{len(exp.nodes)} nodes" + (" (capped)" if exp.truncated else "")
        fig.suptitle(f"GNNExplainer: transaction {tx_id} (time step {exp.time_step}, true label illicit)",
                     x=0.01, ha="left", fontsize=12, fontweight="bold")
        fig.text(
            0.01, 0.905,
            f"{MODEL_LABELS[kind]} score {exp.score:.3f} (threshold {threshold:.3f})  |  with only the "
            f"{exp.edges_kept} of {exp.edges_in_scope} edges marked important: {exp.score_kept_edges:.3f}  |  "
            f"with all edges removed: {exp.score_no_edges:.3f}  |  2-hop neighbourhood, {shown}",
            fontsize=8.5, color=INK_SECONDARY, ha="left")
        out = save(fig, figures_dir(cfg) / f"gnn_explanation_{rank}.png")
        log.info("wrote %s", out)
        strongest = np.argsort(-exp.edge_importance)[:5]
        records.append({
            "figure": out.name, "tx_id": tx_id, "time_step": exp.time_step, "model": kind,
            "score": exp.score, "threshold": threshold,
            "score_all_edges_removed": exp.score_no_edges,
            "score_only_important_edges": exp.score_kept_edges,
            "edges_in_scope": exp.edges_in_scope, "edges_marked_important": exp.edges_kept,
            "neighbourhood_nodes": int(len(exp.nodes)), "neighbourhood_edges": int(exp.edges.shape[1]),
            "top_features": exp.top_features[:5],
            "top_edges": [
                {"from_tx": int(bundle["tx_id"][exp.edges[0, e]]), "to_tx": int(bundle["tx_id"][exp.edges[1, e]]),
                 "importance": float(exp.edge_importance[e])} for e in strongest
            ],
        })
    save_json(records, results_dir(cfg) / "gnn_explanations.json")
    return records


def graph_reliance(cfg: dict, bundle: dict) -> pd.DataFrame:
    """How much do the trained GNNs actually use the graph?

    For every GNN, feature set and seed, the saved model scores its evaluation
    graph twice: once normally and once with ALL edges removed, so that each
    node is scored from its own features only. Nothing is retrained.

        validation rows   model trained on steps 1-29, scored on steps 30-34
        test rows         model trained on steps 1-34, scored on steps 35-49

    If the scores barely change, the model is in effect an MLP. If they get
    worse without edges, the neighbourhood was helping; if they get better,
    what the model learned about neighbourhoods did not carry over to that
    period. (Removing edges is a condition the model never saw in training,
    so this measures reliance on the graph, not what a retrained model
    without a graph would score. The MLP baseline answers that question.)
    """
    from sklearn.metrics import average_precision_score, f1_score

    from .models import GNN_KINDS

    direction = cfg["gnn"]["direction"]
    device = torch.device("cpu")
    rows = []
    for features in cfg["feature_sets"]:
        graphs = build_graph_set(bundle, cfg, features, direction)
        for kind in GNN_KINDS:
            for seed in cfg["seeds"]:
                for split, phase in (("val", "select"), ("test", "final")):
                    path = checkpoint_path(cfg, "main", kind, features, direction, seed, phase)
                    if not path.exists():
                        continue
                    model, _ = load_checkpoint(path)
                    graph = graphs[split]
                    known = graph.labeled.numpy()
                    y = graph.y.numpy()[known]
                    with_edges = predict_scores(model, graph, device)[known]
                    bare = Data(x=graph.x, edge_index=graph.edge_index[:, :0])
                    without_edges = predict_scores(model, bare, device)[known]
                    # The threshold the protocol chose for this run (same for both phases).
                    threshold = float(load_checkpoint(
                        checkpoint_path(cfg, "main", kind, features, direction, seed, "final"))[1]["threshold"])
                    rows.append({
                        "model": kind, "features": features, "seed": seed, "split": split,
                        "pr_auc_with_edges": average_precision_score(y, with_edges),
                        "pr_auc_without_edges": average_precision_score(y, without_edges),
                        "f1_with_edges": f1_score(y, with_edges >= threshold, zero_division=0),
                        "f1_without_edges": f1_score(y, without_edges >= threshold, zero_division=0),
                        "decisions_changed": float(((with_edges >= threshold) != (without_edges >= threshold)).mean()),
                    })
    table = pd.DataFrame(rows)
    table.to_csv(results_dir(cfg) / "graph_reliance.csv", index=False)
    return table


def run(cfg: dict) -> None:
    bundle = load_bundle(cfg)
    if not graph_reliance(cfg, bundle).empty:
        log.info("wrote %s", results_dir(cfg) / "graph_reliance.csv")
    table = tree_importance(cfg, bundle)
    if table is not None:
        log.info("wrote %s", plot_tree_importance(cfg, table))
        share = table.groupby("group")["lgbm_mean_abs_shap"].sum() / table["lgbm_mean_abs_shap"].sum()
        log.info("share of total |SHAP|: %s", {k: round(float(v), 3) for k, v in share.items()})
    gnn_examples(cfg, bundle)


def main() -> None:
    args = config_arg_parser("Explain tree and GNN predictions.").parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
