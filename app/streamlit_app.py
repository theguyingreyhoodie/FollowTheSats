"""Streamlit demo: inspect the dataset, the models and individual transactions.

    streamlit run app/streamlit_app.py

The app only LOADS what run_all.py saved (metrics, per-node scores, model
checkpoints). It never trains a model. The only computation done live is
explaining one transaction at a time (GNNExplainer on that transaction's time
step, SHAP for one row), which takes about a second on a CPU.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import altair as alt  # noqa: E402
import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from btc_aml import analysis  # noqa: E402
from btc_aml.data import FEATURE_NAMES, FEATURE_SETS, ILLICIT, LICIT, UNKNOWN, load_bundle  # noqa: E402
from btc_aml.evaluate import load_metrics, run_tag  # noqa: E402
from btc_aml.explain import draw_explanation, explain_node, illicit_shap_values  # noqa: E402
from btc_aml.models import GNN_KINDS  # noqa: E402
from btc_aml.plotting import CLASS_COLORS, INK_MUTED, MODEL_COLORS, SHUTDOWN_STEP, apply_style  # noqa: E402
from btc_aml.splits import build_graph_set, build_table_set  # noqa: E402
from btc_aml.train_gnn import best_gnn, checkpoint_path, load_checkpoint  # noqa: E402
from btc_aml.utils import load_config, load_json, models_dir, results_dir  # noqa: E402

st.set_page_config(page_title="Illicit Bitcoin transactions", page_icon="🔎", layout="wide")

LABEL_NAMES = {ILLICIT: "illicit", LICIT: "licit", UNKNOWN: "unknown"}


# --------------------------------------------------------------------------- #
# Loading (cached: each of these runs once per session, not on every click)
# --------------------------------------------------------------------------- #
@st.cache_resource
def get_cfg() -> dict:
    """Use the config that produced the saved results, so app and artifacts match.

    Order: BTC_AML_CONFIG environment variable, then results/run_info.json,
    then the quick config.
    """
    path = os.environ.get("BTC_AML_CONFIG")
    if not path:
        info = ROOT / "results" / "run_info.json"
        path = load_json(info)["config_path"] if info.exists() else "configs/quick.yaml"
    if not Path(path).exists():  # e.g. results copied from another machine
        path = ROOT / "configs" / Path(path).name
    return load_config(path)


@st.cache_resource
def get_bundle() -> dict:
    return load_bundle(get_cfg())


@st.cache_data
def get_metrics() -> pd.DataFrame:
    return load_metrics(get_cfg())


@st.cache_data
def get_per_step() -> pd.DataFrame:
    return pd.read_csv(results_dir(get_cfg()) / "per_timestep.csv")


@st.cache_data
def get_summary() -> pd.DataFrame:
    return analysis.summarize(get_metrics())


@st.cache_resource
def get_scores() -> dict:
    """Saved test scores of the first seed, for every model.

    Returns {(model, features): (score per global node index, threshold)}.
    Nodes outside the test period hold NaN.
    """
    cfg, bundle = get_cfg(), get_bundle()
    seed = cfg["seeds"][0]
    n = bundle["x"].shape[0]
    rows = get_metrics()
    rows = rows[(rows["split"] == "test") & (rows["seed"] == seed)
                & rows["experiment"].isin(["main", "hybrid"])]
    out = {}
    for r in rows.itertuples():
        pred = analysis.load_prediction(cfg, r.experiment, r.model, r.features, r.variant, seed)
        if pred is None:
            continue
        idx, score, threshold = pred
        full = np.full(n, np.nan, dtype=np.float32)
        full[idx] = score
        out[(r.model, r.features)] = (full, threshold)
    return out


@st.cache_resource
def get_gnn():
    """The best GNN on AF (final model, first seed) and the test graph it scores."""
    cfg, bundle = get_cfg(), get_bundle()
    direction = cfg["gnn"]["direction"]
    kind = best_gnn(cfg, "AF")
    model, ckpt = load_checkpoint(
        checkpoint_path(cfg, "main", kind, "AF", direction, cfg["seeds"][0], "final"))
    graph = build_graph_set(bundle, cfg, "AF", direction)["test"]
    return kind, model, float(ckpt["threshold"]), graph


@st.cache_resource
def get_lgbm():
    """LightGBM on AF (final model, first seed), its SHAP explainer and the test table."""
    import shap

    cfg = get_cfg()
    path = models_dir(cfg) / f"{run_tag('main', 'lgbm', 'AF', '-', cfg['seeds'][0])}.joblib"
    if not path.exists():
        return None
    model = joblib.load(path)["model"]
    X, _, _, idx = build_table_set(get_bundle(), cfg, "AF")["test"]
    return shap.TreeExplainer(model), X, idx


@st.cache_data(show_spinner="Running GNNExplainer on this transaction's time step…")
def get_explanation(node: int):
    cfg = get_cfg()
    _, model, _, graph = get_gnn()
    return explain_node(model, graph, get_bundle(), node, FEATURE_NAMES[FEATURE_SETS["AF"]],
                        epochs=cfg["explain"]["explainer_epochs"], seed=cfg["seeds"][0])


@st.cache_data
def default_node() -> int:
    """The transaction shown first: a correctly flagged illicit one, with at least
    three neighbours, whose GNN score depends most on its neighbourhood (largest
    drop when all edges are removed). It makes the explanation worth looking at."""
    import torch
    from torch_geometric.data import Data

    from btc_aml.neural import predict_scores

    _, model, threshold, graph = get_gnn()
    cpu = torch.device("cpu")
    with_edges = predict_scores(model, graph, cpu)
    without = predict_scores(model, Data(x=graph.x, edge_index=graph.edge_index[:, :0]), cpu)
    degree = np.bincount(graph.edge_index[0].numpy(), minlength=graph.num_nodes)
    ok = np.flatnonzero((graph.y.numpy() == ILLICIT) & (with_edges >= threshold) & (degree >= 3))
    if len(ok) == 0:
        ok = np.flatnonzero(graph.y.numpy() == ILLICIT)
    return int(graph.node_idx[ok[np.argmax(with_edges[ok] - without[ok])]])


def require_results() -> None:
    cfg = get_cfg()
    if not (results_dir(cfg) / "metrics.csv").exists():
        st.error(
            "No saved results found. Run the pipeline first:\n\n"
            "```\npython run_all.py --config configs/quick.yaml\n```"
        )
        st.stop()


# --------------------------------------------------------------------------- #
# Small chart helpers
# --------------------------------------------------------------------------- #
def model_color_scale(models: list[str]) -> alt.Scale:
    """Fixed colour per model, so a model looks the same in every chart."""
    return alt.Scale(domain=[analysis.label(m) for m in models],
                     range=[MODEL_COLORS.get(m, INK_MUTED) for m in models])


def shutdown_rule() -> alt.Chart:
    return alt.Chart(pd.DataFrame({"time_step": [SHUTDOWN_STEP]})).mark_rule(
        color=INK_MUTED, strokeWidth=1).encode(x="time_step:Q")


def step_axis(title: str = "time step") -> alt.X:
    return alt.X("time_step:Q", title=title, axis=alt.Axis(tickMinStep=1, format="d", grid=False),
                 scale=alt.Scale(nice=False, padding=12))


# --------------------------------------------------------------------------- #
# Page 1: overview
# --------------------------------------------------------------------------- #
def page_overview() -> None:
    cfg = get_cfg()
    stats = load_json(results_dir(cfg) / "dataset_stats.json")
    st.title("Detecting illicit Bitcoin transactions")
    st.write(
        "The Elliptic dataset is a graph of Bitcoin transactions. An edge A → B means an output "
        "of transaction A is spent as an input of B. A small share of transactions is labelled "
        "illicit (scams, ransomware, dark markets…) or licit (exchanges, miners…); most are unlabelled."
    )
    c = st.columns(5)
    c[0].metric("Transactions (nodes)", f"{stats['num_nodes']:,}")
    c[1].metric("Payment flows (edges)", f"{stats['num_edges']:,}")
    c[2].metric("Illicit", f"{stats['num_illicit']:,}")
    c[3].metric("Licit", f"{stats['num_licit']:,}")
    c[4].metric("Unknown", f"{stats['num_unknown']:,}")

    counts = pd.read_csv(results_dir(cfg) / "labels_per_time_step.csv")
    st.subheader("Label distribution over time")
    long = counts.melt(id_vars="time_step", value_vars=["illicit", "licit", "unknown"],
                       var_name="label", value_name="transactions")
    order = ["illicit", "licit", "unknown"]
    long["order"] = long["label"].map({name: i for i, name in enumerate(order)})
    bars = alt.Chart(long).mark_bar(stroke="white", strokeWidth=0.6).encode(
        x=alt.X("time_step:O", title="time step (about two weeks each)",
                axis=alt.Axis(labelAngle=0, labelOverlap=True)),
        y=alt.Y("transactions:Q", title="transactions"),
        color=alt.Color("label:N", scale=alt.Scale(domain=order, range=[CLASS_COLORS[o] for o in order]),
                        legend=alt.Legend(title=None, orient="top")),
        order="order:Q",
        tooltip=["time_step", "label", alt.Tooltip("transactions:Q", format=",")],
    ).properties(height=300)
    st.altair_chart(bars, use_container_width=True)

    counts["illicit_pct"] = 100 * counts["illicit_share"]
    share = alt.Chart(counts).mark_line(color=CLASS_COLORS["illicit"], point=alt.OverlayMarkDef(
        size=50, filled=True, color=CLASS_COLORS["illicit"])).encode(
        x=step_axis("time step"),
        y=alt.Y("illicit_pct:Q", title="% illicit among labelled"),
        tooltip=["time_step", alt.Tooltip("illicit_pct:Q", format=".1f", title="% illicit"),
                 "illicit", "licit"],
    ).properties(height=220)
    st.altair_chart(share + shutdown_rule(), use_container_width=True)
    st.caption(
        f"The vertical line marks time step {SHUTDOWN_STEP}, when a large dark market was shut down. "
        "The illicit share drops sharply right after it, and the illicit activity that remains "
        "looks different from what the models were trained on."
    )

    st.subheader("How the data is split")
    s = cfg["split"]
    rows = []
    for name, purpose in (("train", "fit model weights"),
                          ("val", "choose hyperparameters, early-stopping epoch and threshold"),
                          ("test", "final score, evaluated once")):
        info = stats["splits"][name]
        rows.append({
            "split": name, "time steps": f"{s[name][0]}–{s[name][1]}", "used to": purpose,
            "transactions": f"{info['nodes']:,}", "illicit": f"{info['illicit']:,}",
            "licit": f"{info['licit']:,}",
            "% illicit among labelled": f"{100 * info['illicit_share_of_labeled']:.1f}%",
            "accuracy of 'everything is licit'": f"{100 * info['all_licit_accuracy']:.1f}%",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(
        "The split is by time: train on the past, test on the future. The last column is why "
        "accuracy is never the headline metric: a model that flags nothing looks excellent "
        "on accuracy while catching zero illicit transactions (illicit F1 = 0)."
    )


# --------------------------------------------------------------------------- #
# Page 2: model comparison
# --------------------------------------------------------------------------- #
def page_models() -> None:
    cfg = get_cfg()
    summary, per_step = get_summary(), get_per_step()
    n_seeds = int(get_metrics().query("experiment == 'main'")["seed"].nunique())
    st.title("Model comparison")
    st.caption(
        f"Config `{cfg['run_name']}`, {n_seeds} seed(s). All numbers are on the test period "
        f"(time steps {cfg['split']['test'][0]}–{cfg['split']['test'][1]}) for models trained on "
        f"steps 1–{cfg['split']['val'][1]}; thresholds were chosen on validation."
    )

    features = st.radio("Feature set", cfg["feature_sets"], horizontal=True, index=len(cfg["feature_sets"]) - 1,
                        format_func=lambda f: "LF: 93 local features" if f == "LF" else "AF: all 165 features")
    rows = []
    for model in analysis.MODEL_ORDER:
        r = analysis.pick(summary, "main", model, features)
        if r is not None:
            rows.append({"Model": analysis.label(model),
                         "Illicit precision": analysis.cell(r, "precision"),
                         "Illicit recall": analysis.cell(r, "recall"),
                         "Illicit F1": analysis.cell(r, "f1"),
                         "PR-AUC": analysis.cell(r, "pr_auc"),
                         "ROC-AUC": analysis.cell(r, "roc_auc"),
                         "micro-F1 (= accuracy)": analysis.cell(r, "micro_f1")})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("Mean ± standard deviation over seeds. micro-F1 is listed only for comparison with "
               "the original paper; it rewards predicting 'licit' and should not be used to rank models.")

    st.subheader("Illicit F1 per test time step")
    available = [m for m in analysis.MODEL_ORDER if analysis.pick(summary, "main", m, features) is not None]
    default = [m for m in ("rf", "lgbm", "mlp", best_gnn(cfg, features)) if m in available]
    chosen = st.multiselect("Models", available, default=default, format_func=analysis.label)
    main = per_step[(per_step["experiment"] == "main") & (per_step["split"] == "test")
                    & (per_step["features"] == features) & per_step["model"].isin(chosen)]
    if not main.empty:
        data = main.groupby(["model", "time_step"], as_index=False).agg(
            f1=("f1", "mean"), n_illicit=("n_illicit", "first"))
        data["Model"] = data["model"].map(analysis.label)
        lines = alt.Chart(data).mark_line(point=alt.OverlayMarkDef(size=45, filled=True)).encode(
            x=step_axis(),
            y=alt.Y("f1:Q", title="illicit F1 (mean over seeds)", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Model:N", scale=model_color_scale(chosen),
                            legend=alt.Legend(title=None, orient="top")),
            tooltip=["Model", "time_step", alt.Tooltip("f1:Q", format=".3f", title="illicit F1"),
                     alt.Tooltip("n_illicit:Q", title="illicit txs in step")],
        ).properties(height=340)
        st.altair_chart(lines + shutdown_rule(), use_container_width=True)
        st.caption(
            f"Vertical line: dark-market shutdown at time step {SHUTDOWN_STEP}. Every model collapses "
            "from there on. Steps 45 and 46 contain only a handful of illicit transactions, so their "
            "F1 is noisy."
        )

    with st.expander("All result tables (ablation, hybrid, rolling retraining, random split, run time)"):
        tables = results_dir(cfg) / "tables.md"
        st.markdown(tables.read_text(encoding="utf-8") if tables.exists() else "Run the analysis phase first.")


# --------------------------------------------------------------------------- #
# Page 3: transaction inspector
# --------------------------------------------------------------------------- #
def page_inspector() -> None:
    cfg, bundle = get_cfg(), get_bundle()
    scores = get_scores()
    kind, _, gnn_threshold, graph = get_gnn()
    y_all = bundle["y"].numpy()
    t_all = bundle["time_step"].numpy()
    tx_all = bundle["tx_id"].numpy()
    test_nodes = graph.node_idx.numpy()
    gnn_scores = scores[(kind, "AF")][0]

    st.title("Transaction inspector")
    st.caption(
        f"Pick a transaction from the test period. Flags and the explanation come from the best GNN "
        f"on AF ({analysis.label(kind)}, first seed), trained on time steps 1–{cfg['split']['val'][1]}."
    )

    rng = np.random.default_rng()
    pools = {
        "Random illicit": test_nodes[y_all[test_nodes] == ILLICIT],
        "Random flagged": test_nodes[gnn_scores[test_nodes] >= gnn_threshold],
        "Random licit": test_nodes[y_all[test_nodes] == LICIT],
    }
    if "node" not in st.session_state:
        st.session_state.node = default_node()
    cols = st.columns([1, 1, 1, 2])
    for col, (name, pool) in zip(cols, pools.items()):
        if col.button(name, use_container_width=True) and len(pool):
            st.session_state.node = int(rng.choice(pool))
    typed = cols[3].text_input("…or enter a test transaction id", value="", placeholder="e.g. "
                               + str(int(tx_all[st.session_state.node])))
    if typed.strip():
        match = np.flatnonzero(tx_all == int(typed)) if typed.strip().isdigit() else []
        if len(match) and match[0] in set(test_nodes.tolist()):
            st.session_state.node = int(match[0])
        else:
            st.warning("That id is not a transaction of the test period (time steps "
                       f"{cfg['split']['test'][0]}–{cfg['split']['test'][1]}).")

    node = st.session_state.node
    label_name = LABEL_NAMES[int(y_all[node])]
    c = st.columns(4)
    c[0].metric("Transaction id", str(int(tx_all[node])))
    c[1].metric("Time step", int(t_all[node]))
    c[2].metric("True label", label_name)
    flagged = gnn_scores[node] >= gnn_threshold
    c[3].metric(f"{analysis.label(kind)} score (threshold {gnn_threshold:.3f})",
                f"{gnn_scores[node]:.3f} · " + ("flagged" if flagged else "not flagged"))
    if label_name == "unknown":
        st.info("This transaction has no ground-truth label, so nobody knows whether a flag is right.")

    left, right = st.columns([1, 1])

    with right:
        st.subheader("What every model says")
        rows = []
        for (model, features), (score, threshold) in scores.items():
            if np.isnan(score[node]):
                continue
            name = analysis.label(model).replace("Random Forest", "RF").replace(" (out-of-fold)", "")
            rows.append({"Model": f"{name} · {features}", "P(illicit)": float(score[node]),
                         "Threshold": float(threshold), "Flagged": bool(score[node] >= threshold)})
        table = pd.DataFrame(rows)
        st.dataframe(
            table, hide_index=True, use_container_width=True, height=min(38 * (len(table) + 1), 520),
            column_config={
                "P(illicit)": st.column_config.ProgressColumn(format="%.3f", min_value=0.0, max_value=1.0,
                                                              width="small"),
                "Threshold": st.column_config.NumberColumn(format="%.3f", width="small"),
                "Flagged": st.column_config.CheckboxColumn(width="small"),
            },
        )
        st.caption("First seed of each model. A model flags a transaction when its probability "
                   "reaches that model's own threshold, chosen on the validation period.")

    exp = get_explanation(node)
    with left:
        st.subheader("2-hop neighbourhood")
        apply_style()
        fig, ax = plt.subplots(figsize=(6.6, 5.4))
        draw_explanation(ax, exp, bundle, gnn_threshold, scores=gnn_scores[exp.nodes], legend=False)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        dot = lambda color: f"<span style='color:{color}'>●</span>"
        st.markdown(
            f"<small>{dot(CLASS_COLORS['illicit'])} illicit &nbsp; {dot(CLASS_COLORS['licit'])} licit &nbsp; "
            f"{dot(CLASS_COLORS['unknown'])} unknown &nbsp;·&nbsp; ★ inspected transaction &nbsp;·&nbsp; "
            "dark ring = flagged by the GNN &nbsp;·&nbsp; thick dark arrow = important for the prediction</small>",
            unsafe_allow_html=True)
        n_labels = pd.Series(y_all[exp.nodes]).map(LABEL_NAMES).value_counts()
        st.caption(
            f"{len(exp.nodes)} transactions"
            + (" (capped at 200; nearest and most important kept)" if exp.truncated else "")
            + f", {exp.edges.shape[1]} payment flows. Labels: "
            + ", ".join(f"{v} {k}" for k, v in n_labels.items())
            + ". Arrows point from the transaction whose output is spent to the one spending it. "
              "A 2-layer GNN can only see this far."
        )

    st.subheader("Why was it scored this way?")
    change = exp.score - exp.score_no_edges
    f = st.columns(3)
    f[0].metric("GNN score, full neighbourhood", f"{exp.score:.3f}")
    f[1].metric("GNN score, all edges removed", f"{exp.score_no_edges:.3f}")
    f[2].metric(f"GNN score, only the {exp.edges_kept} edge(s) marked important", f"{exp.score_kept_edges:.3f}")
    if abs(change) < 0.05:
        st.info(
            "Removing every edge changes this score by less than 0.05. The GNN's prediction here comes "
            "from the transaction's **own features**; the neighbourhood plays almost no role, so the edge "
            "importances below carry little meaning for this transaction."
        )
    else:
        st.info(
            f"Removing every edge moves the score by {change:+.3f}, so the neighbourhood matters for this "
            f"prediction. GNNExplainer marked {exp.edges_kept} of the {exp.edges_in_scope} edges within two "
            f"hops as important; with only those edges the score is {exp.score_kept_edges:.3f}."
        )
    a, b = st.columns(2)
    with a:
        st.markdown(f"**{analysis.label(kind)}: most important payment flows (GNNExplainer)**")
        order = np.argsort(-exp.edge_importance)[:8]
        edge_rows = []
        for e in order:
            u, v = int(exp.edges[0, e]), int(exp.edges[1, e])
            other = v if u == node else u
            edge_rows.append({
                "from": str(int(tx_all[u])), "to": str(int(tx_all[v])),
                "importance": float(exp.edge_importance[e]),
                "neighbour": LABEL_NAMES[int(y_all[other])],
                "its score": float(gnn_scores[other]),
            })
        if edge_rows:
            st.dataframe(pd.DataFrame(edge_rows), hide_index=True, use_container_width=True,
                         column_config={
                             "importance": st.column_config.ProgressColumn(format="%.2f", min_value=0.0, max_value=1.0),
                             "its score": st.column_config.NumberColumn(format="%.3f")})
        else:
            st.write("This transaction has no neighbours; the score comes from its own features.")
        st.caption("Importance is GNNExplainer's mask value (0 to 1) for the edge. It searches for the "
                   "smallest set of edges that still reproduces the GNN's prediction. 'neighbour' is the "
                   "transaction at the other end of the edge; 'its score' is the GNN's probability for it.")
        feats = pd.DataFrame(exp.top_features[:6], columns=["feature", "mask value"])
        st.markdown("Features GNNExplainer kept: " + ", ".join(
            f"`{f}` ({v:.2f})" for f, v in feats.itertuples(index=False)))

    with b:
        st.markdown("**LightGBM: feature contributions (SHAP)**")
        lgbm = get_lgbm()
        if lgbm is None:
            st.write("LightGBM model file not found.")
        else:
            explainer, X_test, idx_test = lgbm
            row = int(np.searchsorted(idx_test, node))
            values = illicit_shap_values(explainer, X_test[row:row + 1])[0]
            top = np.argsort(-np.abs(values))[:8]
            contrib = pd.DataFrame({"feature": [FEATURE_NAMES[i] for i in top],
                                    "contribution": values[top],
                                    "scaled value": X_test[row, top]})
            contrib["direction"] = np.where(contrib["contribution"] > 0, "towards illicit", "towards licit")
            chart = alt.Chart(contrib).mark_bar(cornerRadiusEnd=3).encode(
                x=alt.X("contribution:Q", title="SHAP contribution to the score (log-odds)"),
                y=alt.Y("feature:N", sort=None, title=None, scale=alt.Scale(paddingInner=0.35)),
                color=alt.Color("direction:N", scale=alt.Scale(
                    domain=["towards illicit", "towards licit"],
                    range=[CLASS_COLORS["illicit"], CLASS_COLORS["licit"]]),
                    legend=alt.Legend(title=None, orient="top")),
                tooltip=["feature", alt.Tooltip("contribution:Q", format="+.3f"),
                         alt.Tooltip("scaled value:Q", format=".2f"), "direction"],
            ).properties(height=alt.Step(30))
            st.altair_chart(chart, use_container_width=True)
            st.caption("Each bar is how much one feature pushed LightGBM's score for this transaction "
                       "up (towards illicit) or down (towards licit).")

    st.warning(
        "**Limits of these explanations.** The dataset's features are anonymised: we can say that "
        "`local_53` or `agg_13` mattered, but not what they measure. The explanations show which "
        "inputs and which neighbouring transactions the models relied on. They are not evidence "
        "that a transaction is criminal, and a flag should start a human review, never replace one."
    )


# --------------------------------------------------------------------------- #
# Page 4: time travel
# --------------------------------------------------------------------------- #
def page_time_travel() -> None:
    cfg = get_cfg()
    per_step = get_per_step()
    main = per_step[(per_step["experiment"] == "main") & (per_step["split"] == "test")]
    first, last = cfg["split"]["test"]
    k = int(main["k"].iloc[0])

    st.title("Time travel")
    st.caption("Move through the test period one time step (about two weeks) at a time. "
               f"The models were trained once, on steps 1–{cfg['split']['val'][1]}, and never updated.")
    c1, c2 = st.columns([3, 1])
    step = c1.slider("Test time step", first, last, first)
    features = c2.radio("Feature set", cfg["feature_sets"], horizontal=True,
                        index=len(cfg["feature_sets"]) - 1)

    now = main[(main["time_step"] == step) & (main["features"] == features)]
    agg = now.groupby("model").agg(
        precision=("precision", "mean"), recall=("recall", "mean"), f1=("f1", "mean"),
        precision_at_k=("precision_at_k", "mean"), recall_at_k=("recall_at_k", "mean"),
        n_flagged=("n_flagged", "mean")).reindex([m for m in analysis.MODEL_ORDER if m in set(now["model"])])
    n_illicit, n_labeled = int(now["n_illicit"].iloc[0]), int(now["n_labeled"].iloc[0])

    best = agg["f1"].idxmax()
    c = st.columns(5)
    c[0].metric("Labelled transactions", f"{n_labeled:,}")
    c[1].metric("Illicit among them", f"{n_illicit:,} ({100 * n_illicit / n_labeled:.1f}%)")
    c[2].metric(f"Best illicit F1 ({analysis.label(best)})", f"{agg.loc[best, 'f1']:.3f}")
    c[3].metric(f"Its precision@{k}", f"{agg.loc[best, 'precision_at_k']:.3f}")
    c[4].metric(f"Its recall@{k}", f"{agg.loc[best, 'recall_at_k']:.3f}")
    if step >= SHUTDOWN_STEP:
        st.error(f"Time step {step} is after the dark-market shutdown (t = {SHUTDOWN_STEP}). The illicit "
                 "transactions here no longer look like the ones the models learned from.")

    table = agg.reset_index()
    table["model"] = table["model"].map(analysis.label)
    table.columns = ["Model", "Illicit precision", "Illicit recall", "Illicit F1",
                     f"Precision@{k}", f"Recall@{k}", "Transactions flagged"]
    st.dataframe(table, hide_index=True, use_container_width=True,
                 column_config={col: st.column_config.NumberColumn(format="%.3f") for col in table.columns[1:6]}
                 | {"Transactions flagged": st.column_config.NumberColumn(format="%.1f")})
    st.caption(
        f"Mean over seeds. Alert budget: if an analyst reviews only the {k} highest-scored labelled "
        f"transactions of this time step, precision@{k} is the share of those that are illicit and "
        f"recall@{k} is the share of all illicit transactions of the step that they would see. "
        f"With {n_illicit} illicit transactions in this step, precision@{k} cannot exceed "
        f"{min(1.0, n_illicit / min(k, n_labeled)):.2f}."
    )

    st.subheader("Where this step sits")
    models = [m for m in ("rf", "lgbm", "mlp", best_gnn(cfg, features)) if m in agg.index]
    hist = main[(main["features"] == features) & main["model"].isin(models)] \
        .groupby(["model", "time_step"], as_index=False)["f1"].mean()
    hist["Model"] = hist["model"].map(analysis.label)
    lines = alt.Chart(hist).mark_line(point=alt.OverlayMarkDef(size=40, filled=True)).encode(
        x=step_axis(),
        y=alt.Y("f1:Q", title="illicit F1", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Model:N", scale=model_color_scale(models), legend=alt.Legend(title=None, orient="top")),
        tooltip=["Model", "time_step", alt.Tooltip("f1:Q", format=".3f")],
    ).properties(height=280)
    cursor = alt.Chart(pd.DataFrame({"time_step": [step]})).mark_rule(
        color="#0b0b0b", strokeWidth=2, opacity=0.35).encode(x="time_step:Q")
    st.altair_chart(lines + shutdown_rule() + cursor, use_container_width=True)
    st.caption(f"Dark vertical band: the selected step. Thin grey line: shutdown at t = {SHUTDOWN_STEP}.")


# --------------------------------------------------------------------------- #
PAGES = {
    "Overview": page_overview,
    "Model comparison": page_models,
    "Transaction inspector": page_inspector,
    "Time travel": page_time_travel,
}

require_results()
with st.sidebar:
    st.header("Illicit Bitcoin transactions")
    # A link such as ...?page=Time+travel opens that page directly.
    wanted = st.query_params.get("page", "Overview")
    start = list(PAGES).index(wanted) if wanted in PAGES else 0
    page = st.radio("Page", list(PAGES), index=start, label_visibility="collapsed")
    st.divider()
    st.caption(
        f"Results from config `{get_cfg()['run_name']}`. The app loads saved models and scores; "
        "it does not train anything."
    )
PAGES[page]()
