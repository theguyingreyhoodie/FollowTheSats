"""Turn the raw result files into tables and figures.

    python -m btc_aml.analysis --config configs/quick.yaml

Reads   results/metrics.csv, results/per_timestep.csv, results/predictions/*.npz
Writes  results/summary.csv            mean and std over seeds for every model
        results/period_metrics.csv     scores before vs after the t=43 shutdown
        results/tables.md              all tables as Markdown (the README embeds them)
        results/figures/*.png

Nothing is trained here, with one small exception: the t-SNE figure runs the
saved GNN once to get embeddings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, precision_recall_curve

from .data import ILLICIT, UNKNOWN, load_bundle
from .evaluate import binary_metrics, load_metrics, run_tag
from .models import GNN_KINDS
from .neural import node_embeddings
from .plotting import (
    CLASS_COLORS, INK, INK_MUTED, INK_SECONDARY, MODEL_COLORS, MODEL_LABELS, SERIES_COLORS,
    SHUTDOWN_STEP, SURFACE, apply_style, mark_shutdown, save,
)
from .splits import build_graph_set
from .train_gnn import best_gnn, checkpoint_path, load_checkpoint
from .utils import (
    config_arg_parser, figures_dir, get_device, get_logger, load_config, load_json, results_dir,
)

log = get_logger()

METRICS = ["precision", "recall", "f1", "pr_auc", "roc_auc", "micro_f1"]
BASELINES = ["logreg", "rf", "lgbm", "mlp"]
MODEL_ORDER = BASELINES + list(GNN_KINDS)


def label(model: str) -> str:
    """Human-readable model name, e.g. 'rf+oof' -> 'Random Forest + GNN score (out-of-fold)'."""
    if model.endswith("+oof"):
        return MODEL_LABELS[model[:-4]] + " + GNN score (out-of-fold)"
    if model.endswith("+emb"):
        return MODEL_LABELS[model[:-4]] + " + GNN embeddings"
    return MODEL_LABELS.get(model, model)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_prediction(cfg: dict, experiment: str, model: str, features: str, variant: str, seed: int):
    """(node_idx, score, threshold) of one saved test run, or None if absent."""
    path = results_dir(cfg) / "predictions" / f"{run_tag(experiment, model, features, variant, seed)}.npz"
    if not path.exists():
        return None
    z = np.load(path)
    return z["node_idx"], z["score"], float(z["threshold"])


def gnn_variant(cfg: dict, model: str) -> str:
    """GNN rows carry the edge direction as their variant; other models use '-'."""
    return cfg["gnn"]["direction"] if model in GNN_KINDS else "-"


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation over seeds for every run configuration."""
    keys = ["experiment", "model", "features", "variant", "split"]
    grouped = df.groupby(keys, sort=False)
    out = grouped[METRICS].mean().add_suffix("_mean")
    # ddof=1 is the usual sample standard deviation; with one seed it is NaN,
    # which is the honest value ("no spread measured"), not zero.
    out = out.join(grouped[METRICS].std(ddof=1).add_suffix("_std"))
    out["n_seeds"] = grouped["seed"].nunique()
    out["train_seconds_mean"] = grouped["train_seconds"].mean()
    return out.reset_index()


def fmt(mean: float, std: float) -> str:
    """'0.791 ± 0.012', or just '0.791' when there is a single seed."""
    if pd.isna(mean):
        return "n/a"
    return f"{mean:.3f}" if pd.isna(std) else f"{mean:.3f} ± {std:.3f}"


def md_table(rows: list[list[str]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def pick(summary: pd.DataFrame, experiment: str, model: str, features: str,
         split: str = "test", variant: str | None = None):
    rows = summary[
        (summary["experiment"] == experiment) & (summary["model"] == model)
        & (summary["features"] == features) & (summary["split"] == split)
    ]
    if variant is not None:
        rows = rows[rows["variant"] == variant]
    return None if rows.empty else rows.iloc[0]


def cell(row, metric: str) -> str:
    return "n/a" if row is None else fmt(row[f"{metric}_mean"], row[f"{metric}_std"])


def main_tables(cfg: dict, summary: pd.DataFrame) -> str:
    """One table per feature set: every model, every headline metric (test set)."""
    parts = []
    for features in cfg["feature_sets"]:
        rows = []
        for model in MODEL_ORDER:
            r = pick(summary, "main", model, features)
            if r is not None:
                rows.append([label(model)] + [cell(r, m) for m in METRICS])
        n_feat = "93 local features" if features == "LF" else "all 165 features"
        parts.append(
            f"**{features} ({n_feat}), test steps 35-49**\n\n"
            + md_table(rows, ["Model", "Illicit precision", "Illicit recall", "Illicit F1",
                              "PR-AUC", "ROC-AUC", "micro-F1"])
        )
    return "\n\n".join(parts)


def lf_vs_af_table(cfg: dict, summary: pd.DataFrame) -> str:
    """Does a model gain from the hand-made neighbourhood features (AF minus LF)?"""
    rows = []
    for model in MODEL_ORDER:
        lf, af = pick(summary, "main", model, "LF"), pick(summary, "main", model, "AF")
        if lf is None or af is None:
            continue
        rows.append([label(model), cell(lf, "f1"), cell(af, "f1"),
                     f"{af['f1_mean'] - lf['f1_mean']:+.3f}",
                     cell(lf, "pr_auc"), cell(af, "pr_auc"),
                     f"{af['pr_auc_mean'] - lf['pr_auc_mean']:+.3f}"])
    return md_table(rows, ["Model", "F1 on LF", "F1 on AF", "AF − LF (F1)",
                           "PR-AUC on LF", "PR-AUC on AF", "AF − LF (PR-AUC)"])


def ablation_table(cfg: dict, summary: pd.DataFrame) -> str:
    ab = cfg["ablation"]
    rows_ab = summary[(summary["experiment"] == "ablation") & (summary["split"] == "test")]
    if rows_ab.empty:
        return "_not run_"
    model = rows_ab["model"].iloc[0]
    rows = []
    for direction in ab["directions"]:
        experiment = "main" if direction == cfg["gnn"]["direction"] else "ablation"
        r = pick(summary, experiment, model, ab["features"], variant=direction)
        v = pick(summary, experiment, model, ab["features"], split="val", variant=direction)
        if r is not None:
            rows.append([direction, cell(r, "precision"), cell(r, "recall"), cell(r, "f1"),
                         cell(r, "pr_auc"), cell(v, "pr_auc")])
    return (
        f"Model: {label(model)} on {ab['features']}.\n\n"
        + md_table(rows, ["Edge treatment", "Illicit precision", "Illicit recall", "Illicit F1",
                          "PR-AUC (test)", "PR-AUC (validation)"])
    )


def hybrid_table(cfg: dict, summary: pd.DataFrame) -> str:
    hyb = summary[(summary["experiment"] == "hybrid") & (summary["split"] == "test")]
    if hyb.empty:
        return "_not run_"
    rows = []
    for features in cfg["feature_sets"]:
        # The GNN is chosen per feature set (by validation PR-AUC); it is stored as the row's variant.
        used = hyb[hyb["features"] == features]["variant"]
        gnn = label(used.iloc[0]) if len(used) else "n/a"
        for tree in cfg["hybrid"]["trees"]:
            plain = pick(summary, "main", tree, features)
            entries = [("features only", plain)]
            for mode in cfg["hybrid"]["modes"]:
                entries.append((f"+ {gnn} {'embeddings' if mode == 'emb' else 'score (out-of-fold)'}",
                                pick(summary, "hybrid", f"{tree}+{mode}", features)))
            for name, r in entries:
                if r is None:
                    continue
                delta = "" if r is plain else f"{r['f1_mean'] - plain['f1_mean']:+.3f}"
                rows.append([label(tree), features, name, cell(r, "precision"), cell(r, "recall"),
                             cell(r, "f1"), delta, cell(r, "pr_auc")])
    return (
        "The GNN is the one with the best validation PR-AUC on the same feature set.\n\n"
        + md_table(rows, ["Tree model", "Features", "Input", "Illicit precision", "Illicit recall",
                          "Illicit F1", "ΔF1 vs features only", "PR-AUC"])
    )


def period_metrics(cfg: dict, df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Pooled metrics before (35-42) and after (43-49) the dark-market shutdown.

    F1 cannot be averaged over time steps (a step with 2 illicit transactions
    would count as much as one with 200), so the saved predictions are pooled
    per period and the metrics recomputed.
    """
    y_all, t_all = bundle["y"].numpy(), bundle["time_step"].numpy()
    last = cfg["split"]["test"][1]
    periods = {f"{cfg['split']['test'][0]}-{SHUTDOWN_STEP - 1}": (cfg["split"]["test"][0], SHUTDOWN_STEP - 1),
               f"{SHUTDOWN_STEP}-{last}": (SHUTDOWN_STEP, last)}
    rows = []
    test_rows = df[(df["split"] == "test") & (df["experiment"].isin(["main", "hybrid", "rolling", "ablation"]))]
    for r in test_rows.itertuples():
        pred = load_prediction(cfg, r.experiment, r.model, r.features, r.variant, r.seed)
        if pred is None:
            continue
        idx, score, threshold = pred
        y, t = y_all[idx], t_all[idx]
        for name, (lo, hi) in periods.items():
            keep = (y != UNKNOWN) & (t >= lo) & (t <= hi)
            m = binary_metrics(y[keep], score[keep], threshold)
            rows.append({"experiment": r.experiment, "model": r.model, "features": r.features,
                         "variant": r.variant, "seed": r.seed, "period": name, **m})
    return pd.DataFrame(rows)


def period_table(cfg: dict, periods: pd.DataFrame) -> str:
    """Before/after-shutdown F1 and recall for the main models and rolling retraining."""
    if periods.empty:
        return "_not run_"
    names = list(dict.fromkeys(periods["period"]))
    agg = periods.groupby(["experiment", "model", "features", "variant", "period"])[
        ["precision", "recall", "f1"]].agg(["mean", "std"])
    rows = []

    def add(experiment: str, model: str, features: str, shown: str) -> None:
        sub = agg.loc[(agg.index.get_level_values(0) == experiment)
                      & (agg.index.get_level_values(1) == model)
                      & (agg.index.get_level_values(2) == features)]
        if sub.empty:
            return
        row = [shown, features]
        for name in names:
            s = sub.xs(name, level="period")
            for metric in ("f1", "recall", "precision"):
                row.append(fmt(s[(metric, "mean")].iloc[0], s[(metric, "std")].iloc[0]))
        rows.append(row)

    for features in cfg["feature_sets"]:
        for model in MODEL_ORDER:
            add("main", model, features, label(model))
    for model in cfg.get("rolling", {}).get("models", []):
        add("rolling", model, cfg["rolling"]["features"], label(model) + " — rolling retraining")
    header = ["Model", "Features"]
    for name in names:
        header += [f"F1, steps {name}", f"Recall, steps {name}", f"Precision, steps {name}"]
    return md_table(rows, header)


def threshold_diagnostic(cfg: dict, df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """How much does the choice of threshold cost each model on the test set?

    For every main run, three F1 values from the same saved test scores:
        f1_chosen   at the threshold chosen on validation (the reported result)
        f1_default  at the naive threshold 0.5
        f1_oracle   at the best possible threshold FOR THE TEST SET

    !!! f1_oracle is NOT a valid result. Picking a threshold with test labels
    !!! is cheating. It is an upper bound that separates two reasons a model
    !!! can score badly: a poor ranking (even the oracle is low) or a threshold
    !!! that did not transfer from validation to test (oracle much higher).
    """
    y_all = bundle["y"].numpy()
    rows = []
    main = df[(df["experiment"] == "main") & (df["split"] == "test")]
    for r in main.itertuples():
        pred = load_prediction(cfg, r.experiment, r.model, r.features, r.variant, r.seed)
        if pred is None:
            continue
        idx, score, threshold = pred
        known = y_all[idx] != UNKNOWN
        y, s = y_all[idx][known], score[known]
        precision, recall, _ = precision_recall_curve(y, s)
        f1_curve = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
        rows.append({
            "model": r.model, "features": r.features, "seed": r.seed, "threshold": threshold,
            "f1_chosen": binary_metrics(y, s, threshold)["f1"],
            "f1_default": binary_metrics(y, s, 0.5)["f1"],
            "f1_oracle": float(f1_curve.max()),
        })
    return pd.DataFrame(rows)


def threshold_table(cfg: dict, diag: pd.DataFrame) -> str:
    if diag.empty:
        return "_not run_"
    agg = diag.groupby(["model", "features"])[["threshold", "f1_chosen", "f1_default", "f1_oracle"]].agg(["mean", "std"])
    rows = []
    for features in cfg["feature_sets"]:
        for model in MODEL_ORDER:
            if (model, features) not in agg.index:
                continue
            a = agg.loc[(model, features)]
            rows.append([label(model), features, f"{a[('threshold', 'mean')]:.3f}",
                         fmt(a[("f1_chosen", "mean")], a[("f1_chosen", "std")]),
                         fmt(a[("f1_default", "mean")], a[("f1_default", "std")]),
                         fmt(a[("f1_oracle", "mean")], a[("f1_oracle", "std")]),
                         f"{a[('f1_oracle', 'mean')] - a[('f1_chosen', 'mean')]:+.3f}"])
    return (
        "The last two F1 columns are diagnostics, **not results**: the oracle threshold is chosen "
        "with test labels and is only an upper bound on what a better threshold could achieve.\n\n"
        + md_table(rows, ["Model", "Features", "Mean threshold (from validation)",
                          "F1 at validation threshold (reported)", "F1 at threshold 0.5",
                          "F1 at oracle test threshold (upper bound)", "Oracle − reported"])
    )


def graph_reliance_table(cfg: dict) -> str:
    """Scores of the trained GNNs with and without edges (computed in explain.py)."""
    path = results_dir(cfg) / "graph_reliance.csv"
    if not path.exists():
        return "_written by the explain phase_"
    table = pd.read_csv(path)
    columns = ["pr_auc_with_edges", "pr_auc_without_edges", "f1_with_edges", "f1_without_edges",
               "decisions_changed"]
    agg = table.groupby(["model", "features", "split"])[columns].agg(["mean", "std"])
    rows = []
    for features in cfg["feature_sets"]:
        for model in GNN_KINDS:
            row = [label(model), features]
            for split in ("val", "test"):
                if (model, features, split) not in agg.index:
                    row += ["n/a", "n/a"]
                    continue
                a = agg.loc[(model, features, split)]
                row += [fmt(a[("pr_auc_with_edges", "mean")], a[("pr_auc_with_edges", "std")]),
                        fmt(a[("pr_auc_without_edges", "mean")], a[("pr_auc_without_edges", "std")])]
            if (model, features, "test") in agg.index:
                a = agg.loc[(model, features, "test")]
                row += [fmt(a[("f1_with_edges", "mean")], a[("f1_with_edges", "std")]),
                        fmt(a[("f1_without_edges", "mean")], a[("f1_without_edges", "std")]),
                        f"{100 * a[('decisions_changed', 'mean')]:.1f}%"]
            rows.append(row)
    return (
        "Each trained GNN scores its evaluation graph twice: normally, and with every edge removed "
        "(each node then sees only its own features). Nothing is retrained. Validation = model "
        "trained on steps 1-29 scored on steps 30-34; test = model trained on steps 1-34 scored on "
        "steps 35-49.\n\n"
        + md_table(rows, ["Model", "Features", "Validation PR-AUC, with edges", "Validation PR-AUC, edges removed",
                          "Test PR-AUC, with edges", "Test PR-AUC, edges removed",
                          "Test F1, with edges", "Test F1, edges removed", "Test decisions that change"])
    )


def rolling_table(cfg: dict, summary: pd.DataFrame) -> str:
    section = cfg.get("rolling", {})
    rows = []
    for model in section.get("models", []):
        static = pick(summary, "main", model, section["features"])
        rolling = pick(summary, "rolling", model, section["features"])
        if static is None or rolling is None:
            continue
        for name, r in (("trained once on steps 1-34", static), ("retrained before every test step", rolling)):
            rows.append([label(model), name, cell(r, "precision"), cell(r, "recall"),
                         cell(r, "f1"), cell(r, "pr_auc"), int(r["n_seeds"])])
    if not rows:
        return "_not run_"
    return md_table(rows, ["Model", "Training", "Illicit precision", "Illicit recall",
                           "Illicit F1", "PR-AUC", "Seeds"])


def random_split_table(cfg: dict, df: pd.DataFrame, summary: pd.DataFrame) -> str:
    rnd = df[df["experiment"] == "random_split"]
    if rnd.empty:
        return "_not run_"
    rows = []
    for r in rnd.itertuples():
        temporal = df[(df["experiment"] == "main") & (df["model"] == r.model) & (df["features"] == r.features)
                      & (df["split"] == "test") & (df["seed"] == r.seed)]
        t_f1 = temporal["f1"].iloc[0] if not temporal.empty else float("nan")
        rows.append([label(r.model), r.features, f"{t_f1:.3f}", f"{r.f1:.3f}", f"{r.f1 - t_f1:+.3f}"])
    return (
        "One run each (first seed). The random-split column is an illustration of leakage, not a result.\n\n"
        + md_table(rows, ["Model", "Features", "F1, temporal split (real protocol, same seed)",
                          "F1, random split (leaky)", "Inflation"])
    )


def alert_table(cfg: dict, per_step: pd.DataFrame) -> str:
    """Average precision@K / recall@K over the test time steps, before and after t=43."""
    main = per_step[(per_step["experiment"] == "main") & (per_step["split"] == "test")]
    if main.empty:
        return "_not run_"
    k = int(main["k"].iloc[0])
    rows = []
    for features in cfg["feature_sets"]:
        for model in MODEL_ORDER:
            sub = main[(main["model"] == model) & (main["features"] == features)]
            if sub.empty:
                continue
            before, after = sub[sub["time_step"] < SHUTDOWN_STEP], sub[sub["time_step"] >= SHUTDOWN_STEP]
            rows.append([label(model), features,
                         f"{before['precision_at_k'].mean():.3f}", f"{before['recall_at_k'].mean():.3f}",
                         f"{after['precision_at_k'].mean():.3f}", f"{after['recall_at_k'].mean():.3f}"])
    first, last = cfg["split"]["test"]
    return md_table(rows, ["Model", "Features",
                           f"Precision@{k}, steps {first}-{SHUTDOWN_STEP - 1}",
                           f"Recall@{k}, steps {first}-{SHUTDOWN_STEP - 1}",
                           f"Precision@{k}, steps {SHUTDOWN_STEP}-{last}",
                           f"Recall@{k}, steps {SHUTDOWN_STEP}-{last}"])


def trivial_baseline_table(cfg: dict) -> str:
    """What 'predict licit for everything' scores (computed in eda.py)."""
    path = results_dir(cfg) / "dataset_stats.json"
    if not path.exists():
        return "_run btc_aml.eda first_"
    stats = load_json(path)["splits"]
    rows = [[name, f"{s['illicit']:,}", f"{s['licit']:,}", f"{100 * s['illicit_share_of_labeled']:.2f}%",
             f"{100 * s['all_licit_accuracy']:.2f}%", "0.000"]
            for name, s in stats.items() if name != "trainval"]
    return md_table(rows, ["Split", "Illicit", "Licit", "Illicit share of labelled",
                           "Accuracy of 'everything is licit'", "Illicit F1 of 'everything is licit'"])


def timing_table(cfg: dict) -> str:
    path = results_dir(cfg) / "timings.json"
    if not path.exists():
        return "_timings are written by run_all.py_"
    t = load_json(path)
    rows = [[phase, f"{seconds:.0f} s", f"{seconds / 60:.1f} min"] for phase, seconds in t["phases"].items()]
    rows.append(["**total**", f"{t['total_seconds']:.0f} s", f"{t['total_seconds'] / 60:.1f} min"])
    return f"Config `{t['config']}`, {t['seeds']} seed(s), device `{t['device']}`, {t['machine']}.\n\n" + md_table(
        rows, ["Phase", "Seconds", "Minutes"])


def write_tables(cfg: dict, df: pd.DataFrame, summary: pd.DataFrame,
                 periods: pd.DataFrame, per_step: pd.DataFrame, diag: pd.DataFrame) -> str:
    n_seeds = int(df[df["experiment"] == "main"]["seed"].nunique())
    spread = ("mean ± standard deviation over " + f"{n_seeds} seeds") if n_seeds > 1 else (
        "a single seed, so no standard deviation is shown")
    sections = {
        "Main results": f"Config `{cfg['run_name']}`; values are {spread}.\n\n" + main_tables(cfg, summary),
        "LF versus AF": lf_vs_af_table(cfg, summary),
        "Edge-direction ablation": ablation_table(cfg, summary),
        "Hybrid: GNN output as extra tree features": hybrid_table(cfg, summary),
        "Before and after the dark-market shutdown": period_table(cfg, periods),
        "Rolling retraining": rolling_table(cfg, summary),
        f"Alert budget (top {cfg['alert_k']} per time step)": alert_table(cfg, per_step),
        "Do the GNNs use the graph?": graph_reliance_table(cfg),
        "Threshold diagnostic": threshold_table(cfg, diag),
        "Random split versus temporal split": random_split_table(cfg, df, summary),
        "Trivial baseline": trivial_baseline_table(cfg),
        "Run time": timing_table(cfg),
    }
    text = "\n\n".join(f"### {title}\n\n{body}" for title, body in sections.items()) + "\n"
    (results_dir(cfg) / "tables.md").write_text(text, encoding="utf-8")
    return text


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_model_comparison(cfg: dict, summary: pd.DataFrame):
    """Dot plot: test illicit F1 of every model, LF next to AF, with seed spread."""
    apply_style()
    models = [m for m in MODEL_ORDER if pick(summary, "main", m, cfg["feature_sets"][0]) is not None]
    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(models) + 1.6))
    offsets = {"LF": -0.16, "AF": 0.16}
    colors = {"LF": SERIES_COLORS[0], "AF": SERIES_COLORS[1]}
    for features in cfg["feature_sets"]:
        for i, model in enumerate(models):
            r = pick(summary, "main", model, features)
            if r is None:
                continue
            y = i + offsets[features]
            std = 0 if pd.isna(r["f1_std"]) else r["f1_std"]
            ax.errorbar(r["f1_mean"], y, xerr=std, fmt="o", color=colors[features], markersize=8,
                        markeredgecolor=SURFACE, markeredgewidth=1.5, elinewidth=1.5, capsize=0,
                        label=features if i == 0 else None)
            ax.text(r["f1_mean"] + std + 0.012, y, f"{r['f1_mean']:.3f}", va="center", fontsize=8,
                    color=INK_SECONDARY)
    ax.set_yticks(range(len(models)), [label(m) for m in models])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("illicit F1 on test steps 35-49 (higher is better)")
    ax.set_title("Test illicit F1 by model and feature set")
    ax.legend(title="feature set", loc="lower left")
    return save(fig, figures_dir(cfg) / "model_comparison_f1.png")


def step_means(per_step: pd.DataFrame, experiment: str, model: str, features: str, column: str):
    sub = per_step[(per_step["experiment"] == experiment) & (per_step["model"] == model)
                   & (per_step["features"] == features) & (per_step["split"] == "test")]
    return sub.groupby("time_step")[column].mean()


def plot_per_time_step(cfg: dict, per_step: pd.DataFrame, gnn: dict[str, str]):
    """Illicit F1 for each test time step, with the t=43 shutdown marked.

    Top panels: four models per feature set (two trees, the MLP, the best GNN).
    Bottom strip: how many illicit transactions each step contains. It is a
    separate panel, not a second y-axis, and it matters for reading the top:
    a step with 2 illicit transactions gives a very noisy F1.
    """
    apply_style()
    feature_sets = cfg["feature_sets"]
    fig, axes = plt.subplots(
        2, len(feature_sets), figsize=(6.2 * len(feature_sets), 5.8), sharex=True, sharey="row",
        gridspec_kw={"height_ratios": [4, 1.1]}, squeeze=False,
    )
    for col, features in enumerate(feature_sets):
        ax = axes[0, col]
        for model in ("rf", "lgbm", "mlp", gnn[features]):
            series = step_means(per_step, "main", model, features, "f1")
            if series.empty:
                continue
            ax.plot(series.index, series.values, color=MODEL_COLORS[model], marker="o",
                    markeredgecolor=SURFACE, markeredgewidth=1.2,
                    label=label(model) + (" (best GNN)" if model == gnn[features] else ""))
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"Illicit F1 per test time step — {features}")
        mark_shutdown(ax, label=(col == 0))
        ax.legend(loc="lower left", fontsize=8)
        if col == 0:
            ax.set_ylabel("illicit F1")

        strip = axes[1, col]
        counts = per_step[(per_step["experiment"] == "main") & (per_step["split"] == "test")] \
            .groupby("time_step")["n_illicit"].first()
        strip.bar(counts.index, counts.values, width=0.6, color=CLASS_COLORS["illicit"])
        for step, n in counts.items():
            strip.text(step, n, str(int(n)), ha="center", va="bottom", fontsize=7, color=INK_SECONDARY)
        strip.set_ylim(0, counts.max() * 1.35)
        strip.set_xlabel("time step")
        strip.set_xticks(list(counts.index))
        mark_shutdown(strip, label=False)
        if col == 0:
            strip.set_ylabel("illicit txs")
    fig.align_ylabels(axes[:, 0])
    return save(fig, figures_dir(cfg) / "per_time_step_f1.png")


def plot_alert_budget(cfg: dict, per_step: pd.DataFrame, gnn: str):
    """precision@K and recall@K per test step for the strongest tree and the best GNN (AF)."""
    apply_style()
    k = cfg["alert_k"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharex=True)
    for ax, column, title in ((axes[0], "precision_at_k", f"Precision@{k}: share of the top {k} alerts that are illicit"),
                              (axes[1], "recall_at_k", f"Recall@{k}: share of all illicit caught in the top {k}")):
        for model in ("rf", "lgbm", gnn):
            series = step_means(per_step, "main", model, "AF", column)
            if series.empty:
                continue
            ax.plot(series.index, series.values, color=MODEL_COLORS[model], marker="o",
                    markeredgecolor=SURFACE, markeredgewidth=1.2, label=label(model))
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("time step")
        mark_shutdown(ax, label=(ax is axes[0]))
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle(f"Analyst alert budget on AF: review the {k} highest-scored transactions of each time step",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return save(fig, figures_dir(cfg) / "alert_budget.png")


def plot_rolling(cfg: dict, per_step: pd.DataFrame):
    """Static model versus rolling retraining, per test time step."""
    section = cfg.get("rolling", {})
    models = [m for m in section.get("models", [])
              if not step_means(per_step, "rolling", m, section["features"], "f1").empty]
    if not models:
        return None
    apply_style()
    fig, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 3.9), sharey=True, squeeze=False)
    for ax, model in zip(axes[0], models):
        static = step_means(per_step, "main", model, section["features"], "f1")
        rolling = step_means(per_step, "rolling", model, section["features"], "f1")
        ax.plot(static.index, static.values, color=INK_MUTED, marker="o", markeredgecolor=SURFACE,
                markeredgewidth=1.2, label="trained once (steps 1-34)")
        ax.plot(rolling.index, rolling.values, color=MODEL_COLORS[model], marker="o",
                markeredgecolor=SURFACE, markeredgewidth=1.2, label="retrained before every step")
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"{label(model)} on {section['features']}")
        ax.set_xlabel("time step")
        ax.set_xticks(list(static.index))
        mark_shutdown(ax, label=(ax is axes[0, 0]))
        ax.legend(loc="lower left", fontsize=8)
    axes[0, 0].set_ylabel("illicit F1")
    return save(fig, figures_dir(cfg) / "rolling_retraining.png")


def plot_pr_curves(cfg: dict, bundle: dict, gnn: str):
    """Precision-recall curves on the test set (first seed, AF features)."""
    apply_style()
    seed = cfg["seeds"][0]
    y_all = bundle["y"].numpy()
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    prevalence = None
    for model in ("rf", "lgbm", "mlp", gnn):
        pred = load_prediction(cfg, "main", model, "AF", gnn_variant(cfg, model), seed)
        if pred is None:
            continue
        idx, score, threshold = pred
        known = y_all[idx] != UNKNOWN
        y, s = y_all[idx][known], score[known]
        precision, recall, _ = precision_recall_curve(y, s)
        ax.plot(recall, precision, color=MODEL_COLORS[model], label=label(model))
        # Mark the operating point: the threshold chosen on validation.
        flagged = s >= threshold
        tp = (flagged & (y == ILLICIT)).sum()
        if flagged.sum():
            ax.scatter(tp / (y == ILLICIT).sum(), tp / flagged.sum(), s=70, color=MODEL_COLORS[model],
                       edgecolor=SURFACE, linewidth=2, zorder=5)
        prevalence = (y == ILLICIT).mean()
    if prevalence is not None:
        ax.axhline(prevalence, color=INK_MUTED, linewidth=1)
        ax.text(0.88, prevalence + 0.012, f"random ranking = illicit share ({prevalence:.3f})",
                ha="right", fontsize=8, color=INK_SECONDARY)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("recall (share of illicit transactions caught)")
    ax.set_ylabel("precision (share of flags that are illicit)")
    ax.set_title("Precision-recall curves, test steps 35-49 (AF)")
    ax.legend(loc="lower left", title="dot = threshold chosen on validation")
    return save(fig, figures_dir(cfg) / "pr_curves.png")


def plot_confusion_matrices(cfg: dict, bundle: dict, gnn: str):
    """Counts of right and wrong decisions at the chosen threshold (first seed, AF)."""
    apply_style()
    seed = cfg["seeds"][0]
    y_all = bundle["y"].numpy()
    wanted = [("main", "rf", "-"), ("main", "lgbm", "-"), ("main", gnn, cfg["gnn"]["direction"]),
              ("hybrid", "rf+oof", gnn)]
    panels = []
    for experiment, model, variant in wanted:
        pred = load_prediction(cfg, experiment, model, "AF", variant, seed)
        if pred is not None:
            panels.append((model, pred))
    if not panels:
        return None
    cmap = LinearSegmentedColormap.from_list("blue", ["#eef4fd", "#cde2fb", "#86b6ef", "#256abf"])
    fig, axes = plt.subplots(1, len(panels), figsize=(3.5 * len(panels), 3.9), squeeze=False)
    for ax, (model, (idx, score, threshold)) in zip(axes[0], panels):
        known = y_all[idx] != UNKNOWN
        y = y_all[idx][known]
        cm = confusion_matrix(y, (score[known] >= threshold).astype(int), labels=[0, 1])
        # Colour by row share (what happened to the licit / to the illicit ones),
        # so the big licit count does not wash out the illicit row.
        ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap=cmap, vmin=0, vmax=1)
        names = [["true negative", "false positive"], ["false negative", "true positive"]]
        for i in range(2):
            for j in range(2):
                share = cm[i, j] / cm[i].sum()
                ax.text(j, i, f"{cm[i, j]:,}\n{names[i][j]}", ha="center", va="center", fontsize=8.5,
                        color="#ffffff" if share > 0.6 else INK)
        ax.set_xticks([0, 1], ["licit", "illicit"])
        ax.set_yticks([0, 1], ["licit", "illicit"])
        ax.set_xlabel("predicted")
        if ax is axes[0, 0]:
            ax.set_ylabel("true label")
        ax.set_title(label(model).replace(" + GNN score (out-of-fold)", "\n+ GNN score (out-of-fold)"),
                     fontsize=10)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
    fig.suptitle("Confusion matrices on test steps 35-49 (AF, first seed)", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return save(fig, figures_dir(cfg) / "confusion_matrices.png")


def plot_tsne(cfg: dict, bundle: dict, gnn: str):
    """t-SNE of the best GNN's embeddings, training period next to test period.

    t-SNE squeezes the 64-128 dimensional embeddings into 2-D while trying to
    keep similar points close together. Read it for cluster structure only:
    distances between far-apart groups and the axes themselves mean nothing.
    """
    from sklearn.manifold import TSNE

    seed = cfg["seeds"][0]
    direction = cfg["gnn"]["direction"]
    path = checkpoint_path(cfg, "main", gnn, "AF", direction, seed, "final")
    if not path.exists():
        return None
    model, _ = load_checkpoint(path)
    device = get_device(cfg)
    model.to(device)
    graphs = build_graph_set(bundle, cfg, "AF", direction)

    rng = np.random.default_rng(seed)
    per_panel = cfg["analysis"]["tsne_sample"] // 2
    embeddings, labels, panel = [], [], []
    for name in ("trainval", "test"):
        graph = graphs[name]
        emb = node_embeddings(model, graph, device)
        candidates = np.flatnonzero(graph.labeled.numpy())
        chosen = rng.choice(candidates, size=min(per_panel, len(candidates)), replace=False)
        embeddings.append(emb[chosen])
        labels.append(graph.y.numpy()[chosen])
        panel += [name] * len(chosen)
    # One t-SNE for both periods, so the two panels share a coordinate system.
    xy = TSNE(n_components=2, perplexity=30, init="pca", random_state=seed).fit_transform(
        np.vstack(embeddings))
    labels, panel = np.concatenate(labels), np.array(panel)

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    titles = {"trainval": "Training period (steps 1-34): the GNN saw these labels",
              "test": "Test period (steps 35-49): never seen"}
    for ax, name in zip(axes, ("trainval", "test")):
        for cls, cls_name in ((0, "licit"), (1, "illicit")):  # illicit drawn last, on top
            keep = (panel == name) & (labels == cls)
            ax.scatter(xy[keep, 0], xy[keep, 1], s=10, color=CLASS_COLORS[cls_name], alpha=0.75,
                       linewidth=0, label=f"{cls_name} ({keep.sum()})")
        ax.set_title(titles[name], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.legend(loc="upper right", markerscale=2)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(f"t-SNE of {label(gnn)} embeddings (AF, random sample of labelled transactions)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return save(fig, figures_dir(cfg) / "tsne_embeddings.png")


# --------------------------------------------------------------------------- #
def refresh_tables(cfg: dict, bundle: dict | None = None):
    """Rebuild summary.csv, period_metrics.csv and tables.md from the metric files."""
    df = load_metrics(cfg)
    per_step = pd.read_csv(results_dir(cfg) / "per_timestep.csv")
    bundle = bundle or load_bundle(cfg)

    summary = summarize(df)
    summary.to_csv(results_dir(cfg) / "summary.csv", index=False)
    periods = period_metrics(cfg, df, bundle)
    periods.to_csv(results_dir(cfg) / "period_metrics.csv", index=False)
    diag = threshold_diagnostic(cfg, df, bundle)
    diag.to_csv(results_dir(cfg) / "threshold_diagnostic.csv", index=False)
    write_tables(cfg, df, summary, periods, per_step, diag)
    log.info("wrote %s", results_dir(cfg) / "tables.md")
    return summary, per_step


def run(cfg: dict) -> None:
    bundle = load_bundle(cfg)
    summary, per_step = refresh_tables(cfg, bundle)

    gnn = {features: best_gnn(cfg, features) for features in cfg["feature_sets"]}
    figures = [
        plot_model_comparison(cfg, summary),
        plot_per_time_step(cfg, per_step, gnn),
        plot_pr_curves(cfg, bundle, gnn["AF"]),
        plot_confusion_matrices(cfg, bundle, gnn["AF"]),
        plot_alert_budget(cfg, per_step, gnn["AF"]),
        plot_rolling(cfg, per_step),
        plot_tsne(cfg, bundle, gnn["AF"]),
    ]
    for path in figures:
        if path is not None:
            log.info("wrote %s", path)


def main() -> None:
    args = config_arg_parser("Build result tables and figures.").parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
