"""Exploratory data analysis: look at the data before modelling it.

Produces three figures in results/figures/ and a small table of numbers in
results/dataset_stats.json (the README and the app read the numbers from
there, so nothing is typed in by hand):

    eda_labels_per_time_step.png   how many illicit / licit / unknown per step
    eda_class_balance.png          overall label distribution
    eda_degree_distribution.png    how many neighbours transactions have

    python -m btc_aml.eda --config configs/quick.yaml
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .data import ILLICIT, LICIT, UNKNOWN, load_bundle
from .plotting import CLASS_COLORS, INK_SECONDARY, SURFACE, apply_style, mark_shutdown, save
from .splits import split_masks
from .utils import config_arg_parser, figures_dir, get_logger, load_config, results_dir, save_json

log = get_logger()


def label_counts_per_step(bundle: dict) -> pd.DataFrame:
    """One row per time step with the number of illicit / licit / unknown nodes."""
    df = pd.DataFrame({"time_step": bundle["time_step"].numpy(), "y": bundle["y"].numpy()})
    counts = (
        df.groupby("time_step")["y"]
        .value_counts()
        .unstack(fill_value=0)
        .rename(columns={ILLICIT: "illicit", LICIT: "licit", UNKNOWN: "unknown"})
    )[["illicit", "licit", "unknown"]]
    counts["labeled"] = counts["illicit"] + counts["licit"]
    # Share of illicit among LABELLED transactions (unknown ones tell us nothing).
    counts["illicit_share"] = counts["illicit"] / counts["labeled"]
    return counts.reset_index()


def plot_labels_per_step(counts: pd.DataFrame, cfg: dict):
    """Two stacked panels sharing the time axis: counts, then illicit share.

    Two panels rather than two y-axes on one plot: a count and a percentage
    have unrelated scales, and overlaying them would suggest a relationship
    that depends only on how the axes happen to be lined up.
    """
    apply_style()
    fig, (ax_n, ax_share) = plt.subplots(
        2, 1, figsize=(10, 6.2), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
    )
    steps = counts["time_step"].to_numpy()

    bottom = np.zeros(len(counts))
    for name in ("illicit", "licit", "unknown"):
        ax_n.bar(
            steps, counts[name], bottom=bottom, width=0.72, color=CLASS_COLORS[name],
            label=name, edgecolor=SURFACE, linewidth=0.8,
        )
        bottom += counts[name].to_numpy()
    ax_n.set_ylabel("transactions")
    ax_n.set_title("Transactions per time step, by label")
    # Legend sits on the title row so it cannot collide with the split labels.
    ax_n.legend(ncol=3, loc="lower right", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.2)

    ax_share.plot(steps, 100 * counts["illicit_share"], color=CLASS_COLORS["illicit"], marker="o",
                  markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax_share.set_ylabel("% illicit among labelled")
    ax_share.set_xlabel("time step (about two weeks each)")
    ax_share.set_title("Illicit share of labelled transactions")
    ax_share.set_ylim(bottom=0)
    mark_shutdown(ax_share)

    # Shade the validation and test periods so the split is visible on the data.
    for ax in (ax_n, ax_share):
        for name, shade in (("val", 0.05), ("test", 0.10)):
            lo, hi = cfg["split"][name]
            ax.axvspan(lo - 0.5, hi + 0.5, color=INK_SECONDARY, alpha=shade, linewidth=0, zorder=0)
        ax.set_xlim(0.2, 49.8)
    y_top = ax_n.get_ylim()[1]
    for name in ("train", "val", "test"):
        lo, hi = cfg["split"][name]
        ax_n.text((lo + hi) / 2, y_top * 0.97, name, ha="center", va="top",
                  fontsize=9, color=INK_SECONDARY)
    ax_n.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40, 43, 45, 49])
    return save(fig, figures_dir(cfg) / "eda_labels_per_time_step.png")


def plot_class_balance(bundle: dict, cfg: dict):
    """Horizontal bars with the count written at the end of each bar."""
    apply_style()
    y = bundle["y"].numpy()
    names = ["illicit", "licit", "unknown"]
    values = [int((y == ILLICIT).sum()), int((y == LICIT).sum()), int((y == UNKNOWN).sum())]

    fig, ax = plt.subplots(figsize=(8, 2.6))
    ax.barh(names, values, color=[CLASS_COLORS[n] for n in names], height=0.5)
    for name, value in zip(names, values):
        ax.text(value, name, f"  {value:,}  ({100 * value / len(y):.1f}%)", va="center",
                ha="left", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.3)
    ax.grid(False)
    ax.xaxis.set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_title(f"Class balance over all {len(y):,} transactions")
    return save(fig, figures_dir(cfg) / "eda_class_balance.png")


def degree_table(bundle: dict) -> pd.DataFrame:
    """In-degree, out-degree and total degree of every node.

    For an edge A -> B (an output of A is spent by B):
        out-degree of A = how many later transactions spend A's outputs
        in-degree of B  = how many earlier transactions fund B
    """
    n = bundle["x"].shape[0]
    src, dst = bundle["edge_index"].numpy()
    out_deg = np.bincount(src, minlength=n)
    in_deg = np.bincount(dst, minlength=n)
    return pd.DataFrame({"in": in_deg, "out": out_deg, "total": in_deg + out_deg})


def plot_degree_distribution(degrees: pd.DataFrame, cfg: dict):
    """Log-log plot of how many nodes have each degree.

    Both axes are logarithmic because degree is heavy-tailed: most nodes have
    one or two neighbours, a handful have hundreds. On linear axes the plot
    would be a single spike at the left edge.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    styles = {"in": ("#2a78d6", "o", "in-degree (transactions it spends from)"),
              "out": ("#eb6834", "s", "out-degree (transactions that spend it)")}
    for kind, (color, marker, label) in styles.items():
        values, counts = np.unique(degrees[kind][degrees[kind] > 0], return_counts=True)
        ax.scatter(values, counts, s=22, color=color, marker=marker, label=label,
                   edgecolor=SURFACE, linewidth=0.6, alpha=0.9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("degree (number of neighbours)")
    ax.set_ylabel("number of transactions")
    ax.set_title("Degree distribution (log-log)")
    ax.legend(loc="upper right")
    return save(fig, figures_dir(cfg) / "eda_degree_distribution.png")


def dataset_stats(bundle: dict, cfg: dict, counts: pd.DataFrame, degrees: pd.DataFrame) -> dict:
    """Numbers quoted in the README / explainer / app, computed rather than typed."""
    y = bundle["y"]
    n_illicit, n_licit = int((y == ILLICIT).sum()), int((y == LICIT).sum())
    stats = {
        "num_nodes": int(y.numel()),
        "num_edges": int(bundle["edge_index"].shape[1]),
        "num_illicit": n_illicit,
        "num_licit": n_licit,
        "num_unknown": int((y == UNKNOWN).sum()),
        "illicit_share_of_labeled": n_illicit / (n_illicit + n_licit),
        "mean_total_degree": float(degrees["total"].mean()),
        "median_total_degree": float(degrees["total"].median()),
        "max_in_degree": int(degrees["in"].max()),
        "max_out_degree": int(degrees["out"].max()),
        "nodes_per_step_min": int(counts[["illicit", "licit", "unknown"]].sum(axis=1).min()),
        "nodes_per_step_max": int(counts[["illicit", "licit", "unknown"]].sum(axis=1).max()),
        "splits": {},
    }
    masks = split_masks(bundle, cfg)
    for name in ("train", "val", "test", "trainval"):
        m = masks[name]
        illicit, licit = int((y[m] == ILLICIT).sum()), int((y[m] == LICIT).sum())
        stats["splits"][name] = {
            "nodes": int(m.sum()),
            "illicit": illicit,
            "licit": licit,
            "unknown": int((y[m] == UNKNOWN).sum()),
            "illicit_share_of_labeled": illicit / (illicit + licit),
            # What "predict licit for everything" scores on this split:
            "all_licit_accuracy": licit / (illicit + licit),
            "all_licit_illicit_f1": 0.0,
        }
    return stats


def run(cfg: dict) -> None:
    bundle = load_bundle(cfg)
    counts = label_counts_per_step(bundle)
    degrees = degree_table(bundle)

    counts.to_csv(results_dir(cfg) / "labels_per_time_step.csv", index=False)
    save_json(dataset_stats(bundle, cfg, counts, degrees), results_dir(cfg) / "dataset_stats.json")
    for path in (
        plot_labels_per_step(counts, cfg),
        plot_class_balance(bundle, cfg),
        plot_degree_distribution(degrees, cfg),
    ):
        log.info("wrote %s", path)


def main() -> None:
    args = config_arg_parser("Exploratory figures and dataset statistics.").parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
