"""One shared look for every figure, so all charts read as a set.

Colour rules used across the project:
    * a class always has the same colour: illicit = red, licit = blue,
      unknown = grey (grey because it carries no label information);
    * models are coloured from a fixed, colour-blind-checked order
      (SERIES_COLORS), and a model keeps its colour in every chart;
    * text is always dark ink, never the series colour;
    * grid lines are thin, solid and pale so the data stays the loudest thing.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files; no window needed (works on servers too)
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

CLASS_COLORS = {"illicit": "#e34948", "licit": "#2a78d6", "unknown": "#c3c2b7"}

# Fixed categorical order (adjacent colours stay distinguishable under the
# common forms of colour blindness). Never cycle past the end of this list.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

# A model keeps the same colour wherever it appears.
MODEL_COLORS = {
    "rf": SERIES_COLORS[0],
    "lgbm": SERIES_COLORS[1],
    "mlp": SERIES_COLORS[2],
    "logreg": SERIES_COLORS[3],
    "gcn": SERIES_COLORS[4],
    "skip_gcn": SERIES_COLORS[5],
    "sage": SERIES_COLORS[6],
    "gat": SERIES_COLORS[7],
}

MODEL_LABELS = {
    "logreg": "Logistic Regression",
    "rf": "Random Forest",
    "lgbm": "LightGBM",
    "mlp": "MLP",
    "gcn": "GCN",
    "skip_gcn": "Skip-GCN",
    "sage": "GraphSAGE",
    "gat": "GAT",
    "rf+emb": "Random Forest + GNN emb.",
    "lgbm+emb": "LightGBM + GNN emb.",
}

# The dark-market shutdown: from this time step on, every model's recall collapses.
SHUTDOWN_STEP = 43


def apply_style() -> None:
    """Set matplotlib defaults once; call at the top of every plotting function."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "ytick.left": False,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "lines.markersize": 6,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def mark_shutdown(ax, label: bool = True) -> None:
    """Draw the vertical marker for the dark-market shutdown at t = 43."""
    ax.axvline(SHUTDOWN_STEP, color=INK_MUTED, linewidth=1.0, zorder=1)
    if label:
        ax.annotate(
            "dark market\nshut down (t=43)",
            xy=(SHUTDOWN_STEP, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(4, -2),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=8,
            color=INK_SECONDARY,
        )


def save(fig, path: Path) -> Path:
    """Save a figure as PNG and close it (keeps memory flat in long runs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
