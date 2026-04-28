from __future__ import annotations

import matplotlib.pyplot as plt

PALETTE = {
    "bronze": "#b88755",
    "bronze_dark": "#8f633e",
    "forest": "#395347",
    "paper": "#f7f1e7",
    "line": "#d7c9b8",
    "grid": "#e8ddd0",
    "text": "#231d19",
    "muted": "#6a6258",
}


def set_plot_defaults() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 170,
            "savefig.bbox": "tight",
            "font.size": 10,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "axes.edgecolor": PALETTE["line"],
            "axes.linewidth": 0.8,
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "text.color": PALETTE["text"],
        }
    )


def style_axes(ax, xlabel: str | None = None, ylabel: str | None = None, grid_axis: str = "x") -> None:
    ax.set_facecolor(PALETTE["paper"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["line"])
    ax.spines["bottom"].set_color(PALETTE["line"])
    ax.grid(axis=grid_axis, color=PALETTE["grid"], linestyle="--", linewidth=0.8)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=PALETTE["muted"])
    if ylabel:
        ax.set_ylabel(ylabel, color=PALETTE["muted"])
    ax.tick_params(length=0)
