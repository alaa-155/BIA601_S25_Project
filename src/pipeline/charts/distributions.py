from __future__ import annotations

import matplotlib.pyplot as plt

from pipeline.models import PreparedData
from .style import PALETTE, style_axes


def create_distribution_charts(prepared: PreparedData, chart_dir) -> None:
    _save_horizontal_bars(
        prepared.products["category"].value_counts().sort_values(ascending=True),
        chart_dir / "category_distribution.png",
        PALETTE["forest"],
        "Product category distribution",
        "Number of products",
    )
    _save_horizontal_bars(
        prepared.users["country"].value_counts().sort_values(ascending=True),
        chart_dir / "country_distribution.png",
        PALETTE["bronze"],
        "User distribution by country",
        "Number of users",
    )


def _save_horizontal_bars(series, output_path, color: str, title: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.6), facecolor=PALETTE["paper"])
    bars = ax.barh(series.index, series.values, color=color, alpha=0.92, height=0.62)
    style_axes(ax, xlabel=xlabel)
    ax.set_title(title, pad=14)
    ax.set_xlim(0, max(series.values) * 1.18)
    for bar, value in zip(bars, series.values):
        ax.text(
            bar.get_width() + max(series.values) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{int(value)}",
            va="center",
            ha="left",
            color=PALETTE["text"],
            fontweight="bold",
        )
    fig.savefig(output_path, facecolor=PALETTE["paper"])
    plt.close(fig)
