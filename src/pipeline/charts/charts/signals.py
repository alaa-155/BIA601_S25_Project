from __future__ import annotations

import matplotlib.pyplot as plt

from pipeline.models import PreparedData
from .style import PALETTE


def create_behavior_chart(prepared: PreparedData, chart_dir) -> None:
    behavior = prepared.interactions[["viewed", "clicked", "purchased"]].sum()
    labels = ["Viewed", "Clicked", "Purchased"]
    values = [int(behavior["viewed"]), int(behavior["clicked"]), int(behavior["purchased"])]
    max_value = max(values)
    left_offsets = [(max_value - value) / 2 for value in values]
    colors = [PALETTE["bronze"], PALETTE["bronze_dark"], PALETTE["forest"]]

    fig, ax = plt.subplots(figsize=(8.2, 4.6), facecolor=PALETTE["paper"])
    bars = ax.barh(labels, values, left=left_offsets, color=colors, height=0.62)
    ax.set_facecolor(PALETTE["paper"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.tick_params(length=0)
    ax.set_title("Behavior signal funnel", pad=14)
    ax.invert_yaxis()
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_y() + bar.get_height() / 2,
            f"{value:,}",
            va="center",
            ha="center",
            color="white",
            fontweight="bold",
        )
    fig.savefig(chart_dir / "behavior_funnel.png", facecolor=PALETTE["paper"])
    plt.close(fig)
