from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .style import PALETTE, style_axes


def create_metrics_chart(metrics: dict[str, float], chart_dir) -> None:
    labels = ["Recall@10", "NDCG@10", "Diversity@10"]
    baseline = np.array([metrics["baseline_recall_at_10"], metrics["baseline_ndcg_at_10"], metrics["baseline_diversity_at_10"]])
    enhanced = np.array([metrics["ga_recall_at_10"], metrics["ga_ndcg_at_10"], metrics["ga_diversity_at_10"]])
    x = np.arange(len(labels))
    width = 0.32

    fig, ax = plt.subplots(figsize=(8.2, 4.8), facecolor=PALETTE["paper"])
    bars1 = ax.bar(x - width / 2, baseline, width, label="Baseline hybrid", color=PALETTE["bronze"], alpha=0.9)
    bars2 = ax.bar(x + width / 2, enhanced, width, label="Hybrid + GA", color=PALETTE["forest"], alpha=0.95)
    style_axes(ax, ylabel="Score", grid_axis="y")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Offline evaluation comparison", pad=14)
    ax.set_ylim(0, float(max(max(baseline), max(enhanced))) * 1.25)
    ax.legend(frameon=False, loc="upper left")
    for bars in (bars1, bars2):
        _add_labels(ax, bars)
    fig.savefig(chart_dir / "metrics_comparison.png", facecolor=PALETTE["paper"])
    plt.close(fig)


def _add_labels(ax, bars) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + ax.get_ylim()[1] * 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            color=PALETTE["text"],
            fontweight="bold",
            fontsize=9,
        )
