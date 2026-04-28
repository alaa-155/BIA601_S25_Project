from __future__ import annotations

from pipeline.config import PROCESSED
from pipeline.models import PreparedData
from .distributions import create_distribution_charts
from .metrics_plot import create_metrics_chart
from .signals import create_behavior_chart
from .style import set_plot_defaults


def create_charts(prepared: PreparedData, metrics: dict[str, float]) -> None:
    set_plot_defaults()
    chart_dir = PROCESSED / "charts"
    chart_dir.mkdir(exist_ok=True)
    create_distribution_charts(prepared, chart_dir)
    create_behavior_chart(prepared, chart_dir)
    create_metrics_chart(metrics, chart_dir)
