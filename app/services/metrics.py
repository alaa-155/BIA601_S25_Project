from __future__ import annotations

from typing import Any

from app.constants import METRIC_DEFINITIONS
from app.data_store import AppData
from app.services.formatters import format_metric, pct_delta


def build_metric_cards(data: AppData) -> list[dict[str, str]]:
    summary = data.summary
    return [
        {"label": "عدد المستخدمين", "value": f"{summary['users_count']:,}"},
        {"label": "عدد المنتجات", "value": f"{summary['products_count']:,}"},
        {"label": "عدد التقييمات", "value": f"{summary['ratings_count']:,}"},
        {"label": "سجلات التفاعل", "value": f"{summary['interactions_count']:,}"},
    ]


def build_result_cards(data: AppData) -> list[dict[str, Any]]:
    metrics = data.metrics
    entries = [
        (label, float(metrics[baseline_key]), float(metrics[enhanced_key]))
        for label, baseline_key, enhanced_key in METRIC_DEFINITIONS
    ]
    max_value = max(max(baseline, enhanced) for _, baseline, enhanced in entries) or 1.0
    return [
        {
            "label": label,
            "baseline": format_metric(baseline),
            "enhanced": format_metric(enhanced),
            "baseline_pct": round(baseline / max_value * 100, 1),
            "enhanced_pct": round(enhanced / max_value * 100, 1),
            "delta_pct": pct_delta(enhanced, baseline),
        }
        for label, baseline, enhanced in entries
    ]
