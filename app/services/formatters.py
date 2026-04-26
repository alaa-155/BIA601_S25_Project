from __future__ import annotations

from typing import Any

from app.constants import CATEGORY_DISPLAY, COUNTRY_DISPLAY, REASON_DISPLAY


def format_metric(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


def pct_delta(new_value: float, old_value: float) -> float:
    if not old_value:
        return 0.0
    return round(((new_value - old_value) / old_value) * 100.0, 1)


def display_category(value: str) -> str:
    return CATEGORY_DISPLAY.get(value, value)


def display_country(value: str) -> str:
    return COUNTRY_DISPLAY.get(value, value)


def display_reason(value: str) -> str:
    return REASON_DISPLAY.get(value, value)


def recommendation_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": int(row["rank"]),
        "product_id": int(row["product_id"]),
        "category": display_category(row["category"]),
        "price": f"{int(round(float(row['price']))):,}",
        "reason": display_reason(row["reason"]),
    }
