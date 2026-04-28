from __future__ import annotations

from typing import Any

from app.data_store import AppData
from app.services.formatters import display_category, display_country, recommendation_record


def _user_rows(data: AppData, user_id: int, limit: int) -> list[dict[str, Any]]:
    return (
        data.recommendations[data.recommendations["user_id"] == user_id]
        .sort_values("rank")
        .head(limit)
        .to_dict(orient="records")
    )


def preview_cards(data: AppData, user_id: int, limit: int = 3) -> list[dict[str, Any]]:
    return [recommendation_record(row) for row in _user_rows(data, user_id, limit)]


def get_user_payload(data: AppData, user_id: int) -> dict[str, Any] | None:
    if user_id not in data.user_lookup.index:
        return None

    user_row = data.user_lookup.loc[user_id].to_dict()
    rows = _user_rows(data, user_id, 5)
    categories = []
    for category in [row["category"] for row in rows]:
        label = display_category(category)
        if label not in categories:
            categories.append(label)
        if len(categories) == 3:
            break

    return {
        "profile": {
            "user_id": int(user_id),
            "age": int(user_row["age"]),
            "country": display_country(user_row["country"]),
        },
        "categories": categories,
        "recommendations": [recommendation_record(row) for row in rows],
    }
