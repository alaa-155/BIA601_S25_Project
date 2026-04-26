from __future__ import annotations

from typing import Any

from app.constants import DEFAULT_USER_ID
from app.data_store import AppData
from app.services.metrics import build_metric_cards, build_result_cards
from app.services.recommendations import get_user_payload, preview_cards


def home_context(data: AppData) -> dict[str, Any]:
    featured_user = int(data.users["user_id"].iloc[0])
    return {
        "featured_user": featured_user,
        "metric_cards": build_metric_cards(data),
        "result_cards": build_result_cards(data),
        "preview_cards": preview_cards(data, featured_user),
        "summary": data.summary,
        "page_name": "home",
    }


def recommendations_context(data: AppData, user_id: int) -> dict[str, Any]:
    payload = get_user_payload(data, user_id)
    active_user_id = user_id
    if payload is None:
        active_user_id = DEFAULT_USER_ID
        payload = get_user_payload(data, active_user_id)
    return {
        "user_id": active_user_id,
        "payload": payload,
        "summary": data.summary,
        "page_name": "recommendations",
    }
