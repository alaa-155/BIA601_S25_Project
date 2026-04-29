from __future__ import annotations

from typing import Any

from app.data_store import AppData
from app.services.formatters import display_category, display_country, recommendation_record

# Get the top recommendation rows for a specific user.
# The rows are sorted by rank and limited to the requested number.
def _user_rows(data: AppData, user_id: int, limit: int) -> list[dict[str, Any]]:
    return (
        data.recommendations[data.recommendations["user_id"] == user_id]
        .sort_values("rank")
        .head(limit)
        .to_dict(orient="records")
    )

# Prepare a small recommendation preview for the home page.
def preview_cards(data: AppData, user_id: int, limit: int = 3) -> list[dict[str, Any]]:
    return [recommendation_record(row) for row in _user_rows(data, user_id, limit)]

# Prepare all data needed for the recommendations page.
# The page shows user information, top categories, and final recommended products.
def get_user_payload(data: AppData, user_id: int) -> dict[str, Any] | None:
    if user_id not in data.user_lookup.index:
        return None
        
    # Get the selected user's profile information.
    user_row = data.user_lookup.loc[user_id].to_dict()
    # Get the final recommendation rows for the selected user.
    rows = _user_rows(data, user_id, 5)
    # Extract the most common categories from the user's recommendation list.
    categories = []
    for category in [row["category"] for row in rows]:
        label = display_category(category)
        if label not in categories:
            categories.append(label)
        if len(categories) == 3:
            break
   # Return a complete payload that the HTML template can render directly.
    return {
        "profile": {
            "user_id": int(user_id),
            "age": int(user_row["age"]),
            "country": display_country(user_row["country"]),
        },
        "categories": categories,
        "recommendations": [recommendation_record(row) for row in rows],
    }
