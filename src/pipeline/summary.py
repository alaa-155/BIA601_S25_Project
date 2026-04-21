from __future__ import annotations
import json
from pipeline.config import PROCESSED
from pipeline.models import PreparedData


def export_summary(prepared: PreparedData, metrics: dict[str, float]) -> None:
    behavior = prepared.interactions[["viewed", "clicked", "purchased"]].copy()
    summary = {
        "users_count": int(prepared.users.shape[0]),
        "products_count": int(prepared.products.shape[0]),
        "ratings_count": int((prepared.interactions["rating"] > 0).sum()),
        "interactions_count": int(prepared.interactions.shape[0]),
        "countries": prepared.users["country"].value_counts().to_dict(),
        "categories": prepared.products["category"].value_counts().to_dict(),
        "price_min": float(prepared.products["price"].min()),
        "price_max": float(prepared.products["price"].max()),
        "price_mean": round(float(prepared.products["price"].mean()), 2),
        "clicked_without_viewed": int(((behavior["clicked"] == 1) & (behavior["viewed"] == 0)).sum()),
        "purchased_without_clicked": int(((behavior["purchased"] == 1) & (behavior["clicked"] == 0)).sum()),
        "evaluation_metrics": metrics,
        "paper": {
            "title": "E-commerce recommender system based on improved K-means commodity information management model",
            "authors": "Wei Zhang, Zonghua Wu",
            "journal": "Heliyon (2024)",
            "doi": "https://doi.org/10.1016/j.heliyon.2024.e29045",
        },
    }
    with open(PROCESSED / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
