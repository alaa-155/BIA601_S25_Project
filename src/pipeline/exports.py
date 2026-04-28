from __future__ import annotations
import json
import pandas as pd

from pipeline.config import PROCESSED
from pipeline.models import PreparedData


def export_metrics(metrics: dict[str, float]) -> None:
    with open(PROCESSED / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)


def export_csvs(prepared: PreparedData, recommendations: pd.DataFrame) -> None:
    prepared.users.to_csv(PROCESSED / "users_clean.csv", index=False)
    prepared.products.to_csv(PROCESSED / "products_clean.csv", index=False)
    prepared.interactions.to_csv(PROCESSED / "interactions_clean.csv", index=False)
    recommendations.to_csv(PROCESSED / "user_recommendations.csv", index=False)
