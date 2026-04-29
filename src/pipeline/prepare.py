from __future__ import annotations

import pandas as pd

from pipeline.config import RAW
from pipeline.models import PreparedData


def _load_raw_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    users = pd.read_excel(RAW / "users.xlsx")
    products = pd.read_excel(RAW / "products.xlsx")
    ratings = pd.read_excel(RAW / "ratings.xlsx")
    behavior = pd.read_excel(RAW / "behavior_15500.xlsx")
    return users, products, ratings, behavior


def _normalize_columns(
    users: pd.DataFrame,
    products: pd.DataFrame,
    ratings: pd.DataFrame,
    behavior: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    users = users.rename(columns={"id_user": "user_id"})
    products = products.rename(columns={"id_product": "product_id"})
    ratings = ratings.rename(columns={"id_user": "user_id", "id_product": "product_id"})
    behavior = behavior.rename(columns={"id_user": "user_id", "id_product": "product_id"})
    return users, products, ratings, behavior


def _aggregate_interactions(ratings: pd.DataFrame, behavior: pd.DataFrame) -> pd.DataFrame:
    ratings_agg = ratings.groupby(["user_id", "product_id"], as_index=False)["rating"].mean()
    behavior_agg = behavior.groupby(["user_id", "product_id"], as_index=False)[["viewed", "clicked", "purchased"]].max()
    interactions = behavior_agg.merge(ratings_agg, on=["user_id", "product_id"], how="outer").fillna(
        {
            "viewed": 0,
            "clicked": 0,
            "purchased": 0,
            "rating": 0,
        }
    )
    interactions["implicit_strength"] = (
        0.15 * interactions["viewed"]
        + 0.35 * interactions["clicked"]
        + 0.80 * interactions["purchased"]
        + 0.55 * (interactions["rating"] / 5.0)
    )
    interactions["positive"] = (
        (interactions["purchased"] > 0)
        | (interactions["clicked"] > 0)
        | (interactions["rating"] >= 4)
    ).astype(int)
    return interactions


def load_and_prepare() -> PreparedData:
    users, products, ratings, behavior = _normalize_columns(*_load_raw_frames())

    users = users[["user_id", "age", "country"]].copy()
    products = products[["product_id", "category", "price"]].copy().sort_values("product_id")
    interactions = _aggregate_interactions(ratings, behavior)

    user_ids = sorted(users["user_id"].astype(int).tolist())
    product_ids = products["product_id"].astype(int).to_numpy()
    u2i = {user_id: index for index, user_id in enumerate(user_ids)}
    p2i = {product_id: index for index, product_id in enumerate(product_ids)}

    return PreparedData(users, products, interactions, user_ids, product_ids, u2i, p2i)
