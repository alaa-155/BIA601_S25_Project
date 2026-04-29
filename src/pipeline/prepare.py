from __future__ import annotations
import pandas as pd
from pipeline.config import RAW
from pipeline.models import PreparedData

# Load the original Excel files provided with the assignment.
# These files contain users, products, ratings, and behavior data.
def _load_raw_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    users = pd.read_excel(RAW / "users.xlsx")
    products = pd.read_excel(RAW / "products.xlsx")
    ratings = pd.read_excel(RAW / "ratings.xlsx")
    behavior = pd.read_excel(RAW / "behavior_15500.xlsx")
    return users, products, ratings, behavior

# Standardize column names across all datasets.
# This ensures that users, products, ratings, and behavior can be merged
# using the same keys: user_id and product_id.
def _normalize_columns(
    users: pd.DataFrame,
    products: pd.DataFrame,
    ratings: pd.DataFrame,
    behavior: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Rename assignment-specific IDs into common names used by the pipeline.
    users = users.rename(columns={"id_user": "user_id"})
    products = products.rename(columns={"id_product": "product_id"})
    ratings = ratings.rename(columns={"id_user": "user_id", "id_product": "product_id"})
    behavior = behavior.rename(columns={"id_user": "user_id", "id_product": "product_id"})
    return users, products, ratings, behavior

# Combine explicit ratings and implicit user behavior into one interaction table.
# Ratings represent direct feedback, while views, clicks, and purchases
# represent behavioral signals inside the e-commerce system.
def _aggregate_interactions(ratings: pd.DataFrame, behavior: pd.DataFrame) -> pd.DataFrame:
    # Average duplicated ratings for the same user-product pair.
    ratings_agg = ratings.groupby(["user_id", "product_id"], as_index=False)["rating"].mean()
    # Merge duplicated behavior rows by keeping whether the action happened at least once.
    behavior_agg = behavior.groupby(["user_id", "product_id"], as_index=False)[["viewed", "clicked", "purchased"]].max()
    # Combine rating and behavior data so every observed interaction is kept.
    # Missing values are replaced with zero because not every product has both
    # rating and behavior information.
    interactions = behavior_agg.merge(ratings_agg, on=["user_id", "product_id"], how="outer").fillna(
        {
            "viewed": 0,
            "clicked": 0,
            "purchased": 0,
            "rating": 0,
        }
    )
    # Convert user actions into one numerical interaction strength.
    # Purchases receive the highest weight, followed by ratings, clicks, and views.
    interactions["implicit_strength"] = (
        0.15 * interactions["viewed"]
        + 0.35 * interactions["clicked"]
        + 0.80 * interactions["purchased"]
        + 0.55 * (interactions["rating"] / 5.0)
    )
    # Mark useful interactions as positive examples for evaluation.
    # A product is considered positive if it was purchased, clicked, or highly rated.
    interactions["positive"] = (
        (interactions["purchased"] > 0)
        | (interactions["clicked"] > 0)
        | (interactions["rating"] >= 4)
    ).astype(int)
    return interactions

# Run the full data preparation process.
# This function loads raw data, normalizes columns, builds the interaction table,
# and creates index mappings used later by the recommender system.
def load_and_prepare() -> PreparedData:
    # Load the raw Excel files and normalize their column names.
    users, products, ratings, behavior = _normalize_columns(*_load_raw_frames())
   
    # Keep only the columns needed by the recommendation pipeline.
    users = users[["user_id", "age", "country"]].copy()
    products = products[["product_id", "category", "price"]].copy().sort_values("product_id")
    # Build the unified interaction table used for recommendation and evaluation.
    interactions = _aggregate_interactions(ratings, behavior)
    
    # Create ordered user and product ID lists.
    # These lists keep the matrix representation consistent across the pipeline.
    user_ids = sorted(users["user_id"].astype(int).tolist())
    product_ids = products["product_id"].astype(int).to_numpy()
    # Build lookup dictionaries that convert real IDs into matrix indexes.
    u2i = {user_id: index for index, user_id in enumerate(user_ids)}
    p2i = {product_id: index for index, product_id in enumerate(product_ids)}
    # Return all prepared data in one structured object for the next pipeline stages.
    return PreparedData(users, products, interactions, user_ids, product_ids, u2i, p2i)
