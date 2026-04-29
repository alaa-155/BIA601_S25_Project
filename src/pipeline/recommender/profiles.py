from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from pipeline.config import RANDOM_SEED

# Build user and product profiles from the prepared interaction data.
# The profiles summarize behavior patterns, category preferences,
# price preferences, product popularity, and already-seen products.
def build_profiles(self, interactions: pd.DataFrame) -> dict[str, object]:
    # Create a sparse user-product matrix where each cell stores
    # the interaction strength between one user and one product.
    sparse_matrix = csr_matrix(
        (
            interactions["implicit_strength"].to_numpy(dtype=float),
            (
                interactions["user_id"].map(self.prepared.u2i).to_numpy(),
                interactions["product_id"].map(self.prepared.p2i).to_numpy(),
            ),
        ),
        shape=(len(self.prepared.user_ids), len(self.prepared.product_ids)),
    )
    # Use SVD to discover hidden behavioral patterns between users and products.
    # This helps the system recommend products beyond direct category matching.
    svd = TruncatedSVD(n_components=20, random_state=RANDOM_SEED)
    user_factors = svd.fit_transform(sparse_matrix)
    item_factors = svd.components_.T
    latent_scores = user_factors @ item_factors.T
    # Calculate product popularity based on total interaction strength.
    popularity = (
        interactions.groupby("product_id")["implicit_strength"]
        .sum()
        .reindex(self.prepared.product_ids)
        .fillna(0)
        .to_numpy(dtype=float)
    )
    popularity = (popularity - popularity.min()) / (popularity.max() - popularity.min() + 1e-9)
    # Attach product category and price to each interaction.
    interactions_meta = interactions.merge(self.products[["product_id", "category", "price"]], on="product_id", how="left")
    # Prepare containers for each user's category preference,
    # preferred price level, and previously seen products.
    user_cat_pref = np.zeros((len(self.prepared.user_ids), len(self.categories)), dtype=float)
    user_price_pref = np.full(len(self.prepared.user_ids), self.products["price"].mean(), dtype=float)
    seen = {uid: set() for uid in self.prepared.user_ids}
    
    # Build a compact profile for each user based on their interaction history.
    for user_id, group in interactions_meta.groupby("user_id"):
        user_index = self.prepared.u2i[int(user_id)]
        seen[int(user_id)] = set(group["product_id"].astype(int).tolist())
        # Estimate how much the user prefers each product category.
        cat_weights = group.groupby("category")["implicit_strength"].sum()
        if cat_weights.sum() > 0:
            cat_weights = cat_weights / cat_weights.sum()
            for category, value in cat_weights.items():
                user_cat_pref[user_index, self.cat_to_idx[category]] = float(value)             
        # Estimate the user's preferred price using weighted average price.
        weights = group["implicit_strength"].to_numpy(dtype=float)
        if weights.sum() > 0:
            user_price_pref[user_index] = float(np.average(group["price"].to_numpy(dtype=float), weights=weights))

    return {
        "latent_scores": latent_scores,
        "popularity": popularity,
        "user_cat_pref": user_cat_pref,
        "user_price_pref": user_price_pref,
        "seen": seen,
    }

# Calculate recommendation scores for one user.
# The final score combines hidden behavior, category preference,
# price similarity, and general product popularity.
def score_user(self, user_id: int, profiles: dict[str, object]) -> dict[str, np.ndarray]:
    user_index = self.prepared.u2i[user_id]
    # Get and normalize the user's latent behavior scores for all products.
    latent = profiles["latent_scores"][user_index]
    latent = (latent - latent.min()) / (latent.max() - latent.min() + 1e-9)
    # Measure how well each product category matches the user's preferences. 
    category_affinity = profiles["user_cat_pref"][user_index][self.prod_cat_idx]
    # Measure how close each product price is to the user's preferred price range.
    price_affinity = np.maximum(
        0.0,
        1.0 - np.abs(self.prod_price - profiles["user_price_pref"][user_index]) / (self.price_range + 1e-9),
    )
    popularity = profiles["popularity"]
    # Combine all scoring signals into one hybrid recommendation score.
    # Behavior has the highest weight, followed by category, price, and popularity.
    hybrid_score = 0.60 * latent + 0.18 * category_affinity + 0.12 * price_affinity + 0.10 * popularity
    return {
        "latent": latent,
        "category_affinity": category_affinity,
        "price_affinity": price_affinity,
        "popularity": popularity,
        "hybrid_score": hybrid_score,
    }
