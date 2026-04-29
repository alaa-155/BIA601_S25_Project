from __future__ import annotations

import numpy as np

from pipeline.config import CANDIDATE_POOL, TOP_K

# Build the candidate product list for one user before final optimization.
# Products the user already interacted with are removed to avoid repeated recommendations.
def candidate_payload(self, user_id: int, profiles: dict[str, object]) -> dict[str, object]:
    # Calculate recommendation scores for all products using the user profile.
    score_parts = self._score_user(user_id, profiles)
    # Get the products that the user has already viewed, rated, clicked, or purchased.
    seen = profiles["seen"][user_id]
    # Create a filter that keeps only products not seen before by the user.
    mask = np.ones(len(self.product_ids), dtype=bool)
    if seen:
        mask[[self.prepared.p2i[item] for item in seen]] = False
    # Keep only the IDs and scores of products that passed the filter
    candidate_ids = self.product_ids[mask]
    candidate_hybrid = score_parts["hybrid_score"][mask]
    # Sort candidates from highest to lowest hybrid recommendation score.
    order = np.argsort(-candidate_hybrid)
    return {
        "score_parts": score_parts,
        "candidate_ids": candidate_ids[order],
        "candidate_hybrid": candidate_hybrid[order],
    }

# Generate the final recommendation list for one user.
# A baseline list is created first, then the genetic algorithm improves the final ranking.
def recommend_for_user(self, user_id: int, profiles: dict[str, object], k: int = TOP_K) -> dict[str, object]:
    # Build the initial candidate list for the selected user.
    payload = self._candidate_payload(user_id, profiles)
    candidate_ids = payload["candidate_ids"]
    candidate_hybrid = payload["candidate_hybrid"]
    score_parts = payload["score_parts"]
    # Select the top products by direct hybrid score as the baseline list.
    base_ids = candidate_ids[:k]
    # Limit the search space to the strongest candidates before running the genetic algorithm.
    pool_ids = candidate_ids[:CANDIDATE_POOL]
    pool_scores = candidate_hybrid[:CANDIDATE_POOL]
    pool_indices = np.array([self.prepared.p2i[item] for item in pool_ids])
    # Apply the genetic algorithm to improve the final recommendation list
    # using relevance, diversity, and novelty.
    ga_ids = self._ga_select(pool_ids, pool_scores, self.prod_cat_idx[pool_indices], score_parts["popularity"][pool_indices], k=k)

    rows = [_build_row(self, user_id, rank, int(product_id), score_parts) for rank, product_id in enumerate(ga_ids, start=1)]
    return {"base_ids": base_ids, "ga_rows": rows}

# Convert one recommended product into a structured row for saving or display.
# The row includes product details, recommendation scores, and an Arabic reason.
def _build_row(self, user_id: int, rank: int, product_id: int, score_parts: dict[str, np.ndarray]) -> dict[str, object]:
    product_index = self.prepared.p2i[product_id]
    product_row = self.product_lookup.loc[product_id]
    return {
        "user_id": user_id,
        "rank": rank,
        "product_id": product_id,
        "category": product_row["category"],
        "price": float(product_row["price"]),
        "hybrid_score": float(score_parts["hybrid_score"][product_index]),
        "latent_score": float(score_parts["latent"][product_index]),
        "category_score": float(score_parts["category_affinity"][product_index]),
        "price_score": float(score_parts["price_affinity"][product_index]),
        "popularity_score": float(score_parts["popularity"][product_index]),
        "reason": "، ".join(_build_reasons(score_parts, product_index)),
    }

# Create short Arabic explanations for the recommended products.
# These explanations help users understand why each product was selected.
def _build_reasons(score_parts: dict[str, np.ndarray], product_index: int) -> list[str]:
    # Collect readable reasons based on category match, price match, and behavior score.
    reasons: list[str] = []
    if score_parts["category_affinity"][product_index] >= 0.18:
        reasons.append("تطابق قوي مع الفئات المفضلة")
    if score_parts["price_affinity"][product_index] >= 0.75:
        reasons.append("ملاءمة جيدة مع نمط السعر")
    if score_parts["latent"][product_index] >= 0.70:
        reasons.append("إشارة سلوكية/تقييمية قوية")
    return reasons or ["توازن جيد بين التفضيلات والانتشار"]
