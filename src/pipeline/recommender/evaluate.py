from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.config import CANDIDATE_POOL, EVAL_SAMPLE_USERS, TOP_K


def evaluate(self, eval_users: int = EVAL_SAMPLE_USERS) -> dict[str, float]:
    train, holdout_pairs = _split_holdout(self.prepared.interactions.copy(), eval_users)
    profiles = self._build_profiles(train)
    totals = {"base_hits": 0, "ga_hits": 0, "base_ndcg": 0.0, "ga_ndcg": 0.0, "base_diversity": [], "ga_diversity": []}

    for user_id, true_product_id in holdout_pairs:
        _update_metrics(self, user_id, true_product_id, profiles, totals)

    total = len(holdout_pairs)
    return {
        "evaluation_users": total,
        "baseline_recall_at_10": round(totals["base_hits"] / total, 4),
        "ga_recall_at_10": round(totals["ga_hits"] / total, 4),
        "baseline_ndcg_at_10": round(totals["base_ndcg"] / total, 4),
        "ga_ndcg_at_10": round(totals["ga_ndcg"] / total, 4),
        "baseline_diversity_at_10": round(float(np.mean(totals["base_diversity"])), 4),
        "ga_diversity_at_10": round(float(np.mean(totals["ga_diversity"])), 4),
    }


def train_full_and_export(self) -> tuple[dict[str, object], pd.DataFrame]:
    profiles = self._build_profiles(self.prepared.interactions)
    rows: list[dict[str, object]] = []
    for user_id in self.prepared.user_ids:
        rows.extend(self.recommend_for_user(user_id, profiles)["ga_rows"])
    recommendations = pd.DataFrame(rows)
    recommendations["hybrid_score_display"] = recommendations["hybrid_score"].round(3)
    recommendations["price"] = recommendations["price"].round(2)
    return profiles, recommendations


def _split_holdout(interactions: pd.DataFrame, eval_users: int) -> tuple[pd.DataFrame, list[tuple[int, int]]]:
    train = interactions.copy()
    holdout_pairs: list[tuple[int, int]] = []
    positive = interactions[interactions["positive"] == 1]
    for user_id, group in positive.groupby("user_id"):
        if len(group) < 2:
            continue
        row = group.sample(1, random_state=int(user_id)).iloc[0]
        holdout_pairs.append((int(user_id), int(row["product_id"])))
        train = train[~((train["user_id"] == user_id) & (train["product_id"] == row["product_id"]))]
    return train, holdout_pairs[:eval_users]


def _update_metrics(self, user_id: int, true_product_id: int, profiles: dict[str, object], totals: dict[str, object]) -> None:
    payload = self._candidate_payload(user_id, profiles)
    candidate_ids = payload["candidate_ids"]
    candidate_hybrid = payload["candidate_hybrid"]
    score_parts = payload["score_parts"]
    base_ids = candidate_ids[:TOP_K]
    pool_ids = candidate_ids[:CANDIDATE_POOL]
    pool_scores = candidate_hybrid[:CANDIDATE_POOL]
    pool_indices = np.array([self.prepared.p2i[item] for item in pool_ids])
    ga_ids = self._ga_select(pool_ids, pool_scores, self.prod_cat_idx[pool_indices], score_parts["popularity"][pool_indices], k=TOP_K)

    totals["base_diversity"].append(len(set(self.prod_category[[self.prepared.p2i[item] for item in base_ids]])) / TOP_K)
    totals["ga_diversity"].append(len(set(self.prod_category[[self.prepared.p2i[item] for item in ga_ids]])) / TOP_K)
    _update_hit_counts(base_ids, true_product_id, totals, "base")
    _update_hit_counts(ga_ids, true_product_id, totals, "ga")


def _update_hit_counts(predictions: np.ndarray, true_product_id: int, totals: dict[str, object], prefix: str) -> None:
    if true_product_id not in predictions:
        return
    position = int(np.where(predictions == true_product_id)[0][0])
    totals[f"{prefix}_hits"] += 1
    totals[f"{prefix}_ndcg"] += 1.0 / np.log2(position + 2)
