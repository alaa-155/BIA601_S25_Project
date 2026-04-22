from __future__ import annotations

import numpy as np

from pipeline.config import RANDOM_SEED
from pipeline.models import PreparedData
from .evaluate import evaluate, train_full_and_export
from .genetic import ga_select
from .profiles import build_profiles, score_user
from .recommend import candidate_payload, recommend_for_user


class HybridGARecommender:
    def __init__(self, prepared: PreparedData):
        self.prepared = prepared
        self.rng = np.random.default_rng(RANDOM_SEED)
        self.products = prepared.products.sort_values("product_id").reset_index(drop=True)
        self.product_lookup = self.products.set_index("product_id")
        self.product_ids = prepared.product_ids
        self.prod_category = self.products["category"].to_numpy()
        self.prod_price = self.products["price"].to_numpy(dtype=float)
        self.price_range = float(self.prod_price.max() - self.prod_price.min())
        self.categories = sorted(self.products["category"].unique().tolist())
        self.cat_to_idx = {category: index for index, category in enumerate(self.categories)}
        self.prod_cat_idx = np.array([self.cat_to_idx[category] for category in self.prod_category])

    _build_profiles = build_profiles
    _score_user = score_user
    _ga_select = ga_select
    _candidate_payload = candidate_payload
    recommend_for_user = recommend_for_user
    evaluate = evaluate
    train_full_and_export = train_full_and_export
