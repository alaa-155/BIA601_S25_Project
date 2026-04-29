from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PreparedData:
    users: pd.DataFrame
    products: pd.DataFrame
    interactions: pd.DataFrame
    user_ids: list[int]
    product_ids: np.ndarray
    u2i: dict[int, int]
    p2i: dict[int, int]
