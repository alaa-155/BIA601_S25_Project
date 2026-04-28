from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DATA_DIR = ROOT_DIR / "data" / "processed"


@dataclass(frozen=True)
class AppData:
    users: pd.DataFrame
    recommendations: pd.DataFrame
    metrics: dict
    summary: dict
    user_lookup: pd.DataFrame


def load_app_data() -> AppData:
    users = pd.read_csv(DATA_DIR / "users_clean.csv")
    recommendations = pd.read_csv(DATA_DIR / "user_recommendations.csv")
    metrics = json.loads((DATA_DIR / "metrics.json").read_text(encoding="utf-8"))
    summary = json.loads((DATA_DIR / "summary.json").read_text(encoding="utf-8"))
    user_lookup = users.set_index("user_id")
    return AppData(
        users=users,
        recommendations=recommendations,
        metrics=metrics,
        summary=summary,
        user_lookup=user_lookup,
    )


def project_paths() -> dict[str, Path]:
    return {
        "base_dir": BASE_DIR,
        "root_dir": ROOT_DIR,
        "templates_dir": BASE_DIR / "templates",
        "static_dir": BASE_DIR / "static",
    }
