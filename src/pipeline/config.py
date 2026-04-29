from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
EVAL_SAMPLE_USERS = 300
TOP_K = 10
CANDIDATE_POOL = 30
