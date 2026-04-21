from __future__ import annotations
import json
from pipeline.charts import create_charts
from pipeline.config import EVAL_SAMPLE_USERS
from pipeline.exports import export_csvs, export_metrics
from pipeline.prepare import load_and_prepare
from pipeline.recommender import HybridGARecommender
from pipeline.summary import export_summary


def main() -> None:
    prepared = load_and_prepare()
    model = HybridGARecommender(prepared)
    metrics = model.evaluate(EVAL_SAMPLE_USERS)
    export_metrics(metrics)

    _, recommendations = model.train_full_and_export()
    export_csvs(prepared, recommendations)
    export_summary(prepared, metrics)
    create_charts(prepared, metrics)

    print("Pipeline completed successfully.")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
