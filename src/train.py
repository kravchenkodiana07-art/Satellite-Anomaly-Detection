# src/train.py
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from .featureizer import build_schema_from_buckets, vectorize_bucket
from .model import (
    TrainedArtifact,
    train_isolation_forest,
    compute_train_stats_for_contributions,
    save_artifact,
    save_train_stats,
)
from .schemas import TelemetryBucketRequest


def load_jsonl(path: Path) -> List[TelemetryBucketRequest]:
    reqs: List[TelemetryBucketRequest] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            reqs.append(TelemetryBucketRequest.model_validate_json(line))
    return reqs


def main(train_path: str = "data/train.jsonl") -> None:
    train_path_p = Path(train_path)
    reqs = load_jsonl(train_path_p)

    schema = build_schema_from_buckets(reqs)

    rows = [vectorize_bucket(r, schema) for r in reqs]
    X = pd.concat(rows, axis=0, ignore_index=True)

    scaler, model, threshold, p05, p50, p95 = train_isolation_forest(X)

    artifact = TrainedArtifact(
        schema=schema,
        scaler=scaler,
        model=model,
        decision_threshold=threshold,
        score_p05=p05,
        score_p50=p50,
        score_p95=p95,
    )
    save_artifact(artifact)

    # stats for per-feature deviation-based contributions (use only base columns)
    X_base = X[schema.columns].copy()
    stats = compute_train_stats_for_contributions(X_base)
    save_train_stats(stats)

    print("Saved model to models/model.joblib")
    print("Saved train stats to models/train_stats.json")
    print(f"Schema columns: {len(schema.columns)} (+missing indicators: {len(schema.all_columns()) - len(schema.columns)})")
    print(f"Decision threshold (fixed): {threshold:.6f}")
    print(f"Train decision_function quantiles: p05={p05:.6f}, p50={p50:.6f}, p95={p95:.6f}")


if __name__ == "__main__":
    main()

