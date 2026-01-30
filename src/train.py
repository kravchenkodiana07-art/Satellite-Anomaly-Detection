from __future__ import annotations

import json
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

    rows = []
    for r in reqs:
        rows.append(vectorize_bucket(r, schema))
    X = pd.concat(rows, axis=0, ignore_index=True)

    scaler, model, threshold = train_isolation_forest(X)

    artifact = TrainedArtifact(schema=schema, scaler=scaler, model=model, decision_threshold=threshold)
    save_artifact(artifact)

    # stats for per-feature deviation-based contributions (use only base columns)
    X_base = X[schema.columns].copy()
    stats = compute_train_stats_for_contributions(X_base)
    save_train_stats(stats)

    print(f"Saved model to models/model.joblib")
    print(f"Saved train stats to models/train_stats.json")
    print(f"Schema columns: {len(schema.columns)} (+missing indicators: {len(schema.columns)})")
    print(f"Decision threshold: {threshold:.6f}")


if __name__ == "__main__":
    main()
