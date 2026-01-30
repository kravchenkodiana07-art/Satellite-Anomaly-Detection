from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .config import SETTINGS
from .featureizer import FeatureSchema, vectorize_bucket
from .schemas import TelemetryBucketRequest


@dataclass
class TrainedArtifact:
    schema: FeatureSchema
    scaler: StandardScaler
    model: IsolationForest
    decision_threshold: float  # decision_function below => anomaly


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def train_isolation_forest(X: pd.DataFrame) -> tuple[StandardScaler, IsolationForest, float]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)

    model = IsolationForest(
        n_estimators=SETTINGS.n_estimators,
        contamination=SETTINGS.contamination,
        random_state=SETTINGS.random_state,
    )
    model.fit(Xs)

    scores = model.decision_function(Xs)  # higher => more normal
    threshold = float(np.quantile(scores, SETTINGS.contamination))
    return scaler, model, threshold


def save_artifact(artifact: TrainedArtifact, path=SETTINGS.model_path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(artifact, path)


def load_artifact(path=SETTINGS.model_path) -> TrainedArtifact:
    return load(path)


def compute_train_stats_for_contributions(X: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for col in X.columns:
        s = X[col].astype(float)
        q1 = float(np.quantile(s, 0.25))
        q3 = float(np.quantile(s, 0.75))
        iqr = max(q3 - q1, 1e-9)
        stats[col] = {"median": float(np.median(s)), "iqr": float(iqr)}
    return stats


def save_train_stats(stats: Dict[str, Dict[str, float]], path=SETTINGS.train_stats_path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


def load_train_stats(path=SETTINGS.train_stats_path) -> Dict[str, Dict[str, float]]:
    return json.loads(path.read_text(encoding="utf-8"))


def score_bucket(
    req: TelemetryBucketRequest,
    artifact: TrainedArtifact,
    train_stats: Dict[str, Dict[str, float]],
    top_n: int = 3,
) -> tuple[float, float, Dict[str, float], List[tuple[str, float]]]:
    X = vectorize_bucket(req, artifact.schema)
    Xs = artifact.scaler.transform(X.values)

    decision = float(artifact.model.decision_function(Xs)[0])

    margin = decision - artifact.decision_threshold
    anomaly_score = 1.0 - _sigmoid(8.0 * margin)
    anomaly_score = float(np.clip(anomaly_score, 0.0, 1.0))

    # FIX: Use absolute margin for confidence (higher margin = more confident)
    confidence = float(np.clip(abs(margin) / 3.0, 0.0, 1.0))

    # Rest of the code stays the same...
    deviations: Dict[str, float] = {}
    for col in artifact.schema.columns:
        v = float(X[col].iloc[0])
        st = train_stats.get(col)
        if st is None:
            continue
        med = float(st["median"])
        iqr = float(st["iqr"])
        dev = abs(v - med) / iqr
        deviations[col] = float(1.0 - np.exp(-dev))

    per_signal: Dict[str, float] = {}
    for feat_key, dev in deviations.items():
        if "." not in feat_key:
            continue
        signal = feat_key.split(".", 1)[0]
        per_signal[signal] = max(per_signal.get(signal, 0.0), dev)

    top = sorted(deviations.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    s = sum(w for _, w in top) or 1.0
    top_norm = [(k, float(w / s)) for k, w in top]

    return anomaly_score, confidence, per_signal, top_norm
