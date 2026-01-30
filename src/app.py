# src/app.py
from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .config import SETTINGS
from .model import load_artifact, load_train_stats, score_bucket
from .schemas import (
    TelemetryBucketRequest,
    TelemetryBucketResponse,
    MLBlock,
    ModelInfo,
    Contributor,
)

app = FastAPI(title="Telemetry Anomaly Scoring", version="1.0")

_artifact = None
_stats = None


@app.on_event("startup")
def _startup() -> None:
    global _artifact, _stats
    try:
        _artifact = load_artifact(SETTINGS.model_path)
        _stats = load_train_stats(SETTINGS.train_stats_path)
        print(f"Loaded model artifact from {SETTINGS.model_path}")
        print(f"Loaded train stats from {SETTINGS.train_stats_path}")
    except Exception as e:
        _artifact = None
        _stats = None
        print(f"WARNING: Failed to load artifacts: {e}")


@app.post("/score", response_model=TelemetryBucketResponse)
def score(req: TelemetryBucketRequest) -> TelemetryBucketResponse:
    if _artifact is None or _stats is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded. Train first.")

    anomaly_score, confidence, per_signal, top = score_bucket(req, _artifact, _stats, top_n=3)

    return TelemetryBucketResponse(
        schema_version=req.schema_version,
        bucket_start=req.bucket_start,
        ml=MLBlock(
            model=ModelInfo(name="isolation_forest", version="v1.0"),
            anomaly_score=anomaly_score,
            confidence=confidence,
            per_signal_score={k: float(v) for k, v in per_signal.items()},
            top_contributors=[Contributor(key=k, weight=w) for k, w in top],
        ),
    )
