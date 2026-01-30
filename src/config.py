from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model_path: Path = Path("models/model.joblib")
    train_stats_path: Path = Path("models/train_stats.json")
    default_schema_version: str = "v1"
    bucket_sec_default: int = 60

    contamination: float = 0.03
    n_estimators: int = 300
    random_state: int = 42

    known_feature_keys: tuple[str, ...] = (
        "mean",
        "min",
        "max",
        "std",
        "slope",
        "p95",
        "missing_rate",
    )


SETTINGS = Settings()
