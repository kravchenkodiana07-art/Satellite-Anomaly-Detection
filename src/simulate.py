# src/simulate.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict

import numpy as np

from .schemas import TelemetryBucketRequest, SignalFeatures


@dataclass(frozen=True)
class SignalSpec:
    mean: float
    std: float
    slope_mean: float
    slope_std: float
    missing_rate_mean: float = 0.0
    missing_rate_std: float = 0.01


DEFAULT_SPECS: Dict[str, SignalSpec] = {
    "cpu_temperature": SignalSpec(mean=45.0, std=1.1, slope_mean=0.35, slope_std=0.15),
    "battery_voltage": SignalSpec(mean=12.5, std=0.05, slope_mean=-0.02, slope_std=0.02),
    "pressure": SignalSpec(mean=101325.0, std=90.0, slope_mean=5.0, slope_std=4.0),
    "gyro_speed": SignalSpec(mean=0.10, std=0.06, slope_mean=0.01, slope_std=0.02),
    "signal_strength": SignalSpec(mean=-70.0, std=3.2, slope_mean=-0.40, slope_std=0.25),
    "power_consumption": SignalSpec(mean=150.0, std=18.0, slope_mean=0.8, slope_std=0.7),
}


def _make_features(rng: np.random.Generator, spec: SignalSpec) -> SignalFeatures:
    mean = float(rng.normal(spec.mean, spec.std))
    std = float(abs(rng.normal(spec.std, max(spec.std * 0.2, 1e-6))))
    min_v = float(mean - abs(rng.normal(2.0 * std, max(std, 1e-6))))
    max_v = float(mean + abs(rng.normal(2.0 * std, max(std, 1e-6))))
    p95 = float(mean + abs(rng.normal(1.6 * std, max(0.5 * std, 1e-6))))
    p95 = float(min(max(p95, min_v), max_v))
    slope = float(rng.normal(spec.slope_mean, spec.slope_std))
    missing_rate = float(np.clip(rng.normal(spec.missing_rate_mean, spec.missing_rate_std), 0.0, 1.0))
    return SignalFeatures(
        mean=mean,
        min=min_v,
        max=max_v,
        std=std,
        slope=slope,
        p95=p95,
        missing_rate=missing_rate,
    )


def generate_bucket(
    bucket_start: datetime,
    rng: np.random.Generator,
    specs: Dict[str, SignalSpec] = DEFAULT_SPECS,
    anomaly_prob: float = 0.0,
    drop_signal_prob: float = 0.05,
) -> TelemetryBucketRequest:
    signals = {name: _make_features(rng, spec) for name, spec in specs.items()}

    # emulate dynamic set: sometimes a signal missing
    if rng.random() < drop_signal_prob:
        name = rng.choice(list(signals.keys()))
        signals.pop(name, None)

    # inject anomalies sometimes (USED ONLY FOR TEST DATA)
    if rng.random() < anomaly_prob and signals:
        keys = list(signals.keys())
        k = int(rng.integers(1, min(3, len(keys)) + 1))
        chosen = rng.choice(keys, size=k, replace=False)
        for name in chosen:
            f = signals[name]
            bump = float(abs(rng.normal(0, 1)) * 4.0)
            signals[name] = SignalFeatures(
                mean=f.mean + bump * (1 if rng.random() < 0.5 else -1),
                min=f.min,
                max=f.max + bump,
                std=f.std * (1.0 + bump * 0.3),
                slope=f.slope + bump,
                p95=f.p95 + bump,
                missing_rate=min(1.0, f.missing_rate + 0.1),
            )

    return TelemetryBucketRequest(
        schema_version="v1",
        bucket_start=bucket_start.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
        bucket_sec=60,
        signals=signals,
    )


def write_jsonl(
    out_path: Path,
    n_minutes: int,
    seed: int,
    anomaly_prob: float,
    drop_signal_prob: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    start = datetime(2026, 1, 29, 22, 32, 0, tzinfo=timezone.utc)

    with out_path.open("w", encoding="utf-8") as f:
        for i in range(n_minutes):
            req = generate_bucket(
                start + timedelta(minutes=i),
                rng,
                anomaly_prob=anomaly_prob,
                drop_signal_prob=drop_signal_prob,
            )
            f.write(req.model_dump_json())
            f.write("\n")


def write_sample_request(out_path: Path = Path("data/sample_request.json"), seed: int = 7) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    start = datetime(2026, 1, 29, 22, 32, 0, tzinfo=timezone.utc)
    req = generate_bucket(start, rng, anomaly_prob=0.0, drop_signal_prob=0.0)
    out_path.write_text(req.model_dump_json(indent=2), encoding="utf-8")


if __name__ == "__main__":
    # TRAIN: normal-only
    write_jsonl(
        out_path=Path("data/train.jsonl"),
        n_minutes=2000,
        seed=42,
        anomaly_prob=0.0,
        drop_signal_prob=0.05,
    )

    # TEST: contains anomalies (for evaluation only)
    write_jsonl(
        out_path=Path("data/test_anoms.jsonl"),
        n_minutes=400,
        seed=43,
        anomaly_prob=0.10,
        drop_signal_prob=0.05,
    )

    write_sample_request()
    print("Wrote data/train.jsonl (normal-only)")
    print("Wrote data/test_anoms.jsonl (with anomalies)")
    print("Wrote data/sample_request.json")
