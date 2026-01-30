from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import SETTINGS
from .schemas import TelemetryBucketRequest


@dataclass
class FeatureSchema:
    columns: List[str]
    add_missing_indicators: bool = True

    def all_columns(self) -> List[str]:
        if not self.add_missing_indicators:
            return list(self.columns)
        return list(self.columns) + [f"{c}.__missing__" for c in self.columns]


def _flatten_bucket(req: TelemetryBucketRequest) -> Dict[str, float]:
    flat: Dict[str, float] = {}
    for signal_name, feats in req.signals.items():
        for fk in SETTINGS.known_feature_keys:
            flat[f"{signal_name}.{fk}"] = float(getattr(feats, fk))
    return flat


def vectorize_bucket(req: TelemetryBucketRequest, schema: FeatureSchema) -> pd.DataFrame:
    flat = _flatten_bucket(req)

    row = {col: flat.get(col, np.nan) for col in schema.columns}
    df = pd.DataFrame([row], columns=schema.columns)

    if schema.add_missing_indicators:
        miss = df.isna().astype(float)
        miss.columns = [f"{c}.__missing__" for c in schema.columns]
        df = pd.concat([df, miss], axis=1)

    # simple imputation; missing indicators keep info
    df = df.fillna(0.0)
    return df


def build_schema_from_buckets(reqs: List[TelemetryBucketRequest]) -> FeatureSchema:
    cols = set()
    for req in reqs:
        cols.update(_flatten_bucket(req).keys())
    return FeatureSchema(columns=sorted(cols), add_missing_indicators=True)
