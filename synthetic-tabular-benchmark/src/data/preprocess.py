from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class PreprocessResult:
    features: Any
    target: Any | None
    transformer: ColumnTransformer
    metadata: Dict[str, Any]


def build_preprocess_pipeline(df: pd.DataFrame, target: str | None = None) -> Tuple[Pipeline, Dict[str, Any]]:
    feature_cols = [c for c in df.columns if c != target]
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in feature_cols if c not in num_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=True)),
        ]
    )
    transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ]
    )

    metadata: Dict[str, Any] = {
        "feature_columns": feature_cols,
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "target": target,
    }
    return transformer, metadata


def apply_preprocess(df: pd.DataFrame, target: str | None = None) -> PreprocessResult:
    transformer, metadata = build_preprocess_pipeline(df, target)
    X = df.drop(columns=[target]) if target and target in df.columns else df
    y = df[target] if target and target in df.columns else None
    features = transformer.fit_transform(X)
    return PreprocessResult(features=features, target=y, transformer=transformer, metadata=metadata)


