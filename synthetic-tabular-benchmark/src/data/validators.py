from __future__ import annotations

from typing import Dict, Any

import pandas as pd


def validate_dataframe(df: pd.DataFrame, target: str | None = None) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Beklenen pandas.DataFrame")
    if df.empty:
        raise ValueError("Veri çerçevesi boş")
    if df.columns.duplicated().any():
        raise ValueError("Yinelenen sütun isimleri var")
    if target is not None and target not in df.columns:
        raise ValueError(f"Hedef sütun bulunamadı: {target}")


def infer_metadata(df: pd.DataFrame, target: str | None = None) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "columns": [],
        "target": target,
        "dtypes": {},
        "categorical_levels": {},
        "numerical_stats": {},
    }

    for col in df.columns:
        if col == target:
            continue
        dtype = str(df[col].dtype)
        metadata["columns"].append(col)
        metadata["dtypes"][col] = dtype
        if pd.api.types.is_numeric_dtype(df[col]):
            s = df[col].dropna()
            metadata["numerical_stats"][col] = {
                "min": float(s.min()) if not s.empty else None,
                "max": float(s.max()) if not s.empty else None,
                "mean": float(s.mean()) if not s.empty else None,
                "std": float(s.std()) if not s.empty else None,
            }
        else:
            levels = (
                df[col].astype("category").cat.categories.tolist()
                if not pd.api.types.is_numeric_dtype(df[col])
                else []
            )
            metadata["categorical_levels"][col] = levels

    return metadata


