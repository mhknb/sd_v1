from __future__ import annotations

from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .interface import DatasetInterface
from .validators import validate_dataframe, infer_metadata
from .preprocess import apply_preprocess, PreprocessResult


class ParquetDataset(DatasetInterface):
    def __init__(self, path: str | None = None, target: str | None = None) -> None:
        self.path = path
        self.target = target
        self._metadata: Dict[str, Any] | None = None
        self._df: pd.DataFrame | None = None

    def load_data(self, config: Dict[str, Any]) -> pd.DataFrame:
        path = config.get("path", self.path)
        if path is None:
            raise ValueError("Parquet için 'path' gereklidir")
        read_kwargs = config.get("read_kwargs", {})
        df = pd.read_parquet(path, **read_kwargs)
        validate_dataframe(df, target=self.target)
        self._df = df
        self._metadata = infer_metadata(df, target=self.target)
        return df

    def preprocess(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        result: PreprocessResult = apply_preprocess(data, target=self.target)
        self._metadata = {**(self._metadata or {}), **result.metadata}
        return result.features, result.metadata

    def split_data(self, data: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_df, test_df = train_test_split(data, test_size=test_size, random_state=42, stratify=data[self.target] if self.target else None)
        return train_df, test_df

    def get_metadata(self) -> Dict[str, Any]:
        return self._metadata or {}


