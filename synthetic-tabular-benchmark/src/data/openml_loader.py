from __future__ import annotations

from typing import Any, Dict, Tuple

import openml
import pandas as pd
from sklearn.model_selection import train_test_split

from .interface import DatasetInterface
from .validators import validate_dataframe, infer_metadata
from .preprocess import apply_preprocess, PreprocessResult


class OpenMLDataset(DatasetInterface):
    def __init__(self, name: str, target: str | None = None) -> None:
        self.name = name
        self.target = target
        self._metadata: Dict[str, Any] | None = None
        self._df: pd.DataFrame | None = None

    def load_data(self, config: Dict[str, Any]) -> pd.DataFrame:
        dataset_id = config.get("openml_id")
        if dataset_id is None:
            raise ValueError("OpenML dataset için 'openml_id' gereklidir")
        oml = openml.datasets.get_dataset(dataset_id)
        X, y, categorical_indicator, attribute_names = oml.get_data(
            dataset_format="dataframe", target=self.target
        )
        df = X.copy()
        if y is not None:
            df[self.target] = y
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


