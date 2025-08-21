from __future__ import annotations

from .registry import register_dataset
from .openml_loader import OpenMLDataset
from .csv_loader import CSVDataset
from .parquet_loader import ParquetDataset


def setup_registry() -> None:
    register_dataset("openml", OpenMLDataset)
    register_dataset("csv", CSVDataset)
    register_dataset("parquet", ParquetDataset)


