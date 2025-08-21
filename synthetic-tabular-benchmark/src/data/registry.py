from __future__ import annotations

from typing import Dict, Type

from .interface import DatasetInterface


_REGISTRY: Dict[str, Type[DatasetInterface]] = {}


def register_dataset(name: str, cls: Type[DatasetInterface]) -> None:
    key = name.lower()
    if key in _REGISTRY:
        raise ValueError(f"Dataset '{name}' zaten kayıtlı.")
    _REGISTRY[key] = cls


def get_dataset_class(name: str) -> Type[DatasetInterface]:
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"Dataset '{name}' bulunamadı. Kayıtlı: {list(_REGISTRY)}")
    return _REGISTRY[key]


