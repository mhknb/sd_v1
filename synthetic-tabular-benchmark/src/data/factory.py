from __future__ import annotations

from typing import Any, Dict

from .registry import get_dataset_class


def create_dataset(dataset_key: str, datasets_cfg: Dict[str, Any]):
    """Config'e göre dataset örneği oluştur.

    datasets.yaml içinde beklenen şema:
    registry:
      <dataset_key>:
        loader: <openml|csv|parquet>
        ... diğer loader spesifik alanlar
    """
    entry = datasets_cfg.get("registry", {}).get(dataset_key)
    if entry is None:
        raise KeyError(f"Dataset kaydı bulunamadı: {dataset_key}")
    loader_name = entry.get("loader")
    if not loader_name:
        raise ValueError(f"Dataset '{dataset_key}' için 'loader' alanı gerekli")
    cls = get_dataset_class(loader_name)

    # Hedef adı (varsa) çıkar
    target = entry.get("target")

    # OpenML vs dosya tabanlı loader'larda kurucu imzaları değişebilir,
    # ancak hepsi (name/path, target) ikilisini kabul edecek şekilde ayarlandı.
    if loader_name == "openml":
        instance = cls(name=dataset_key, target=target)
    else:
        # csv/parquet gibi dosya tabanlı loader'lar için opsiyonel path beklenir
        path = entry.get("path")
        instance = cls(path=path, target=target)

    return instance, entry


