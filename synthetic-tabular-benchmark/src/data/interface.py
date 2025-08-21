from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple


class DatasetInterface(ABC):
    """Veri seti arayüzü.

    Beklenen akış: load_data -> preprocess -> split_data -> get_metadata
    """

    @abstractmethod
    def load_data(self, config: Dict[str, Any]) -> Any:
        """Ham veriyi yükler (ör. pandas.DataFrame, target kolonu dahil)."""

    @abstractmethod
    def preprocess(self, data: Any) -> Tuple[Any, Dict[str, Any]]:
        """Ön işleme uygular ve (işlenmiş_veri, metadata) döner."""

    @abstractmethod
    def split_data(self, data: Any, test_size: float) -> Tuple[Any, Any]:
        """Veriyi eğitim/teste böler."""

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Öznitelik tipleri, kategorik seviyeler, aralıklar vb."""


