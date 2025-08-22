from __future__ import annotations
import pandas as pd
from ctgan import CTGAN

from .base import BaseSynthesizer


class CTGANWrapper(BaseSynthesizer):
	def __init__(
		self,
		epochs: int = 10,
		batch_size: int = 500,
		pac: int = 10,
		cuda: bool = False,
		verbose: bool = False,
	) -> None:
		self.model = CTGAN(epochs=epochs, batch_size=batch_size, pac=pac, cuda=cuda, verbose=verbose)
		self._fitted = False

	def fit(self, data: pd.DataFrame, discrete_columns: list[str] | None = None) -> None:
		self.model.fit(data, discrete_columns=discrete_columns or [])
		self._fitted = True

	def sample(self, num_rows: int) -> pd.DataFrame:
		if not self._fitted:
			raise RuntimeError("Model is not fitted.")
		return self.model.sample(num_rows)
