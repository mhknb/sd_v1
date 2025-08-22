from __future__ import annotations
import pandas as pd
from ctgan import TVAE

from .base import BaseSynthesizer


class TVAEWrapper(BaseSynthesizer):
	def __init__(
		self,
		epochs: int = 10,
		batch_size: int = 500,
		cuda: bool = False,
		verbose: bool = False,
		compress_dims: tuple[int, ...] = (128, 128),
		decompress_dims: tuple[int, ...] = (128, 128),
		loss_factor: int = 2,
		l2scale: float = 1e-5,
	) -> None:
		self.model = TVAE(
			epochs=epochs,
			batch_size=batch_size,
			cuda=cuda,
			verbose=verbose,
			compress_dims=compress_dims,
			decompress_dims=decompress_dims,
			loss_factor=loss_factor,
			l2scale=l2scale,
		)
		self._fitted = False

	def fit(self, data: pd.DataFrame, discrete_columns: list[str] | None = None) -> None:
		self.model.fit(data, discrete_columns=discrete_columns or [])
		self._fitted = True

	def sample(self, num_rows: int) -> pd.DataFrame:
		if not self._fitted:
			raise RuntimeError("Model is not fitted.")
		return self.model.sample(num_rows)
