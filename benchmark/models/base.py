from abc import ABC, abstractmethod
import pandas as pd


class BaseSynthesizer(ABC):
	@abstractmethod
	def fit(self, data: pd.DataFrame, discrete_columns: list[str] | None = None) -> None:
		...

	@abstractmethod
	def sample(self, num_rows: int) -> pd.DataFrame:
		...
