from __future__ import annotations
import sys
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from .base import BaseSynthesizer


class CTABGANPlusWrapper(BaseSynthesizer):
	def __init__(
		self,
		target_col: str,
		test_ratio: float = 0.2,
		epochs: int = 10,
		batch_size: int = 500,
		cuda: bool = False,
		verbose: bool = False,
	) -> None:
		self.target_col = target_col
		self.test_ratio = test_ratio
		self.epochs = epochs
		self.batch_size = batch_size
		self.cuda = cuda
		self.verbose = verbose

		self._ctabgan = None
		self._temp_csv_path = None

	def _ensure_third_party_path(self) -> Path:
		root = Path(__file__).resolve().parents[2]
		tp = root / "third_party" / "CTAB-GAN-Plus"
		if str(tp) not in sys.path:
			sys.path.insert(0, str(tp))
		return tp

	def fit(self, data: pd.DataFrame, discrete_columns: List[str] | None = None) -> None:
		discrete_columns = list(discrete_columns or [])
		if self.target_col not in discrete_columns:
			discrete_columns.append(self.target_col)
		
		# Save data to temporary CSV for CTABGAN
		temp_dir = Path("/tmp") / "ctabgan_temp"
		temp_dir.mkdir(exist_ok=True)
		self._temp_csv_path = temp_dir / "temp_data.csv"
		data.to_csv(self._temp_csv_path, index=False)
		
		# Derive column types based on Adult dataset structure
		categorical_columns = discrete_columns
		log_columns = []
		mixed_columns = {'capital_loss': [0.0], 'capital_gain': [0.0]}
		general_columns = ['age']
		non_categorical_columns = []
		integer_columns = ['age', 'fnlwgt', 'capital_gain', 'capital_loss', 'hours_per_week']
		
		self._ensure_third_party_path()
		from model.ctabgan import CTABGAN  # type: ignore
		
		# Create CTABGAN instance with our data
		self._ctabgan = CTABGAN(
			raw_csv_path=str(self._temp_csv_path),
			test_ratio=self.test_ratio,
			categorical_columns=categorical_columns,
			log_columns=log_columns,
			mixed_columns=mixed_columns,
			general_columns=general_columns,
			non_categorical_columns=non_categorical_columns,
			integer_columns=integer_columns,
			problem_type={"Classification": self.target_col}
		)
		
		# Override epochs and batch_size in synthesizer
		if hasattr(self._ctabgan.synthesizer, 'epochs'):
			self._ctabgan.synthesizer.epochs = self.epochs
		if hasattr(self._ctabgan.synthesizer, 'batch_size'):
			self._ctabgan.synthesizer.batch_size = self.batch_size
		
		# Fit the model
		self._ctabgan.fit()

	def sample(self, num_rows: int) -> pd.DataFrame:
		if self._ctabgan is None:
			raise RuntimeError("Model is not fitted.")
		
		# Generate samples using original method
		sample_df = self._ctabgan.generate_samples()
		
		# Return requested number of rows
		if len(sample_df) >= num_rows:
			return sample_df.head(num_rows)
		else:
			# If not enough samples, duplicate some rows
			remaining = num_rows - len(sample_df)
			extra_rows = sample_df.sample(n=min(remaining, len(sample_df)), replace=True)
			return pd.concat([sample_df, extra_rows], ignore_index=True).head(num_rows)

	def __del__(self):
		# Clean up temporary file
		if self._temp_csv_path and self._temp_csv_path.exists():
			try:
				self._temp_csv_path.unlink()
			except:
				pass
