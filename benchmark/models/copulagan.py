from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import norm
from scipy.stats import uniform
import warnings

warnings.filterwarnings("ignore")

from .base import BaseSynthesizer


def _mlp(in_dim: int, hidden: List[int], out_dim: int, out_act: nn.Module | None = None) -> nn.Sequential:
	layers: list[nn.Module] = []
	prev = in_dim
	for h in hidden:
		layers += [nn.Linear(prev, h), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.1)]
		prev = h
	layers.append(nn.Linear(prev, out_dim))
	if out_act is not None:
		layers.append(out_act)
	return nn.Sequential(*layers)


class CopulaGANGenerator(nn.Module):
	"""Generator for CopulaGAN."""
	
	def __init__(self, latent_dim: int, hidden_dims: List[int], output_dim: int):
		super().__init__()
		self.latent_dim = latent_dim
		self.output_dim = output_dim
		
		self.generator = _mlp(latent_dim, hidden_dims, output_dim, out_act=nn.Sigmoid())
	
	def forward(self, noise: torch.Tensor) -> torch.Tensor:
		return self.generator(noise)


class CopulaGANDiscriminator(nn.Module):
	"""Discriminator for CopulaGAN."""
	
	def __init__(self, input_dim: int, hidden_dims: List[int]):
		super().__init__()
		self.discriminator = _mlp(input_dim, hidden_dims, 1, out_act=None)
	
	def forward(self, data: torch.Tensor) -> torch.Tensor:
		return self.discriminator(data)


class CopulaGANWrapper(BaseSynthesizer):
	"""
	Custom CopulaGAN implementation using PyTorch.
	
	CopulaGAN uses copula functions to model multivariate distributions
	by separating the modeling of individual variables (margins) from their
	dependencies (copula).
	"""
	
	def __init__(
		self,
		latent_dim: int = 128,
		gen_hidden: List[int] | None = None,
		disc_hidden: List[int] | None = None,
		epochs: int = 100,
		batch_size: int = 500,
		lr: float = 2e-4,
		beta1: float = 0.5,
		beta2: float = 0.9,
		cuda: bool = False,
		verbose: bool = False,
	) -> None:
		self.latent_dim = latent_dim
		self.gen_hidden = gen_hidden or [256, 256, 256]
		self.disc_hidden = disc_hidden or [256, 256, 256]
		self.epochs = epochs
		self.batch_size = batch_size
		self.lr = lr
		self.beta1 = beta1
		self.beta2 = beta2
		self.device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
		self.verbose = verbose

		self._fitted = False
		self._generator: CopulaGANGenerator | None = None
		self._discriminator: CopulaGANDiscriminator | None = None
		self._data_dim: int | None = None
		self._categorical_columns: List[str] | None = None
		self._numerical_columns: List[str] | None = None
		self._categories: Dict[str, List[str]] | None = None
		self._numerical_ranges: Dict[str, tuple] | None = None
		self._marginal_distributions: Dict[str, Any] | None = None

	def _prepare_data(self, df: pd.DataFrame, discrete_columns: List[str] | None) -> torch.Tensor:
		"""Prepare data for CopulaGAN training."""
		discrete_columns = list(discrete_columns or [])
		
		# Store column info
		self._categorical_columns = discrete_columns
		self._numerical_columns = [c for c in df.columns if c not in discrete_columns and c not in ['class']]
		
		# Store numerical ranges for denormalization
		self._numerical_ranges = {}
		for col in self._numerical_columns:
			col_min = df[col].min()
			col_max = df[col].max()
			self._numerical_ranges[col] = (col_min, col_max)
		
		# Create categories mapping
		self._categories = {}
		df_encoded = df.copy()
		
		# One-hot encode categorical columns
		for col in discrete_columns:
			unique_vals = sorted(df[col].astype(str).unique())
			self._categories[col] = unique_vals
			
			# Create one-hot columns
			onehot_cols = []
			for val in unique_vals:
				col_name = f"{col}_{val}"
				onehot_cols.append((df[col].astype(str) == val).astype(int))
			
			# Add all one-hot columns at once
			onehot_df = pd.concat(onehot_cols, axis=1, keys=[f"{col}_{val}" for val in unique_vals])
			df_encoded = pd.concat([df_encoded, onehot_df], axis=1)
		
		# Remove ALL original categorical columns AFTER adding one-hot columns
		for col in discrete_columns:
			if col in df_encoded.columns:
				df_encoded = df_encoded.drop(columns=[col])
		
		# Also remove target column if it exists (like 'class')
		target_cols = ['class', 'income']
		for col in target_cols:
			if col in df_encoded.columns:
				df_encoded = df_encoded.drop(columns=[col])
		
		self._data_dim = df_encoded.shape[1]
		
		# Fit marginal distributions for numerical columns
		self._marginal_distributions = {}
		for col in self._numerical_columns:
			if col in df_encoded.columns:  # Only process columns that still exist
				col_data = df_encoded[col].dropna()
				if len(col_data) > 0:
					# Fit normal distribution to numerical columns
					mean_val = col_data.mean()
					std_val = col_data.std()
					if std_val > 0:
						self._marginal_distributions[col] = {
							'type': 'normal',
							'mean': mean_val,
							'std': std_val
						}
					else:
						self._marginal_distributions[col] = {
							'type': 'uniform',
							'min': col_data.min(),
							'max': col_data.max()
						}
		
		# Normalize to [0, 1] range - only for numerical columns
		df_normalized = df_encoded.astype(float)
		
		# Get the list of one-hot encoded column names
		onehot_columns = []
		for col, values in self._categories.items():
			for val in values:
				onehot_columns.append(f"{col}_{val}")
		
		# Only normalize columns that were originally numerical (not one-hot encoded)
		for col in df_normalized.columns:
			if col in self._numerical_columns and col not in onehot_columns:
				col_min = df_normalized[col].min()
				col_max = df_normalized[col].max()
				if col_max > col_min:
					df_normalized[col] = (df_normalized[col] - col_min) / (col_max - col_min)
		
		return torch.from_numpy(df_normalized.values).float()

	def _reconstruct_data(self, df_encoded: pd.DataFrame) -> pd.DataFrame:
		"""Reconstruct original data format from encoded data."""
		if self._categories is None or self._numerical_ranges is None:
			return df_encoded
		
		df_reconstructed = df_encoded.copy()
		
		# Reconstruct categorical columns
		for col, values in self._categories.items():
			# Find one-hot columns for this categorical
			onehot_cols = [c for c in df_encoded.columns if c.startswith(f"{col}_")]
			
			if onehot_cols:
				# Get the column with highest value (most likely category)
				onehot_data = df_encoded[onehot_cols]
				most_likely = onehot_data.idxmax(axis=1)
				
				# Extract the actual value from column name
				reconstructed_values = []
				for col_name in most_likely:
					value = col_name.replace(f"{col}_", "")
					reconstructed_values.append(value)
				
				df_reconstructed[col] = reconstructed_values
				
				# Remove one-hot columns
				df_reconstructed = df_reconstructed.drop(columns=onehot_cols)
		
		# Denormalize numerical columns
		for col, (col_min, col_max) in self._numerical_ranges.items():
			if col in df_reconstructed.columns:
				# Denormalize from [0, 1] back to original range
				df_reconstructed[col] = df_reconstructed[col] * (col_max - col_min) + col_min
				
				# Round integer columns
				if col in ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']:
					df_reconstructed[col] = df_reconstructed[col].round().astype(int)
		
		# Ensure all original columns are present
		all_original_cols = self._categorical_columns + self._numerical_columns + ['class']
		for col in all_original_cols:
			if col not in df_reconstructed.columns:
				# Fill missing columns with default values
				if col in self._categorical_columns:
					df_reconstructed[col] = self._categories[col][0] if self._categories else "Unknown"
				elif col == 'class':
					df_reconstructed[col] = '<=50K'  # Default class value
				else:
					df_reconstructed[col] = 0
		
		# Return with original column order
		return df_reconstructed[all_original_cols]

	def _apply_copula_transform(self, data: torch.Tensor) -> torch.Tensor:
		"""Apply copula transformation to data."""
		if self._marginal_distributions is None:
			return data
		
		data_np = data.cpu().numpy()
		transformed_data = data_np.copy()
		
		# Apply inverse CDF transformation for numerical columns
		col_idx = 0
		for col in self._numerical_columns:
			if col in self._marginal_distributions:
				marginal = self._marginal_distributions[col]
				
				if marginal['type'] == 'normal':
					# Transform from uniform [0,1] to normal distribution
					transformed_data[:, col_idx] = norm.ppf(
						np.clip(data_np[:, col_idx], 0.001, 0.999)
					)
				elif marginal['type'] == 'uniform':
					# Transform from uniform [0,1] to uniform [min, max]
					transformed_data[:, col_idx] = (
						data_np[:, col_idx] * (marginal['max'] - marginal['min']) + marginal['min']
					)
			
			col_idx += 1
		
		return torch.from_numpy(transformed_data).float().to(self.device)

	def fit(self, data: pd.DataFrame, discrete_columns: list[str] | None = None) -> None:
		"""Train the CopulaGAN model."""
		X = self._prepare_data(data, discrete_columns)
		
		# Build models
		self._generator = CopulaGANGenerator(
			self.latent_dim, self.gen_hidden, self._data_dim
		).to(self.device)
		
		self._discriminator = CopulaGANDiscriminator(
			self._data_dim, self.disc_hidden
		).to(self.device)
		
		# Optimizers
		opt_G = optim.Adam(self._generator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
		opt_D = optim.Adam(self._discriminator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
		
		# Data loader
		dataset = TensorDataset(X)
		loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
		
		# Training loop
		self._generator.train()
		self._discriminator.train()
		
		for epoch in range(self.epochs):
			total_g_loss = 0
			total_d_loss = 0
			
			for batch_idx, (real_batch,) in enumerate(loader):
				real_batch = real_batch.to(self.device)
				batch_size = real_batch.size(0)
				
				# Train discriminator
				opt_D.zero_grad()
				
				# Real data
				real_labels = torch.ones(batch_size, 1).to(self.device)
				real_output = self._discriminator(real_batch)
				d_real_loss = nn.BCEWithLogitsLoss()(real_output, real_labels)
				
				# Fake data
				noise = torch.randn(batch_size, self.latent_dim).to(self.device)
				fake_data = self._generator(noise)
				fake_labels = torch.zeros(batch_size, 1).to(self.device)
				fake_output = self._discriminator(fake_data.detach())
				d_fake_loss = nn.BCEWithLogitsLoss()(fake_output, fake_labels)
				
				d_loss = (d_real_loss + d_fake_loss) / 2
				d_loss.backward()
				opt_D.step()
				
				# Train generator
				opt_G.zero_grad()
				
				# Generate fake data
				noise = torch.randn(batch_size, self.latent_dim).to(self.device)
				fake_data = self._generator(noise)
				fake_output = self._discriminator(fake_data)
				
				g_loss = nn.BCEWithLogitsLoss()(fake_output, real_labels)
				g_loss.backward()
				opt_G.step()
				
				total_g_loss += g_loss.item()
				total_d_loss += d_loss.item()
			
			if self.verbose and (epoch + 1) % 10 == 0:
				print(f"Epoch {epoch + 1}/{self.epochs}, G Loss: {total_g_loss / len(loader):.6f}, D Loss: {total_d_loss / len(loader):.6f}")
		
		self._fitted = True

	def sample(self, num_rows: int) -> pd.DataFrame:
		"""Generate synthetic samples."""
		if not self._fitted or self._generator is None:
			raise RuntimeError("Model is not fitted.")
		
		self._generator.eval()
		
		with torch.no_grad():
			# Generate noise
			noise = torch.randn(num_rows, self.latent_dim).to(self.device)
			
			# Generate samples
			samples = self._generator(noise)
			samples = samples.cpu().numpy()
			samples = np.clip(samples, 0, 1)  # Clip to [0, 1]
		
		# Create DataFrame with encoded columns
		encoded_columns = []
		for col in self._numerical_columns:
			encoded_columns.append(col)
		for col, values in self._categories.items():
			for val in values:
				encoded_columns.append(f"{col}_{val}")
		
		# Ensure we have the right number of columns
		if len(encoded_columns) != samples.shape[1]:
			# If column count doesn't match, use generic names
			encoded_columns = [f"col_{i}" for i in range(samples.shape[1])]
		
		df_encoded = pd.DataFrame(samples, columns=encoded_columns)
		
		# Reconstruct original format
		df_reconstructed = self._reconstruct_data(df_encoded)
		
		return df_reconstructed
