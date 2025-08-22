from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseSynthesizer


def _mlp(in_dim: int, hidden: List[int], out_dim: int, out_act: nn.Module | None = None) -> nn.Sequential:
	layers: list[nn.Module] = []
	prev = in_dim
	for h in hidden:
		layers += [nn.Linear(prev, h), nn.LeakyReLU(0.2, inplace=True)]
		prev = h
	layers.append(nn.Linear(prev, out_dim))
	if out_act is not None:
		layers.append(out_act)
	return nn.Sequential(*layers)


class TabDDPMWrapper(BaseSynthesizer):
	"""
	Simple Tabular DDPM (Denoising Diffusion Probabilistic Model) implementation.
	"""
	
	def __init__(
		self,
		target_col: str,
		latent_dim: int = 128,
		hidden_dims: List[int] | None = None,
		epochs: int = 10,
		batch_size: int = 512,
		lr: float = 2e-4,
		beta_start: float = 1e-4,
		beta_end: float = 0.02,
		timesteps: int = 100,  # Reduced for faster training
		cuda: bool = False,
		verbose: bool = False,
	) -> None:
		self.target_col = target_col
		self.latent_dim = latent_dim
		self.hidden_dims = hidden_dims or [256, 256, 256]
		self.epochs = epochs
		self.batch_size = batch_size
		self.lr = lr
		self.beta_start = beta_start
		self.beta_end = beta_end
		self.timesteps = timesteps
		self.device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
		self.verbose = verbose

		self._fitted = False
		self._model: nn.Module | None = None
		self._betas: torch.Tensor | None = None
		self._alphas: torch.Tensor | None = None
		self._alphas_cumprod: torch.Tensor | None = None
		self._data_dim: int | None = None
		self._original_columns: List[str] | None = None
		self._categorical_columns: List[str] | None = None
		self._numerical_columns: List[str] | None = None
		self._onehot_mapping: Dict[str, List[str]] | None = None
		self._numerical_ranges: Dict[str, tuple] | None = None

	def _setup_diffusion(self) -> None:
		"""Setup diffusion schedule."""
		self._betas = torch.linspace(self.beta_start, self.beta_end, self.timesteps)
		self._alphas = 1.0 - self._betas
		self._alphas_cumprod = torch.cumprod(self._alphas, dim=0)

	def _prepare_data(self, df: pd.DataFrame, discrete_columns: List[str] | None) -> torch.Tensor:
		"""Prepare data for diffusion model."""
		discrete_columns = list(discrete_columns or [])
		if self.target_col not in discrete_columns:
			discrete_columns.append(self.target_col)
		
		# Store original column info
		self._categorical_columns = discrete_columns
		self._numerical_columns = [c for c in df.columns if c not in discrete_columns]
		
		# Store numerical ranges for denormalization
		self._numerical_ranges = {}
		for col in self._numerical_columns:
			col_min = df[col].min()
			col_max = df[col].max()
			self._numerical_ranges[col] = (col_min, col_max)
		
		# Create one-hot mapping for reconstruction
		self._onehot_mapping = {}
		df_encoded = df.copy()
		
		# One-hot encode categorical columns
		for col in discrete_columns:
			unique_vals = sorted(df[col].astype(str).unique())
			self._onehot_mapping[col] = unique_vals
			
			# Create one-hot columns
			onehot_cols = []
			for val in unique_vals:
				col_name = f"{col}_{val}"
				onehot_cols.append((df[col].astype(str) == val).astype(int))
			
			# Add all one-hot columns at once
			onehot_df = pd.concat(onehot_cols, axis=1, keys=[f"{col}_{val}" for val in unique_vals])
			df_encoded = pd.concat([df_encoded, onehot_df], axis=1)
			
			# Remove original column
			df_encoded = df_encoded.drop(columns=[col])
		
		self._original_columns = df_encoded.columns.tolist()
		self._data_dim = df_encoded.shape[1]
		
		# Normalize to [0, 1] range
		df_normalized = df_encoded.astype(float)
		for col in df_normalized.columns:
			if df_normalized[col].dtype in ['int64', 'float64']:
				col_min = df_normalized[col].min()
				col_max = df_normalized[col].max()
				if col_max > col_min:
					df_normalized[col] = (df_normalized[col] - col_min) / (col_max - col_min)
		
		return torch.from_numpy(df_normalized.values).float()

	def _reconstruct_data(self, df_encoded: pd.DataFrame) -> pd.DataFrame:
		"""Reconstruct original data format from encoded data."""
		if self._onehot_mapping is None or self._numerical_ranges is None:
			return df_encoded
		
		df_reconstructed = df_encoded.copy()
		
		# Reconstruct categorical columns
		for col, values in self._onehot_mapping.items():
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
		all_original_cols = self._categorical_columns + self._numerical_columns
		for col in all_original_cols:
			if col not in df_reconstructed.columns:
				# Fill missing columns with default values
				if col in self._categorical_columns:
					df_reconstructed[col] = self._onehot_mapping[col][0] if self._onehot_mapping else "Unknown"
				else:
					df_reconstructed[col] = 0
		
		# Return with original column order
		return df_reconstructed[all_original_cols]

	def fit(self, data: pd.DataFrame, discrete_columns: list[str] | None = None) -> None:
		"""Train the diffusion model."""
		X = self._prepare_data(data, discrete_columns)
		self._setup_diffusion()
		
		# Create U-Net like model for noise prediction
		self._model = _mlp(
			self._data_dim + 1,  # +1 for timestep
			self.hidden_dims,
			self._data_dim,
			out_act=nn.Sigmoid()
		).to(self.device)
		
		optimizer = optim.Adam(self._model.parameters(), lr=self.lr)
		dataset = TensorDataset(X)
		loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
		
		self._model.train()
		for epoch in range(self.epochs):
			total_loss = 0
			for batch_idx, (x,) in enumerate(loader):
				x = x.to(self.device)
				b = x.size(0)
				
				# Sample random timesteps
				t = torch.randint(0, self.timesteps, (b,), device=self.device).long()
				
				# Add noise according to diffusion schedule
				noise = torch.randn_like(x)
				sqrt_alphas_cumprod_t = self._alphas_cumprod[t].sqrt().view(-1, 1)
				sqrt_one_minus_alphas_cumprod_t = (1 - self._alphas_cumprod[t]).sqrt().view(-1, 1)
				
				noisy_x = sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise
				
				# Predict noise
				t_input = t.float() / self.timesteps  # Normalize timestep to [0, 1]
				model_input = torch.cat([noisy_x, t_input.unsqueeze(1)], dim=1)
				predicted_noise = self._model(model_input)
				
				# MSE loss
				loss = nn.MSELoss()(predicted_noise, noise)
				
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				
				total_loss += loss.item()
			
			if self.verbose and (epoch + 1) % 5 == 0:
				print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss / len(loader):.6f}")
		
		self._fitted = True

	def sample(self, num_rows: int) -> pd.DataFrame:
		"""Generate samples using reverse diffusion."""
		if not self._fitted or self._model is None:
			raise RuntimeError("Model is not fitted.")
		
		self._model.eval()
		
		# Start from pure noise
		x = torch.randn(num_rows, self._data_dim, device=self.device)
		
		# Reverse diffusion process
		with torch.no_grad():
			for t in reversed(range(self.timesteps)):
				t_batch = torch.full((num_rows,), t, device=self.device, dtype=torch.long)
				t_input = t_batch.float() / self.timesteps
				
				# Predict noise
				model_input = torch.cat([x, t_input.unsqueeze(1)], dim=1)
				predicted_noise = self._model(model_input)
				
				# Reverse step
				alpha_t = self._alphas[t]
				alpha_t_cumprod = self._alphas_cumprod[t]
				beta_t = self._betas[t]
				
				if t > 0:
					noise = torch.randn_like(x)
				else:
					noise = torch.zeros_like(x)
				
				x = (1 / alpha_t.sqrt()) * (
					x - ((1 - alpha_t) / (1 - alpha_t_cumprod).sqrt()) * predicted_noise
				) + beta_t.sqrt() * noise
		
		# Convert back to DataFrame
		samples = x.cpu().numpy()
		samples = np.clip(samples, 0, 1)  # Clip to [0, 1]
		
		# Create DataFrame with one-hot encoded columns
		df_encoded = pd.DataFrame(samples, columns=self._original_columns)
		
		# Reconstruct original format
		df_reconstructed = self._reconstruct_data(df_encoded)
		
		return df_reconstructed
