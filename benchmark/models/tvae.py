from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import warnings

warnings.filterwarnings("ignore")

from .base import BaseSynthesizer


def _mlp(in_dim: int, hidden: List[int], out_dim: int, out_act: nn.Module | None = None) -> nn.Sequential:
	"""Create MLP with specified architecture."""
	layers: list[nn.Module] = []
	prev = in_dim
	for h in hidden:
		layers += [nn.Linear(prev, h), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.1)]
		prev = h
	layers.append(nn.Linear(prev, out_dim))
	if out_act is not None:
		layers.append(out_act)
	return nn.Sequential(*layers)


class TVAEEncoder(nn.Module):
	"""Encoder for TVAE."""
	
	def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
		super().__init__()
		self.encoder = _mlp(input_dim, hidden_dims, hidden_dims[-1])
		self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
		self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
	
	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		h = self.encoder(x)
		mu = self.fc_mu(h)
		logvar = self.fc_logvar(h)
		return mu, logvar


class TVAEDecoder(nn.Module):
	"""Decoder for TVAE."""
	
	def __init__(self, latent_dim: int, hidden_dims: List[int], output_dim: int):
		super().__init__()
		# Reverse the hidden dimensions for decoder
		decoder_hidden = list(reversed(hidden_dims))
		self.decoder = _mlp(latent_dim, decoder_hidden, output_dim, out_act=nn.Sigmoid())
	
	def forward(self, z: torch.Tensor) -> torch.Tensor:
		return self.decoder(z)


class DataTransformer:
	"""Handle data transformation for TVAE."""
	
	def __init__(self):
		self.numerical_columns = []
		self.categorical_columns = []
		self.scalers = {}
		self.encoders = {}
		self.categories = {}
		self.output_dim = 0
		self.column_info = []
		
	def fit(self, data: pd.DataFrame, discrete_columns: List[str]):
		"""Fit the transformer on data."""
		self.categorical_columns = list(discrete_columns)
		
		# Include target column if it's string-like
		target_cols = ['class', 'income']
		for col in target_cols:
			if col not in discrete_columns and col in data.columns:
				if data[col].dtype == 'object' or pd.api.types.is_categorical_dtype(data[col]):
					self.categorical_columns.append(col)
		
		self.numerical_columns = [c for c in data.columns if c not in self.categorical_columns]
		
		# Store categories for each categorical column
		for col in self.categorical_columns:
			unique_vals = sorted(data[col].astype(str).unique())
			self.categories[col] = unique_vals
			
			# One-hot encoder for each categorical column
			encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
			encoded = encoder.fit_transform(data[[col]].astype(str))
			self.encoders[col] = encoder
			
			self.column_info.append({
				'name': col,
				'type': 'categorical',
				'size': len(unique_vals),
				'encoder': encoder
			})
		
		# Scalers for numerical columns
		for col in self.numerical_columns:
			scaler = StandardScaler()
			scaled = scaler.fit_transform(data[[col]])
			self.scalers[col] = scaler
			
			self.column_info.append({
				'name': col,
				'type': 'numerical',
				'size': 1,
				'scaler': scaler
			})
		
		# Calculate output dimension
		self.output_dim = sum(info['size'] for info in self.column_info)
	
	def transform(self, data: pd.DataFrame) -> torch.Tensor:
		"""Transform data to neural network input format."""
		transformed_parts = []
		
		# Transform each column
		for info in self.column_info:
			col = info['name']
			if info['type'] == 'categorical':
				encoded = info['encoder'].transform(data[[col]].astype(str))
				transformed_parts.append(encoded)
			else:  # numerical
				scaled = info['scaler'].transform(data[[col]])
				transformed_parts.append(scaled)
		
		# Concatenate all transformed parts
		if transformed_parts:
			transformed_data = np.concatenate(transformed_parts, axis=1)
		else:
			transformed_data = np.zeros((len(data), 0))
		
		return torch.FloatTensor(transformed_data)
	
	def inverse_transform(self, data: torch.Tensor) -> pd.DataFrame:
		"""Transform neural network output back to original format."""
		data_np = data.cpu().numpy()
		
		result_data = {}
		start_idx = 0
		
		# Inverse transform each column
		for info in self.column_info:
			col = info['name']
			end_idx = start_idx + info['size']
			col_data = data_np[:, start_idx:end_idx]
			
			if info['type'] == 'categorical':
				# For categorical, find the category with highest probability
				category_idx = np.argmax(col_data, axis=1)
				categories = info['encoder'].categories_[0]
				result_data[col] = [categories[idx] for idx in category_idx]
			else:  # numerical
				# Inverse transform numerical data
				result_data[col] = info['scaler'].inverse_transform(col_data).flatten()
			
			start_idx = end_idx
		
		return pd.DataFrame(result_data)


class TVAEWrapper(BaseSynthesizer):
	"""
	Custom TVAE implementation using PyTorch.
	
	TVAE (Table Variational Autoencoder) generates synthetic tabular data
	using a variational autoencoder approach with continuous latent space.
	"""
	
	def __init__(
		self,
		latent_dim: int = 128,
		hidden_dims: List[int] | None = None,
		epochs: int = 100,
		batch_size: int = 500,
		lr: float = 2e-4,
		beta1: float = 0.5,
		beta2: float = 0.9,
		kl_weight: float = 1.0,  # Weight for KL divergence loss
		cuda: bool = False,
		verbose: bool = False,
	) -> None:
		self.latent_dim = latent_dim
		self.hidden_dims = hidden_dims or [256, 256]
		self.epochs = epochs
		self.batch_size = batch_size
		self.lr = lr
		self.beta1 = beta1
		self.beta2 = beta2
		self.kl_weight = kl_weight
		self.device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
		self.verbose = verbose

		self._fitted = False
		self._encoder: TVAEEncoder | None = None
		self._decoder: TVAEDecoder | None = None
		self._transformer: DataTransformer | None = None

	def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
		"""Reparameterization trick for VAE."""
		std = torch.exp(0.5 * logvar)
		eps = torch.randn_like(std)
		return mu + eps * std

	def _kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
		"""Compute KL divergence loss."""
		return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

	def _reconstruction_loss(self, x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
		"""Compute reconstruction loss (MSE for numerical, BCE for categorical)."""
		# Use MSE loss for all data since we have mixed numerical and categorical
		return nn.MSELoss(reduction='sum')(x_recon, x)

	def fit(self, data: pd.DataFrame, discrete_columns: list[str] | None = None) -> None:
		"""Train the TVAE model."""
		discrete_columns = list(discrete_columns or [])
		
		# Initialize transformer
		self._transformer = DataTransformer()
		self._transformer.fit(data, discrete_columns)
		
		# Transform data
		transformed_data = self._transformer.transform(data)
		
		# Build models
		self._encoder = TVAEEncoder(
			self._transformer.output_dim,
			self.hidden_dims,
			self.latent_dim
		).to(self.device)
		
		self._decoder = TVAEDecoder(
			self.latent_dim,
			self.hidden_dims,
			self._transformer.output_dim
		).to(self.device)
		
		# Optimizer
		encoder_params = list(self._encoder.parameters())
		decoder_params = list(self._decoder.parameters())
		all_params = encoder_params + decoder_params
		optimizer = optim.Adam(all_params, lr=self.lr, betas=(self.beta1, self.beta2))
		
		# Data loader
		dataset = TensorDataset(transformed_data)
		loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
		
		# Training loop
		self._encoder.train()
		self._decoder.train()
		
		for epoch in range(self.epochs):
			total_loss = 0
			total_recon_loss = 0
			total_kl_loss = 0
			
			for batch_idx, (real_batch,) in enumerate(loader):
				real = real_batch.to(self.device)
				batch_size = real.size(0)
				
				# Forward pass
				mu, logvar = self._encoder(real)
				z = self._reparameterize(mu, logvar)
				recon = self._decoder(z)
				
				# Compute losses
				recon_loss = self._reconstruction_loss(real, recon)
				kl_loss = self._kl_loss(mu, logvar)
				total_loss_vae = recon_loss + self.kl_weight * kl_loss
				
				# Backward pass
				optimizer.zero_grad()
				total_loss_vae.backward()
				optimizer.step()
				
				total_loss += total_loss_vae.item()
				total_recon_loss += recon_loss.item()
				total_kl_loss += kl_loss.item()
			
			if self.verbose and (epoch + 1) % 10 == 0:
				avg_loss = total_loss / len(loader)
				avg_recon = total_recon_loss / len(loader)
				avg_kl = total_kl_loss / len(loader)
				print(f"Epoch {epoch + 1}/{self.epochs}, Total Loss: {avg_loss:.6f}, Recon: {avg_recon:.6f}, KL: {avg_kl:.6f}")
		
		self._fitted = True

	def sample(self, num_rows: int) -> pd.DataFrame:
		"""Generate synthetic samples."""
		if not self._fitted or self._decoder is None or self._transformer is None:
			raise RuntimeError("Model is not fitted.")
		
		self._decoder.eval()
		
		with torch.no_grad():
			# Sample from standard normal distribution
			z = torch.randn(num_rows, self.latent_dim).to(self.device)
			
			# Decode to generate samples
			samples = self._decoder(z)
			
			# Transform back to original format
			synthetic_data = self._transformer.inverse_transform(samples)
		
		return synthetic_data
