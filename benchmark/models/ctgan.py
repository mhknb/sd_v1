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


class CTGANGenerator(nn.Module):
	"""Generator for CTGAN."""
	
	def __init__(self, latent_dim: int, condition_dim: int, hidden_dims: List[int], output_dim: int):
		super().__init__()
		input_dim = latent_dim + condition_dim
		self.generator = _mlp(input_dim, hidden_dims, output_dim, out_act=nn.Tanh())
	
	def forward(self, noise: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
		input_tensor = torch.cat([noise, conditions], dim=1)
		return self.generator(input_tensor)


class CTGANDiscriminator(nn.Module):
	"""Discriminator for CTGAN."""
	
	def __init__(self, input_dim: int, condition_dim: int, hidden_dims: List[int]):
		super().__init__()
		total_input_dim = input_dim + condition_dim
		self.discriminator = _mlp(total_input_dim, hidden_dims, 1, out_act=None)
	
	def forward(self, data: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
		input_tensor = torch.cat([data, conditions], dim=1)
		return self.discriminator(input_tensor)


class DataTransformer:
	"""Handle data transformation for CTGAN."""
	
	def __init__(self):
		self.numerical_columns = []
		self.categorical_columns = []
		self.target_column = None
		self.scalers = {}
		self.encoders = {}
		self.categories = {}
		self.output_dim = 0
		self.condition_dim = 0
		self.column_info = []
		
	def fit(self, data: pd.DataFrame, discrete_columns: List[str], target_col: str):
		"""Fit the transformer on data."""
		self.target_column = target_col
		self.categorical_columns = list(discrete_columns)
		
		# Include target_col in categorical if it's string-like
		if target_col not in discrete_columns:
			if data[target_col].dtype == 'object' or pd.api.types.is_categorical_dtype(data[target_col]):
				self.categorical_columns.append(target_col)
		
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
		
		# Condition dimension (target column categories)
		if target_col in self.categories:
			self.condition_dim = len(self.categories[target_col])
		else:
			self.condition_dim = 1
	
	def transform(self, data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
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
		
		# Prepare conditions (target column)
		if self.target_column in self.encoders:
			conditions = self.encoders[self.target_column].transform(
				data[[self.target_column]].astype(str)
			)
		else:
			# If target is numerical, use binning or simple encoding
			target_values = data[self.target_column].values
			conditions = np.zeros((len(data), 1))
		
		return torch.FloatTensor(transformed_data), torch.FloatTensor(conditions)
	
	def inverse_transform(self, data: torch.Tensor, conditions: torch.Tensor) -> pd.DataFrame:
		"""Transform neural network output back to original format."""
		data_np = data.cpu().numpy()
		conditions_np = conditions.cpu().numpy()
		
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
		
		# Add target column from conditions
		if self.target_column in self.encoders:
			target_idx = np.argmax(conditions_np, axis=1)
			target_categories = self.encoders[self.target_column].categories_[0]
			result_data[self.target_column] = [target_categories[idx] for idx in target_idx]
		else:
			result_data[self.target_column] = conditions_np.flatten()
		
		return pd.DataFrame(result_data)


class CTGANWrapper(BaseSynthesizer):
	"""
	Custom CTGAN implementation using PyTorch.
	
	CTGAN (Conditional Tabular GAN) generates synthetic tabular data
	by conditioning on discrete columns to ensure mode coverage.
	"""
	
	def __init__(
		self,
		latent_dim: int = 128,
		gen_hidden: List[int] | None = None,
		disc_hidden: List[int] | None = None,
		epochs: int = 300,
		batch_size: int = 500,
		lr: float = 2e-4,
		beta1: float = 0.5,
		beta2: float = 0.9,
		pac: int = 10,  # PacGAN parameter
		cuda: bool = False,
		verbose: bool = False,
		target_col: str = "class",
	) -> None:
		self.latent_dim = latent_dim
		self.gen_hidden = gen_hidden or [256, 256]
		self.disc_hidden = disc_hidden or [256, 256]
		self.epochs = epochs
		self.batch_size = batch_size
		self.lr = lr
		self.beta1 = beta1
		self.beta2 = beta2
		self.pac = pac
		self.device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
		self.verbose = verbose
		self.target_col = target_col

		self._fitted = False
		self._generator: CTGANGenerator | None = None
		self._discriminator: CTGANDiscriminator | None = None
		self._transformer: DataTransformer | None = None
		self._condition_sampler: Dict[str, float] | None = None

	def _sample_conditions(self, n_samples: int) -> torch.Tensor:
		"""Sample conditions for training."""
		if self._condition_sampler is None:
			return torch.zeros(n_samples, self._transformer.condition_dim)
		
		# Sample from target categories based on their frequency
		categories = list(self._condition_sampler.keys())
		probabilities = list(self._condition_sampler.values())
		
		sampled_categories = np.random.choice(
			categories, size=n_samples, p=probabilities
		)
		
		# Convert to one-hot encoding
		conditions = torch.zeros(n_samples, self._transformer.condition_dim)
		for i, cat in enumerate(sampled_categories):
			cat_idx = self._transformer.categories[self.target_col].index(cat)
			conditions[i, cat_idx] = 1.0
		
		return conditions

	def fit(self, data: pd.DataFrame, discrete_columns: list[str] | None = None) -> None:
		"""Train the CTGAN model."""
		discrete_columns = list(discrete_columns or [])
		
		# Initialize transformer
		self._transformer = DataTransformer()
		self._transformer.fit(data, discrete_columns, self.target_col)
		
		# Prepare condition sampler
		if self.target_col in self._transformer.categories:
			target_counts = data[self.target_col].value_counts()
			total = len(data)
			self._condition_sampler = {
				str(cat): count / total for cat, count in target_counts.items()
			}
		
		# Transform data
		transformed_data, conditions = self._transformer.transform(data)
		
		# Build models
		self._generator = CTGANGenerator(
			self.latent_dim,
			self._transformer.condition_dim,
			self.gen_hidden,
			self._transformer.output_dim
		).to(self.device)
		
		self._discriminator = CTGANDiscriminator(
			self._transformer.output_dim,
			self._transformer.condition_dim,
			self.disc_hidden
		).to(self.device)
		
		# Optimizers
		opt_G = optim.Adam(self._generator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
		opt_D = optim.Adam(self._discriminator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
		
		# Data loader
		dataset = TensorDataset(transformed_data, conditions)
		loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
		
		# Training loop
		self._generator.train()
		self._discriminator.train()
		
		for epoch in range(self.epochs):
			total_g_loss = 0
			total_d_loss = 0
			
			for batch_idx, (real_batch, real_conditions) in enumerate(loader):
				real_batch = real_batch.to(self.device)
				real_conditions = real_conditions.to(self.device)
				batch_size = real_batch.size(0)
				
				# Train discriminator
				opt_D.zero_grad()
				
				# Real data
				real_labels = torch.ones(batch_size, 1).to(self.device)
				real_output = self._discriminator(real_batch, real_conditions)
				d_real_loss = nn.BCEWithLogitsLoss()(real_output, real_labels)
				
				# Fake data
				noise = torch.randn(batch_size, self.latent_dim).to(self.device)
				fake_conditions = self._sample_conditions(batch_size).to(self.device)
				fake_data = self._generator(noise, fake_conditions)
				fake_labels = torch.zeros(batch_size, 1).to(self.device)
				fake_output = self._discriminator(fake_data.detach(), fake_conditions)
				d_fake_loss = nn.BCEWithLogitsLoss()(fake_output, fake_labels)
				
				d_loss = (d_real_loss + d_fake_loss) / 2
				d_loss.backward()
				opt_D.step()
				
				# Train generator
				opt_G.zero_grad()
				
				# Generate fake data
				noise = torch.randn(batch_size, self.latent_dim).to(self.device)
				fake_conditions = self._sample_conditions(batch_size).to(self.device)
				fake_data = self._generator(noise, fake_conditions)
				fake_output = self._discriminator(fake_data, fake_conditions)
				
				g_loss = nn.BCEWithLogitsLoss()(fake_output, real_labels)
				g_loss.backward()
				opt_G.step()
				
				total_g_loss += g_loss.item()
				total_d_loss += d_loss.item()
			
			if self.verbose and (epoch + 1) % 50 == 0:
				print(f"Epoch {epoch + 1}/{self.epochs}, G Loss: {total_g_loss / len(loader):.6f}, D Loss: {total_d_loss / len(loader):.6f}")
		
		self._fitted = True

	def sample(self, num_rows: int) -> pd.DataFrame:
		"""Generate synthetic samples."""
		if not self._fitted or self._generator is None or self._transformer is None:
			raise RuntimeError("Model is not fitted.")
		
		self._generator.eval()
		
		with torch.no_grad():
			# Generate noise
			noise = torch.randn(num_rows, self.latent_dim).to(self.device)
			
			# Sample conditions
			conditions = self._sample_conditions(num_rows).to(self.device)
			
			# Generate samples
			samples = self._generator(noise, conditions)
			
			# Transform back to original format
			synthetic_data = self._transformer.inverse_transform(samples, conditions)
		
		return synthetic_data
