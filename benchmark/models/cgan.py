from __future__ import annotations
from dataclasses import dataclass
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


@dataclass
class _TableSpec:
	numeric_cols: List[str]
	categorical_cols: List[str]
	categories: Dict[str, List[Any]]
	num_min: Dict[str, float]
	num_max: Dict[str, float]
	onehot_slices: Dict[str, slice]
	output_dim: int


class CGANWrapper(BaseSynthesizer):
	def __init__(
		self,
		target_col: str | None = None,
		latent_dim: int = 128,
		gen_hidden: List[int] | None = None,
		disc_hidden: List[int] | None = None,
		epochs: int = 10,
		batch_size: int = 512,
		lr: float = 2e-4,
		cuda: bool = False,
		verbose: bool = False,
	) -> None:
		self.target_col = target_col
		self.latent_dim = latent_dim
		self.gen_hidden = gen_hidden or [256, 256]
		self.disc_hidden = disc_hidden or [256, 256]
		self.epochs = epochs
		self.batch_size = batch_size
		self.lr = lr
		self.device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
		self.verbose = verbose

		self._fitted = False
		self._G: nn.Module | None = None
		self._D: nn.Module | None = None
		self._spec: _TableSpec | None = None
		self._classes: List[Any] = []

	def _fit_table_spec(self, df: pd.DataFrame, discrete_columns: List[str] | None) -> _TableSpec:
		discrete_columns = discrete_columns or []
		numeric_cols = [c for c in df.columns if c not in discrete_columns]
		categorical_cols = list(discrete_columns)

		categories: Dict[str, List[Any]] = {}
		for c in categorical_cols:
			categories[c] = sorted(df[c].dropna().astype(str).unique().tolist())

		num_min: Dict[str, float] = {}
		num_max: Dict[str, float] = {}
		for c in numeric_cols:
			col = pd.to_numeric(df[c], errors='coerce')
			num_min[c] = float(np.nanmin(col)) if np.isfinite(np.nanmin(col)) else 0.0
			num_max[c] = float(np.nanmax(col)) if np.isfinite(np.nanmax(col)) else 1.0
			if num_max[c] - num_min[c] < 1e-9:
				num_max[c] = num_min[c] + 1.0

		# one-hot slices offsets after numeric
		offset = len(numeric_cols)
		onehot_slices: Dict[str, slice] = {}
		for c in categorical_cols:
			n = len(categories[c])
			onehot_slices[c] = slice(offset, offset + n)
			offset += n

		output_dim = len(numeric_cols) + sum(len(categories[c]) for c in categorical_cols)
		return _TableSpec(
			numeric_cols=numeric_cols,
			categorical_cols=categorical_cols,
			categories=categories,
			num_min=num_min,
			num_max=num_max,
			onehot_slices=onehot_slices,
			output_dim=output_dim,
		)

	def _transform(self, df: pd.DataFrame) -> np.ndarray:
		assert self._spec is not None
		spec = self._spec
		num_arr = []
		for c in spec.numeric_cols:
			col = pd.to_numeric(df[c], errors='coerce').fillna(spec.num_min[c])
			v = (col - spec.num_min[c]) / (spec.num_max[c] - spec.num_min[c])
			num_arr.append(v.values.reshape(-1, 1))
		num_mat = np.concatenate(num_arr, axis=1) if num_arr else np.zeros((len(df), 0))

		cat_arr = []
		for c in spec.categorical_cols:
			vals = df[c].astype(str).fillna(spec.categories[c][0])
			one_hot = np.zeros((len(df), len(spec.categories[c])), dtype=np.float32)
			index_map = {v: i for i, v in enumerate(spec.categories[c])}
			idx = vals.map(index_map).fillna(0).astype(int).values
			one_hot[np.arange(len(df)), idx] = 1.0
			cat_arr.append(one_hot)
		cat_mat = np.concatenate(cat_arr, axis=1) if cat_arr else np.zeros((len(df), 0))

		mat = np.concatenate([num_mat, cat_mat], axis=1)
		return mat.astype(np.float32)

	def _inverse(self, mat: np.ndarray) -> pd.DataFrame:
		assert self._spec is not None
		spec = self._spec
		out: Dict[str, Any] = {}
		# numeric inverse
		for i, c in enumerate(spec.numeric_cols):
			v = mat[:, i]
			v = np.clip(v, 0.0, 1.0)
			orig = v * (spec.num_max[c] - spec.num_min[c]) + spec.num_min[c]
			out[c] = orig
		# categorical inverse
		for c in spec.categorical_cols:
			s = spec.onehot_slices[c]
			scores = mat[:, s]
			idx = np.argmax(scores, axis=1)
			cats = [spec.categories[c][int(i)] for i in idx]
			out[c] = cats
		return pd.DataFrame(out)

	def fit(self, data: pd.DataFrame, discrete_columns: list[str] | None = None) -> None:
		if self.target_col is None:
			raise ValueError("CGANWrapper requires target_col")
		# separate target and features
		classes = sorted(data[self.target_col].astype(str).unique().tolist())
		self._classes = classes
		Xdf = data.drop(columns=[self.target_col])
		self._spec = self._fit_table_spec(Xdf, discrete_columns)
		X = self._transform(Xdf)
		k = len(classes)
		# conditional one-hot labels for training
		Y = data[self.target_col].astype(str).map({c: i for i, c in enumerate(classes)}).values
		Y_oh = np.zeros((len(Y), k), dtype=np.float32)
		Y_oh[np.arange(len(Y)), Y] = 1.0

		G = _mlp(self.latent_dim + k, self.gen_hidden, self._spec.output_dim, out_act=nn.Sigmoid())
		D = _mlp(self._spec.output_dim + k, self.disc_hidden, 1, out_act=None)
		G.to(self.device)
		D.to(self.device)
		opt_G = optim.Adam(G.parameters(), lr=self.lr, betas=(0.5, 0.9))
		opt_D = optim.Adam(D.parameters(), lr=self.lr, betas=(0.5, 0.9))
		dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y_oh))
		loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
		bce = nn.BCEWithLogitsLoss()
		for _ in range(self.epochs):
			for xb, yb in loader:
				xb = xb.to(self.device)
				yb = yb.to(self.device)
				b = xb.size(0)
				# D step
				z = torch.randn(b, self.latent_dim, device=self.device)
				fake = G(torch.cat([z, yb], dim=1)).detach()
				pred_real = D(torch.cat([xb, yb], dim=1))
				pred_fake = D(torch.cat([fake, yb], dim=1))
				loss_D = bce(pred_real, torch.ones_like(pred_real)) + bce(pred_fake, torch.zeros_like(pred_fake))
				opt_D.zero_grad(set_to_none=True)
				loss_D.backward()
				opt_D.step()
				# G step
				z = torch.randn(b, self.latent_dim, device=self.device)
				fake = G(torch.cat([z, yb], dim=1))
				pred = D(torch.cat([fake, yb], dim=1))
				loss_G = bce(pred, torch.ones_like(pred))
				opt_G.zero_grad(set_to_none=True)
				loss_G.backward()
				opt_G.step()
		self._G, self._D = G, D
		self._fitted = True

	def sample(self, num_rows: int) -> pd.DataFrame:
		if not self._fitted or self._G is None or self._spec is None:
			raise RuntimeError("Model is not fitted.")
		G = self._G
		G.eval()
		classes = self._classes
		k = len(classes)
		rows = []
		labels = []
		with torch.no_grad():
			bs = self.batch_size
			for start in range(0, num_rows, bs):
				b = min(bs, num_rows - start)
				y = np.random.randint(0, k, size=b)
				labels.extend([classes[int(i)] for i in y])
				y_oh = np.zeros((b, k), dtype=np.float32)
				y_oh[np.arange(b), y] = 1.0
				z = torch.randn(b, self.latent_dim)
				inp = torch.from_numpy(np.concatenate([z.numpy(), y_oh], axis=1)).float()
				y_gen = G(inp).numpy()
				rows.append(y_gen)
		Xgen = np.concatenate(rows, axis=0)
		features_df = self._inverse(Xgen)
		features_df[self.target_col] = labels
		# reorder columns to numeric + categorical (including target at end for stability)
		cols = self._spec.numeric_cols + self._spec.categorical_cols
		if self.target_col not in cols:
			cols = cols + [self.target_col]
		return features_df[cols]
