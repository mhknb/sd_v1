from __future__ import annotations
from typing import Dict, List
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, entropy


def _jsd_from_counts(p_counts: np.ndarray, q_counts: np.ndarray) -> float:
	p = p_counts / (p_counts.sum() + 1e-12)
	q = q_counts / (q_counts.sum() + 1e-12)
	m = 0.5 * (p + q)
	return 0.5 * (entropy(p, m, base=2) + entropy(q, m, base=2))


def compute_fidelity(
	real: pd.DataFrame,
	synth: pd.DataFrame,
	categorical_columns: List[str],
	numerical_columns: List[str],
	num_bins: int = 20,
) -> Dict[str, float]:
	metrics: Dict[str, float] = {}

	# Per-numeric KS
	ks_values = []
	for col in numerical_columns:
		r = real[col].dropna().astype(float)
		s = synth[col].dropna().astype(float)
		if len(r) > 0 and len(s) > 0:
			stat = ks_2samp(r, s).statistic
			ks_values.append(stat)
	metrics["ks_mean"] = float(np.mean(ks_values)) if ks_values else float("nan")

	# JSD per categorical
	jsd_values = []
	for col in categorical_columns:
		rvc = real[col].astype(str).value_counts()
		svc = synth[col].astype(str).value_counts()
		cats = list(set(rvc.index).union(set(svc.index)))
		r_counts = np.array([rvc.get(c, 0) for c in cats], dtype=float)
		s_counts = np.array([svc.get(c, 0) for c in cats], dtype=float)
		jsd = _jsd_from_counts(r_counts, s_counts)
		jsd_values.append(jsd)
	metrics["jsd_mean"] = float(np.mean(jsd_values)) if jsd_values else float("nan")

	# Correlation difference (numeric Pearson)
	if len(numerical_columns) >= 2:
		r_corr = real[numerical_columns].corr().fillna(0.0).to_numpy()
		s_corr = synth[numerical_columns].corr().fillna(0.0).to_numpy()
		corr_diff = np.mean(np.abs(r_corr - s_corr))
		metrics["corr_abs_diff_mean"] = float(corr_diff)
	else:
		metrics["corr_abs_diff_mean"] = float("nan")

	return metrics
