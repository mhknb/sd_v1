import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
	"""Basic cleanup for categorical strings."""
	df = df.copy()
	for col in df.select_dtypes(include=["object", "category"]).columns:
		df[col] = (
			df[col]
			.astype(str)
			.str.strip()
			.replace({"?": None, "nan": None})
		)
	return df


def load_adult(test_size: float = 0.2, random_state: int = 42):
	"""
	Load and preprocess the UCI Adult dataset from OpenML.

	Returns
	-------
	train_df: pd.DataFrame
	test_df: pd.DataFrame
	meta: dict with keys: target, categorical_columns, numerical_columns, discrete_columns
	"""
	dataset = fetch_openml(name="adult", version=2, as_frame=True)
	df: pd.DataFrame = dataset.frame

	# Standardize column names
	df.columns = [c.replace('-', '_').replace(' ', '_') for c in df.columns]

	# Ensure target name
	target_col = "class" if "class" in df.columns else "income"
	if target_col not in df.columns:
		if "income" in df.columns:
			target_col = "income"
		else:
			raise ValueError("Adult dataset target column not found.")

	df = _clean_dataframe(df)

	# Identify column types
	categorical_columns = [
		c for c in df.columns if df[c].dtype == 'object' or str(df[c].dtype) == 'category'
	]
	if target_col not in categorical_columns:
		categorical_columns.append(target_col)

	numerical_columns = [c for c in df.columns if c not in categorical_columns]

	# Train test split
	train_df, test_df = train_test_split(
		df, test_size=test_size, random_state=random_state, stratify=df[target_col]
	)

	meta = {
		"target": target_col,
		"categorical_columns": categorical_columns,
		"numerical_columns": numerical_columns,
		"discrete_columns": [c for c in categorical_columns if c != target_col],
	}
	return train_df.reset_index(drop=True), test_df.reset_index(drop=True), meta
