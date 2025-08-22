import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from benchmark.data.adult import load_adult
from benchmark.models.cgan import CGANWrapper
from benchmark.models.great_wrapper import GReaTWrapper
from benchmark.models.tabddpm import TabDDPMWrapper
from benchmark.models.copulagan import CopulaGANWrapper
from benchmark.models.ctgan import CTGANWrapper
from benchmark.models.tvae import TVAEWrapper
from benchmark.metrics.fidelity import compute_fidelity
from benchmark.metrics.utility import compute_utility
from benchmark.utils.io import ensure_dir, save_csv, save_json


def get_dataset(name: str):
	name = name.lower()
	if name == "adult":
		return load_adult()
	raise ValueError(f"Unknown dataset: {name}")


def get_model(name: str, **kwargs):
	name = name.lower()
	if name == "cgan":
		kwargs.pop("pac", None)
		return CGANWrapper(**kwargs)
	if name == "great":
		kwargs.pop("pac", None)
		return GReaTWrapper(**kwargs)
	if name == "tabddpm":
		kwargs.pop("pac", None)
		return TabDDPMWrapper(**kwargs)
	if name == "copulagan":
		kwargs.pop("pac", None)
		return CopulaGANWrapper(**kwargs)
	if name == "ctgan":
		return CTGANWrapper(**kwargs)
	if name == "tvae":
		return TVAEWrapper(**kwargs)
	raise ValueError(f"Unknown model: {name}")


def main():
	parser = argparse.ArgumentParser(description="Tabular GAN benchmark runner")
	parser.add_argument("--model", required=True, choices=["cgan", "great", "tabddpm", "copulagan", "ctgan", "tvae"], help="Model adı")
	parser.add_argument("--dataset", required=True, choices=["adult"], help="Veri seti adı")
	parser.add_argument("--output-dir", default="outputs", help="Çıktı klasörü")
	parser.add_argument("--epochs", type=int, default=5, help="Eğitim epoch sayısı")
	parser.add_argument("--batch-size", type=int, default=500)
	parser.add_argument("--pac", type=int, default=10)
	parser.add_argument("--no-cuda", action="store_true")
	args = parser.parse_args()

	print(f"Loading {args.dataset} dataset...")
	train_df, test_df, meta = get_dataset(args.dataset)
	print(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")

	model_kwargs = dict(
		epochs=args.epochs,
		batch_size=args.batch_size,
		cuda=not args.no_cuda,
		verbose=True,
	)
	if args.model in ("cgan", "great", "tabddpm", "ctgan"):
		model_kwargs["target_col"] = meta["target"]

	print(f"Training {args.model} model...")
	model = get_model(args.model, **model_kwargs)
	model.fit(train_df, discrete_columns=meta["discrete_columns"]) 
	
	print(f"Generating {len(train_df)} synthetic samples...")
	n_synth = len(train_df)
	synth_df: pd.DataFrame = model.sample(n_synth)

	print("Computing fidelity metrics...")
	fidelity_metrics = compute_fidelity(
		real=train_df,
		synth=synth_df,
		categorical_columns=meta["categorical_columns"],
		numerical_columns=meta["numerical_columns"],
	)

	all_metrics = fidelity_metrics.copy()

	# Compute utility metrics
	print("Computing utility metrics (TSTR/TRTS)...")
	utility_metrics = compute_utility(
		real_train=train_df,
		real_test=test_df,
		synth_train=synth_df,
		target_col=meta["target"],
		categorical_columns=meta["categorical_columns"],
		numerical_columns=meta["numerical_columns"],
		model_types=["rf", "lr"]
	)
	all_metrics.update(utility_metrics)

	# Save results
	out_dir = Path(args.output_dir) / args.dataset / args.model
	ensure_dir(out_dir)
	
	print(f"Saving results to {out_dir}...")
	save_csv(out_dir / "synthetic.csv", synth_df)
	save_json(out_dir / "fidelity.json", fidelity_metrics)
	save_json(out_dir / "utility.json", utility_metrics)
	save_json(out_dir / "all_metrics.json", all_metrics)
	save_json(
		out_dir / "meta.json",
		{"dataset": args.dataset, "model": args.model, "num_rows": n_synth, **meta},
	)

	print(f"\n✅ Results saved to: {out_dir}")
	print("\n📊 Fidelity Metrics:")
	for key, value in fidelity_metrics.items():
		print(f"  {key}: {value:.4f}")
	
	print("\n📈 Utility Metrics:")
	for key, value in utility_metrics.items():
		if not np.isnan(value):
			print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
	main()
