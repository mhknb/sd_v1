#!/usr/bin/env python3
"""Run full benchmark for all models with 500 epochs and full dataset.
Shows a progress bar across models and streams logs to console and files.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from tqdm import tqdm

MODELS = [
	"cgan",
	"great",
	"tabddpm",
	"copulagan",
	"ctgan",
]

DATASET = "adult"
EPOCHS = 500
BATCH_SIZE = 256
OUTPUT_DIR = Path("outputs")
LOG_DIR = Path("logs/full_benchmark")


def run_model(model: str) -> int:
	"""Run a single model via run_benchmark.py and stream output to a log file."""
	LOG_DIR.mkdir(parents=True, exist_ok=True)
	log_path = LOG_DIR / f"{DATASET}_{model}.log"
	cmd = [
		sys.executable,
		"run_benchmark.py",
		"--model", model,
		"--dataset", DATASET,
		"--epochs", str(EPOCHS),
		"--batch-size", str(BATCH_SIZE),
	]
	# Stream stdout/stderr to both console and file
	with open(log_path, "w") as lf:
		proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
		for line in proc.stdout:  # type: ignore[arg-type]
			print(line, end="")
			lf.write(line)
			lf.flush()
		return_code = proc.wait()
	return return_code


def read_results(model: str) -> dict:
	model_dir = OUTPUT_DIR / DATASET / model
	results = {"model": model}
	fidelity = model_dir / "fidelity.json"
	utility = model_dir / "utility.json"
	if fidelity.exists():
		with open(fidelity) as f:
			results["fidelity"] = json.load(f)
	if utility.exists():
		with open(utility) as f:
			results["utility"] = json.load(f)
	return results


def main():
	print(f"Starting full benchmark: models={MODELS}, dataset={DATASET}, epochs={EPOCHS}, batch_size={BATCH_SIZE}")
	print(f"Logs: {LOG_DIR.resolve()}")
	all_results: list[dict] = []
	failures: list[str] = []
	
	for model in tqdm(MODELS, desc="Models", unit="model"):
		start = time.time()
		code = run_model(model)
		elapsed = time.time() - start
		if code != 0:
			print(f"[ERROR] Model {model} exited with code {code} after {elapsed:.1f}s. See logs.")
			failures.append(model)
		else:
			print(f"[OK] Model {model} completed in {elapsed/60:.1f} min")
			all_results.append(read_results(model))
	
	# Summary
	print("\n===== SUMMARY =====")
	for r in all_results:
		m = r.get("model")
		fid = r.get("fidelity", {})
		util = r.get("utility", {})
		print(f"\nModel: {m}")
		if fid:
			print(f"  Fidelity: ks={fid.get('ks_mean')}, jsd={fid.get('jsd_mean')}, corr={fid.get('corr_abs_diff_mean')}")
		else:
			print("  Fidelity: (missing)")
		if util:
			print("  Utility:")
			for k, v in util.items():
				print(f"    {k}: {v}")
		else:
			print("  Utility: (missing)")
	
	if failures:
		print(f"\nFailures: {failures}")
	else:
		print("\nAll models completed successfully.")


if __name__ == "__main__":
	main()
