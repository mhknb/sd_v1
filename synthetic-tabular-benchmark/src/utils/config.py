from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class BenchmarkConfig:
    models: Dict[str, Any]
    datasets: Dict[str, Any]
    evaluation: Dict[str, Any]


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_all_configs(base_dir: str | Path) -> BenchmarkConfig:
    base = Path(base_dir)
    models = load_yaml(base / "configs" / "models.yaml")
    datasets = load_yaml(base / "configs" / "datasets.yaml")
    evaluation = load_yaml(base / "configs" / "evaluation.yaml")
    return BenchmarkConfig(models=models, datasets=datasets, evaluation=evaluation)


