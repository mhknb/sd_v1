from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
	p = Path(path)
	p.mkdir(parents=True, exist_ok=True)
	return p


def save_json(path: str | Path, data: Dict[str, Any]) -> None:
	p = Path(path)
	ensure_dir(p.parent)
	with p.open("w", encoding="utf-8") as f:
		json.dump(data, f, indent=2, ensure_ascii=False)


def save_csv(path: str | Path, df: pd.DataFrame) -> None:
	p = Path(path)
	ensure_dir(p.parent)
	df.to_csv(p, index=False)
