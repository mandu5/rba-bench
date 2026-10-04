"""Run directory and file logging utilities."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if is_dataclass(obj):
            return asdict(obj)
        return super().default(obj)


def prepare_run_dir(base_dir: str | Path, run_name: str) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(base_dir) / f"{timestamp}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: str | Path, payload: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, cls=NumpyJSONEncoder)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, cls=NumpyJSONEncoder) + "\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        with Path(path).open("w", encoding="utf-8") as handle:
            handle.write("")
        return
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def snapshot_config(path: str | Path, payload: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
