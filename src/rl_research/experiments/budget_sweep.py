"""Helpers for controlled update-budget multiplier sweeps."""

from __future__ import annotations

import argparse
import math
from typing import Any

from rl_research.config import ExperimentConfig


BUDGET_METADATA_KEYS = (
    "budget_multiplier",
    "base_max_updates_per_window",
    "effective_max_updates_per_window",
    "budget_window_steps",
)


def positive_budget_multiplier(value: str) -> float:
    try:
        multiplier = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--budget-multiplier must be a positive float.") from exc
    if multiplier <= 0.0:
        raise argparse.ArgumentTypeError("--budget-multiplier must be > 0.")
    return multiplier


def format_budget_multiplier_suffix(multiplier: float) -> str:
    multiplier = positive_budget_multiplier(str(multiplier))
    if math.isclose(multiplier, 1.0):
        return ""
    text = f"{multiplier:.6g}"
    if "." not in text:
        text = f"{text}.0"
    return f"_budget{text.replace('.', 'p')}x"


def apply_budget_multiplier(config: ExperimentConfig, multiplier: float) -> dict[str, Any]:
    multiplier = positive_budget_multiplier(str(multiplier))
    base_updates = int(config.budget.max_updates_per_window)
    effective_updates = max(1, math.floor(base_updates * multiplier))
    suffix = format_budget_multiplier_suffix(multiplier)

    config.budget.max_updates_per_window = effective_updates
    if suffix and not config.name.endswith(suffix):
        config.name = f"{config.name}{suffix}"

    if suffix:
        tags = [
            f"budget_multiplier={multiplier:g}",
            f"base_max_updates_per_window={base_updates}",
            f"effective_max_updates_per_window={effective_updates}",
        ]
        for tag in tags:
            if tag not in config.tags:
                config.tags.append(tag)

    return {
        "budget_multiplier": float(multiplier),
        "base_max_updates_per_window": base_updates,
        "effective_max_updates_per_window": effective_updates,
        "budget_window_steps": int(config.budget.window_steps),
        "name_suffix": suffix,
    }


def budget_metadata_payload(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return {}
    return {key: metadata[key] for key in BUDGET_METADATA_KEYS if key in metadata}
