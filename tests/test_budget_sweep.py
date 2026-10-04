from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from rl_research.cli import build_parser
from rl_research.config import load_experiment_config
from rl_research.experiments.budget_sweep import apply_budget_multiplier
from rl_research.experiments.sweep import run_seed_sweep


def test_budget_multiplier_default_preserves_budget_and_name() -> None:
    config = load_experiment_config("configs/train/mock_smoke.yaml")
    base_budget = config.budget.max_updates_per_window
    base_name = config.name

    metadata = apply_budget_multiplier(config, 1.0)

    assert config.budget.max_updates_per_window == base_budget
    assert config.name == base_name
    assert metadata["budget_multiplier"] == 1.0
    assert metadata["base_max_updates_per_window"] == base_budget
    assert metadata["effective_max_updates_per_window"] == base_budget
    assert metadata["budget_window_steps"] == config.budget.window_steps
    assert metadata["name_suffix"] == ""


def test_budget_multiplier_scales_budget_and_adds_suffix_and_tags() -> None:
    config = load_experiment_config("configs/train/mock_smoke.yaml")

    metadata = apply_budget_multiplier(config, 0.5)

    assert config.budget.max_updates_per_window == 60
    assert config.name == "mock_smoke_budget0p5x"
    assert metadata["budget_multiplier"] == 0.5
    assert metadata["base_max_updates_per_window"] == 120
    assert metadata["effective_max_updates_per_window"] == 60
    assert "budget_multiplier=0.5" in config.tags
    assert "base_max_updates_per_window=120" in config.tags
    assert "effective_max_updates_per_window=60" in config.tags


def test_budget_multiplier_uses_floor_and_minimum_one_update() -> None:
    config = load_experiment_config("configs/train/mock_smoke.yaml")
    config.budget.max_updates_per_window = 1

    metadata = apply_budget_multiplier(config, 0.5)

    assert config.budget.max_updates_per_window == 1
    assert metadata["effective_max_updates_per_window"] == 1


def test_cli_rejects_invalid_budget_multipliers() -> None:
    parser = build_parser()
    base_args = [
        "sweep-seeds",
        "--config",
        "configs/train/mock_smoke.yaml",
        "--seeds",
        "1",
    ]

    for bad_value in ("0", "-1", "not-a-number"):
        with pytest.raises(SystemExit):
            parser.parse_args([*base_args, "--budget-multiplier", bad_value])


def test_seed_sweep_dry_run_emits_budget_metadata(tmp_path: Path) -> None:
    config = load_experiment_config("configs/train/mock_smoke.yaml")
    config.output_dir = str(tmp_path)
    metadata = apply_budget_multiplier(config, 0.5)

    output = run_seed_sweep(
        config,
        seeds=[1],
        policies=["frozen", "uniform"],
        dry_run=True,
        budget_metadata=metadata,
    )

    assert "budget0p5x" in Path(output["seed_sweep_dir"]).name

    rows = pd.read_csv(output["rows_path"])
    assert set(rows["budget_multiplier"]) == {0.5}
    assert set(rows["base_max_updates_per_window"]) == {120}
    assert set(rows["effective_max_updates_per_window"]) == {60}
    assert set(rows["budget_window_steps"]) == {200}

    payload = json.loads(Path(output["json_path"]).read_text(encoding="utf-8"))
    assert payload["budget_multiplier"] == 0.5
    assert payload["base_max_updates_per_window"] == 120
    assert payload["effective_max_updates_per_window"] == 60
    assert payload["budget_window_steps"] == 200
