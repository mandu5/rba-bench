"""Configuration models and YAML loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainConfig:
    algo: str = "sac"
    env_id: str = "MockContinuousControl-v0"
    seed: int = 7
    total_steps: int = 10_000
    pretrain_steps: int = 4_000
    device: str = "cpu"
    log_interval: int = 1_000
    eval_window: int = 10


@dataclass
class DriftScenario:
    shift_type: str = "actuator_gain"
    shift_step: int = 2_000
    severity: float = 0.25
    repeat_interval: int = 0
    repeat_count: int = 1


@dataclass
class AdaptationBudget:
    window_steps: int = 500
    max_updates_per_window: int = 250
    recent_replay_ratio: float = 0.5


@dataclass
class EvalConfig:
    eval_episodes: int = 3
    eval_interval_steps: int = 1_000
    deterministic: bool = True
    eval_seed_offset: int = 10_000


@dataclass
class CostConfig:
    include_update_count: bool = True
    include_replay_samples: bool = True
    include_wallclock_update_time: bool = True
    include_reward_ctrl: bool = True
    budget_score_update_penalty: float = 0.05


@dataclass
class ModelConfig:
    hidden_sizes: tuple[int, int] = (256, 256)
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha: float = 0.2
    autotune: bool = True
    batch_size: int = 256
    buffer_size: int = 100_000
    recent_buffer_size: int = 20_000
    learning_starts: int = 1_000
    policy_frequency: int = 2
    target_network_frequency: int = 1


@dataclass
class EvalRecord:
    return_mean: float
    return_auc: float
    recovery_auc: float
    steps_to_recovery: int
    update_cost: int
    replay_samples: int
    wallclock_update_time: float
    forgetting: float
    reward_per_update: float | None
    improvement_over_frozen: float | None = None
    improvement_per_1k_updates: float | None = None
    normalized_budgeted_score: float | None = None
    reward_ctrl_mean: float | None = None


@dataclass
class ExperimentConfig:
    name: str = "mock_smoke"
    train: TrainConfig = field(default_factory=TrainConfig)
    drift: DriftScenario = field(default_factory=DriftScenario)
    budget: AdaptationBudget = field(default_factory=AdaptationBudget)
    eval: EvalConfig = field(default_factory=EvalConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    adaptation_policy: str = "proposed"
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    output_dir: str = "results"
    tags: list[str] = field(default_factory=list)


def _merge_dataclass(instance: Any, raw: dict[str, Any]) -> Any:
    values = asdict(instance)
    values.update(raw)
    if "hidden_sizes" in values and isinstance(values["hidden_sizes"], list):
        values["hidden_sizes"] = tuple(int(v) for v in values["hidden_sizes"])
    return type(instance)(**values)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    cfg = ExperimentConfig()
    if "name" in raw:
        cfg.name = str(raw["name"])
    if "train" in raw:
        cfg.train = _merge_dataclass(cfg.train, raw["train"])
    if "drift" in raw:
        cfg.drift = _merge_dataclass(cfg.drift, raw["drift"])
    if "budget" in raw:
        cfg.budget = _merge_dataclass(cfg.budget, raw["budget"])
    if "eval" in raw:
        cfg.eval = _merge_dataclass(cfg.eval, raw["eval"])
    if "cost" in raw:
        cfg.cost = _merge_dataclass(cfg.cost, raw["cost"])
    if "model" in raw:
        cfg.model = _merge_dataclass(cfg.model, raw["model"])
    if "adaptation_policy" in raw:
        cfg.adaptation_policy = str(raw["adaptation_policy"])
    if "env_kwargs" in raw:
        cfg.env_kwargs = dict(raw["env_kwargs"] or {})
    if "output_dir" in raw:
        cfg.output_dir = str(raw["output_dir"])
    if "tags" in raw:
        cfg.tags = [str(tag) for tag in raw["tags"]]

    return cfg


def dump_experiment_config(config: ExperimentConfig) -> dict[str, Any]:
    return asdict(config)
