"""Policy-sweep utilities."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from rl_research.config import ExperimentConfig
from rl_research.experiments.budget_sweep import budget_metadata_payload
from rl_research.experiments.runner import _apply_step_override, prepare_pretrained_state, run_deployment_from_pretrained
from rl_research.utils.logging import prepare_run_dir, write_csv, write_json


DEFAULT_POLICIES = ["frozen", "fixedadapt", "recentonly", "staticmix", "proposed"]

# v2.1.1 default baseline roster — same total update budget for all 9 entries.
# Added `recency-weighted` (Soft-Recency) after v2.1 pilot showed it as a
# strong baseline in its own right. `budgeted-novelty` was renamed from
# `budgeted-pred` for honest naming (state-delta novelty, not predictability
# residual); both registry keys are accepted.
DEFAULT_POLICIES_V21 = [
    "frozen",
    "uniform",
    "recentonly",
    "recency-weighted",
    "per",
    "staticmix",
    "budgeted-td",
    "budgeted-grad",
    "budgeted-novelty",
]


def run_policy_sweep(
    config: ExperimentConfig,
    policies: list[str] | None = None,
    total_steps_override: int | None = None,
    dry_run: bool = False,
    sweep_parent: str | Path | None = None,
    budget_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected_policies = policies or DEFAULT_POLICIES
    budget_payload = budget_metadata_payload(budget_metadata)
    config_for_pretrain = deepcopy(config)
    pretrained_state = None if dry_run else prepare_pretrained_state(config_for_pretrain, total_steps_override)
    parent = Path(sweep_parent) if sweep_parent else Path(config.output_dir) / "sweeps"
    sweep_dir = prepare_run_dir(parent, f"{config.name}_policy_sweep")
    if pretrained_state is not None:
        torch.save(
            {
                "agent_state": pretrained_state["agent_state"],
                "replay_state": pretrained_state["replay_state"],
                "reference_eval": pretrained_state["reference_eval"],
                "pretrain_summary": {
                    "reference_return": pretrained_state["pretrain"]["reference_return"],
                    "reference_reward_rate": pretrained_state["pretrain"]["reference_reward_rate"],
                    "total_updates": pretrained_state["pretrain"]["total_updates"],
                },
            },
            sweep_dir / "pretrained_state.pt",
        )

    summaries: list[dict[str, Any]] = []

    for policy in selected_policies:
        policy_config = deepcopy(config)
        _apply_step_override(policy_config, total_steps_override)
        policy_config.adaptation_policy = policy
        policy_config.output_dir = str(sweep_dir)
        run_name = f"{config.name}_{policy}"
        if dry_run:
            policy_config.train.total_steps = total_steps_override or policy_config.train.total_steps
            summary = {
                "run_name": run_name,
                "adaptation_policy": policy,
                "env_id": policy_config.train.env_id,
                "seed": policy_config.train.seed,
                "deploy_eval": {},
                "run_dir": str(sweep_dir / run_name),
            }
        else:
            summary = run_deployment_from_pretrained(
            config=policy_config,
            pretrained_state=pretrained_state,
            run_name_override=run_name,
            dry_run=dry_run,
            )
        row = _flatten_summary(summary)
        row.update(budget_payload)
        summaries.append(row)

    summaries = _attach_frozen_improvements(summaries)
    summaries = _attach_normalized_budgeted_score_v2(summaries)
    sweep_name = f"{config.name}_policy_sweep"
    csv_path = sweep_dir / f"{sweep_name}.csv"
    json_path = sweep_dir / f"{sweep_name}.json"
    write_csv(csv_path, summaries)
    rankings = rank_summaries(summaries)
    write_json(
        json_path,
        {
            "config_name": config.name,
            "policies": selected_policies,
            "total_steps_override": total_steps_override,
            "dry_run": dry_run,
            "sweep_dir": str(sweep_dir),
            "rows": summaries,
            "rankings": rankings,
            "ranking_disagreement_vs_reward_only": ranking_disagreements(rankings),
            **budget_payload,
        },
    )
    return {
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "sweep_dir": str(sweep_dir),
        "rows": summaries,
        "rankings": rankings,
        "ranking_disagreement_vs_reward_only": ranking_disagreements(rankings),
        **budget_payload,
    }


def run_seed_sweep(
    config: ExperimentConfig,
    seeds: list[int],
    policies: list[str] | None = None,
    total_steps_override: int | None = None,
    dry_run: bool = False,
    budget_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    budget_payload = budget_metadata_payload(budget_metadata)
    seed_root = prepare_run_dir(Path(config.output_dir) / "sweeps", f"{config.name}_seed_sweep")
    all_rows: list[dict[str, Any]] = []
    seed_outputs: list[dict[str, Any]] = []

    for seed in seeds:
        seed_config = deepcopy(config)
        seed_config.train.seed = int(seed)
        seed_config.name = f"{config.name}_seed{seed}"
        seed_config.output_dir = str(seed_root)
        output = run_policy_sweep(
            seed_config,
            policies=policies,
            total_steps_override=total_steps_override,
            dry_run=dry_run,
            sweep_parent=seed_root,
            budget_metadata=budget_metadata,
        )
        seed_outputs.append(output)
        for row in output["rows"]:
            row = dict(row)
            row["seed"] = int(seed)
            all_rows.append(row)

    aggregate_rows = aggregate_seed_rows(all_rows)
    for row in aggregate_rows:
        row.update(budget_payload)
    rows_path = seed_root / f"{config.name}_seed_sweep_rows.csv"
    aggregate_path = seed_root / f"{config.name}_seed_sweep_aggregate.csv"
    json_path = seed_root / f"{config.name}_seed_sweep.json"
    write_csv(rows_path, all_rows)
    write_csv(aggregate_path, aggregate_rows)
    write_json(
        json_path,
        {
            "config_name": config.name,
            "seeds": seeds,
            "policies": policies or DEFAULT_POLICIES,
            "total_steps_override": total_steps_override,
            "dry_run": dry_run,
            "seed_outputs": seed_outputs,
            "rows_path": str(rows_path),
            "aggregate_path": str(aggregate_path),
            "aggregate_rows": aggregate_rows,
            **budget_payload,
        },
    )
    return {
        "seed_sweep_dir": str(seed_root),
        "rows_path": str(rows_path),
        "aggregate_path": str(aggregate_path),
        "json_path": str(json_path),
        "aggregate_rows": aggregate_rows,
        **budget_payload,
    }


def aggregate_seed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    policies = sorted({str(row["adaptation_policy"]) for row in rows})
    metrics = [
        "return_mean",
        "recovery_auc",
        "forgetting",
        "update_cost",
        "improvement_over_frozen",
        "improvement_per_1k_updates",
        "normalized_budgeted_score",
        "reward_ctrl_mean",
    ]
    aggregate: list[dict[str, Any]] = []
    for policy in policies:
        policy_rows = [row for row in rows if str(row["adaptation_policy"]) == policy]
        output: dict[str, Any] = {
            "adaptation_policy": policy,
            "n_seeds": len(policy_rows),
        }
        for metric in metrics:
            values = [row.get(metric) for row in policy_rows if row.get(metric) is not None]
            if values:
                numeric = [float(value) for value in values]
                output[f"{metric}_mean"] = sum(numeric) / len(numeric)
                output[f"{metric}_std"] = _std(numeric)
            else:
                output[f"{metric}_mean"] = None
                output[f"{metric}_std"] = None
        aggregate.append(output)
    return aggregate


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return variance**0.5


def _flatten_summary(summary: dict[str, Any]) -> dict[str, Any]:
    deploy_eval = summary.get("deploy_eval", {})
    return {
        "run_name": summary.get("run_name"),
        "adaptation_policy": summary.get("adaptation_policy"),
        "env_id": summary.get("env_id"),
        "seed": summary.get("seed"),
        "return_mean": deploy_eval.get("return_mean"),
        "return_auc": deploy_eval.get("return_auc"),
        "recovery_auc": deploy_eval.get("recovery_auc"),
        "steps_to_recovery": deploy_eval.get("steps_to_recovery"),
        "update_cost": deploy_eval.get("update_cost"),
        "replay_samples": deploy_eval.get("replay_samples"),
        "wallclock_update_time": deploy_eval.get("wallclock_update_time"),
        "forgetting": deploy_eval.get("forgetting"),
        "reward_per_update": deploy_eval.get("reward_per_update"),
        "improvement_over_frozen": deploy_eval.get("improvement_over_frozen"),
        "improvement_per_1k_updates": deploy_eval.get("improvement_per_1k_updates"),
        "normalized_budgeted_score": deploy_eval.get("normalized_budgeted_score"),
        "normalized_budgeted_score_v2": None,  # populated by sweep aggregator
        "reward_ctrl_mean": deploy_eval.get("reward_ctrl_mean"),
        "run_dir": summary.get("run_dir"),
    }


def _attach_frozen_improvements(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frozen_rows = [row for row in rows if str(row.get("adaptation_policy", "")).lower() == "frozen"]
    if not frozen_rows:
        return rows
    frozen_return = float(frozen_rows[0].get("return_mean") or 0.0)
    for row in rows:
        return_mean = row.get("return_mean")
        update_cost = int(row.get("update_cost") or 0)
        if return_mean is None:
            continue
        improvement = float(return_mean) - frozen_return
        row["improvement_over_frozen"] = improvement
        row["improvement_per_1k_updates"] = None if update_cost <= 0 else improvement / update_cost * 1000.0
    return rows


def _attach_normalized_budgeted_score_v2(
    rows: list[dict[str, Any]], update_penalty: float = 0.5
) -> list[dict[str, Any]]:
    """v2.1.1 recalibrated score.

    Per-run `normalized_budgeted_score` divides update_cost by a hardcoded
    1000, which becomes cost-dominated when update budgets are large
    (e.g. Frozen always wins for max_budget ≈ 35k updates). v2.1.1 attaches
    a sweep-relative score `normalized_budgeted_score_v2` that uses the
    sweep-wide max cost as the denominator, keeping cost_ratio ∈ [0, 1].
    """
    max_cost = max((float(row.get("update_cost") or 0.0) for row in rows), default=0.0)
    if max_cost <= 0:
        for row in rows:
            row["normalized_budgeted_score_v2"] = None
        return rows
    non_frozen_returns = [
        abs(float(row.get("return_mean") or 0.0))
        for row in rows
        if str(row.get("adaptation_policy", "")).lower() != "frozen"
        and row.get("return_mean") is not None
    ]
    return_scale = max(non_frozen_returns, default=0.0)
    if return_scale < 1e-6:
        for row in rows:
            row["normalized_budgeted_score_v2"] = None
        return rows
    for row in rows:
        ret = row.get("return_mean")
        cost = float(row.get("update_cost") or 0.0)
        if ret is None:
            row["normalized_budgeted_score_v2"] = None
            continue
        normalized_return = float(ret) / return_scale
        cost_ratio = cost / max_cost
        row["normalized_budgeted_score_v2"] = float(normalized_return - update_penalty * cost_ratio)
    return rows


def rank_summaries(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    def sorted_names(metric: str, reverse: bool = True) -> list[str]:
        def metric_value(row: dict[str, Any]) -> float:
            value = row.get(metric)
            if value is None:
                return float("-inf") if reverse else float("inf")
            return float(value)

        ranked = sorted(
            rows,
            key=metric_value,
            reverse=reverse,
        )
        return [str(row["adaptation_policy"]) for row in ranked]

    return {
        "reward_only": sorted_names("return_mean", reverse=True),
        "recovery": sorted_names("recovery_auc", reverse=True),
        "reward_per_update": sorted_names("reward_per_update", reverse=True),
        "normalized_budgeted_score": sorted_names("normalized_budgeted_score", reverse=True),
        "normalized_budgeted_score_v2": sorted_names("normalized_budgeted_score_v2", reverse=True),
        "forgetting": sorted_names("forgetting", reverse=False),
        "wallclock_update_time": sorted_names("wallclock_update_time", reverse=False),
    }


def ranking_disagreements(rankings: dict[str, list[str]]) -> dict[str, int]:
    reward_ranking = rankings.get("reward_only", [])
    return {
        metric: _pairwise_disagreement(reward_ranking, ranking)
        for metric, ranking in rankings.items()
        if metric != "reward_only"
    }


def _pairwise_disagreement(left: list[str], right: list[str]) -> int:
    left_position = {name: idx for idx, name in enumerate(left)}
    right_position = {name: idx for idx, name in enumerate(right)}
    names = sorted(set(left_position) & set(right_position))
    disagreements = 0
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            left_order = left_position[first] < left_position[second]
            right_order = right_position[first] < right_position[second]
            if left_order != right_order:
                disagreements += 1
    return disagreements
