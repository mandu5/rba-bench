"""Evaluation metrics for reward, cost, and recovery."""

from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np

from rl_research.config import EvalRecord


def area_under_curve(values: Sequence[float]) -> float:
    series = np.asarray(values, dtype=np.float32)
    if len(series) == 0:
        return 0.0
    if len(series) == 1:
        return float(series[0])
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(series, dx=1.0))


def steps_to_recovery(values: Sequence[float], target: float) -> int:
    for idx, value in enumerate(values):
        if value >= target:
            return idx
    return -1


def recovery_auc(values: Sequence[float], target: float) -> float:
    series = np.asarray(values, dtype=np.float32)
    if len(series) == 0:
        return 0.0
    scale = max(abs(target), 1e-6)
    clipped = 1.0 - np.maximum(target - series, 0.0) / scale
    clipped = np.clip(clipped, 0.0, 1.0)
    return area_under_curve(clipped) / max(len(clipped), 1)


def forgetting(reference_value: float, values: Sequence[float]) -> float:
    if len(values) == 0:
        return 0.0
    return float(max(reference_value - np.min(values), 0.0))


def reward_per_cost(total_reward: float, update_cost: int) -> float:
    return float(total_reward / max(update_cost, 1))


def reward_per_update(total_reward: float, update_cost: int) -> float | None:
    if update_cost <= 0:
        return None
    return float(total_reward / update_cost)


def ranking_disagreement(metric_a: dict[str, float], metric_b: dict[str, float]) -> int:
    keys = sorted(set(metric_a) & set(metric_b))
    disagreement = 0
    for left, right in combinations(keys, 2):
        order_a = metric_a[left] >= metric_a[right]
        order_b = metric_b[left] >= metric_b[right]
        if order_a != order_b:
            disagreement += 1
    return disagreement


def summarize_eval(
    pretrain_returns: Sequence[float],
    deploy_returns: Sequence[float],
    update_cost: int,
    replay_samples: int,
    wallclock_update_time: float,
    reference_return: float | None = None,
    frozen_return: float | None = None,
    update_penalty: float = 0.05,
    reward_ctrl_values: Sequence[float] | None = None,
) -> EvalRecord:
    reference = reference_return
    if reference is None:
        reference = float(np.mean(pretrain_returns[-5:])) if pretrain_returns else 0.0
    deploy_series = np.asarray(deploy_returns, dtype=np.float32)
    mean_return = float(np.mean(deploy_series)) if len(deploy_series) else 0.0
    target = reference - 0.1 * abs(reference) if reference != 0.0 else 0.0
    improvement = None if frozen_return is None else mean_return - frozen_return
    improvement_per_1k = None
    if improvement is not None and update_cost > 0:
        improvement_per_1k = improvement / update_cost * 1000.0
    scale = max(abs(reference), abs(frozen_return or 0.0), 1e-6)
    normalized_return = mean_return / scale
    normalized_cost = update_cost / 1000.0
    budgeted_score = normalized_return - update_penalty * normalized_cost
    reward_ctrl_mean = None
    if reward_ctrl_values:
        reward_ctrl_mean = float(np.mean(np.asarray(reward_ctrl_values, dtype=np.float32)))
    return EvalRecord(
        return_mean=mean_return,
        return_auc=area_under_curve(deploy_series.tolist()),
        recovery_auc=recovery_auc(deploy_series.tolist(), target=target),
        steps_to_recovery=steps_to_recovery(deploy_series.tolist(), target),
        update_cost=update_cost,
        replay_samples=replay_samples,
        wallclock_update_time=wallclock_update_time,
        forgetting=forgetting(reference, deploy_series.tolist()),
        reward_per_update=reward_per_update(float(np.sum(deploy_series)), update_cost),
        improvement_over_frozen=improvement,
        improvement_per_1k_updates=improvement_per_1k,
        normalized_budgeted_score=budgeted_score,
        reward_ctrl_mean=reward_ctrl_mean,
    )
