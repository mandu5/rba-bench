"""Experiment runner for the RL research scaffold."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import time
from typing import Any

import numpy as np

from rl_research.adaptation.budget import BudgetDecision, WindowStats, scheduled_updates_this_step
from rl_research.adaptation.policies import AdaptationPolicy, build_adaptation_policy
from rl_research.agents.sac import SACAgent
from rl_research.config import ExperimentConfig, dump_experiment_config
from rl_research.data.replay import DualReplayBuffer
from rl_research.envs.base import get_space_bounds, reset_env, step_env
from rl_research.envs.drift import DriftWrapper
from rl_research.envs.factory import create_env
from rl_research.evaluation.metrics import summarize_eval
from rl_research.evaluation.plasticity import compute_plasticity_snapshot, snapshot_to_dict
from rl_research.utils.logging import prepare_run_dir, snapshot_config, write_csv, write_json, write_jsonl
from rl_research.utils.seed import set_global_seeds


def _space_dim(space: Any) -> int:
    return int(np.prod(space.shape))


def _finalize_window(
    window_index: int,
    step_rewards: list[float],
    td_error_ema: float,
    drift_events: int,
    updates_used: int,
    replay_samples: int,
    update_seconds: float,
    reference_reward_rate: float,
    grad_norm_ema: float = 0.0,
    predictability_residual_ema: float = 0.0,
) -> WindowStats:
    mean_return = float(np.mean(step_rewards)) if step_rewards else 0.0
    denom = max(abs(reference_reward_rate), 1e-6)
    return_drop = max(reference_reward_rate - mean_return, 0.0) / denom
    return WindowStats(
        window_index=window_index,
        mean_return=mean_return,
        reference_return=reference_reward_rate,
        return_drop=return_drop,
        td_error_ema=td_error_ema,
        drift_events=drift_events,
        updates_used=updates_used,
        replay_samples=replay_samples,
        mean_update_time=update_seconds / max(updates_used, 1),
        steps_elapsed=len(step_rewards),
        grad_norm_ema=grad_norm_ema,
        predictability_residual_ema=predictability_residual_ema,
    )


def _random_action(space: Any, rng: np.random.Generator) -> np.ndarray:
    if hasattr(space, "sample"):
        try:
            return np.asarray(space.sample(), dtype=np.float32)
        except TypeError:
            return np.asarray(space.sample(rng), dtype=np.float32)
    low, high = get_space_bounds(space)
    return rng.uniform(low, high).astype(np.float32)


def evaluate_policy(
    agent: SACAgent,
    config: ExperimentConfig,
    *,
    step: int,
    phase: str,
    seed: int,
    drift_events: int = 0,
) -> dict[str, Any]:
    env = create_env(config.train.env_id, config.env_kwargs)
    if drift_events > 0:
        env = DriftWrapper(env, config.drift)
        env.force_apply_events(drift_events)

    episode_returns: list[float] = []
    reward_ctrl_values: list[float] = []
    try:
        for episode in range(config.eval.eval_episodes):
            obs, _ = reset_env(env, seed=seed + episode)
            done = False
            total_return = 0.0
            total_reward_ctrl = 0.0
            while not done:
                action = agent.act(obs, deterministic=config.eval.deterministic)
                obs, reward, terminated, truncated, info = step_env(env, action)
                done = terminated or truncated
                total_return += reward
                if "reward_ctrl" in info:
                    total_reward_ctrl += float(info["reward_ctrl"])
            episode_returns.append(total_return)
            reward_ctrl_values.append(total_reward_ctrl)
    finally:
        if hasattr(env, "close"):
            env.close()

    return {
        "phase": phase,
        "step": step,
        "drift_events": drift_events,
        "return_mean": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "return_std": float(np.std(episode_returns)) if episode_returns else 0.0,
        "episode_returns": episode_returns,
        "reward_ctrl_mean": float(np.mean(reward_ctrl_values)) if reward_ctrl_values else None,
    }


def _run_pretrain_phase(
    env: Any,
    agent: SACAgent,
    replay: DualReplayBuffer,
    config: ExperimentConfig,
    rng: np.random.Generator,
) -> dict[str, Any]:
    obs, _ = reset_env(env, seed=config.train.seed)
    episode_return = 0.0
    episode_length = 0
    episodes: list[dict[str, Any]] = []
    update_logs: list[dict[str, Any]] = []
    td_error_ema = 0.0
    total_reward = 0.0
    total_updates = 0

    for step in range(config.train.pretrain_steps):
        if step < config.model.learning_starts:
            action = _random_action(env.action_space, rng)
        else:
            action = agent.act(obs, deterministic=False)

        next_obs, reward, terminated, truncated, _ = step_env(env, action)
        done = terminated or truncated
        replay.add(obs, action, reward, next_obs, done)
        obs = next_obs

        total_reward += reward
        episode_return += reward
        episode_length += 1

        if len(replay) >= config.model.learning_starts:
            batch, sample_meta = replay.sample_mixed(
                batch_size=config.model.batch_size,
                recent_ratio=0.5,
                rng=rng,
            )
            metrics = agent.update(batch, global_step=step + 1)
            td_error_ema = 0.95 * td_error_ema + 0.05 * float(metrics["td_error_mean"])
            total_updates += 1
            update_logs.append(
                {
                    "phase": "pretrain",
                    "step": step + 1,
                    "recent_ratio": 0.5,
                    "updates": 1,
                    "recent_samples": sample_meta["recent_samples"],
                    "historical_samples": sample_meta["historical_samples"],
                    **{k: v for k, v in metrics.items() if k != "td_errors_per_sample"},
                }
            )

        if done:
            episodes.append(
                {
                    "phase": "pretrain",
                    "episode_index": len(episodes),
                    "return": episode_return,
                    "length": episode_length,
                    "end_step": step + 1,
                    "partial": False,
                }
            )
            obs, _ = reset_env(env)
            episode_return = 0.0
            episode_length = 0

    if episode_length > 0:
        episodes.append(
            {
                "phase": "pretrain",
                "episode_index": len(episodes),
                "return": episode_return,
                "length": episode_length,
                "end_step": config.train.pretrain_steps,
                "partial": True,
            }
        )

    reward_rate = total_reward / max(config.train.pretrain_steps, 1)
    reference_return = float(np.mean([row["return"] for row in episodes[-5:]])) if episodes else 0.0
    return {
        "episodes": episodes,
        "update_logs": update_logs,
        "reference_return": reference_return,
        "reference_reward_rate": reward_rate,
        "td_error_ema": td_error_ema,
        "total_updates": total_updates,
    }


def _run_deploy_phase(
    env: Any,
    agent: SACAgent,
    replay: DualReplayBuffer,
    policy: AdaptationPolicy,
    config: ExperimentConfig,
    rng: np.random.Generator,
    reference_return: float,
    reference_reward_rate: float,
) -> dict[str, Any]:
    deploy_steps = max(config.train.total_steps - config.train.pretrain_steps, 0)
    obs, _ = reset_env(env, seed=config.train.seed + 1)
    episode_return = 0.0
    episode_length = 0
    episodes: list[dict[str, Any]] = []
    update_logs: list[dict[str, Any]] = []
    window_logs: list[dict[str, Any]] = []
    eval_records: list[dict[str, Any]] = []
    td_error_ema = 0.0

    decision = policy.decide(stats=None, budget=config.budget)
    updates_remaining = decision.max_updates
    window_rewards: list[float] = []
    window_updates = 0
    window_replay_samples = 0
    window_update_time = 0.0
    window_index = 0
    total_updates = 0
    total_replay_samples = 0
    total_update_time = 0.0
    grad_norm_ema = 0.0
    predictability_residual_ema = 0.0
    plasticity_snapshots: list[dict[str, float | int]] = []

    for step in range(deploy_steps):
        if len(replay) < config.model.learning_starts:
            action = _random_action(env.action_space, rng)
        else:
            action = agent.act(obs, deterministic=False)

        next_obs, reward, terminated, truncated, info = step_env(env, action)
        done = terminated or truncated
        replay.add(obs, action, reward, next_obs, done)
        obs = next_obs

        episode_return += reward
        episode_length += 1
        window_rewards.append(reward)

        if len(replay) >= config.model.learning_starts and updates_remaining > 0:
            step_in_window = (step % config.budget.window_steps) + 1
            remaining_steps = config.budget.window_steps - step_in_window + 1
            updates_now = scheduled_updates_this_step(updates_remaining, remaining_steps)
            for _ in range(updates_now):
                start = time.perf_counter()
                if decision.sampling_mode == "weighted" and decision.weight_fn is not None:
                    batch, sample_meta = replay.sample_weighted(
                        batch_size=config.model.batch_size,
                        weight_fn=decision.weight_fn,
                        rng=rng,
                        beta_is=decision.beta_is,
                        source="historical",
                    )
                else:
                    batch, sample_meta = replay.sample_mixed(
                        batch_size=config.model.batch_size,
                        recent_ratio=decision.recent_ratio,
                        rng=rng,
                    )
                metrics = agent.update(batch, global_step=config.train.pretrain_steps + step + 1)
                duration = time.perf_counter() - start
                td_error_ema = 0.95 * td_error_ema + 0.05 * metrics["td_error_mean"]
                grad_norm_ema = 0.95 * grad_norm_ema + 0.05 * float(metrics.get("grad_norm_mean", 0.0))
                predictability_residual_ema = (
                    0.95 * predictability_residual_ema
                    + 0.05 * float(metrics.get("predictability_residual_mean", 0.0))
                )
                # Update PER priorities with fresh per-sample TD-errors when available.
                if (
                    decision.sampling_mode == "weighted"
                    and batch.indices is not None
                    and "td_errors_per_sample" in metrics
                ):
                    replay.update_priorities(
                        indices=batch.indices,
                        priorities=np.asarray(metrics["td_errors_per_sample"], dtype=np.float32),
                        source=sample_meta.get("weighted_source", "historical"),
                    )
                # v2.1.1: Backfill per-sample input gradient norms (Budgeted-Grad).
                if (
                    batch.indices is not None
                    and metrics.get("input_grads_per_sample") is not None
                ):
                    replay.update_input_grads(
                        indices=batch.indices,
                        grads=np.asarray(metrics["input_grads_per_sample"], dtype=np.float32),
                        source=sample_meta.get("weighted_source", "historical"),
                    )
                updates_remaining -= 1
                window_updates += 1
                total_updates += 1
                replay_used = sample_meta.get("recent_samples", 0) + sample_meta.get(
                    "historical_samples", 0
                )
                if replay_used == 0:
                    replay_used = sample_meta.get("weighted_samples", config.model.batch_size)
                window_replay_samples += replay_used
                total_replay_samples += replay_used
                window_update_time += duration
                total_update_time += duration
                update_logs.append(
                    {
                        "phase": "deploy",
                        "step": config.train.pretrain_steps + step + 1,
                        "window_index": window_index,
                        "sampling_mode": decision.sampling_mode,
                        "recent_ratio": decision.recent_ratio,
                        "updates": 1,
                        "recent_samples": sample_meta.get("recent_samples", 0),
                        "historical_samples": sample_meta.get("historical_samples", 0),
                        "update_seconds": duration,
                        **{k: v for k, v in metrics.items() if k not in ("td_errors_per_sample", "input_grads_per_sample")},
                    }
                )

        if done:
            episodes.append(
                {
                    "phase": "deploy",
                    "episode_index": len(episodes),
                    "return": episode_return,
                    "length": episode_length,
                    "end_step": config.train.pretrain_steps + step + 1,
                    "partial": False,
                }
            )
            obs, _ = reset_env(env)
            episode_return = 0.0
            episode_length = 0

        window_complete = ((step + 1) % config.budget.window_steps == 0) or (step + 1 == deploy_steps)
        if window_complete:
            current_drift_events = int(info.get("drift_events", 0))
            stats = _finalize_window(
                window_index=window_index,
                step_rewards=window_rewards,
                td_error_ema=td_error_ema,
                drift_events=current_drift_events,
                updates_used=window_updates,
                replay_samples=window_replay_samples,
                update_seconds=window_update_time,
                reference_reward_rate=reference_reward_rate,
                grad_norm_ema=grad_norm_ema,
                predictability_residual_ema=predictability_residual_ema,
            )
            window_logs.append(
                {
                    "window_index": stats.window_index,
                    "mean_return": stats.mean_return,
                    "reference_return": stats.reference_return,
                    "return_drop": stats.return_drop,
                    "td_error_ema": stats.td_error_ema,
                    "drift_events": stats.drift_events,
                    "updates_used": stats.updates_used,
                    "replay_samples": stats.replay_samples,
                    "mean_update_time": stats.mean_update_time,
                    "steps_elapsed": stats.steps_elapsed,
                    "decision_recent_ratio": decision.recent_ratio,
                    "decision_reason": decision.reason,
                }
            )
            global_step = config.train.pretrain_steps + step + 1
            should_eval = (
                (step + 1) == deploy_steps
                or (step + 1) % max(config.eval.eval_interval_steps, 1) == 0
            )
            if should_eval:
                eval_records.append(
                    evaluate_policy(
                        agent,
                        config,
                        step=global_step,
                        phase="deploy_eval",
                        seed=config.train.seed + config.eval.eval_seed_offset + global_step,
                        drift_events=current_drift_events,
                    )
                )
            # Plasticity snapshot at window boundary, using a fresh batch from
            # the historical buffer as the probe set.
            try:
                if len(replay) >= config.model.batch_size:
                    probe_batch = replay.historical.sample(
                        batch_size=min(config.model.batch_size, len(replay)),
                        rng=rng,
                    )
                    snap = compute_plasticity_snapshot(
                        agent.actor,
                        sample_obs=probe_batch.observations,
                        step=global_step,
                        window_index=window_index,
                        grad_norm_ema=grad_norm_ema,
                        include_ntk=False,
                        device=agent.device,
                    )
                    plasticity_snapshots.append(snapshot_to_dict(snap))
            except Exception:  # pragma: no cover - diagnostic must not break runs
                pass
            window_index += 1
            decision = policy.decide(stats=stats, budget=config.budget)
            updates_remaining = decision.max_updates
            window_rewards = []
            window_updates = 0
            window_replay_samples = 0
            window_update_time = 0.0

    if episode_length > 0:
        episodes.append(
            {
                "phase": "deploy",
                "episode_index": len(episodes),
                "return": episode_return,
                "length": episode_length,
                "end_step": config.train.total_steps,
                "partial": True,
            }
        )
    return {
        "episodes": episodes,
        "update_logs": update_logs,
        "window_logs": window_logs,
        "eval_records": eval_records,
        "plasticity_snapshots": plasticity_snapshots,
        "total_updates": total_updates,
        "total_replay_samples": total_replay_samples,
        "total_update_time": total_update_time,
        "reference_return": reference_return,
    }


def _apply_step_override(config: ExperimentConfig, total_steps_override: int | None) -> None:
    if total_steps_override is not None:
        config.train.total_steps = int(total_steps_override)
        if config.train.pretrain_steps >= config.train.total_steps:
            config.train.pretrain_steps = max(config.train.total_steps // 2, 1)
        else:
            config.train.pretrain_steps = min(config.train.pretrain_steps, config.train.total_steps)
        if config.model.learning_starts >= config.train.pretrain_steps:
            config.model.learning_starts = min(
                config.model.learning_starts,
                max(config.train.pretrain_steps // 4, 16),
            )
        config.model.batch_size = min(config.model.batch_size, 64)
        config.eval.eval_interval_steps = min(
            config.eval.eval_interval_steps,
            max(config.train.total_steps - config.train.pretrain_steps, 1),
        )


def _make_agent_and_replay(config: ExperimentConfig) -> tuple[SACAgent, DualReplayBuffer]:
    env = create_env(config.train.env_id, config.env_kwargs)
    try:
        obs_dim = _space_dim(env.observation_space)
        action_low, action_high = get_space_bounds(env.action_space)
        act_dim = int(np.prod(action_low.shape))
    finally:
        if hasattr(env, "close"):
            env.close()

    agent = SACAgent(
        obs_dim=obs_dim,
        action_low=action_low,
        action_high=action_high,
        config=config.model,
        device=config.train.device,
    )
    replay = DualReplayBuffer(
        obs_dim=obs_dim,
        act_dim=act_dim,
        capacity=config.model.buffer_size,
        recent_capacity=config.model.recent_buffer_size,
    )
    return agent, replay


def prepare_pretrained_state(
    config: ExperimentConfig,
    total_steps_override: int | None = None,
) -> dict[str, Any]:
    _apply_step_override(config, total_steps_override)
    set_global_seeds(config.train.seed)
    rng = np.random.default_rng(config.train.seed)

    base_env = create_env(config.train.env_id, config.env_kwargs)
    agent, replay = _make_agent_and_replay(config)
    try:
        pretrain = _run_pretrain_phase(base_env, agent, replay, config, rng)
        reference_eval = evaluate_policy(
            agent,
            config,
            step=config.train.pretrain_steps,
            phase="pretrain_reference_eval",
            seed=config.train.seed + config.eval.eval_seed_offset,
            drift_events=0,
        )
    finally:
        if hasattr(base_env, "close"):
            base_env.close()

    return {
        "agent_state": deepcopy(agent.state_dict()),
        "replay_state": deepcopy(replay.state_dict()),
        "pretrain": pretrain,
        "reference_eval": reference_eval,
        "config": dump_experiment_config(config),
    }


def run_deployment_from_pretrained(
    config: ExperimentConfig,
    pretrained_state: dict[str, Any],
    run_name_override: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    set_global_seeds(config.train.seed)
    rng = np.random.default_rng(config.train.seed + 1)
    run_dir = prepare_run_dir(config.output_dir, run_name_override or config.name)
    snapshot_config(run_dir / "config.snapshot.yaml", dump_experiment_config(config))

    if dry_run:
        payload = {
            "run_dir": str(run_dir),
            "config": dump_experiment_config(config),
        }
        write_json(run_dir / "dry_run.json", payload)
        return payload

    deploy_env = DriftWrapper(create_env(config.train.env_id, config.env_kwargs), config.drift)
    agent, replay = _make_agent_and_replay(config)
    agent.load_state_dict(deepcopy(pretrained_state["agent_state"]))
    replay.load_state_dict(deepcopy(pretrained_state["replay_state"]))
    policy = build_adaptation_policy(config.adaptation_policy)

    pretrain = pretrained_state["pretrain"]
    reference_eval = pretrained_state["reference_eval"]
    deploy = _run_deploy_phase(
        deploy_env,
        agent,
        replay,
        policy,
        config,
        rng,
        reference_return=reference_eval["return_mean"],
        reference_reward_rate=pretrain["reference_reward_rate"],
    )
    deploy_eval_returns = [row["return_mean"] for row in deploy["eval_records"]]
    if not deploy_eval_returns:
        deploy_eval_returns = [row["return"] for row in deploy["episodes"]]
    reward_ctrl_values = [
        row["reward_ctrl_mean"]
        for row in deploy["eval_records"]
        if row.get("reward_ctrl_mean") is not None
    ]

    summary = {
        "run_name": run_name_override or config.name,
        "adaptation_policy": policy.name,
        "env_id": config.train.env_id,
        "seed": config.train.seed,
        "pretrain_reference_return": pretrain["reference_return"],
        "pretrain_reference_eval_return": reference_eval["return_mean"],
        "pretrain_reference_reward_rate": pretrain["reference_reward_rate"],
        "deploy_eval": asdict(
            summarize_eval(
                pretrain_returns=[reference_eval["return_mean"]],
                deploy_returns=deploy_eval_returns,
                update_cost=deploy["total_updates"],
                replay_samples=deploy["total_replay_samples"],
                wallclock_update_time=deploy["total_update_time"],
                reference_return=reference_eval["return_mean"],
                update_penalty=config.cost.budget_score_update_penalty,
                reward_ctrl_values=reward_ctrl_values,
            )
        ),
        "run_dir": str(run_dir),
    }

    all_episodes = pretrain["episodes"] + deploy["episodes"]
    all_updates = pretrain["update_logs"] + deploy["update_logs"]
    write_csv(run_dir / "episodes.csv", all_episodes)
    write_jsonl(run_dir / "updates.jsonl", all_updates)
    write_csv(run_dir / "windows.csv", deploy["window_logs"])
    write_csv(run_dir / "eval_records.csv", [reference_eval] + deploy["eval_records"])
    write_csv(run_dir / "plasticity_snapshots.csv", deploy["plasticity_snapshots"])
    write_json(run_dir / "summary.json", summary)

    if hasattr(deploy_env, "close"):
        deploy_env.close()

    return summary


def run_experiment(
    config: ExperimentConfig,
    run_name_override: str | None = None,
    total_steps_override: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    _apply_step_override(config, total_steps_override)
    if dry_run:
        run_dir = prepare_run_dir(config.output_dir, run_name_override or config.name)
        snapshot_config(run_dir / "config.snapshot.yaml", dump_experiment_config(config))
        payload = {
            "run_dir": str(run_dir),
            "config": dump_experiment_config(config),
        }
        write_json(run_dir / "dry_run.json", payload)
        return payload

    pretrained_state = prepare_pretrained_state(config)
    return run_deployment_from_pretrained(config, pretrained_state, run_name_override=run_name_override)
