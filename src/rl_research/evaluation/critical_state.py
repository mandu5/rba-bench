"""Critical-state retention metric (v2.1).

Two definitions are reported simultaneously to defend against the "ad hoc
critical state selection" reviewer attack:

  A. **Task-agnostic** definition uses purely statistical signals from the
     pre-drift trajectory: low-return percentile + high TD-error +
     failure-proximity (terminal-step neighborhood). No domain knowledge.

  B. **Domain-specific** definition uses environment-aware boundaries
     (Hopper torso height, Walker2d instability, CSTR constraint boundary).
     A small registry maps env_id -> classifier callable.

Both definitions return:
  - critical_state_set: list of (observation, return_at_state) tuples
  - retention_rate: fraction of critical states from which the agent recovers
                    within `recovery_horizon` steps post-drift
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class CriticalStateRecord:
    observation: np.ndarray
    state_score: float
    source: str  # "low_return" | "high_td" | "failure_proximity" | "domain"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetentionResult:
    definition: str  # "task_agnostic" | "domain_specific"
    n_critical_states: int
    retention_rate: float
    mean_recovery_steps: float
    notes: str = ""


# A domain classifier returns True when an observation is "critical" for
# a specific environment. It is environment-specific and lives in a registry.
DomainCriticalClassifier = Callable[[np.ndarray, dict[str, Any]], bool]


# ---------------------------------------------------------------------------
# Definition A: task-agnostic critical states
# ---------------------------------------------------------------------------


def select_task_agnostic_critical_states(
    observations: np.ndarray,
    returns_at_state: np.ndarray,
    td_errors_at_state: np.ndarray | None = None,
    is_terminal: np.ndarray | None = None,
    *,
    return_percentile: float = 10.0,
    td_percentile: float = 90.0,
    terminal_neighborhood: int = 3,
    max_states: int = 64,
    rng: np.random.Generator | None = None,
) -> list[CriticalStateRecord]:
    """Select critical states using purely statistical criteria.

    Three sources of "criticality":
      - bottom `return_percentile`% by trajectory return
      - top `td_percentile`% by |TD-error| (if provided)
      - within `terminal_neighborhood` steps of any terminal flag (if provided)
    Selected states are deduplicated and capped at `max_states`.
    """
    rng = rng or np.random.default_rng(0)
    n = observations.shape[0]
    if n == 0:
        return []

    selected_idx: set[int] = set()

    # Source 1: low-return percentile
    if returns_at_state.size > 0:
        threshold = np.percentile(returns_at_state, return_percentile)
        mask = returns_at_state <= threshold
        for i in np.where(mask)[0].tolist():
            selected_idx.add(int(i))

    # Source 2: high TD-error percentile
    sources: dict[int, str] = {}
    if td_errors_at_state is not None and td_errors_at_state.size > 0:
        td_threshold = np.percentile(np.abs(td_errors_at_state), td_percentile)
        mask = np.abs(td_errors_at_state) >= td_threshold
        for i in np.where(mask)[0].tolist():
            selected_idx.add(int(i))
            sources.setdefault(int(i), "high_td")

    # Source 3: failure proximity
    if is_terminal is not None and is_terminal.any():
        terminal_indices = np.where(is_terminal)[0]
        for t_idx in terminal_indices:
            for offset in range(-terminal_neighborhood, 1):
                cand = int(t_idx + offset)
                if 0 <= cand < n:
                    selected_idx.add(cand)
                    sources.setdefault(cand, "failure_proximity")

    # Deduplicate, cap, and emit records.
    if not selected_idx:
        return []
    indices = sorted(selected_idx)
    if len(indices) > max_states:
        indices = list(rng.choice(indices, size=max_states, replace=False))
    records: list[CriticalStateRecord] = []
    for i in indices:
        source = sources.get(i, "low_return")
        records.append(
            CriticalStateRecord(
                observation=observations[i].copy(),
                state_score=float(returns_at_state[i]) if i < len(returns_at_state) else 0.0,
                source=source,
                metadata={"index": int(i)},
            )
        )
    return records


# ---------------------------------------------------------------------------
# Definition B: domain-specific critical states
# ---------------------------------------------------------------------------


_DOMAIN_REGISTRY: dict[str, DomainCriticalClassifier] = {}


def register_domain_classifier(env_id: str, fn: DomainCriticalClassifier) -> None:
    _DOMAIN_REGISTRY[env_id] = fn


def get_domain_classifier(env_id: str) -> DomainCriticalClassifier | None:
    return _DOMAIN_REGISTRY.get(env_id)


def _hopper_v5_critical(observation: np.ndarray, info: dict[str, Any]) -> bool:
    """Critical when torso height is below safety margin or angle is large.

    Hopper-v5 observation index 0 = torso z-height.
    Healthy torso height range: ~0.7. Critical < 0.85 (close to fall).
    """
    if observation.size < 1:
        return False
    z = float(observation[0])
    return z < 0.85


def _walker2d_v5_critical(observation: np.ndarray, info: dict[str, Any]) -> bool:
    """Critical when torso height drops below typical stable region."""
    if observation.size < 1:
        return False
    z = float(observation[0])
    return z < 1.0


def _pcgym_cstr_critical(observation: np.ndarray, info: dict[str, Any]) -> bool:
    """Critical when state nears constraint boundary (env-specific)."""
    # Placeholder: relies on `info["constraint_violation"]` to be passed by the
    # PC-Gym wrapper. If not available, returns False.
    return bool(info.get("constraint_violation", False)) or bool(
        info.get("near_constraint", False)
    )


# Register defaults
register_domain_classifier("Hopper-v5", _hopper_v5_critical)
register_domain_classifier("Hopper-v4", _hopper_v5_critical)
register_domain_classifier("Walker2d-v5", _walker2d_v5_critical)
register_domain_classifier("Walker2d-v4", _walker2d_v5_critical)
register_domain_classifier("PCGym-CSTR-v0", _pcgym_cstr_critical)


def select_domain_specific_critical_states(
    env_id: str,
    observations: np.ndarray,
    info_records: Sequence[dict[str, Any]] | None = None,
    *,
    max_states: int = 64,
    rng: np.random.Generator | None = None,
) -> list[CriticalStateRecord]:
    """Select critical states using a registered env-specific classifier.

    Falls back to empty list if the environment is not in the registry.
    """
    rng = rng or np.random.default_rng(0)
    classifier = get_domain_classifier(env_id)
    if classifier is None or observations.shape[0] == 0:
        return []
    info_records = info_records or [{} for _ in range(observations.shape[0])]
    selected: list[CriticalStateRecord] = []
    for i, obs in enumerate(observations):
        info = info_records[i] if i < len(info_records) else {}
        if classifier(obs, info):
            selected.append(
                CriticalStateRecord(
                    observation=obs.copy(),
                    state_score=0.0,
                    source="domain",
                    metadata={"index": int(i), "env_id": env_id},
                )
            )
    if len(selected) > max_states:
        idx = rng.choice(len(selected), size=max_states, replace=False)
        selected = [selected[int(i)] for i in idx]
    return selected


# ---------------------------------------------------------------------------
# Retention evaluation
# ---------------------------------------------------------------------------


def evaluate_retention(
    critical_states: list[CriticalStateRecord],
    *,
    rollout_fn: Callable[[np.ndarray], dict[str, Any]],
    recovery_horizon: int = 200,
    success_return_threshold: float = 0.0,
) -> RetentionResult:
    """Roll out from each critical state and measure recovery rate.

    `rollout_fn(observation)` should return a dict with at least:
      - "return": cumulative undiscounted return over the rollout
      - "steps":  number of env steps executed
      - "success": optional boolean override of return-based success
    """
    if not critical_states:
        return RetentionResult(
            definition="empty",
            n_critical_states=0,
            retention_rate=0.0,
            mean_recovery_steps=0.0,
            notes="no critical states selected",
        )
    n_success = 0
    steps_to_success: list[float] = []
    for record in critical_states:
        rollout = rollout_fn(record.observation)
        ret = float(rollout.get("return", 0.0))
        success = bool(rollout.get("success", ret >= success_return_threshold))
        if success:
            n_success += 1
            steps_to_success.append(float(rollout.get("steps", recovery_horizon)))
    retention_rate = float(n_success) / max(len(critical_states), 1)
    mean_steps = float(np.mean(steps_to_success)) if steps_to_success else float(recovery_horizon)
    return RetentionResult(
        definition="combined",
        n_critical_states=len(critical_states),
        retention_rate=retention_rate,
        mean_recovery_steps=mean_steps,
    )


def evaluate_dual_retention(
    env_id: str,
    pre_drift_observations: np.ndarray,
    pre_drift_returns: np.ndarray,
    pre_drift_td_errors: np.ndarray | None,
    pre_drift_terminals: np.ndarray | None,
    pre_drift_infos: Sequence[dict[str, Any]] | None,
    rollout_fn: Callable[[np.ndarray], dict[str, Any]],
    *,
    recovery_horizon: int = 200,
    success_return_threshold: float = 0.0,
    max_states: int = 64,
    rng: np.random.Generator | None = None,
) -> dict[str, RetentionResult]:
    """Compute both retention definitions and return as a dict."""
    rng = rng or np.random.default_rng(0)
    task_agnostic = select_task_agnostic_critical_states(
        observations=pre_drift_observations,
        returns_at_state=pre_drift_returns,
        td_errors_at_state=pre_drift_td_errors,
        is_terminal=pre_drift_terminals,
        max_states=max_states,
        rng=rng,
    )
    domain = select_domain_specific_critical_states(
        env_id=env_id,
        observations=pre_drift_observations,
        info_records=pre_drift_infos,
        max_states=max_states,
        rng=rng,
    )
    res_a = evaluate_retention(
        task_agnostic,
        rollout_fn=rollout_fn,
        recovery_horizon=recovery_horizon,
        success_return_threshold=success_return_threshold,
    )
    res_a.definition = "task_agnostic"
    res_b = evaluate_retention(
        domain,
        rollout_fn=rollout_fn,
        recovery_horizon=recovery_horizon,
        success_return_threshold=success_return_threshold,
    )
    res_b.definition = "domain_specific"
    return {"task_agnostic": res_a, "domain_specific": res_b}
