"""Optional PC-Gym environment adapter.

The adapter keeps PC-Gym optional for the main MuJoCo workflow. It normalizes
constraint-related info keys so the critical-state metrics can consume PC-Gym
case-study runs without hard-coding PC-Gym internals throughout the runner.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np


class PCGymInfoAdapter:
    """Normalize PC-Gym step/reset metadata to the RBA-Bench convention."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self._action_scale = 1.0
        self._last_action: np.ndarray | None = None

    @property
    def action_space(self) -> Any:
        return self.env.action_space

    @property
    def observation_space(self) -> Any:
        return self.env.observation_space

    @property
    def unwrapped(self) -> Any:
        return getattr(self.env, "unwrapped", self.env)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        result = self.env.reset(*args, **kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
            return obs, self._normalize_info(info)
        return result

    def step(self, action: Any) -> Any:
        scaled_action = self._scale_action(action)
        self._last_action = np.asarray(scaled_action, dtype=np.float32)
        result = self.env.step(scaled_action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            return obs, reward, terminated, truncated, self._normalize_info(info)
        if len(result) == 4:
            obs, reward, done, info = result
            return obs, reward, done, self._normalize_info(info)
        return result

    def close(self) -> None:
        if hasattr(self.env, "close"):
            self.env.close()

    def _normalize_info(self, raw_info: Any) -> dict[str, Any]:
        info = dict(raw_info or {})
        constraint_violation, constraint_margin = self._constraint_status(info)
        if "constraint_violation" not in info:
            for key in ("cons_viol", "constraint_violated", "violated_constraints"):
                if key in info:
                    info["constraint_violation"] = bool(info[key])
                    break
        if "constraint_violation" not in info:
            info["constraint_violation"] = constraint_violation
        if "constraint_margin" not in info:
            for key in ("cons_margin", "constraint_distance", "safety_margin"):
                if key in info:
                    info["constraint_margin"] = info[key]
                    break
        if "constraint_margin" not in info and constraint_margin is not None:
            info["constraint_margin"] = constraint_margin
        if "near_constraint" not in info and constraint_margin is not None:
            threshold = float(getattr(self.env, "env_params", {}).get("near_constraint_margin", 0.0))
            info["near_constraint"] = bool(constraint_margin <= threshold)
        info["pcgym_action_scale"] = self._action_scale
        return info

    def _scale_action(self, action: Any) -> Any:
        if self._action_scale == 1.0:
            return action
        arr = np.asarray(action, dtype=np.float32) * float(self._action_scale)
        low = getattr(self.action_space, "low", None)
        high = getattr(self.action_space, "high", None)
        if low is not None and high is not None:
            arr = np.clip(arr, low, high)
        return arr

    def _constraint_status(self, info: dict[str, Any]) -> tuple[bool, float | None]:
        margins = self._constraint_margins()
        if not margins:
            cons_info = info.get("cons_info")
            if cons_info is not None:
                try:
                    return bool(np.any(np.asarray(cons_info, dtype=float) > 0.0)), None
                except Exception:
                    return False, None
            return False, None
        min_margin = float(min(margins))
        return bool(min_margin <= 0.0), min_margin

    def _constraint_margins(self) -> list[float]:
        env_params = getattr(self.env, "env_params", {}) or {}
        constraints = getattr(self.env, "constraints", None) or env_params.get("constraints")
        cons_type = getattr(self.env, "cons_type", None) or env_params.get("cons_type")
        state = getattr(self.env, "state", None)
        if constraints is None or cons_type is None or state is None:
            return []
        model = getattr(self.env, "model", None)
        if model is None or not hasattr(model, "info"):
            return []
        model_info = model.info()
        states = list(model_info.get("states", []))
        inputs = list(model_info.get("inputs", []))
        values: dict[str, float] = {}
        raw_state = np.asarray(state, dtype=float).reshape(-1)
        for index, name in enumerate(states):
            if index < raw_state.size:
                values[str(name)] = float(raw_state[index])
        if self._last_action is not None:
            raw_action = np.asarray(self._last_action, dtype=float).reshape(-1)
            for index, name in enumerate(inputs):
                if index < raw_action.size:
                    values[str(name)] = float(raw_action[index])

        margins: list[float] = []
        for key, bounds in dict(constraints).items():
            if str(key) not in values:
                continue
            bound_types = dict(cons_type).get(key, dict(cons_type).get(str(key), []))
            for bound, bound_type in zip(bounds, bound_types):
                value = values[str(key)]
                bound_value = float(bound)
                if str(bound_type) == "<=":
                    margins.append(bound_value - value)
                elif str(bound_type) == ">=":
                    margins.append(value - bound_value)
        return margins

    def apply_drift(self, shift_type: str, severity: float, event_index: int) -> dict[str, Any]:
        if shift_type != "actuator_gain":
            raise RuntimeError(f"PCGymInfoAdapter does not support drift type `{shift_type}`.")
        factor = max(0.2, 1.0 - float(severity) * float(event_index + 1))
        self._action_scale = factor
        return {"pcgym_action_scale": factor}


def create_pcgym_env(env_kwargs: dict[str, Any] | None = None) -> PCGymInfoAdapter:
    """Create a PC-Gym environment from an explicit env_params block.

    Expected config shape:

    ```yaml
    train:
      env_id: PCGym-CSTR-v0
    env_kwargs:
      env_params:
        model: cstr_ode
        ...
    ```

    PC-Gym's exact parameter schema is intentionally passed through rather than
    re-modeled here because the project treats PC-Gym as a post-Phase-G case
    study and the final CSTR schedule is still locked in `docs/pcgym_case_study_runbook.md`.
    """
    kwargs = dict(env_kwargs or {})
    env_params = kwargs.pop("env_params", None)
    if env_params is None:
        raise ValueError(
            "PCGym-CSTR-v0 requires `env_kwargs.env_params`. "
            "See docs/pcgym_case_study_runbook.md before running Phase F."
        )

    try:
        import pcgym  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise ImportError(
            "PC-Gym is required for PCGym-CSTR-v0. Install it with "
            "`python3 -m pip install -e .[pcgym]`."
        ) from exc

    env = pcgym.make_env(materialize_pcgym_env_params(env_params))
    return PCGymInfoAdapter(env)


def materialize_pcgym_env_params(env_params: dict[str, Any]) -> dict[str, Any]:
    """Convert YAML-friendly PC-Gym config values into the objects PC-Gym expects."""
    params = deepcopy(env_params)
    n_steps = int(params["N"])

    if "SP" in params:
        params["SP"] = {
            str(key): _materialize_schedule(value, n_steps)
            for key, value in dict(params["SP"]).items()
        }
    if "disturbances" in params:
        params["disturbances"] = {
            str(key): _materialize_schedule(value, n_steps)
            for key, value in dict(params["disturbances"]).items()
        }

    for space_key in ("o_space", "a_space", "disturbance_bounds"):
        if space_key in params:
            params[space_key] = _materialize_space(params[space_key])
    if "x0" in params:
        params["x0"] = np.asarray(params["x0"], dtype=np.float32)
    return params


def _materialize_space(space: dict[str, Any]) -> dict[str, Any]:
    out = dict(space)
    if "low" in out:
        out["low"] = np.asarray(out["low"], dtype=np.float32)
    if "high" in out:
        out["high"] = np.asarray(out["high"], dtype=np.float32)
    return out


def _materialize_schedule(value: Any, n_steps: int) -> np.ndarray:
    if isinstance(value, dict):
        if "constant" in value:
            return np.full(n_steps, float(value["constant"]), dtype=np.float32)
        if "piecewise" in value:
            series = np.zeros(n_steps, dtype=np.float32)
            points = sorted((int(start), float(val)) for start, val in value["piecewise"])
            if not points:
                raise ValueError("Piecewise schedule requires at least one [start, value] pair.")
            current = points[0][1]
            point_idx = 0
            for step in range(n_steps):
                while point_idx + 1 < len(points) and step >= points[point_idx + 1][0]:
                    point_idx += 1
                    current = points[point_idx][1]
                series[step] = current
            return series
        raise ValueError(f"Unsupported schedule spec: {value}")
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 1:
        return np.full(n_steps, float(arr.reshape(-1)[0]), dtype=np.float32)
    if arr.shape[0] != n_steps:
        raise ValueError(f"Schedule length {arr.shape[0]} does not match N={n_steps}.")
    return arr
