"""Deployment-time drift wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rl_research.config import DriftScenario


@dataclass
class DriftEvent:
    applied: bool
    event_index: int
    metadata: dict[str, Any]


class DriftWrapper:
    """Applies drift during deployment according to a scenario."""

    def __init__(self, env: Any, scenario: DriftScenario) -> None:
        self.env = env
        self.scenario = scenario
        self.global_step = 0
        self.drift_events = 0
        self._base_snapshots = self._capture_base_snapshots()

    def _capture_base_snapshots(self) -> dict[str, np.ndarray | float]:
        base_env = getattr(self.env, "unwrapped", self.env)
        snapshots: dict[str, np.ndarray | float] = {}
        if hasattr(base_env, "model"):
            model = base_env.model
            if hasattr(model, "body_mass"):
                snapshots["body_mass"] = np.array(model.body_mass, copy=True)
            if hasattr(model, "geom_friction"):
                snapshots["geom_friction"] = np.array(model.geom_friction, copy=True)
            if hasattr(model, "actuator_gear"):
                snapshots["actuator_gear"] = np.array(model.actuator_gear, copy=True)
        if hasattr(base_env, "m"):
            snapshots["pendulum_m"] = float(base_env.m)
        if hasattr(base_env, "g"):
            snapshots["pendulum_g"] = float(base_env.g)
        if hasattr(base_env, "max_torque"):
            snapshots["pendulum_max_torque"] = float(base_env.max_torque)
        return snapshots

    @property
    def action_space(self) -> Any:
        return self.env.action_space

    @property
    def observation_space(self) -> Any:
        return self.env.observation_space

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def reset(self, seed: int | None = None) -> Any:
        return self.env.reset(seed=seed)

    def step(self, action: np.ndarray) -> Any:
        result = self.env.step(action)
        self.global_step += 1
        drift_event = self._maybe_apply_drift()
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        else:
            obs, reward, done, info = result
            terminated, truncated = done, False
        info = dict(info)
        info["drift_events"] = self.drift_events
        info["drift_applied"] = drift_event.applied
        info["drift_event_index"] = drift_event.event_index
        if drift_event.metadata:
            info["drift_metadata"] = drift_event.metadata
        return obs, reward, terminated, truncated, info

    def _maybe_apply_drift(self) -> DriftEvent:
        if self.drift_events >= self.scenario.repeat_count:
            return DriftEvent(False, self.drift_events, {})

        event_step = self.scenario.shift_step + self.drift_events * max(self.scenario.repeat_interval, 0)
        if self.global_step < event_step:
            return DriftEvent(False, self.drift_events, {})

        metadata = self._apply_drift(self.drift_events)
        self.drift_events += 1
        return DriftEvent(True, self.drift_events, metadata)

    def force_apply_events(self, event_count: int) -> None:
        target_count = min(event_count, self.scenario.repeat_count)
        while self.drift_events < target_count:
            self._apply_drift(self.drift_events)
            self.drift_events += 1

    def _apply_drift(self, event_index: int) -> dict[str, Any]:
        if hasattr(self.env, "apply_drift"):
            return self.env.apply_drift(self.scenario.shift_type, self.scenario.severity, event_index)

        base_env = getattr(self.env, "unwrapped", self.env)
        severity = self.scenario.severity
        increase_factor = 1.0 + severity * float(event_index + 1)
        decrease_factor = max(0.2, 1.0 - severity * float(event_index + 1))

        if hasattr(base_env, "model"):
            model = base_env.model
            if self.scenario.shift_type == "body_mass" and "body_mass" in self._base_snapshots:
                model.body_mass[:] = self._base_snapshots["body_mass"] * increase_factor
                return {"body_mass_factor": increase_factor}
            if self.scenario.shift_type == "friction" and "geom_friction" in self._base_snapshots:
                model.geom_friction[:] = self._base_snapshots["geom_friction"] * increase_factor
                return {"geom_friction_factor": increase_factor}
            if self.scenario.shift_type == "actuator_gain" and "actuator_gear" in self._base_snapshots:
                model.actuator_gear[:] = self._base_snapshots["actuator_gear"] * decrease_factor
                return {"actuator_gear_factor": decrease_factor}

        if hasattr(base_env, "m") and self.scenario.shift_type == "body_mass":
            base_env.m = float(self._base_snapshots["pendulum_m"]) * increase_factor
            return {"pendulum_mass_factor": increase_factor}
        if hasattr(base_env, "g") and self.scenario.shift_type == "friction":
            base_env.g = float(self._base_snapshots["pendulum_g"]) * increase_factor
            return {"pendulum_gravity_factor": increase_factor}
        if hasattr(base_env, "max_torque") and self.scenario.shift_type == "actuator_gain":
            base_env.max_torque = float(self._base_snapshots["pendulum_max_torque"]) * decrease_factor
            return {"pendulum_max_torque_factor": decrease_factor}

        raise RuntimeError(
            f"Drift type `{self.scenario.shift_type}` is not supported for environment "
            f"{type(base_env).__name__}. Add an environment-specific drift adapter."
        )
