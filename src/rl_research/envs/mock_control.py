"""Built-in continuous-control environment for smoke tests."""

from __future__ import annotations

from typing import Any

import numpy as np

from rl_research.envs.base import BoxSpace


class MockContinuousControlEnv:
    """A tiny continuous-control environment with controllable drift."""

    def __init__(
        self,
        episode_length: int = 200,
        dt: float = 0.05,
        target: float = 0.0,
    ) -> None:
        self.episode_length = episode_length
        self.dt = dt
        self.target = target
        self.observation_space = BoxSpace(
            low=np.array([-5.0, -5.0, -5.0], dtype=np.float32),
            high=np.array([5.0, 5.0, 5.0], dtype=np.float32),
        )
        self.action_space = BoxSpace(
            low=np.array([-2.0], dtype=np.float32),
            high=np.array([2.0], dtype=np.float32),
        )
        self._base_inertia = 1.0
        self._base_friction = 0.12
        self._base_actuator_gain = 1.0
        self._base_stiffness = 0.55
        self._rng = np.random.default_rng()
        self.reset()

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.step_count = 0
        self.position = float(self._rng.uniform(-0.75, 0.75))
        self.velocity = float(self._rng.uniform(-0.25, 0.25))
        self.inertia = self._base_inertia
        self.friction = self._base_friction
        self.actuator_gain = self._base_actuator_gain
        self.stiffness = self._base_stiffness
        return self._get_obs(), self._info()

    def _get_obs(self) -> np.ndarray:
        return np.array(
            [self.position, self.velocity, self.position - self.target],
            dtype=np.float32,
        )

    def _info(self) -> dict[str, Any]:
        return {
            "inertia": self.inertia,
            "friction": self.friction,
            "actuator_gain": self.actuator_gain,
        }

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        u = float(np.clip(action[0], self.action_space.low[0], self.action_space.high[0]))
        accel = (
            self.actuator_gain * u
            - self.stiffness * (self.position - self.target)
            - self.friction * self.velocity
        ) / self.inertia
        noise = float(self._rng.normal(0.0, 0.01))
        self.velocity += self.dt * accel + noise
        self.position += self.dt * self.velocity
        self.step_count += 1

        tracking_error = self.position - self.target
        reward = -(
            tracking_error * tracking_error
            + 0.1 * self.velocity * self.velocity
            + 0.01 * u * u
        )
        terminated = False
        truncated = self.step_count >= self.episode_length
        info = self._info()
        return self._get_obs(), reward, terminated, truncated, info

    def apply_drift(
        self,
        shift_type: str,
        severity: float,
        event_index: int,
    ) -> dict[str, Any]:
        multiplier = 1.0 + severity * float(event_index + 1)
        if shift_type == "body_mass":
            self.inertia = self._base_inertia * multiplier
        elif shift_type == "friction":
            self.friction = self._base_friction * multiplier
        elif shift_type == "actuator_gain":
            self.actuator_gain = self._base_actuator_gain * max(0.2, 1.0 - severity * (event_index + 1))
        elif shift_type == "stiffness":
            self.stiffness = self._base_stiffness * multiplier
        else:
            raise ValueError(f"Unsupported mock drift type: {shift_type}")
        return self._info()

    def close(self) -> None:
        return None
