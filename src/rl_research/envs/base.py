"""Environment helpers shared across custom and gym-like environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class BoxSpace:
    low: np.ndarray
    high: np.ndarray

    def __post_init__(self) -> None:
        self.low = np.asarray(self.low, dtype=np.float32)
        self.high = np.asarray(self.high, dtype=np.float32)
        self.shape = self.low.shape

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        generator = rng or np.random.default_rng()
        return generator.uniform(self.low, self.high).astype(np.float32)


def reset_env(env: Any, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    result = env.reset(seed=seed)
    if isinstance(result, tuple) and len(result) == 2:
        obs, info = result
    else:
        obs, info = result, {}
    return np.asarray(obs, dtype=np.float32), dict(info)


def step_env(env: Any, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
    elif len(result) == 4:
        obs, reward, done, info = result
        terminated, truncated = bool(done), False
    else:
        raise ValueError(f"Unsupported step result length: {len(result)}")
    return np.asarray(obs, dtype=np.float32), float(reward), bool(terminated), bool(truncated), dict(info)


def get_space_bounds(space: Any) -> tuple[np.ndarray, np.ndarray]:
    low = np.asarray(space.low, dtype=np.float32)
    high = np.asarray(space.high, dtype=np.float32)
    return low, high
