"""Environment construction."""

from __future__ import annotations

from typing import Any

from rl_research.envs.mock_control import MockContinuousControlEnv
from rl_research.envs.pcgym_adapter import create_pcgym_env

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - optional dependency
    gym = None


def create_env(env_id: str, env_kwargs: dict[str, Any] | None = None) -> Any:
    kwargs = env_kwargs or {}
    if env_id == "MockContinuousControl-v0":
        return MockContinuousControlEnv(**kwargs)
    if env_id == "PCGym-CSTR-v0":
        return create_pcgym_env(kwargs)
    if gym is None:
        raise ImportError(
            f"`gymnasium` is required for env_id={env_id}. "
            "Install the optional dependency group: `python3 -m pip install -e .[envs]` "
            "or `.[mujoco]` for MuJoCo environments."
        )
    return gym.make(env_id, **kwargs)
