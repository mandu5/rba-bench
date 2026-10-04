import numpy as np
import pytest

from rl_research.config import DriftScenario
from rl_research.data.replay import DualReplayBuffer
from rl_research.envs.drift import DriftWrapper
from rl_research.envs.factory import create_env


def test_replay_mixed_sampling_reports_counts() -> None:
    rng = np.random.default_rng(0)
    replay = DualReplayBuffer(obs_dim=2, act_dim=1, capacity=100, recent_capacity=20)
    for index in range(30):
        obs = np.array([index, index + 1], dtype=np.float32)
        action = np.array([0.1], dtype=np.float32)
        replay.add(obs, action, float(index), obs + 1, False)

    batch, meta = replay.sample_mixed(batch_size=10, recent_ratio=0.3, rng=rng)

    assert batch.observations.shape == (10, 2)
    assert meta["recent_samples"] == 3
    assert meta["historical_samples"] == 7


def test_hopper_v5_drift_wrapper_supports_body_mass() -> None:
    gymnasium = pytest.importorskip("gymnasium")
    _ = gymnasium
    env = DriftWrapper(
        create_env("Hopper-v5"),
        DriftScenario(shift_type="body_mass", shift_step=1, severity=0.1, repeat_count=1),
    )
    try:
        env.force_apply_events(1)
        assert env.drift_events == 1
    finally:
        env.close()


def test_walker2d_v5_drift_wrapper_supports_friction() -> None:
    gymnasium = pytest.importorskip("gymnasium")
    _ = gymnasium
    env = DriftWrapper(
        create_env("Walker2d-v5"),
        DriftScenario(shift_type="friction", shift_step=1, severity=0.1, repeat_count=1),
    )
    try:
        env.force_apply_events(1)
        assert env.drift_events == 1
    finally:
        env.close()
