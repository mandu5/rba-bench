from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from rl_research.envs.pcgym_adapter import PCGymInfoAdapter, materialize_pcgym_env_params


class _FakePCGymEnv:
    observation_space = SimpleNamespace(shape=(3,), low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]))
    action_space = SimpleNamespace(shape=(1,), low=np.array([-1]), high=np.array([1]))
    env_params = {
        "constraints": {"T": [380.0]},
        "cons_type": {"T": ["<="]},
        "near_constraint_margin": 5.0,
    }

    def __init__(self):
        self.state = np.array([0.5, 377.0, 340.0], dtype=np.float32)
        self.actions = []
        self.model = SimpleNamespace(info=lambda: {"states": ["Ca", "T"], "inputs": ["Tc"]})

    def reset(self, *args, **kwargs):
        return np.zeros(3, dtype=np.float32), {"cons_viol": True}

    def step(self, action):
        self.actions.append(np.asarray(action, dtype=np.float32))
        return np.zeros(3, dtype=np.float32), 0.0, False, False, {}


def test_pcgym_info_adapter_normalizes_constraint_keys() -> None:
    env = PCGymInfoAdapter(_FakePCGymEnv())

    _, reset_info = env.reset(seed=1)
    assert reset_info["constraint_violation"] is True

    *_, step_info = env.step(np.zeros(1, dtype=np.float32))
    assert step_info["constraint_violation"] is False
    assert step_info["constraint_margin"] == 3.0
    assert step_info["near_constraint"] is True


def test_pcgym_adapter_action_gain_drift_scales_actions() -> None:
    raw_env = _FakePCGymEnv()
    env = PCGymInfoAdapter(raw_env)

    metadata = env.apply_drift("actuator_gain", severity=0.25, event_index=0)
    assert metadata == {"pcgym_action_scale": 0.75}

    env.step(np.array([0.8], dtype=np.float32))
    assert np.allclose(raw_env.actions[-1], np.array([0.6], dtype=np.float32))


def test_materialize_pcgym_env_params_expands_yaml_friendly_values() -> None:
    params = materialize_pcgym_env_params(
        {
            "N": 5,
            "SP": {"T": {"piecewise": [[0, 340.0], [3, 350.0]]}},
            "disturbances": {"Ti": {"constant": 350.0}},
            "o_space": {"low": [0.0, 300.0, 300.0], "high": [1.0, 450.0, 380.0]},
            "a_space": {"low": [250.0], "high": [350.0]},
            "x0": [0.8, 330.0],
            "model": "cstr_ode",
        }
    )

    assert np.allclose(params["SP"]["T"], np.array([340, 340, 340, 350, 350], dtype=np.float32))
    assert np.allclose(params["disturbances"]["Ti"], np.full(5, 350.0, dtype=np.float32))
    assert params["o_space"]["low"].shape == (3,)
    assert params["a_space"]["high"].shape == (1,)
    assert params["x0"].shape == (2,)
