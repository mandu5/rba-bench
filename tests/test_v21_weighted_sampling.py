"""Unit tests for v2.1 additions: weighted sampling, baselines, plasticity, critical state."""

from __future__ import annotations

import numpy as np
import pytest

from rl_research.adaptation.policies import (
    BudgetedGradPolicy,
    BudgetedPredPolicy,
    BudgetedTDPolicy,
    PERPolicy,
    UniformPolicy,
    all_policy_names,
    build_adaptation_policy,
)
from rl_research.adaptation.budget import WindowStats
from rl_research.config import AdaptationBudget
from rl_research.data.replay import DualReplayBuffer, SimpleReplayBuffer
from rl_research.evaluation.critical_state import (
    evaluate_dual_retention,
    evaluate_retention,
    select_domain_specific_critical_states,
    select_task_agnostic_critical_states,
)


# ---------------------------------------------------------------------------
# Replay weighted sampling
# ---------------------------------------------------------------------------


def _fill_buffer(buffer: SimpleReplayBuffer, n: int) -> None:
    for i in range(n):
        obs = np.array([i, i + 1], dtype=np.float32)
        buffer.add(obs, np.array([0.1], dtype=np.float32), float(i), obs + 1, False, priority=float(i + 1))


def test_simple_replay_tracks_timestamps() -> None:
    buf = SimpleReplayBuffer(obs_dim=2, act_dim=1, capacity=20)
    _fill_buffer(buf, 5)
    view = buf.view()
    # ages should be 4, 3, 2, 1, 0 in insertion order.
    assert view["ages"].shape == (5,)
    np.testing.assert_array_equal(view["ages"], np.array([4, 3, 2, 1, 0], dtype=np.float32))


def test_weighted_sampling_concentrates_on_high_weights() -> None:
    buf = SimpleReplayBuffer(obs_dim=2, act_dim=1, capacity=10)
    _fill_buffer(buf, 5)
    rng = np.random.default_rng(123)

    # Weight only the last (newest) sample heavily.
    def weight_fn(view):
        w = np.zeros(view["ages"].shape[0], dtype=np.float64)
        w[view["ages"] == 0] = 1.0
        return w

    batch = buf.sample_weighted(batch_size=64, weight_fn=weight_fn, rng=rng)
    # The newest (insertion index 4) carries reward 4.
    assert batch.indices is not None
    assert (batch.indices == 4).mean() > 0.95


def test_per_weight_function_uses_priority() -> None:
    buf = SimpleReplayBuffer(obs_dim=2, act_dim=1, capacity=10)
    _fill_buffer(buf, 5)
    rng = np.random.default_rng(7)
    policy = PERPolicy(alpha=1.0)
    decision = policy.decide(stats=None, budget=AdaptationBudget(max_updates_per_window=10))
    assert decision.weight_fn is not None
    batch = buf.sample_weighted(
        batch_size=200, weight_fn=decision.weight_fn, rng=rng, beta_is=decision.beta_is
    )
    # With α=1 and priorities = [1,2,3,4,5], expected highest-prio sample is index 4.
    counts = np.bincount(batch.indices.astype(int), minlength=5)
    assert counts[4] >= counts[0] * 2
    assert batch.is_weights is not None and batch.is_weights.shape == (200,)


def test_dual_replay_supports_weighted_sampling() -> None:
    dr = DualReplayBuffer(obs_dim=2, act_dim=1, capacity=50, recent_capacity=10)
    for i in range(20):
        obs = np.array([i, i + 1], dtype=np.float32)
        dr.add(obs, np.array([0.0], dtype=np.float32), float(i), obs + 1, False)
    rng = np.random.default_rng(0)

    def weight_fn(view):
        return np.ones(view["ages"].shape[0], dtype=np.float64)

    batch, meta = dr.sample_weighted(batch_size=8, weight_fn=weight_fn, rng=rng)
    assert batch.observations.shape == (8, 2)
    assert "weighted_samples" in meta
    assert meta["weighted_samples"] == 8
    # recent + historical share counts should sum to batch size.
    assert meta["recent_samples"] + meta["historical_samples"] == 8


# ---------------------------------------------------------------------------
# Adaptation policies
# ---------------------------------------------------------------------------


def test_all_v21_policies_constructable() -> None:
    """v2.1.1 honest names: budgeted-novelty replaces v2.1's budgeted-pred."""
    expected = {"uniform", "per", "budgeted-td", "budgeted-grad", "budgeted-novelty"}
    canonical_names = set(all_policy_names())
    assert expected.issubset(canonical_names)
    for name in expected:
        policy = build_adaptation_policy(name)
        decision = policy.decide(stats=None, budget=AdaptationBudget(max_updates_per_window=10))
        assert decision.sampling_mode == "weighted"
        assert decision.weight_fn is not None


def test_budgeted_pred_is_alias_for_budgeted_novelty() -> None:
    """Backward-compatibility: v2.1 'budgeted-pred' resolves to v2.1.1 BudgetedNoveltyPolicy."""
    policy = build_adaptation_policy("budgeted-pred")
    assert policy.name == "Budgeted-Novelty"


def test_budgeted_td_policy_recency_plus_utility_combination() -> None:
    policy = BudgetedTDPolicy(T_recency=10.0, alpha_util=2.0, c_max=5.0, w_min=0.0)
    decision = policy.decide(stats=None, budget=AdaptationBudget(max_updates_per_window=5))
    fn = decision.weight_fn
    view = {
        "ages": np.array([0.0, 5.0, 10.0], dtype=np.float64),
        "td_priorities": np.array([1.0, 1.0, 1.0], dtype=np.float64),
    }
    w = fn(view)
    # Recency term: [1, 0.5, 0]; utility term scaled by 2*1.0 each = [2,2,2]
    np.testing.assert_allclose(w, np.array([3.0, 2.5, 2.0]), atol=1e-6)


def test_budgeted_grad_policy_uses_per_sample_input_grad() -> None:
    """v2.1.1: utility is per-sample (input_grad), not a global multiplier."""
    policy = BudgetedGradPolicy(T_recency=10.0, alpha_util=2.0, c_max=5.0, w_min=0.0)
    decision = policy.decide(stats=None, budget=AdaptationBudget(max_updates_per_window=5))
    view_high = {
        "ages": np.array([0.0, 0.0]),
        "td_priorities": np.array([1.0, 1.0]),
        "novelty": np.array([1.0, 1.0]),
        "input_grad": np.array([0.1, 1.0]),  # second sample much higher grad
    }
    w = decision.weight_fn(view_high)
    # Same age but higher input_grad → higher weight on sample 1.
    assert w[1] > w[0]


def test_budgeted_pred_policy_uses_per_sample_novelty() -> None:
    """v2.1.1: utility is per-sample novelty (||s_{t+1}-s_t||)."""
    policy = BudgetedPredPolicy(T_recency=10.0, alpha_util=2.0, c_max=5.0, w_min=0.0)
    decision = policy.decide(stats=None, budget=AdaptationBudget(max_updates_per_window=5))
    view = {
        "ages": np.array([0.0, 0.0]),
        "td_priorities": np.array([1.0, 1.0]),
        "novelty": np.array([0.1, 1.0]),  # second sample much higher novelty
        "input_grad": np.array([1.0, 1.0]),
    }
    w = decision.weight_fn(view)
    assert w[1] > w[0]


def test_uniform_policy_yields_uniform_weights() -> None:
    policy = UniformPolicy()
    decision = policy.decide(stats=None, budget=AdaptationBudget(max_updates_per_window=5))
    view = {
        "ages": np.array([0.0, 50.0, 100.0]),
        "td_priorities": np.array([0.1, 100.0, 0.5]),
        "novelty": np.array([0.5, 1.0, 2.0]),
        "input_grad": np.array([0.1, 0.5, 0.9]),
    }
    w = decision.weight_fn(view)
    assert np.allclose(w, np.ones_like(w))


# ---------------------------------------------------------------------------
# v2.1.1 regression: Budgeted-Grad and legacy Budgeted-Pred/Budgeted-Novelty
# MUST produce different sampling distributions when given different
# per-sample utility signals. This guards against the v2.1 bug where both
# policies used global scalar multipliers that vanished under probability
# normalization.
# ---------------------------------------------------------------------------


def test_v211_grad_and_pred_diverge_with_different_utility_signals() -> None:
    """Different per-sample utility → different sampling distribution."""
    from rl_research.adaptation.policies import BudgetedGradPolicy, BudgetedPredPolicy

    grad_policy = BudgetedGradPolicy(T_recency=10.0, alpha_util=2.0, c_max=5.0, w_min=0.0)
    pred_policy = BudgetedPredPolicy(T_recency=10.0, alpha_util=2.0, c_max=5.0, w_min=0.0)
    budget = AdaptationBudget(max_updates_per_window=5)

    g_decision = grad_policy.decide(stats=None, budget=budget)
    p_decision = pred_policy.decide(stats=None, budget=budget)

    # Same ages → same recency contribution.
    # But novelty and input_grad differ → utility per sample differs.
    view = {
        "ages": np.array([0.0, 5.0, 10.0]),
        "td_priorities": np.array([1.0, 1.0, 1.0]),
        "novelty": np.array([0.1, 0.5, 1.0]),       # increasing novelty
        "input_grad": np.array([1.0, 0.5, 0.1]),    # decreasing input grad
    }
    w_grad = g_decision.weight_fn(view)
    w_pred = p_decision.weight_fn(view)
    p_grad = w_grad / w_grad.sum()
    p_pred = w_pred / w_pred.sum()

    # Distributions must differ by at least L1 = 0.05 with these inputs.
    l1 = float(np.abs(p_grad - p_pred).sum())
    assert l1 > 0.05, (
        f"Budgeted-Grad and Budgeted-Novelty produced near-identical "
        f"sampling distributions (L1={l1:.4f}). v2.1 bug regression."
    )


def test_v211_recency_weighted_baseline_works() -> None:
    """RecencyWeighted must be available and use per-sample recency."""
    from rl_research.adaptation.policies import RecencyWeightedPolicy

    policy = RecencyWeightedPolicy(T_recency=20.0, w_min=0.05)
    decision = policy.decide(stats=None, budget=AdaptationBudget(max_updates_per_window=5))
    view = {
        "ages": np.array([0.0, 10.0, 20.0]),
        "td_priorities": np.array([1.0, 1.0, 1.0]),
        "novelty": np.array([1.0, 1.0, 1.0]),
        "input_grad": np.array([1.0, 1.0, 1.0]),
    }
    w = decision.weight_fn(view)
    # Newer transitions get higher weight; oldest hits w_min floor.
    assert w[0] > w[1] > w[2]
    assert abs(w[2] - 0.05) < 1e-6


# ---------------------------------------------------------------------------
# Plasticity proxy
# ---------------------------------------------------------------------------


def test_plasticity_snapshot_runs_on_simple_mlp() -> None:
    torch = pytest.importorskip("torch")
    from rl_research.evaluation.plasticity import compute_plasticity_snapshot

    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 2),
    )
    obs = np.random.RandomState(0).randn(16, 4).astype(np.float32)
    snap = compute_plasticity_snapshot(
        model, sample_obs=obs, step=100, window_index=2, grad_norm_ema=0.5,
    )
    assert 0.0 <= snap.dormant_neuron_ratio <= 1.0
    assert snap.effective_rank >= 0.0
    assert snap.window_index == 2
    assert snap.grad_norm_ema == 0.5


# ---------------------------------------------------------------------------
# Critical-state retention metric
# ---------------------------------------------------------------------------


def test_task_agnostic_critical_state_selection_picks_low_return() -> None:
    rng = np.random.default_rng(0)
    n = 50
    obs = rng.standard_normal((n, 3)).astype(np.float32)
    returns = np.linspace(0.0, 100.0, n).astype(np.float32)
    records = select_task_agnostic_critical_states(
        obs, returns, td_errors_at_state=None, is_terminal=None, return_percentile=10.0,
        max_states=8, rng=rng,
    )
    assert len(records) > 0
    indices = [r.metadata["index"] for r in records]
    assert max(indices) < int(n * 0.15)  # all from bottom 10–15%


def test_domain_specific_critical_state_uses_hopper_classifier() -> None:
    obs = np.zeros((10, 11), dtype=np.float32)
    obs[3, 0] = 0.5  # very low torso → critical
    obs[7, 0] = 0.6
    obs[5, 0] = 1.5  # healthy
    records = select_domain_specific_critical_states(
        env_id="Hopper-v5", observations=obs, info_records=None, max_states=10,
    )
    indices = sorted(r.metadata["index"] for r in records)
    assert 3 in indices and 7 in indices
    assert 5 not in indices


def test_evaluate_retention_returns_zero_when_all_fail() -> None:
    obs = np.zeros((4, 3), dtype=np.float32)
    records = select_task_agnostic_critical_states(
        obs, np.array([0.0, 1.0, 2.0, 3.0]), max_states=4, return_percentile=100.0,
    )

    def rollout(_obs):
        return {"return": -10.0, "steps": 100, "success": False}

    result = evaluate_retention(records, rollout_fn=rollout, success_return_threshold=0.0)
    assert result.retention_rate == 0.0


def test_evaluate_dual_retention_returns_both_definitions() -> None:
    rng = np.random.default_rng(42)
    obs = np.zeros((20, 11), dtype=np.float32)
    obs[2, 0] = 0.4
    returns = np.linspace(0.0, 50.0, 20)

    def rollout(_obs):
        return {"return": 10.0, "steps": 100, "success": True}

    out = evaluate_dual_retention(
        env_id="Hopper-v5",
        pre_drift_observations=obs,
        pre_drift_returns=returns,
        pre_drift_td_errors=None,
        pre_drift_terminals=None,
        pre_drift_infos=None,
        rollout_fn=rollout,
        rng=rng,
    )
    assert "task_agnostic" in out and "domain_specific" in out
    assert out["task_agnostic"].retention_rate == 1.0
