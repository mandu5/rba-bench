"""Adaptation policy baselines (v2.1).

v2.0 policies (Frozen, FixedAdapt, RecentOnly, StaticMix, DriftAware/Proposed)
are preserved for backward compatibility. v2.1 adds:

- `Uniform`            : standard uniform replay
- `PER`                : Prioritized Experience Replay (Schaul et al. 2016) baseline
- `Budgeted-TD`        : recency + TD-error EMA utility (proposed main)
- `Budgeted-Grad`      : recency + gradient-norm utility (proposed alt)
- `Budgeted-Novelty`   : recency + state-delta novelty (proposed secondary)

All v2.1 policies share the same total update budget B. They differ only in the
sampling weight function attached to `BudgetDecision.weight_fn`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl_research.adaptation.budget import BudgetDecision, WindowStats
from rl_research.config import AdaptationBudget


# ---------------------------------------------------------------------------
# Base + v2.0 policies (unchanged behavior, preserved for compatibility)
# ---------------------------------------------------------------------------


@dataclass
class AdaptationPolicy:
    name: str

    def decide(self, stats: WindowStats | None, budget: AdaptationBudget) -> BudgetDecision:
        raise NotImplementedError


class FrozenPolicy(AdaptationPolicy):
    def __init__(self) -> None:
        super().__init__("Frozen")

    def decide(self, stats: WindowStats | None, budget: AdaptationBudget) -> BudgetDecision:
        return BudgetDecision(
            max_updates=0,
            recent_ratio=budget.recent_replay_ratio,
            reason="deployment frozen",
            sampling_mode="mixed",
        )


class FixedAdaptPolicy(AdaptationPolicy):
    def __init__(self) -> None:
        super().__init__("FixedAdapt")

    def decide(self, stats: WindowStats | None, budget: AdaptationBudget) -> BudgetDecision:
        return BudgetDecision(
            max_updates=budget.max_updates_per_window,
            recent_ratio=0.5,
            reason="fixed update schedule with balanced replay",
            sampling_mode="mixed",
        )


class RecentOnlyPolicy(AdaptationPolicy):
    def __init__(self) -> None:
        super().__init__("RecentOnly")

    def decide(self, stats: WindowStats | None, budget: AdaptationBudget) -> BudgetDecision:
        return BudgetDecision(
            max_updates=budget.max_updates_per_window,
            recent_ratio=1.0,
            reason="all adaptation budget spent on recent transitions",
            sampling_mode="mixed",
        )


class StaticMixPolicy(AdaptationPolicy):
    def __init__(self) -> None:
        super().__init__("StaticMix")

    def decide(self, stats: WindowStats | None, budget: AdaptationBudget) -> BudgetDecision:
        return BudgetDecision(
            max_updates=budget.max_updates_per_window,
            recent_ratio=float(np.clip(budget.recent_replay_ratio, 0.0, 1.0)),
            reason="static replay mix under fixed update budget",
            sampling_mode="mixed",
        )


class DriftAwarePolicy(AdaptationPolicy):
    """v2.0 'Proposed' — kept for legacy comparison."""

    def __init__(self) -> None:
        super().__init__("Proposed")

    def decide(self, stats: WindowStats | None, budget: AdaptationBudget) -> BudgetDecision:
        if stats is None:
            return BudgetDecision(
                max_updates=budget.max_updates_per_window,
                recent_ratio=float(np.clip(budget.recent_replay_ratio, 0.0, 1.0)),
                reason="warm start before first deployment window closes",
                sampling_mode="mixed",
            )

        drop_score = max(stats.return_drop, 0.0)
        td_score = min(stats.td_error_ema / 2.0, 1.0)
        boost = 0.35 * max(drop_score, td_score)
        ratio = float(np.clip(budget.recent_replay_ratio + boost, 0.2, 0.95))
        return BudgetDecision(
            max_updates=budget.max_updates_per_window,
            recent_ratio=ratio,
            reason=f"drift-aware replay shift (drop={drop_score:.3f}, td={stats.td_error_ema:.3f})",
            sampling_mode="mixed",
        )


# ---------------------------------------------------------------------------
# v2.1 policies (weighted sampling via BudgetDecision.weight_fn)
# ---------------------------------------------------------------------------


def _uniform_weight_fn(view: dict[str, np.ndarray]) -> np.ndarray:
    n = view["ages"].shape[0]
    return np.ones(n, dtype=np.float64)


def _make_per_weight_fn(alpha: float = 0.6) -> "WeightFnLike":
    def fn(view: dict[str, np.ndarray]) -> np.ndarray:
        # PER: w_i ∝ priority_i^α with priority = TD-error magnitude (kept ≥ 1e-6)
        return np.power(view["td_priorities"].astype(np.float64), float(alpha))

    return fn


def _make_recency_term(ages: np.ndarray, T: float, w_min: float) -> np.ndarray:
    """Linear age decay clipped at w_min, as in v2.1 spec."""
    raw = 1.0 - ages / max(float(T), 1.0)
    return np.maximum(raw, float(w_min))


def _make_budgeted_td_weight_fn(
    T: float,
    alpha_util: float,
    c_max: float,
    w_min: float,
) -> "WeightFnLike":
    def fn(view: dict[str, np.ndarray]) -> np.ndarray:
        ages = view["ages"].astype(np.float64)
        priorities = view["td_priorities"].astype(np.float64)
        # Normalize priorities to 0..1 within the buffer for stable scaling.
        max_p = max(float(priorities.max()), 1e-6)
        utility = priorities / max_p
        utility = np.clip(utility, 0.0, float(c_max))
        recency = _make_recency_term(ages, T=T, w_min=w_min)
        return recency + float(alpha_util) * utility

    return fn


def _make_budgeted_grad_weight_fn(
    T: float,
    alpha_util: float,
    c_max: float,
    w_min: float,
) -> "WeightFnLike":
    """v2.1.1 main proposed alt: recency + per-sample input-gradient utility.

    Uses ||∂Q1_i/∂obs_i|| backfilled into buffer.input_grad by the SAC update.
    Critically per-sample (not a global multiplier) so it does not vanish
    under probability normalization (cf. v2.1 → v2.1.1 bug fix).
    """
    def fn(view: dict[str, np.ndarray]) -> np.ndarray:
        ages = view["ages"].astype(np.float64)
        input_grad = view["input_grad"].astype(np.float64)
        max_g = max(float(input_grad.max()), 1e-6)
        utility = np.clip(input_grad / max_g, 0.0, float(c_max))
        recency = _make_recency_term(ages, T=T, w_min=w_min)
        return recency + float(alpha_util) * utility

    return fn


def _make_budgeted_pred_weight_fn(
    T: float,
    alpha_util: float,
    c_max: float,
    w_min: float,
) -> "WeightFnLike":
    """v2.1.1 secondary: recency + per-sample state-novelty (||s_{t+1}-s_t||).

    Uses buffer.novelty populated at add() time. No auxiliary forward model
    required. Per-sample so it does not vanish under normalization.
    """
    def fn(view: dict[str, np.ndarray]) -> np.ndarray:
        ages = view["ages"].astype(np.float64)
        novelty = view["novelty"].astype(np.float64)
        max_n = max(float(novelty.max()), 1e-6)
        utility = np.clip(novelty / max_n, 0.0, float(c_max))
        recency = _make_recency_term(ages, T=T, w_min=w_min)
        return recency + float(alpha_util) * utility

    return fn


def _make_recency_only_weight_fn(T: float, w_min: float) -> "WeightFnLike":
    """v2.1.1 added: pure recency-weighted sampling on full buffer.

    Discovered from v2.1 pilot as an unexpectedly strong baseline (the
    de-facto behavior of the bugged Budgeted-Grad/Pred). Promoted to a
    first-class baseline so we can credit it honestly and isolate the
    contribution of utility signals on top of it.
    """
    def fn(view: dict[str, np.ndarray]) -> np.ndarray:
        ages = view["ages"].astype(np.float64)
        return _make_recency_term(ages, T=T, w_min=w_min)
    return fn


# Type alias for static analysis
try:
    from rl_research.data.replay import WeightFn as WeightFnLike  # noqa: F401
except ImportError:  # pragma: no cover
    WeightFnLike = object  # type: ignore[assignment]


class UniformPolicy(AdaptationPolicy):
    def __init__(self) -> None:
        super().__init__("Uniform")

    def decide(self, stats: WindowStats | None, budget: AdaptationBudget) -> BudgetDecision:
        return BudgetDecision(
            max_updates=budget.max_updates_per_window,
            recent_ratio=budget.recent_replay_ratio,
            reason="uniform replay (v2.1 baseline)",
            weight_fn=_uniform_weight_fn,
            sampling_mode="weighted",
        )


class PERPolicy(AdaptationPolicy):
    """Prioritized Experience Replay (Schaul et al. 2016) baseline."""

    def __init__(self, alpha: float = 0.6, beta_is: float = 0.4) -> None:
        super().__init__("PER")
        self.alpha = float(alpha)
        self.beta_is = float(beta_is)

    def decide(self, stats: WindowStats | None, budget: AdaptationBudget) -> BudgetDecision:
        return BudgetDecision(
            max_updates=budget.max_updates_per_window,
            recent_ratio=budget.recent_replay_ratio,
            reason=f"PER (alpha={self.alpha}, beta_is={self.beta_is})",
            weight_fn=_make_per_weight_fn(alpha=self.alpha),
            beta_is=self.beta_is,
            sampling_mode="weighted",
        )


class BudgetedTDPolicy(AdaptationPolicy):
    """v2.1 main proposed: recency + TD-error EMA utility under fixed budget."""

    def __init__(self, T_recency: float = 50_000.0, alpha_util: float = 1.0,
                 c_max: float = 5.0, w_min: float = 0.05) -> None:
        super().__init__("Budgeted-TD")
        self.T = float(T_recency)
        self.alpha_util = float(alpha_util)
        self.c_max = float(c_max)
        self.w_min = float(w_min)

    def decide(self, stats: WindowStats | None, budget: AdaptationBudget) -> BudgetDecision:
        return BudgetDecision(
            max_updates=budget.max_updates_per_window,
            recent_ratio=budget.recent_replay_ratio,
            reason=f"Budgeted-TD (T={self.T}, alpha={self.alpha_util})",
            weight_fn=_make_budgeted_td_weight_fn(
                T=self.T, alpha_util=self.alpha_util, c_max=self.c_max, w_min=self.w_min,
            ),
            sampling_mode="weighted",
        )


class BudgetedGradPolicy(AdaptationPolicy):
    """v2.1.1 main alt: recency + per-sample input-gradient utility.

    Utility = ||∂Q1_i/∂obs_i|| backfilled into buffer.input_grad by SAC update.
    """

    def __init__(self, T_recency: float = 50_000.0, alpha_util: float = 1.0,
                 c_max: float = 5.0, w_min: float = 0.05) -> None:
        super().__init__("Budgeted-Grad")
        self.T = float(T_recency)
        self.alpha_util = float(alpha_util)
        self.c_max = float(c_max)
        self.w_min = float(w_min)

    def decide(self, stats: WindowStats | None, budget: AdaptationBudget) -> BudgetDecision:
        return BudgetDecision(
            max_updates=budget.max_updates_per_window,
            recent_ratio=budget.recent_replay_ratio,
            reason=f"Budgeted-Grad (T={self.T}, alpha={self.alpha_util}) — per-sample input grad",
            weight_fn=_make_budgeted_grad_weight_fn(
                T=self.T, alpha_util=self.alpha_util, c_max=self.c_max, w_min=self.w_min,
            ),
            sampling_mode="weighted",
        )


class BudgetedNoveltyPolicy(AdaptationPolicy):
    """v2.1.1 secondary: recency + per-sample state-delta novelty (||s_{t+1}-s_t||).

    Display name: `Budgeted-Novelty`.
    Registry alias `budgeted-pred` is preserved for backward compatibility with
    v2.1 pilot CSV files; the original v2.1 framing as "predictability residual"
    was honest at design time but the cheap implementation we ship is in fact
    a state-delta novelty proxy, not a true forward-model predictability
    residual. v2.1.1 names the policy honestly.
    """

    def __init__(self, T_recency: float = 50_000.0, alpha_util: float = 1.0,
                 c_max: float = 5.0, w_min: float = 0.05) -> None:
        super().__init__("Budgeted-Novelty")
        self.T = float(T_recency)
        self.alpha_util = float(alpha_util)
        self.c_max = float(c_max)
        self.w_min = float(w_min)

    def decide(self, stats: WindowStats | None, budget: AdaptationBudget) -> BudgetDecision:
        return BudgetDecision(
            max_updates=budget.max_updates_per_window,
            recent_ratio=budget.recent_replay_ratio,
            reason=f"Budgeted-Novelty (T={self.T}, alpha={self.alpha_util}) — per-sample state-delta novelty",
            weight_fn=_make_budgeted_pred_weight_fn(
                T=self.T, alpha_util=self.alpha_util, c_max=self.c_max, w_min=self.w_min,
            ),
            sampling_mode="weighted",
        )


# Backward-compatibility alias for v2.1 pilot data and external scripts.
BudgetedPredPolicy = BudgetedNoveltyPolicy


class RecencyWeightedPolicy(AdaptationPolicy):
    """v2.1.1 added: pure recency-weighted sampling on full historical buffer.

    Discovered as a strong baseline in v2.1 pilot (was the de-facto behavior
    of the bugged Budgeted-Grad/Pred). First-class baseline now so we can
    isolate "what does the per-sample utility add on top of recency alone?"
    """

    def __init__(self, T_recency: float = 50_000.0, w_min: float = 0.05) -> None:
        super().__init__("RecencyWeighted")
        self.T = float(T_recency)
        self.w_min = float(w_min)

    def decide(self, stats: WindowStats | None, budget: AdaptationBudget) -> BudgetDecision:
        return BudgetDecision(
            max_updates=budget.max_updates_per_window,
            recent_ratio=budget.recent_replay_ratio,
            reason=f"RecencyWeighted (T={self.T}, w_min={self.w_min})",
            weight_fn=_make_recency_only_weight_fn(T=self.T, w_min=self.w_min),
            sampling_mode="weighted",
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


_POLICY_REGISTRY: dict[str, type[AdaptationPolicy]] = {
    # v2.0 (kept for backward compatibility)
    "frozen": FrozenPolicy,
    "fixedadapt": FixedAdaptPolicy,
    "recentonly": RecentOnlyPolicy,
    "staticmix": StaticMixPolicy,
    "proposed": DriftAwarePolicy,
    "driftaware": DriftAwarePolicy,
    "drift_aware": DriftAwarePolicy,
    # v2.1 / v2.1.1
    "uniform": UniformPolicy,
    "per": PERPolicy,
    "budgeted-td": BudgetedTDPolicy,
    "budgeted_td": BudgetedTDPolicy,
    "budgetedtd": BudgetedTDPolicy,
    "budgeted-grad": BudgetedGradPolicy,
    "budgeted_grad": BudgetedGradPolicy,
    "budgetedgrad": BudgetedGradPolicy,
    # Honest name (v2.1.1+)
    "budgeted-novelty": BudgetedNoveltyPolicy,
    "budgeted_novelty": BudgetedNoveltyPolicy,
    "budgetednovelty": BudgetedNoveltyPolicy,
    # Backward-compat aliases (v2.1 pilot data uses these)
    "budgeted-pred": BudgetedNoveltyPolicy,
    "budgeted_pred": BudgetedNoveltyPolicy,
    "budgetedpred": BudgetedNoveltyPolicy,
    # v2.1.1 added
    "recency-weighted": RecencyWeightedPolicy,
    "recency_weighted": RecencyWeightedPolicy,
    "recencyweighted": RecencyWeightedPolicy,
}


def build_adaptation_policy(name: str) -> AdaptationPolicy:
    normalized = name.strip().lower()
    if normalized not in _POLICY_REGISTRY:
        raise ValueError(f"Unknown adaptation policy: {name}")
    return _POLICY_REGISTRY[normalized]()


def all_policy_names() -> list[str]:
    """Canonical names of all built-in policies (one entry per policy)."""
    return [
        "frozen", "fixedadapt", "recentonly", "staticmix", "proposed",
        "uniform", "per", "budgeted-td", "budgeted-grad", "budgeted-novelty",
        "recency-weighted",
    ]
