"""Budget helper dataclasses (v2.1).

Extends `BudgetDecision` with an optional `weight_fn` that selects sampling
weights for v2.1 baselines (Uniform, PER, Budgeted-{TD,Grad,Novelty}).
When `weight_fn is None`, the runner falls back to v2.0 `sample_mixed` using
`recent_ratio`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional

from rl_research.data.replay import WeightFn


@dataclass
class WindowStats:
    window_index: int
    mean_return: float
    reference_return: float
    return_drop: float
    td_error_ema: float
    drift_events: int
    updates_used: int
    replay_samples: int
    mean_update_time: float
    steps_elapsed: int
    grad_norm_ema: float = 0.0
    predictability_residual_ema: float = 0.0


@dataclass
class BudgetDecision:
    max_updates: int
    recent_ratio: float
    reason: str
    # v2.1 extensions (optional, no v2.0 breakage)
    weight_fn: Optional[WeightFn] = field(default=None, repr=False)
    beta_is: float = 0.0
    sampling_mode: str = "mixed"  # "mixed" | "weighted" | "uniform"


def scheduled_updates_this_step(remaining_updates: int, remaining_steps: int) -> int:
    if remaining_updates <= 0 or remaining_steps <= 0:
        return 0
    return max(1, math.ceil(remaining_updates / remaining_steps))
