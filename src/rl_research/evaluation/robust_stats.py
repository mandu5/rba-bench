"""Robust summary statistics for RL seed sweeps.

The project reports IQM and bootstrap intervals because mean-only reporting is
too brittle for small-N deep RL runs.
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable, Iterable

import numpy as np


def finite_values(values: Iterable[object]) -> np.ndarray:
    """Return finite float values from a generic iterable."""
    out: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            out.append(number)
    return np.asarray(out, dtype=np.float64)


def sample_std(values: Iterable[object]) -> float | None:
    vals = finite_values(values)
    if len(vals) == 0:
        return None
    if len(vals) == 1:
        return 0.0
    return float(np.std(vals, ddof=1))


def mean(values: Iterable[object]) -> float | None:
    vals = finite_values(values)
    if len(vals) == 0:
        return None
    return float(np.mean(vals))


def iqm(values: Iterable[object]) -> float | None:
    """Finite-sample interquartile mean.

    For n=5 this averages the middle 3 values. For n<4, it falls back to the
    mean because trimming would discard too much data.
    """
    vals = np.sort(finite_values(values))
    n = len(vals)
    if n == 0:
        return None
    if n < 4:
        return float(np.mean(vals))
    trim = int(np.floor(0.25 * n))
    kept = vals[trim : n - trim]
    if len(kept) == 0:
        kept = vals
    return float(np.mean(kept))


def bootstrap_ci(
    values: Iterable[object],
    statistic: Callable[[Iterable[object]], float | None] = iqm,
    *,
    confidence: float = 0.95,
    n_resamples: int = 5000,
    seed: int = 0,
) -> tuple[float | None, float | None]:
    """Percentile bootstrap confidence interval for a scalar statistic."""
    vals = finite_values(values)
    if len(vals) == 0:
        return None, None
    if len(vals) == 1 or n_resamples <= 0:
        stat = statistic(vals)
        return stat, stat
    rng = np.random.default_rng(seed)
    stats = np.empty(n_resamples, dtype=np.float64)
    for idx in range(n_resamples):
        sample = rng.choice(vals, size=len(vals), replace=True)
        stat = statistic(sample)
        stats[idx] = np.nan if stat is None else float(stat)
    stats = stats[np.isfinite(stats)]
    if len(stats) == 0:
        return None, None
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(stats, alpha)),
        float(np.quantile(stats, 1.0 - alpha)),
    )


def rank_names(
    rows: list[dict[str, object]],
    metric: str,
    *,
    higher_is_better: bool = True,
    value_suffix: str = "_iqm",
) -> list[str]:
    """Rank policy names by a metric column in aggregated rows."""
    column = metric + value_suffix

    def key(row: dict[str, object]) -> float:
        value = row.get(column)
        if value is None:
            return float("-inf") if higher_is_better else float("inf")
        number = float(value)
        if not np.isfinite(number):
            return float("-inf") if higher_is_better else float("inf")
        return number

    return [
        str(row["adaptation_policy"])
        for row in sorted(rows, key=key, reverse=higher_is_better)
    ]


def kendall_tau(left: list[str], right: list[str]) -> tuple[float | None, int, int]:
    """Kendall tau over two ranked policy-name lists, returning tau/concordant/discordant."""
    left_pos = {name: idx for idx, name in enumerate(left)}
    right_pos = {name: idx for idx, name in enumerate(right)}
    names = sorted(set(left_pos) & set(right_pos))
    if len(names) < 2:
        return None, 0, 0
    concordant = 0
    discordant = 0
    for first, second in combinations(names, 2):
        left_order = left_pos[first] < left_pos[second]
        right_order = right_pos[first] < right_pos[second]
        if left_order == right_order:
            concordant += 1
        else:
            discordant += 1
    total = concordant + discordant
    if total == 0:
        return None, concordant, discordant
    return float((concordant - discordant) / total), concordant, discordant
