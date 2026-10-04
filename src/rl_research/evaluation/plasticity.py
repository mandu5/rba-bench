"""Plasticity proxy module (v2.1).

Implements the three v2.1 main plasticity diagnostics plus two supporting
diagnostics. All functions are intentionally cheap and do not require
additional trained networks.

Main metrics
------------
1. relearning_speed_after_drift(returns_per_window, drift_window_index)
2. dormant_neuron_ratio(model, observations, threshold=...)
3. gradient_norm_ema (already tracked per-window in WindowStats)

Supporting diagnostics
----------------------
4. effective_rank_of_representation(model, observations)
5. ntk_rank_proxy(model, observations) — optional, expensive

These are diagnostics, not method components. They are computed once per
deployment window via `compute_plasticity_snapshot` and recorded alongside
the existing window logs.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np
import torch
from torch import nn


# ---------------------------------------------------------------------------
# Snapshot dataclass
# ---------------------------------------------------------------------------


@dataclass
class PlasticitySnapshot:
    step: int
    window_index: int
    dormant_neuron_ratio: float
    activation_mean_abs: float
    effective_rank: float
    grad_norm_ema: float = 0.0
    # NTK rank is optional and may be NaN if not computed
    ntk_rank_proxy: float = float("nan")


# ---------------------------------------------------------------------------
# Main metrics
# ---------------------------------------------------------------------------


def relearning_speed_after_drift(
    returns_per_window: Sequence[float],
    drift_window_index: int,
    pre_drift_reference: float,
    recovery_fraction: float = 0.9,
) -> dict[str, float]:
    """Number of post-drift windows needed to reach `recovery_fraction` of pre-drift reference.

    Returns a dict with:
      - `windows_to_recovery`: int (or len(post) if not recovered)
      - `recovered`: 1.0 / 0.0
      - `final_gap`: pre_drift_reference - last post-drift return
    """
    series = np.asarray(returns_per_window, dtype=np.float64)
    if drift_window_index >= len(series):
        return {"windows_to_recovery": 0.0, "recovered": 0.0, "final_gap": 0.0}
    post = series[drift_window_index:]
    target = float(pre_drift_reference) * float(recovery_fraction)
    above = np.where(post >= target)[0]
    if above.size == 0:
        return {
            "windows_to_recovery": float(len(post)),
            "recovered": 0.0,
            "final_gap": float(pre_drift_reference - post[-1]),
        }
    return {
        "windows_to_recovery": float(int(above[0])),
        "recovered": 1.0,
        "final_gap": float(pre_drift_reference - post[-1]),
    }


def _hook_collect_post_relu_activations(
    model: nn.Module, sample_obs: torch.Tensor
) -> list[np.ndarray]:
    """Run a forward pass and capture post-activation outputs of nn.ReLU layers.

    Returns one ndarray per ReLU layer of shape (batch, units).
    """
    captured: list[np.ndarray] = []
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def _hook(_module, _inputs, outputs):
        captured.append(outputs.detach().cpu().numpy())

    for module in model.modules():
        if isinstance(module, nn.ReLU):
            handles.append(module.register_forward_hook(_hook))

    try:
        with torch.no_grad():
            model(sample_obs)
    finally:
        for h in handles:
            h.remove()
    return captured


def dormant_neuron_ratio(
    model: nn.Module,
    sample_obs: torch.Tensor,
    threshold: float = 1e-3,
) -> dict[str, float]:
    """Fraction of post-ReLU units whose batch-mean absolute activation is below `threshold`.

    Inspired by Sokar et al. and the Dohare et al. plasticity-loss diagnostics.
    A unit is "dormant" if its mean absolute activation across the batch is
    near zero, i.e. it is contributing nothing to the network output.
    """
    activations_list = _hook_collect_post_relu_activations(model, sample_obs)
    if not activations_list:
        # Model has no nn.ReLU layers — return safe defaults.
        return {"dormant_neuron_ratio": 0.0, "activation_mean_abs": 0.0}
    total_units = 0
    dormant = 0
    abs_means: list[float] = []
    for act in activations_list:
        # act shape: (batch, units) for MLP layers
        if act.ndim < 2:
            continue
        unit_means = np.abs(act).mean(axis=0)  # (units,)
        total_units += unit_means.shape[0]
        dormant += int((unit_means < threshold).sum())
        abs_means.append(float(unit_means.mean()))
    ratio = float(dormant) / max(total_units, 1)
    return {
        "dormant_neuron_ratio": ratio,
        "activation_mean_abs": float(np.mean(abs_means)) if abs_means else 0.0,
    }


# ---------------------------------------------------------------------------
# Supplementary metrics
# ---------------------------------------------------------------------------


def effective_rank_of_representation(
    model: nn.Module,
    sample_obs: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """Roy & Vetterli (2007) effective rank of the last hidden representation.

    eff_rank(F) = exp(- Σ p_i log p_i),  p_i = σ_i / Σ σ_j

    where {σ_i} are singular values of the activation matrix F (n_samples, n_units).
    """
    activations_list = _hook_collect_post_relu_activations(model, sample_obs)
    if not activations_list:
        return 0.0
    # Use the last ReLU layer's activations as the representation proxy.
    F = activations_list[-1]
    if F.ndim != 2 or F.shape[0] < 2 or F.shape[1] < 2:
        return 0.0
    F = F - F.mean(axis=0, keepdims=True)
    try:
        sv = np.linalg.svd(F, compute_uv=False)
    except np.linalg.LinAlgError:
        return 0.0
    sv = sv[sv > eps]
    if sv.size == 0:
        return 0.0
    p = sv / sv.sum()
    entropy = -float((p * np.log(p + eps)).sum())
    return float(np.exp(entropy))


def ntk_rank_proxy(
    model: nn.Module,
    sample_obs: torch.Tensor,
    n_features: int = 64,
    eps: float = 1e-6,
) -> float:
    """Cheap NTK-rank proxy using only the last-layer Jacobian.

    Computes the Jacobian of the network output w.r.t. the last linear layer
    weights for each sample, stacks them, and reports effective rank. This is
    much cheaper than full NTK rank but captures the "feature-direction
    diversity" intuition. Reported only when explicitly requested because
    even this proxy is significantly more expensive than dormant-ratio.
    """
    if sample_obs.shape[0] < 2:
        return float("nan")
    # Find the last nn.Linear layer
    last_linear: nn.Linear | None = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            last_linear = module
    if last_linear is None:
        return float("nan")
    # Use the activations feeding into last_linear as features.
    # We approximate by running the full model and capturing the last ReLU
    # output (which feeds into the final Linear in our DoubleQNetwork / Policy).
    activations = _hook_collect_post_relu_activations(model, sample_obs)
    if not activations:
        return float("nan")
    F = activations[-1]
    if F.ndim != 2 or F.shape[0] < 2 or F.shape[1] < 2:
        return float("nan")
    # Subsample for cost control
    if F.shape[0] > n_features:
        idx = np.linspace(0, F.shape[0] - 1, n_features).astype(int)
        F = F[idx]
    # NTK rank proxy = effective rank of F F^T spectrum
    try:
        gram = F @ F.T
        sv = np.linalg.svd(gram, compute_uv=False)
    except np.linalg.LinAlgError:
        return float("nan")
    sv = sv[sv > eps]
    if sv.size == 0:
        return float("nan")
    p = sv / sv.sum()
    entropy = -float((p * np.log(p + eps)).sum())
    return float(np.exp(entropy))


# ---------------------------------------------------------------------------
# Snapshot orchestrator
# ---------------------------------------------------------------------------


def compute_plasticity_snapshot(
    model: nn.Module,
    sample_obs: np.ndarray,
    *,
    step: int,
    window_index: int,
    grad_norm_ema: float = 0.0,
    include_ntk: bool = False,
    device: torch.device | str = "cpu",
    dormant_threshold: float = 1e-3,
) -> PlasticitySnapshot:
    """Compute the v2.1 plasticity snapshot for one window.

    `sample_obs` should be a numpy batch of observations — typically the most
    recent batch used for an update, or a fresh draw from the replay buffer.
    """
    obs_t = torch.as_tensor(sample_obs, dtype=torch.float32, device=device)
    if obs_t.ndim == 1:
        obs_t = obs_t.unsqueeze(0)
    dormant = dormant_neuron_ratio(model, obs_t, threshold=dormant_threshold)
    eff_rank = effective_rank_of_representation(model, obs_t)
    ntk = float("nan")
    if include_ntk:
        ntk = ntk_rank_proxy(model, obs_t)
    return PlasticitySnapshot(
        step=int(step),
        window_index=int(window_index),
        dormant_neuron_ratio=float(dormant["dormant_neuron_ratio"]),
        activation_mean_abs=float(dormant["activation_mean_abs"]),
        effective_rank=float(eff_rank),
        grad_norm_ema=float(grad_norm_ema),
        ntk_rank_proxy=float(ntk),
    )


def snapshot_to_dict(snap: PlasticitySnapshot) -> dict[str, float | int]:
    return asdict(snap)
