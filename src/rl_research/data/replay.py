"""Replay buffers with weighted sampling support (v2.1).

Adds per-transition timestamps and a generalized `sample_weighted` API used by
v2.1 baselines (PER, Budgeted-TD, Budgeted-Grad, Budgeted-Novelty) while preserving
the existing `sample_mixed` interface for backward compatibility with v2.0
adaptation policies.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class TransitionBatch:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    indices: np.ndarray | None = None
    is_weights: np.ndarray | None = None


# A WeightFn maps the buffer's current state to per-sample non-negative weights.
# It receives a "view" dict with arrays of length self.size:
#   - "ages": number of inserts since each transition was added (0 = newest)
#   - "td_priorities": priority proxy (defaults to 1.0 if not yet set)
WeightFn = Callable[[dict[str, np.ndarray]], np.ndarray]


class SimpleReplayBuffer:
    """Replay buffer with per-transition timestamps and TD priorities.

    `add` accepts an optional `priority` value. Untracked priorities default to
    1.0. The buffer maintains an `insert_step` counter so that `ages` can be
    computed as `insert_step - timestamps[i]`.
    """

    def __init__(self, obs_dim: int, act_dim: int, capacity: int) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.timestamps = np.zeros((capacity,), dtype=np.int64)
        self.priorities = np.ones((capacity,), dtype=np.float32)
        # v2.1.1 per-sample utility fields
        self.novelty = np.ones((capacity,), dtype=np.float32)        # ||s_{t+1} - s_t||
        self.input_grad = np.ones((capacity,), dtype=np.float32)      # ||∂Q/∂s_i||, backfilled by SAC update
        self.ptr = 0
        self.size = 0
        self.insert_step = 0  # monotonically increasing on every add

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
        priority: float = 1.0,
        novelty: float | None = None,
    ) -> int:
        """Insert a transition and return its slot index.

        `novelty` defaults to ||next_observation - observation||, an additional-
        free state-delta proxy used by Budgeted-Novelty. Persists per-sample so it
        is not collapsed by global multiplier normalization (cf. v2.1.1 bug fix).
        """
        slot = self.ptr
        self.obs[slot] = observation
        self.actions[slot] = action
        self.rewards[slot] = reward
        self.next_obs[slot] = next_observation
        self.dones[slot] = float(done)
        self.timestamps[slot] = self.insert_step
        self.priorities[slot] = float(max(priority, 1e-6))
        if novelty is None:
            novelty = float(np.linalg.norm(next_observation - observation))
        self.novelty[slot] = float(max(novelty, 1e-6))
        self.input_grad[slot] = 1.0  # default until first update backfills
        self.insert_step += 1
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return slot

    def __len__(self) -> int:
        return self.size

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        if len(indices) == 0:
            return
        clipped = np.maximum(priorities.astype(np.float32), 1e-6)
        self.priorities[indices] = clipped

    def update_input_grads(self, indices: np.ndarray, grads: np.ndarray) -> None:
        """Backfill per-sample input gradient norms from a recent SAC update."""
        if len(indices) == 0:
            return
        clipped = np.maximum(grads.astype(np.float32), 1e-6)
        self.input_grad[indices] = clipped

    def view(self) -> dict[str, np.ndarray]:
        """Return arrays of length `self.size` for use by WeightFn callers."""
        active_timestamps = self.timestamps[: self.size]
        ages = (self.insert_step - 1 - active_timestamps).astype(np.float32)
        return {
            "ages": ages,
            "td_priorities": self.priorities[: self.size].copy(),
            "novelty": self.novelty[: self.size].copy(),
            "input_grad": self.input_grad[: self.size].copy(),
        }

    def _gather(self, idx: np.ndarray) -> TransitionBatch:
        return TransitionBatch(
            observations=self.obs[idx],
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            next_observations=self.next_obs[idx],
            dones=self.dones[idx],
            indices=idx.copy(),
        )

    def sample(self, batch_size: int, rng: np.random.Generator) -> TransitionBatch:
        """Uniform sampling (v2.0 compatibility)."""
        if self.size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        idx = rng.integers(0, self.size, size=batch_size)
        return self._gather(idx)

    def sample_weighted(
        self,
        batch_size: int,
        weight_fn: WeightFn,
        rng: np.random.Generator,
        beta_is: float = 0.0,
    ) -> TransitionBatch:
        """Sample with per-transition probability proportional to `weight_fn(view)`.

        Optional importance-sampling weights are returned in `is_weights` if
        `beta_is > 0` (PER convention).
        """
        if self.size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        view = self.view()
        weights = np.asarray(weight_fn(view), dtype=np.float64)
        if weights.shape[0] != self.size:
            raise ValueError(
                f"weight_fn returned shape {weights.shape}, expected ({self.size},)."
            )
        weights = np.clip(weights, 0.0, None)
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0.0:
            # Degenerate fallback: uniform.
            probs = np.full(self.size, 1.0 / self.size, dtype=np.float64)
        else:
            probs = weights / total
        idx = rng.choice(self.size, size=batch_size, replace=True, p=probs)
        batch = self._gather(idx)
        if beta_is > 0.0:
            sampled_probs = probs[idx]
            n = float(self.size)
            is_w = (n * sampled_probs) ** (-beta_is)
            is_w /= max(is_w.max(), 1e-8)
            batch.is_weights = is_w.astype(np.float32)
        return batch

    def state_dict(self) -> dict[str, object]:
        return {
            "capacity": self.capacity,
            "obs": self.obs.copy(),
            "next_obs": self.next_obs.copy(),
            "actions": self.actions.copy(),
            "rewards": self.rewards.copy(),
            "dones": self.dones.copy(),
            "timestamps": self.timestamps.copy(),
            "priorities": self.priorities.copy(),
            "novelty": self.novelty.copy(),
            "input_grad": self.input_grad.copy(),
            "ptr": self.ptr,
            "size": self.size,
            "insert_step": self.insert_step,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.capacity = int(state["capacity"])
        self.obs = np.array(state["obs"], copy=True)
        self.next_obs = np.array(state["next_obs"], copy=True)
        self.actions = np.array(state["actions"], copy=True)
        self.rewards = np.array(state["rewards"], copy=True)
        self.dones = np.array(state["dones"], copy=True)
        # Backward compat: older snapshots may lack timestamps/priorities.
        if "timestamps" in state:
            self.timestamps = np.array(state["timestamps"], copy=True)
        else:
            self.timestamps = np.arange(self.capacity, dtype=np.int64)
        if "priorities" in state:
            self.priorities = np.array(state["priorities"], copy=True)
        else:
            self.priorities = np.ones((self.capacity,), dtype=np.float32)
        if "novelty" in state:
            self.novelty = np.array(state["novelty"], copy=True)
        else:
            self.novelty = np.ones((self.capacity,), dtype=np.float32)
        if "input_grad" in state:
            self.input_grad = np.array(state["input_grad"], copy=True)
        else:
            self.input_grad = np.ones((self.capacity,), dtype=np.float32)
        self.ptr = int(state["ptr"])
        self.size = int(state["size"])
        self.insert_step = int(state.get("insert_step", self.size))


class DualReplayBuffer:
    """Two parallel buffers (historical and recent).

    v2.0 API (`sample_mixed(recent_ratio)`) is preserved.
    v2.1 adds:
      - `sample_weighted(weight_fn)` — sample from `historical` (the union view)
      - per-buffer priority updates via `update_priorities`
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        capacity: int,
        recent_capacity: int,
    ) -> None:
        self.historical = SimpleReplayBuffer(obs_dim, act_dim, capacity)
        self.recent = SimpleReplayBuffer(obs_dim, act_dim, recent_capacity)

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
        priority: float = 1.0,
        novelty: float | None = None,
    ) -> tuple[int, int]:
        h_idx = self.historical.add(observation, action, reward, next_observation, done, priority, novelty)
        r_idx = self.recent.add(observation, action, reward, next_observation, done, priority, novelty)
        return h_idx, r_idx

    def __len__(self) -> int:
        return len(self.historical)

    def state_dict(self) -> dict[str, object]:
        return {
            "historical": self.historical.state_dict(),
            "recent": self.recent.state_dict(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.historical.load_state_dict(deepcopy(state["historical"]))
        self.recent.load_state_dict(deepcopy(state["recent"]))

    # ---- v2.0 compatible API ------------------------------------------------

    def sample_mixed(
        self,
        batch_size: int,
        recent_ratio: float,
        rng: np.random.Generator,
    ) -> tuple[TransitionBatch, dict[str, int]]:
        if len(self.historical) == 0:
            raise ValueError("Cannot sample before replay has data.")

        recent_ratio = float(np.clip(recent_ratio, 0.0, 1.0))
        recent_count = int(round(batch_size * recent_ratio))
        historical_count = batch_size - recent_count

        if len(self.recent) == 0:
            recent_count = 0
            historical_count = batch_size
        if len(self.historical) == 0:
            historical_count = 0
            recent_count = batch_size

        pieces: list[TransitionBatch] = []
        if recent_count > 0:
            pieces.append(self.recent.sample(recent_count, rng))
        if historical_count > 0:
            pieces.append(self.historical.sample(historical_count, rng))

        observations = np.concatenate([p.observations for p in pieces], axis=0)
        actions = np.concatenate([p.actions for p in pieces], axis=0)
        rewards = np.concatenate([p.rewards for p in pieces], axis=0)
        next_observations = np.concatenate([p.next_observations for p in pieces], axis=0)
        dones = np.concatenate([p.dones for p in pieces], axis=0)

        order = rng.permutation(len(observations))
        batch = TransitionBatch(
            observations=observations[order],
            actions=actions[order],
            rewards=rewards[order],
            next_observations=next_observations[order],
            dones=dones[order],
        )
        return batch, {
            "recent_samples": recent_count,
            "historical_samples": historical_count,
        }

    # ---- v2.1 weighted API --------------------------------------------------

    def sample_weighted(
        self,
        batch_size: int,
        weight_fn: WeightFn,
        rng: np.random.Generator,
        beta_is: float = 0.0,
        source: str = "historical",
    ) -> tuple[TransitionBatch, dict[str, int]]:
        """Sample from a single underlying buffer using arbitrary weights.

        `source` selects which buffer to draw from. The default is "historical"
        because the v2.1 budgeted policies operate on the union of all stored
        transitions, with the recency vs. retention tradeoff encoded in the
        weight function (via the `ages` field of the buffer view).
        """
        buf: SimpleReplayBuffer
        if source == "recent":
            buf = self.recent
        else:
            buf = self.historical
        if len(buf) == 0:
            raise ValueError("Cannot sample before replay has data.")
        batch = buf.sample_weighted(batch_size, weight_fn=weight_fn, rng=rng, beta_is=beta_is)
        meta = {
            "weighted_source": source,
            "weighted_samples": int(batch_size),
        }
        # Compute approximate recent vs. historical share for downstream logging.
        # A transition counts as "recent-shareable" if its age is within the
        # recent buffer's effective horizon.
        view = buf.view()
        if batch.indices is not None:
            ages = view["ages"][batch.indices]
            recent_horizon = float(self.recent.capacity)
            recent_mask = ages < recent_horizon
            meta["recent_samples"] = int(recent_mask.sum())
            meta["historical_samples"] = int((~recent_mask).sum())
        else:
            meta["recent_samples"] = 0
            meta["historical_samples"] = batch_size
        return batch, meta

    def update_priorities(
        self,
        indices: np.ndarray,
        priorities: np.ndarray,
        source: str = "historical",
    ) -> None:
        if source == "recent":
            self.recent.update_priorities(indices, priorities)
        else:
            self.historical.update_priorities(indices, priorities)

    def update_input_grads(
        self,
        indices: np.ndarray,
        grads: np.ndarray,
        source: str = "historical",
    ) -> None:
        if source == "recent":
            self.recent.update_input_grads(indices, grads)
        else:
            self.historical.update_input_grads(indices, grads)
