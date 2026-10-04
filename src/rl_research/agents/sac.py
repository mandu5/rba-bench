"""Minimal SAC agent inspired by CleanRL structure."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from rl_research.agents.networks import DoubleQNetwork, GaussianPolicy
from rl_research.config import ModelConfig
from rl_research.data.replay import TransitionBatch


class SACAgent:
    def __init__(
        self,
        obs_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        config: ModelConfig,
        device: str,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.act_dim = int(np.prod(action_low.shape))
        self.actor = GaussianPolicy(obs_dim, self.act_dim, config.hidden_sizes, action_low, action_high).to(self.device)
        self.critic = DoubleQNetwork(obs_dim, self.act_dim, config.hidden_sizes).to(self.device)
        self.critic_target = DoubleQNetwork(obs_dim, self.act_dim, config.hidden_sizes).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        self.autotune = config.autotune
        if self.autotune:
            self.target_entropy = -float(self.act_dim)
            self.log_alpha = torch.zeros(1, device=self.device, requires_grad=True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.actor_lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = config.alpha

    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                _, _, action = self.actor.sample(obs)
            else:
                action, _, _ = self.actor.sample(obs)
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def state_dict(self) -> dict[str, object]:
        state: dict[str, object] = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha": self.alpha,
            "autotune": self.autotune,
        }
        if self.autotune:
            state["log_alpha"] = self.log_alpha.detach().clone()
            state["alpha_optimizer"] = self.alpha_optimizer.state_dict()
        return state

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.alpha = float(state["alpha"])
        if self.autotune and "log_alpha" in state:
            self.log_alpha.data.copy_(state["log_alpha"].to(self.device))
            self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])

    def update(self, batch: TransitionBatch, global_step: int) -> dict[str, object]:
        obs = torch.as_tensor(batch.observations, dtype=torch.float32, device=self.device)
        # Enable per-sample input-gradient capture (used by Budgeted-Grad utility).
        # Critic forward is independent across batch dim, so summing Q1 and
        # backpropagating once gives ∂Q1_i/∂obs_i without separate backwards.
        obs.requires_grad_(True)
        actions = torch.as_tensor(batch.actions, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch.next_observations, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_actions, next_log_prob, _ = self.actor.sample(next_obs)
            target_q1, target_q2 = self.critic_target(next_obs, next_actions)
            min_target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target = rewards + (1.0 - dones) * self.config.gamma * min_target_q

        current_q1, current_q2 = self.critic(obs, actions)
        # Per-sample input gradient: ∂(Σ_i Q1_i) / ∂obs_i = ∂Q1_i / ∂obs_i
        # because Q1_j depends only on obs_j (batch-independent).
        try:
            input_grads = torch.autograd.grad(
                current_q1.sum(), obs, retain_graph=True, create_graph=False,
            )[0]
            per_sample_input_grad = input_grads.detach().norm(dim=-1).cpu().numpy()
        except RuntimeError:
            per_sample_input_grad = None
        # Detach obs from autograd graph for downstream reuse safety.
        obs = obs.detach()
        # Optional importance-sampling weights from PER baseline
        if batch.is_weights is not None:
            is_w = torch.as_tensor(batch.is_weights, dtype=torch.float32, device=self.device).view(-1, 1)
            critic_loss = (
                (is_w * (current_q1 - target).pow(2)).mean()
                + (is_w * (current_q2 - target).pow(2)).mean()
            )
        else:
            critic_loss = nn.functional.mse_loss(current_q1, target) + nn.functional.mse_loss(current_q2, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        # Compute critic grad-norm BEFORE the optimizer step.
        critic_grad_norm = 0.0
        for p in self.critic.parameters():
            if p.grad is not None:
                critic_grad_norm += float(p.grad.detach().pow(2).sum().item())
        critic_grad_norm = float(critic_grad_norm ** 0.5)
        self.critic_optimizer.step()

        actor_loss_value = 0.0
        alpha_loss_value = 0.0

        if global_step % self.config.policy_frequency == 0:
            pi, log_prob, _ = self.actor.sample(obs)
            q1_pi, q2_pi = self.critic(obs, pi)
            min_q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (self.alpha * log_prob - min_q_pi).mean()

            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            actor_loss_value = float(actor_loss.item())

            if self.autotune:
                alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad(set_to_none=True)
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = float(self.log_alpha.exp().item())
                alpha_loss_value = float(alpha_loss.item())

        if global_step % self.config.target_network_frequency == 0:
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.config.tau * param.data + (1.0 - self.config.tau) * target_param.data)

        # Per-sample TD-error magnitude (used by PER priority updates).
        td_per_sample_tensor = (current_q1.detach() - target.detach()).abs().squeeze(-1)
        td_per_sample = td_per_sample_tensor.cpu().numpy()
        td_error = float(td_per_sample_tensor.mean().item())
        # Cheap predictability proxy: mean L2 magnitude of state delta.
        # Without a learned forward model, this approximates "how unexpected
        # the next state is on average within the batch."
        predictability_residual = float((next_obs - obs).norm(dim=-1).mean().item())
        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": actor_loss_value,
            "alpha": float(self.alpha),
            "alpha_loss": alpha_loss_value,
            "td_error_mean": td_error,
            "grad_norm_mean": critic_grad_norm,
            "predictability_residual_mean": predictability_residual,
            "td_errors_per_sample": td_per_sample,
            "input_grads_per_sample": per_sample_input_grad,
        }
