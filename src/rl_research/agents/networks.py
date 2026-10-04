"""Neural networks for SAC."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

LOG_STD_MAX = 2.0
LOG_STD_MIN = -5.0


def build_mlp(input_dim: int, hidden_sizes: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for hidden in hidden_sizes:
        layers.append(nn.Linear(prev_dim, hidden))
        layers.append(nn.ReLU())
        prev_dim = hidden
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: tuple[int, ...],
        action_low: np.ndarray,
        action_high: np.ndarray,
    ) -> None:
        super().__init__()
        self.backbone = build_mlp(obs_dim, hidden_sizes, hidden_sizes[-1])
        self.mean = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std = nn.Linear(hidden_sizes[-1], act_dim)
        scale = (action_high - action_low) / 2.0
        bias = (action_high + action_low) / 2.0
        self.register_buffer("action_scale", torch.as_tensor(scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.as_tensor(bias, dtype=torch.float32))

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(observation)
        mean = self.mean(features)
        log_std = torch.clamp(self.log_std(features), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(observation)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        latent = dist.rsample()
        squashed = torch.tanh(latent)
        action = squashed * self.action_scale + self.action_bias
        log_prob = dist.log_prob(latent)
        log_prob -= torch.log(self.action_scale * (1 - squashed.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action


class DoubleQNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: tuple[int, ...]) -> None:
        super().__init__()
        input_dim = obs_dim + act_dim
        self.q1 = build_mlp(input_dim, hidden_sizes, 1)
        self.q2 = build_mlp(input_dim, hidden_sizes, 1)

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([observation, action], dim=-1)
        return self.q1(x), self.q2(x)
