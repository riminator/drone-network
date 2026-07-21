"""
actor.py
Per-drone policy network (the Actor in Actor-Critic / MAPPO).

Architecture: MLP with configurable hidden layers.
Input:  observation vector (DroneAgent.OBS_DIM = 15)
Output: mean and log-std of a Gaussian over action space (4 dims)
        Action is sampled during training; argmax at evaluation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class Actor(nn.Module):
    """
    Stochastic Gaussian policy for continuous action spaces.

    Forward pass returns a (mean, log_std) pair.
    Use .get_action() for sampling + log-prob in one call (needed by PPO).
    """

    # Tighter ceiling prevents std from saturating to e^2 ≈ 7.4 and
    # pinning entropy near its maximum. e^0.5 ≈ 1.65 is plenty of exploration.
    LOG_STD_MIN = -3.0
    LOG_STD_MAX = 0.0   # tightened: +0.5 ceiling let log_std pin at max (entropy 7.65)

    def __init__(
        self,
        obs_dim: int = 15,
        act_dim: int = 4,
        hidden_sizes: list[int] = None,
    ):
        super().__init__()
        hidden_sizes = hidden_sizes or [256, 256]

        layers: list[nn.Module] = []
        in_size = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_size, h), nn.Tanh()]
            in_size = h

        self.shared_net = nn.Sequential(*layers)
        self.mean_head = nn.Linear(in_size, act_dim)

        # State-independent log_std parameter vector (standard stable-PPO design).
        # A separate network head for log_std can saturate its weights and
        # drive entropy to the ceiling. A plain parameter is directly
        # regularised by the entropy coefficient gradient.
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Small init on mean head for stable early updates
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        # Start log_std at 0 → std = 1.0 → moderate exploration
        nn.init.constant_(self.log_std, 0.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean, log_std) — both shape (..., act_dim)."""
        features = self.shared_net(obs)
        mean = self.mean_head(features)
        # Expand to batch dims and clamp to safe range
        log_std = torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        log_std = log_std.expand_as(mean)
        return mean, log_std

    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action and return (squashed_action, raw_action, log_prob).

        squashed_action — sent to the environment:
            spatial [:3] squashed with tanh  → [-1, 1]
            tool    [3]  squashed with sigmoid → [0, 1]

        raw_action — stored in the replay buffer (pre-squash Gaussian sample).
            Must be passed back to evaluate_actions() during the PPO update
            so log-probs are computed in the same (unbounded) space.

        log_prob — corrected for the tanh/sigmoid change-of-variables so that
            the PPO ratio is numerically valid.
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)

        if deterministic:
            raw_action = mean
        else:
            raw_action = dist.rsample()

        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        # Tanh squashing log-det-Jacobian correction (spatial dims 0-2):
        #   log|d(tanh(x))/dx| = log(1 - tanh(x)^2)  (numerically stable form below)
        tanh_correction = (
            2.0 * (np.log(2.0) - raw_action[..., :3] - torch.nn.functional.softplus(-2.0 * raw_action[..., :3]))
        ).sum(dim=-1)
        # Sigmoid squashing correction (tool dim 3):
        #   sigmoid(x) = (tanh(x/2) + 1) / 2  →  correction: -log(sigmoid(x)*(1-sigmoid(x)))
        sig_val = torch.sigmoid(raw_action[..., 3])
        sigmoid_correction = -(torch.log(sig_val + 1e-6) + torch.log(1.0 - sig_val + 1e-6))
        log_prob = log_prob - tanh_correction - sigmoid_correction

        # Squash for env
        spatial = torch.tanh(raw_action[..., :3])
        tool = torch.sigmoid(raw_action[..., 3:4])
        squashed_action = torch.cat([spatial, tool], dim=-1)

        return squashed_action, raw_action, log_prob

    def evaluate_actions(
        self, obs: torch.Tensor, raw_actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Given stored obs and RAW (pre-squash) actions from the buffer,
        return (log_probs, entropy) with squashing correction applied.

        raw_actions must be the pre-tanh/sigmoid samples stored by get_action(),
        NOT the squashed actions sent to the environment.
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)

        log_probs = dist.log_prob(raw_actions).sum(dim=-1)
        # Apply same squashing correction as in get_action()
        tanh_correction = (
            2.0 * (np.log(2.0) - raw_actions[..., :3] - torch.nn.functional.softplus(-2.0 * raw_actions[..., :3]))
        ).sum(dim=-1)
        sig_val = torch.sigmoid(raw_actions[..., 3])
        sigmoid_correction = -(torch.log(sig_val + 1e-6) + torch.log(1.0 - sig_val + 1e-6))
        log_probs = log_probs - tanh_correction - sigmoid_correction

        # Entropy of the underlying Gaussian (approximate — before squashing)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, entropy
