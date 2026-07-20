"""
critic.py
Centralised value network for MAPPO (the Critic).

In CTDE (Centralised Training Decentralised Execution):
  - The critic sees the FULL global state (all drone obs concatenated).
  - At runtime, only the Actor is deployed on each drone.
  
Input:  concatenated observations of ALL drones  (n_drones * obs_dim)
Output: scalar state-value V(s)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class CentralCritic(nn.Module):
    """
    MLP that maps the joint observation of all drones to a scalar value.

    Usage in MAPPO:
        global_obs = torch.cat([obs_drone_0, obs_drone_1, ..., obs_drone_n], dim=-1)
        value = critic(global_obs)   # shape (batch,)
    """

    def __init__(
        self,
        obs_dim: int = 15,
        n_drones: int = 3,
        hidden_sizes: list[int] = None,
    ):
        super().__init__()
        hidden_sizes = hidden_sizes or [512, 256]
        input_dim = obs_dim * n_drones

        layers: list[nn.Module] = []
        in_size = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_size, h), nn.Tanh()]
            in_size = h
        layers.append(nn.Linear(in_size, 1))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Output layer: small init
        last = list(self.net.children())[-1]
        nn.init.orthogonal_(last.weight, gain=1.0)

    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            global_obs: (batch, n_drones * obs_dim) tensor
        Returns:
            values: (batch,) tensor
        """
        return self.net(global_obs).squeeze(-1)
