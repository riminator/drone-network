"""
replay_buffer.py
Fixed-size on-policy rollout buffer for MAPPO.

Stores trajectories for all agents and computes GAE (Generalised Advantage
Estimation) before each policy update.
"""

from __future__ import annotations

import numpy as np
import torch


class RolloutBuffer:
    """
    Stores one epoch of experience across all agents.
    
    Usage:
        buf = RolloutBuffer(n_steps=2048, n_agents=3, obs_dim=15, act_dim=4)
        
        # collect:
        buf.add(obs_dict, action_dict, reward_dict, done_dict,
                log_prob_dict, value_dict)
        
        # after rollout ends:
        buf.compute_returns_and_advantages(last_values, gamma, gae_lambda)
        
        # iterate mini-batches:
        for batch in buf.get_batches(mini_batch_size=64):
            ...
    """

    def __init__(
        self,
        n_steps: int,
        n_agents: int,
        obs_dim: int,
        act_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = "cpu",
    ):
        self.n_steps = n_steps
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

        self._ptr = 0  # current write index
        self._full = False

        # Flat storage: shape (n_steps, n_agents, ...)
        self.obs = np.zeros((n_steps, n_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_agents, act_dim), dtype=np.float32)
        self.rewards = np.zeros((n_steps, n_agents), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_agents), dtype=np.float32)
        self.log_probs = np.zeros((n_steps, n_agents), dtype=np.float32)
        self.values = np.zeros((n_steps, n_agents), dtype=np.float32)

        # Computed after rollout
        self.returns = np.zeros((n_steps, n_agents), dtype=np.float32)
        self.advantages = np.zeros((n_steps, n_agents), dtype=np.float32)

        self._agent_ids: list[str] = []

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def set_agent_ids(self, agent_ids: list[str]):
        self._agent_ids = sorted(agent_ids)

    def add(
        self,
        obs_dict: dict[str, np.ndarray],
        action_dict: dict[str, np.ndarray],
        reward_dict: dict[str, float],
        done_dict: dict[str, bool],
        log_prob_dict: dict[str, float],
        value_dict: dict[str, float],
    ):
        """Store one timestep of experience for all agents."""
        idx = self._ptr
        for i, aid in enumerate(self._agent_ids):
            self.obs[idx, i] = obs_dict.get(aid, np.zeros(self.obs_dim))
            self.actions[idx, i] = action_dict.get(aid, np.zeros(self.act_dim))
            self.rewards[idx, i] = reward_dict.get(aid, 0.0)
            self.dones[idx, i] = float(done_dict.get(aid, False))
            self.log_probs[idx, i] = log_prob_dict.get(aid, 0.0)
            self.values[idx, i] = value_dict.get(aid, 0.0)

        self._ptr += 1
        if self._ptr >= self.n_steps:
            self._full = True
            self._ptr = 0

    # ------------------------------------------------------------------
    # GAE computation
    # ------------------------------------------------------------------

    def compute_returns_and_advantages(
        self, last_values: dict[str, float]
    ):
        """
        Compute GAE advantages and discounted returns in-place.
        Call this once the rollout is complete, before calling get_batches().
        
        last_values: value estimates for the state AFTER the final step.
        """
        last_val = np.array(
            [last_values.get(aid, 0.0) for aid in self._agent_ids],
            dtype=np.float32,
        )

        n = self.n_steps
        gae = np.zeros(self.n_agents, dtype=np.float32)

        for t in reversed(range(n)):
            next_val = last_val if t == n - 1 else self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = (
                self.rewards[t]
                + self.gamma * next_val * next_non_terminal
                - self.values[t]
            )
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]

        # Normalise advantages across all agents and timesteps
        flat_adv = self.advantages.reshape(-1)
        self.advantages = (
            (self.advantages - flat_adv.mean()) / (flat_adv.std() + 1e-8)
        )

    # ------------------------------------------------------------------
    # Mini-batch iteration
    # ------------------------------------------------------------------

    def get_batches(self, mini_batch_size: int):
        """
        Yields mini-batches of flattened (timestep × agent) transitions.
        Each batch is a dict of torch tensors on self.device.
        """
        n = self.n_steps
        # Flatten: (n_steps * n_agents, ...)
        flat = {
            "obs":        torch.tensor(self.obs.reshape(-1, self.obs_dim),   device=self.device),
            "actions":    torch.tensor(self.actions.reshape(-1, self.act_dim), device=self.device),
            "log_probs":  torch.tensor(self.log_probs.reshape(-1),           device=self.device),
            "values":     torch.tensor(self.values.reshape(-1),              device=self.device),
            "returns":    torch.tensor(self.returns.reshape(-1),             device=self.device),
            "advantages": torch.tensor(self.advantages.reshape(-1),         device=self.device),
        }
        # Global obs for critic: concatenate all agents per timestep
        global_obs = self.obs.reshape(n, -1)  # (n_steps, n_agents * obs_dim)
        flat["global_obs"] = torch.tensor(
            np.repeat(global_obs, self.n_agents, axis=0),
            device=self.device,
        )

        total = n * self.n_agents
        indices = np.random.permutation(total)
        for start in range(0, total, mini_batch_size):
            idx = indices[start: start + mini_batch_size]
            yield {k: v[idx] for k, v in flat.items()}

    def reset(self):
        self._ptr = 0
        self._full = False
