"""
bid_policy.py
Neural network that produces a scalar bid value for a (drone, task) pair,
plus a separate learned marginal-value score for co-assignment decisions.

Architecture
------------
Input:  BID_OBS_DIM = 14 features describing one (drone, task) pairing
Output:
  - primary bid logit  → sigmoid → bid ∈ (0, 1)   (used for primary assignment)
  - marginal bid logit → sigmoid → marginal ∈ (0, 1) (used for co-assignment;
    this head is trained to predict extra value of joining an in-progress task)

The two heads share the same hidden-layer trunk so marginal prediction is
cheap and benefits from the shared representation.

Observation layout (14 dims)
-----------------------------
 0-2   drone position (x, y, z)
 3     battery level [0, 1]
 4     current task progress [0, 1]  (drone's active task, 0 if idle)
 5-7   vector to candidate task (dx, dy, dz)
 8     remaining_work() of candidate task [0, 1]
 9     n drones already assigned to candidate task (int, cast to float)
 10    step / max_steps  (urgency)
 11-13 task type one-hot  [water_plant, sweep_floor, toggle_light]
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

BID_OBS_DIM = 14
_TASK_TYPES = ["water_plant", "sweep_floor", "toggle_light"]
_TYPE_INDEX = {t: i for i, t in enumerate(_TASK_TYPES)}


def build_bid_obs(
    drone_pos: np.ndarray,
    battery: float,
    current_progress: float,
    candidate_task,           # BaseTask instance
    step: int,
    max_steps: int,
) -> np.ndarray:
    """
    Construct the 14-dim observation vector for a (drone, task) pair.
    Called once per (drone, candidate_task) combination in each auction round.
    """
    vec_to_task = candidate_task.spec.target_position - drone_pos
    remaining = float(candidate_task.remaining_work())
    n_assigned = float(len(candidate_task.assigned_drone_ids))
    urgency = step / max(max_steps, 1)

    one_hot = np.zeros(3, dtype=np.float32)
    t_idx = _TYPE_INDEX.get(candidate_task.spec.task_type, 0)
    one_hot[t_idx] = 1.0

    obs = np.concatenate([
        drone_pos.astype(np.float32),           # 0-2
        np.array([battery,                      # 3
                  current_progress,             # 4
                  vec_to_task[0],               # 5
                  vec_to_task[1],               # 6
                  vec_to_task[2],               # 7
                  remaining,                    # 8
                  n_assigned,                   # 9
                  urgency], dtype=np.float32),  # 10
        one_hot,                                # 11-13
    ])
    return obs  # shape (14,)


class BidPolicy(nn.Module):
    """
    Maps a (drone, task) observation to:
      - a primary bid scalar ∈ (0, 1)         — used for primary task assignment
      - a marginal bid scalar ∈ (0, 1)        — used for co-assignment decisions;
        represents the learned extra value of adding this drone to a task that
        already has at least one assignee.

    forward(obs) → (primary_logit, marginal_logit)  (used during training)
    bid(obs)     → sigmoid(primary_logit)            (used during primary assignment)
    marginal_bid(obs) → sigmoid(marginal_logit)      (used during co-assignment)

    Both heads share the same hidden-layer trunk (same weights), so the marginal
    head learns from the same feature representation as the primary head with no
    extra parameters in the trunk.
    """

    def __init__(self, obs_dim: int = BID_OBS_DIM, hidden: list[int] | None = None):
        super().__init__()
        hidden = hidden or [64, 64]
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        self.trunk = nn.Sequential(*layers)
        # Two independent output heads — each maps trunk output → scalar logit
        self.primary_head   = nn.Linear(in_dim, 1)
        self.marginal_head  = nn.Linear(in_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Small output init keeps bids near 0.5 at start
        nn.init.orthogonal_(self.primary_head.weight,  gain=0.01)
        nn.init.orthogonal_(self.marginal_head.weight, gain=0.01)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (primary_logit, marginal_logit), each shape (...,)."""
        features = self.trunk(obs)
        return (
            self.primary_head(features).squeeze(-1),
            self.marginal_head(features).squeeze(-1),
        )

    def bid(self, obs: torch.Tensor) -> torch.Tensor:
        """Primary bid ∈ (0, 1), shape (...,)."""
        primary_logit, _ = self.forward(obs)
        return torch.sigmoid(primary_logit)

    def marginal_bid(self, obs: torch.Tensor) -> torch.Tensor:
        """Marginal co-assignment bid ∈ (0, 1), shape (...,)."""
        _, marginal_logit = self.forward(obs)
        return torch.sigmoid(marginal_logit)

    def bid_numpy(self, obs: np.ndarray) -> float:
        """Primary bid — accepts numpy array, returns Python float."""
        with torch.no_grad():
            t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            return float(self.bid(t).item())

    def marginal_bid_numpy(self, obs: np.ndarray) -> float:
        """Marginal bid — accepts numpy array, returns Python float."""
        with torch.no_grad():
            t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            return float(self.marginal_bid(t).item())
