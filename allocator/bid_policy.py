"""
bid_policy.py
Neural network that produces a scalar bid value for a (drone, task) pair.

Architecture
------------
Input:  BID_OBS_DIM = 14 features describing one (drone, task) pairing
Output: single scalar logit, passed through sigmoid → bid ∈ (0, 1)

The network is intentionally small (two 64-unit hidden layers) so it trains
quickly on CPU.  The execution actors (Actor, 256×256) are frozen throughout.

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
    Maps a (drone, task) observation to a scalar bid in (0, 1).

    forward(obs) → raw logit  (used during training for log-prob computation)
    bid(obs)     → sigmoid(logit)  (used during allocation)
    """

    def __init__(self, obs_dim: int = BID_OBS_DIM, hidden: list[int] | None = None):
        super().__init__()
        hidden = hidden or [64, 64]
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Small output init keeps bids near 0.5 at start
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns raw logit, shape (...,)."""
        return self.net(obs).squeeze(-1)

    def bid(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns bid ∈ (0, 1), shape (...,)."""
        return torch.sigmoid(self.forward(obs))

    def bid_numpy(self, obs: np.ndarray) -> float:
        """Convenience: accepts numpy array, returns Python float."""
        with torch.no_grad():
            t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            return float(self.bid(t).item())
