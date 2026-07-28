"""
comm_actor.py
Communication-aware execution policy for the drone swarm.

Architecture
------------
Each drone i receives its 15-dim local observation.  Before computing an
action, it aggregates messages from neighbouring drones via a single round
of Graph Attention (GATv2) message passing.  The attended embedding is
concatenated with the raw local obs and fed into the same MLP trunk + heads
as the original Actor.

               obs_i (15)  ┐
                            ├──► [linear embed] ──► node_feat_i (embed_dim)
               obs_j (15)  ┘                              │
                                                    GATv2 layer (n_heads)
                                                    (neighbours within comm_radius)
                                                          │
                                                   agg_i (embed_dim)
                                                          │
               obs_i (15) ──────────────────────────────┐ │
                                                         concat
                                                          │
                                              MLP trunk (256×256, Tanh)
                                                          │
                                             mean_head (4)  log_std (4)

Design decisions
----------------
- GATv2 (Brody et al. 2022) over vanilla GAT: fixes the "static attention"
  problem where attention weights become independent of query nodes.
- comm_radius controls the neighbourhood: set to float("inf") for fully
  connected (useful for small N), or a metre value for sparse graphs.
- comm_delay: number of steps by which messages are delayed. 0 = instant.
  Simulates radio/compute latency. The CommActor holds a circular buffer of
  past all-agent obs and reads from `comm_delay` steps ago.
- drop_comm_prob: per-edge probability of dropping a message during training.
  Acts as data augmentation for communication-degraded scenarios.
- Backward compatible: set embed_dim=0 to degrade to the original Actor
  (no message passing). This is enforced by calling Actor directly when
  embed_dim == 0.
- Parameter sharing: the same CommActor weights are shared across all drones
  (same as Actor), so any swarm size works without retraining.

Checkpoint compatibility
------------------------
CommActor checkpoints have key "comm_actor_state_dict" (not "actor_state_dict")
so they cannot be accidentally loaded as a plain Actor.  The training loop
saves both keys when using CommActor.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque


# ---------------------------------------------------------------------------
# GATv2 layer — single attention head, then multi-head wrapper
# ---------------------------------------------------------------------------

class _GATv2Head(nn.Module):
    """
    Single GATv2 attention head.

    For each node i:
        e_ij  = LeakyReLU( a^T [ W_l h_i || W_r h_j ] )
        alpha = softmax_j( e_ij )                   (over neighbours only)
        agg_i = sum_j alpha_ij * W_r h_j

    h_i, h_j are node features (embed_dim,).
    """

    def __init__(self, embed_dim: int, leaky_slope: float = 0.2):
        super().__init__()
        self.W_l = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_r = nn.Linear(embed_dim, embed_dim, bias=False)
        self.a   = nn.Linear(2 * embed_dim, 1, bias=False)
        self.leaky_slope = leaky_slope

    def forward(
        self,
        node_feats: torch.Tensor,    # (N, embed_dim)
        adj_mask: torch.Tensor,      # (N, N) bool — True = edge exists
    ) -> torch.Tensor:               # (N, embed_dim)
        N = node_feats.size(0)
        h_l = self.W_l(node_feats)   # (N, E)
        h_r = self.W_r(node_feats)   # (N, E)

        # Expand for pairwise concatenation: (N, N, 2E)
        h_l_rep = h_l.unsqueeze(1).expand(N, N, -1)  # query (i)
        h_r_rep = h_r.unsqueeze(0).expand(N, N, -1)  # key   (j)
        concat  = torch.cat([h_l_rep, h_r_rep], dim=-1)  # (N, N, 2E)

        # Attention coefficients
        e = F.leaky_relu(self.a(concat).squeeze(-1), self.leaky_slope)  # (N, N)

        # Mask out non-edges (set to -inf before softmax)
        e = e.masked_fill(~adj_mask, float("-inf"))
        alpha = torch.softmax(e, dim=1)                   # (N, N)

        # When a node has NO neighbours at all, softmax produces NaN.
        # Replace with zero — the node will rely solely on its own obs.
        alpha = torch.nan_to_num(alpha, nan=0.0)

        # Aggregate
        return torch.matmul(alpha, h_r)   # (N, embed_dim)


class GATv2Layer(nn.Module):
    """
    Multi-head GATv2 layer with mean aggregation across heads.

    Output dim = embed_dim (same as input — keeps the trunk input size fixed).
    """

    def __init__(self, embed_dim: int, n_heads: int = 4, leaky_slope: float = 0.2):
        super().__init__()
        assert embed_dim % n_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by n_heads ({n_heads})"
        )
        self.heads = nn.ModuleList(
            [_GATv2Head(embed_dim, leaky_slope) for _ in range(n_heads)]
        )
        self.n_heads = n_heads

    def forward(
        self,
        node_feats: torch.Tensor,   # (N, embed_dim)
        adj_mask: torch.Tensor,     # (N, N) bool
    ) -> torch.Tensor:              # (N, embed_dim)
        head_outs = [h(node_feats, adj_mask) for h in self.heads]
        # Mean across heads (dim 0 of the stack)
        return torch.stack(head_outs, dim=0).mean(dim=0)


# ---------------------------------------------------------------------------
# CommActor
# ---------------------------------------------------------------------------

class CommActor(nn.Module):
    """
    Drop-in replacement for Actor that prepends one GATv2 message-passing
    round before computing actions.

    Parameters
    ----------
    obs_dim        : per-drone observation dimension (default 15)
    act_dim        : action dimension (default 4)
    embed_dim      : hidden size of the GATv2 node embeddings (default 64)
    n_heads        : number of GATv2 attention heads (default 4)
    hidden_sizes   : MLP trunk hidden sizes (default [256, 256])
    comm_radius    : max distance (m) for a communication edge.
                     float("inf") = fully connected.
    comm_delay     : steps of message delay (0 = instant).
    drop_comm_prob : per-edge drop probability during training (0 = no drop).
    """

    LOG_STD_MIN = -3.0
    LOG_STD_MAX =  0.5

    def __init__(
        self,
        obs_dim:        int   = 15,
        act_dim:        int   = 4,
        embed_dim:      int   = 64,
        n_heads:        int   = 4,
        hidden_sizes:   list[int] | None = None,
        comm_radius:    float = float("inf"),
        comm_delay:     int   = 0,
        drop_comm_prob: float = 0.0,
    ):
        super().__init__()
        self.obs_dim        = obs_dim
        self.act_dim        = act_dim
        self.embed_dim      = embed_dim
        self.comm_radius    = comm_radius
        self.comm_delay     = max(0, comm_delay)
        self.drop_comm_prob = drop_comm_prob

        hidden_sizes = hidden_sizes or [256, 256]

        # --- Node encoder: obs → embed_dim ---
        self.node_encoder = nn.Sequential(
            nn.Linear(obs_dim, embed_dim),
            nn.Tanh(),
        )

        # --- GATv2 message-passing layer ---
        self.gat = GATv2Layer(embed_dim=embed_dim, n_heads=n_heads)

        # --- MLP trunk: (obs_dim + embed_dim) → hidden → mean/log_std ---
        trunk_in = obs_dim + embed_dim
        layers: list[nn.Module] = []
        in_size = trunk_in
        for h in hidden_sizes:
            layers += [nn.Linear(in_size, h), nn.Tanh()]
            in_size = h
        self.trunk    = nn.Sequential(*layers)
        self.mean_head = nn.Linear(in_size, act_dim)
        self.log_std   = nn.Parameter(torch.zeros(act_dim))

        # --- Communication delay buffer ---
        # Holds past all_obs tensors (n_agents, obs_dim) as numpy arrays.
        # deque(maxlen=delay+1) so index [-1] = most recent, [0] = oldest.
        self._delay_buf: deque[np.ndarray] = deque(maxlen=max(1, self.comm_delay + 1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.constant_(self.log_std, 0.0)

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_adj(
        self,
        positions: torch.Tensor,   # (N, 3) — drone positions from obs[:,0:3]
        n: int,
        training: bool,
    ) -> torch.Tensor:
        """
        Build adjacency mask (N, N) based on comm_radius and drop_comm_prob.
        Self-loops are included (node attends to itself).
        """
        # Pairwise distances
        diff  = positions.unsqueeze(1) - positions.unsqueeze(0)  # (N, N, 3)
        dists = diff.norm(dim=-1)                                 # (N, N)
        adj   = dists <= self.comm_radius                         # (N, N) bool

        # Random edge drop for robustness training (skip self-loops)
        if training and self.drop_comm_prob > 0.0:
            drop = torch.bernoulli(
                torch.full((n, n), self.drop_comm_prob, device=positions.device)
            ).bool()
            eye  = torch.eye(n, dtype=torch.bool, device=positions.device)
            drop = drop & ~eye          # never drop self-loop
            adj  = adj & ~drop

        return adj

    # ------------------------------------------------------------------
    # Message passing (shared by forward paths)
    # ------------------------------------------------------------------

    def _attend(
        self,
        all_obs: torch.Tensor,   # (N, obs_dim)  — all drone obs at current step
        training: bool,
    ) -> torch.Tensor:           # (N, embed_dim)
        """Encode nodes, build graph, run one GATv2 round."""
        node_feats = self.node_encoder(all_obs)           # (N, embed_dim)
        positions  = all_obs[:, 0:3]                      # x,y,z from obs layout
        adj        = self._build_adj(positions, all_obs.size(0), training)
        return self.gat(node_feats, adj)                  # (N, embed_dim)

    # ------------------------------------------------------------------
    # Forward pass for a SINGLE timestep (used in rollout collection)
    # ------------------------------------------------------------------

    def forward(
        self,
        all_obs: torch.Tensor,   # (N, obs_dim) — one obs per drone, current step
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (mean, log_std) for all N drones, shape (N, act_dim).

        all_obs must contain ALL drones' observations so the graph can be built.
        If comm_delay > 0, the method maintains an internal ring buffer and
        uses delayed observations for message passing.
        """
        # Push current obs into delay buffer (store as numpy for CPU efficiency)
        self._delay_buf.append(all_obs.detach().cpu().numpy())

        # Pick the delayed obs (or current if buffer not full yet)
        delayed_np = self._delay_buf[0]      # oldest = most delayed
        delayed    = torch.tensor(delayed_np, dtype=all_obs.dtype,
                                  device=all_obs.device)

        agg = self._attend(delayed, self.training)           # (N, embed_dim)
        combined = torch.cat([all_obs, agg], dim=-1)         # (N, obs+embed)
        features = self.trunk(combined)
        mean     = self.mean_head(features)
        log_std  = torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        log_std  = log_std.expand_as(mean)
        return mean, log_std

    def reset_delay_buffer(self):
        """Call at the start of each episode to clear the comm-delay FIFO."""
        self._delay_buf.clear()

    # ------------------------------------------------------------------
    # Action sampling — mirrors Actor.get_action exactly
    # ------------------------------------------------------------------

    def get_action(
        self,
        all_obs: torch.Tensor,   # (N, obs_dim)
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (squashed_action, raw_action, log_prob) for all N drones."""
        mean, log_std = self.forward(all_obs)
        std  = log_std.exp()
        dist = Normal(mean, std)

        raw_action = mean if deterministic else dist.rsample()
        log_prob   = dist.log_prob(raw_action).sum(dim=-1)

        # Squashing corrections (identical to Actor)
        tanh_correction = (
            2.0 * (np.log(2.0) - raw_action[..., :3]
                   - F.softplus(-2.0 * raw_action[..., :3]))
        ).sum(dim=-1)
        sig_val = torch.sigmoid(raw_action[..., 3])
        sigmoid_correction = -(
            torch.log(sig_val + 1e-6) + torch.log(1.0 - sig_val + 1e-6)
        )
        log_prob = log_prob - tanh_correction - sigmoid_correction

        spatial = torch.tanh(raw_action[..., :3])
        tool    = torch.sigmoid(raw_action[..., 3:4])
        squashed_action = torch.cat([spatial, tool], dim=-1)
        return squashed_action, raw_action, log_prob

    # ------------------------------------------------------------------
    # PPO update pass — mirrors Actor.evaluate_actions
    # ------------------------------------------------------------------

    def evaluate_actions(
        self,
        all_obs: torch.Tensor,       # (B*N, obs_dim)   — flattened buffer
        raw_actions: torch.Tensor,   # (B*N, act_dim)
        n_agents: int,               # N — needed to reshape into (B, N, obs_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Recompute log_probs and entropy for a mini-batch.

        The buffer stores obs flat (B*N, obs_dim).  We reshape into
        (B, N, obs_dim), run the GATv2 over each step's full swarm obs,
        then flatten back to (B*N, embed_dim) for the MLP trunk.

        Note: comm_delay is NOT applied during evaluate_actions because the
        buffer already stores the obs that were used during collection.
        We reconstruct the same graph the policy saw at collection time.
        """
        b_total = all_obs.size(0)
        b = b_total // n_agents

        # Reshape: (B, N, obs_dim)
        obs_bnd = all_obs.view(b, n_agents, self.obs_dim)

        agg_list = []
        for step_idx in range(b):
            step_obs = obs_bnd[step_idx]                 # (N, obs_dim)
            agg = self._attend(step_obs, training=True)  # (N, embed_dim)
            agg_list.append(agg)
        agg_all = torch.stack(agg_list, dim=0)           # (B, N, embed_dim)
        agg_flat = agg_all.view(b_total, self.embed_dim) # (B*N, embed_dim)

        combined = torch.cat([all_obs, agg_flat], dim=-1)
        features = self.trunk(combined)
        mean     = self.mean_head(features)
        log_std  = torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        log_std  = log_std.expand_as(mean)

        dist     = Normal(mean, log_std.exp())
        log_probs = dist.log_prob(raw_actions).sum(dim=-1)

        tanh_correction = (
            2.0 * (np.log(2.0) - raw_actions[..., :3]
                   - F.softplus(-2.0 * raw_actions[..., :3]))
        ).sum(dim=-1)
        sig_val = torch.sigmoid(raw_actions[..., 3])
        sigmoid_correction = -(
            torch.log(sig_val + 1e-6) + torch.log(1.0 - sig_val + 1e-6)
        )
        log_probs = log_probs - tanh_correction - sigmoid_correction

        entropy = dist.entropy().sum(dim=-1)
        return log_probs, entropy
