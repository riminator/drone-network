"""
learned_bidder.py
LearnedBidder — the Phase 3 allocator that uses a trained BidPolicy.

Wraps a loaded BidPolicy checkpoint and exposes the standard BaseAllocator
interface.  At inference time there is no gradient computation and no buffer
writes — it is a pure forward pass.

Usage
-----
    from allocator.learned_bidder import LearnedBidder
    allocator = LearnedBidder.from_checkpoint("checkpoints/bid_policy_final.pt")
    env = HomeEnv({"n_drones": 6, "allocator": allocator})
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from allocator.base_allocator import BaseAllocator, WorldSnapshot, AllocationResult, Bid
from allocator.bid_policy import BidPolicy, build_bid_obs
from envs.tasks.base_task import TaskStatus

_ACTIVE = {TaskStatus.PENDING, TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS}


class LearnedBidder(BaseAllocator):
    """
    Allocator backed by a trained BidPolicy network.

    Auction round (identical structure to GreedyAuction / CBBA):
      1. For every (drone, task) pair, compute the 14-dim bid observation.
      2. Forward through BidPolicy → primary bid ∈ (0, 1).
      3. Resolve: highest-bidding unassigned drone wins each task
         (most-contested-first to handle ties, same as CBBA).
      4. Co-assignment pass for shareable tasks using the learned marginal
         head — BidPolicy.marginal_bid() — rather than a hand-crafted formula.
    """

    def __init__(self, policy: BidPolicy, device: str = "cpu", obs_delay: int = 0):
        self.policy = policy.to(device)
        self.policy.eval()
        self.device = device
        self.obs_delay = obs_delay
        # History buffer: list of (positions_dict, batteries_dict), oldest first
        self._obs_history: list[tuple[dict, dict]] = []

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: str = "cpu",
        obs_delay: int = 0,
    ) -> "LearnedBidder":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt.get("bid_policy_config", {})
        policy = BidPolicy(
            obs_dim=cfg.get("obs_dim", 14),
            hidden=cfg.get("hidden", [64, 64]),
        )

        sd = ckpt["bid_policy_state_dict"]

        # Migration: checkpoints saved before the dual-head refactor used a
        # single nn.Sequential called "net".  Map those keys to the new
        # "trunk" + "primary_head" layout.  The marginal head will use its
        # freshly-initialised weights (equivalent to a zero-initialised marginal
        # prior — it will be trained/fine-tuned on next bid-policy training run).
        if any(k.startswith("net.") for k in sd):
            hidden = cfg.get("hidden", [64, 64])
            n_layers = len(hidden)
            new_sd = {}
            for k, v in sd.items():
                if k.startswith("net."):
                    parts = k.split(".", 2)   # ["net", "idx", "weight"/"bias"]
                    layer_idx = int(parts[1])
                    suffix    = parts[2]
                    # Each hidden layer = 2 sequential modules (Linear + Tanh)
                    # so trunk indices 0..2*(n_layers-1) are the hidden layers.
                    # The last index in "net" is the single output Linear.
                    last_linear_idx = n_layers * 2
                    if layer_idx < last_linear_idx:
                        new_sd[f"trunk.{layer_idx}.{suffix}"] = v
                    else:
                        # Old single output → primary head; marginal head keeps init
                        new_sd[f"primary_head.{suffix}"] = v
                else:
                    new_sd[k] = v
            sd = new_sd

        policy.load_state_dict(sd, strict=False)
        return cls(policy, device=device, obs_delay=obs_delay)

    # ------------------------------------------------------------------
    # BaseAllocator interface
    # ------------------------------------------------------------------

    def allocate(self, snapshot: WorldSnapshot) -> AllocationResult:
        active_tasks = [
            (i, t) for i, t in enumerate(snapshot.tasks)
            if t.status in _ACTIVE
        ]

        if not active_tasks:
            return AllocationResult(assignments={d: None for d in snapshot.drone_positions})

        # Maintain obs history for delay simulation
        self._obs_history.append(
            (dict(snapshot.drone_positions), dict(snapshot.drone_batteries))
        )
        if len(self._obs_history) > max(self.obs_delay, 1):
            self._obs_history.pop(0)

        if self.obs_delay > 0 and len(self._obs_history) >= self.obs_delay:
            delayed_positions, delayed_batteries = self._obs_history[0]
        else:
            delayed_positions  = snapshot.drone_positions
            delayed_batteries  = snapshot.drone_batteries

        all_bids: list[Bid] = []

        for drone_id, pos in snapshot.drone_positions.items():
            delayed_pos = delayed_positions.get(drone_id, pos)
            batt = delayed_batteries.get(drone_id, 1.0)
            progress = snapshot.drone_task_progress.get(drone_id, 0.0)
            for task_idx, task in active_tasks:
                obs = build_bid_obs(
                    delayed_pos, batt, progress, task,
                    snapshot.step, snapshot.max_steps,
                )
                bid_val     = self.policy.bid_numpy(obs)
                # Use the dedicated marginal head for co-assignment scoring
                marginal_val = (
                    self.policy.marginal_bid_numpy(obs)
                    if task.spec.is_shareable else 0.0
                )
                all_bids.append(Bid(
                    drone_id=drone_id,
                    task_idx=task_idx,
                    bid_value=bid_val,
                    marginal=marginal_val,
                ))

        # Resolve most-contested tasks first (same tie-breaking as CBBA)
        assignments: dict[str, int | None] = {d: None for d in snapshot.drone_positions}
        assigned_drones: set[str] = set()

        task_bids: dict[int, list[Bid]] = {}
        for b in all_bids:
            task_bids.setdefault(b.task_idx, []).append(b)

        task_order = sorted(
            task_bids.keys(),
            key=lambda i: max(b.bid_value for b in task_bids[i]),
            reverse=True,
        )
        for tidx in task_order:
            for b in sorted(task_bids[tidx], key=lambda b: (b.bid_value, b.drone_id), reverse=True):
                if b.drone_id not in assigned_drones:
                    assignments[b.drone_id] = tidx
                    assigned_drones.add(b.drone_id)
                    break

        # Co-assignment pass for shareable tasks
        idle_drones = [d for d in snapshot.drone_positions if d not in assigned_drones]
        for tidx, task in active_tasks:
            if not task.spec.is_shareable or task.remaining_work() < 0.4 or not idle_drones:
                continue
            candidates = sorted(
                [b for b in all_bids if b.task_idx == tidx and b.drone_id in idle_drones],
                key=lambda b: b.marginal, reverse=True,
            )
            if candidates and candidates[0].marginal > 0:
                best = candidates[0]
                assignments[best.drone_id] = tidx
                idle_drones.remove(best.drone_id)
                assigned_drones.add(best.drone_id)
            if not idle_drones:
                break

        return AllocationResult(assignments=assignments, bids=all_bids)
