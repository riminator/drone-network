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
      2. Forward through BidPolicy → bid ∈ (0, 1).
      3. Resolve: highest-bidding unassigned drone wins each task
         (most-contested-first to handle ties, same as CBBA).
      4. Co-assignment pass for shareable tasks.
    """

    def __init__(self, policy: BidPolicy, device: str = "cpu"):
        self.policy = policy.to(device)
        self.policy.eval()
        self.device = device

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "cpu") -> "LearnedBidder":
        ckpt = torch.load(path, map_location=device)
        cfg = ckpt.get("bid_policy_config", {})
        policy = BidPolicy(
            obs_dim=cfg.get("obs_dim", 14),
            hidden=cfg.get("hidden", [64, 64]),
        )
        policy.load_state_dict(ckpt["bid_policy_state_dict"])
        return cls(policy, device=device)

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

        all_bids: list[Bid] = []

        for drone_id, pos in snapshot.drone_positions.items():
            batt = snapshot.drone_batteries.get(drone_id, 1.0)
            progress = snapshot.drone_task_progress.get(drone_id, 0.0)
            for task_idx, task in active_tasks:
                obs = build_bid_obs(
                    pos, batt, progress, task,
                    snapshot.step, snapshot.max_steps,
                )
                bid_val = self.policy.bid_numpy(obs)
                all_bids.append(Bid(
                    drone_id=drone_id,
                    task_idx=task_idx,
                    bid_value=bid_val,
                    marginal=bid_val * task.remaining_work() / (len(task.assigned_drone_ids) + 1)
                    if task.spec.is_shareable else 0.0,
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
