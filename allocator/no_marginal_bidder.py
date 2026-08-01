"""
no_marginal_bidder.py
Ablation variant of LearnedBidder that disables the marginal head entirely.

Used in the Phase 4 ablation study: run this alongside LearnedBidder on the
surge scenario to isolate the contribution of the dual-head co-assignment.

Differences from LearnedBidder:
  - marginal bid is hardwired to 0.0 for every task (co-assignment disabled)
  - primary assignment logic is identical to LearnedBidder
"""

from __future__ import annotations
from pathlib import Path

import torch

from allocator.base_allocator import BaseAllocator, WorldSnapshot, AllocationResult, Bid
from allocator.bid_policy import BidPolicy, build_bid_obs
from envs.tasks.base_task import TaskStatus

_ACTIVE = {TaskStatus.PENDING, TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS}


class NoMarginalBidder(BaseAllocator):
    """
    LearnedBidder with the marginal head ablated out.
    Primary assignment uses the trained BidPolicy primary head.
    Co-assignment pass is fully disabled (marginal always 0).
    """

    def __init__(self, policy: BidPolicy, device: str = "cpu"):
        self.policy = policy.to(device)
        self.policy.eval()
        self.device = device

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "cpu") -> "NoMarginalBidder":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt.get("bid_policy_config", {})
        policy = BidPolicy(
            obs_dim=cfg.get("obs_dim", 14),
            hidden=cfg.get("hidden", [64, 64]),
        )
        policy.load_state_dict(ckpt["bid_policy_state_dict"], strict=False)
        return cls(policy, device=device)

    def allocate(self, snapshot: WorldSnapshot) -> AllocationResult:
        active_tasks = [(i, t) for i, t in enumerate(snapshot.tasks) if t.status in _ACTIVE]
        if not active_tasks:
            return AllocationResult(assignments={d: None for d in snapshot.drone_positions})

        all_bids: list[Bid] = []
        for drone_id, pos in snapshot.drone_positions.items():
            batt = snapshot.drone_batteries.get(drone_id, 1.0)
            progress = snapshot.drone_task_progress.get(drone_id, 0.0)
            for task_idx, task in active_tasks:
                obs = build_bid_obs(pos, batt, progress, task, snapshot.step, snapshot.max_steps)
                bid_val = self.policy.bid_numpy(obs)
                # marginal hardwired to 0.0 — co-assignment fully disabled
                all_bids.append(Bid(drone_id=drone_id, task_idx=task_idx, bid_value=bid_val, marginal=0.0))

        assignments: dict[str, int | None] = {d: None for d in snapshot.drone_positions}
        assigned_drones: set[str] = set()
        task_bids: dict[int, list[Bid]] = {}
        for b in all_bids:
            task_bids.setdefault(b.task_idx, []).append(b)
        task_order = sorted(task_bids.keys(), key=lambda i: max(b.bid_value for b in task_bids[i]), reverse=True)
        for tidx in task_order:
            for b in sorted(task_bids[tidx], key=lambda b: (b.bid_value, b.drone_id), reverse=True):
                if b.drone_id not in assigned_drones:
                    assignments[b.drone_id] = tidx
                    assigned_drones.add(b.drone_id)
                    break

        # No co-assignment pass — this is the ablation point
        return AllocationResult(assignments=assignments, bids=all_bids)
