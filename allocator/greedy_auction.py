"""
greedy_auction.py
Greedy distance-based auction allocator.

Algorithm
---------
Each drone bids on every active task.  Bid value = battery_weight / distance,
so closer drones with more battery bid higher.  For each task the highest
bidder wins.  Drones that already hold a still-valid assignment keep it
(no unnecessary churn) unless forced to re-bid (e.g. after a disruption).

For shareable tasks the allocator also computes a *marginal* bid — the extra
value of adding a second drone given the task's remaining_work().  If the
marginal exceeds SHARE_THRESHOLD and the task is marked is_shareable, the
idle / lowest-priority drone is co-assigned.

This is the Phase 2a baseline that the learned bidder (Phase 3) must beat.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from allocator.base_allocator import (
    BaseAllocator,
    WorldSnapshot,
    AllocationResult,
    Bid,
)
from envs.tasks.base_task import TaskStatus

# Minimum remaining_work fraction to bother co-assigning a second drone.
# Below this threshold the solo drone will finish before help arrives.
SHARE_THRESHOLD = 0.4

# Weight given to battery level when computing bid value.
# bid = (BATTERY_WEIGHT * battery + (1 - BATTERY_WEIGHT)) / distance
BATTERY_WEIGHT = 0.3

# Small epsilon to avoid division by zero when a drone is exactly on target.
_EPS = 1e-6

# Active task statuses eligible for bidding.
_ACTIVE = {TaskStatus.PENDING, TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS}


class GreedyAuction(BaseAllocator):
    """
    Greedy single-round sealed-bid auction.

    Round structure (called once per allocate() invocation):
      1. Every drone produces one bid per active task.
      2. For each task, the highest-bidding *unassigned* drone wins.
      3. Co-assignment pass: for shareable tasks with remaining_work > threshold,
         assign the best-bidding idle drone as a second worker.
      4. Any remaining idle drones stay idle (no task left to claim).

    Complexity: O(D × T) per round — fast enough to run every step if needed.
    """

    # ------------------------------------------------------------------
    # BaseAllocator interface
    # ------------------------------------------------------------------

    def allocate(self, snapshot: WorldSnapshot) -> AllocationResult:
        active_tasks = [
            (i, t) for i, t in enumerate(snapshot.tasks)
            if t.status in _ACTIVE
        ]

        if not active_tasks:
            return AllocationResult(
                assignments={d: None for d in snapshot.drone_positions},
                bids=[],
            )

        # Step 1 — compute all bids
        all_bids: list[Bid] = []
        for drone_id, pos in snapshot.drone_positions.items():
            batt = snapshot.drone_batteries.get(drone_id, 1.0)
            for task_idx, task in active_tasks:
                dist = float(
                    sum((pos[k] - task.spec.target_position[k]) ** 2 for k in range(3)) ** 0.5
                )
                dist = max(dist, _EPS)
                value = (BATTERY_WEIGHT * batt + (1.0 - BATTERY_WEIGHT)) / dist
                # Marginal: scaled by remaining work and inversely by n already assigned
                n_assigned = len(task.assigned_drone_ids)
                marginal = (
                    value * task.remaining_work() / (n_assigned + 1)
                    if task.spec.is_shareable
                    else 0.0
                )
                all_bids.append(Bid(
                    drone_id=drone_id,
                    task_idx=task_idx,
                    bid_value=value,
                    marginal=marginal,
                ))

        # Step 2 — primary assignment: best bidder per task, each drone wins at most once
        assignments: dict[str, int | None] = {d: None for d in snapshot.drone_positions}
        assigned_drones: set[str] = set()

        # Sort tasks so the one with the highest top-bid is resolved first
        # (reduces conflicts when multiple tasks are highly desirable).
        task_best: dict[int, list[Bid]] = {}
        for b in all_bids:
            task_best.setdefault(b.task_idx, []).append(b)

        task_order = sorted(
            task_best.keys(),
            key=lambda idx: max(b.bid_value for b in task_best[idx]),
            reverse=True,
        )

        for task_idx in task_order:
            bids_for_task = sorted(
                task_best[task_idx],
                key=lambda b: (b.bid_value, b.drone_id),
                reverse=True,
            )
            for b in bids_for_task:
                if b.drone_id not in assigned_drones:
                    assignments[b.drone_id] = task_idx
                    assigned_drones.add(b.drone_id)
                    break

        # Step 3 — co-assignment pass for shareable tasks
        idle_drones = [d for d in snapshot.drone_positions if d not in assigned_drones]
        if idle_drones:
            for task_idx, task in active_tasks:
                if not task.spec.is_shareable:
                    continue
                if task.remaining_work() < SHARE_THRESHOLD:
                    continue
                # Find the best marginal bid from an idle drone for this task
                candidates = sorted(
                    [b for b in all_bids
                     if b.task_idx == task_idx and b.drone_id in idle_drones],
                    key=lambda b: (b.marginal, b.drone_id),
                    reverse=True,
                )
                if candidates and candidates[0].marginal > 0:
                    best = candidates[0]
                    assignments[best.drone_id] = task_idx
                    idle_drones.remove(best.drone_id)
                    assigned_drones.add(best.drone_id)
                if not idle_drones:
                    break

        return AllocationResult(assignments=assignments, bids=all_bids)
