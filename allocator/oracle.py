"""
oracle.py
Oracle allocator — Hungarian-algorithm optimal assignment.

Used as:
  1. A reward baseline during BidPolicy training (solution_quality = makespan / oracle_makespan)
  2. An upper-bound reference in evaluation tables

The oracle minimises total travel distance across all (drone, task) pairs using
scipy's linear_sum_assignment (the Hungarian algorithm), which is O(n³).

It has access to full world state (same WorldSnapshot as all other allocators),
so it is not a fair policy comparison — it is a theoretical optimum for single-
round assignment with distance as the objective.  It does NOT handle co-assignment
or task bundling; it assigns each drone to at most one task.
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

from allocator.base_allocator import BaseAllocator, WorldSnapshot, AllocationResult, Bid
from envs.tasks.base_task import TaskStatus

_ACTIVE = {TaskStatus.PENDING, TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS}
_BIG = 1e9   # cost for impossible assignments


class OracleAllocator(BaseAllocator):
    """
    Optimal single-round assignment using the Hungarian algorithm.

    If scipy is not installed, falls back to greedy distance matching.
    """

    def allocate(self, snapshot: WorldSnapshot) -> AllocationResult:
        drone_ids = sorted(snapshot.drone_positions.keys())
        active = [(i, t) for i, t in enumerate(snapshot.tasks) if t.status in _ACTIVE]

        if not active:
            return AllocationResult(assignments={d: None for d in drone_ids})

        n_d = len(drone_ids)
        n_t = len(active)
        size = max(n_d, n_t)

        # Cost matrix: rows = drones, cols = tasks (padded to square)
        cost = np.full((size, size), _BIG, dtype=np.float64)
        for di, drone_id in enumerate(drone_ids):
            pos = snapshot.drone_positions[drone_id]
            for tj, (task_idx, task) in enumerate(active):
                dist = float(np.linalg.norm(pos - task.spec.target_position))
                cost[di, tj] = dist

        assignments: dict[str, int | None] = {d: None for d in drone_ids}
        all_bids: list[Bid] = []

        if _SCIPY_OK:
            row_ind, col_ind = linear_sum_assignment(cost)
            for di, tj in zip(row_ind, col_ind):
                if di < n_d and tj < n_t and cost[di, tj] < _BIG:
                    drone_id = drone_ids[di]
                    task_idx = active[tj][0]
                    assignments[drone_id] = task_idx
                    all_bids.append(Bid(
                        drone_id=drone_id,
                        task_idx=task_idx,
                        bid_value=1.0 / max(cost[di, tj], 1e-6),
                    ))
        else:
            # Greedy fallback: assign each drone to its nearest unclaimed task
            claimed: set[int] = set()
            for di, drone_id in enumerate(drone_ids):
                best_j, best_cost = None, _BIG
                for tj in range(n_t):
                    if tj not in claimed and cost[di, tj] < best_cost:
                        best_cost = cost[di, tj]
                        best_j = tj
                if best_j is not None:
                    task_idx = active[best_j][0]
                    assignments[drone_id] = task_idx
                    claimed.add(best_j)
                    all_bids.append(Bid(
                        drone_id=drone_id,
                        task_idx=task_idx,
                        bid_value=1.0 / max(best_cost, 1e-6),
                    ))

        return AllocationResult(assignments=assignments, bids=all_bids)
