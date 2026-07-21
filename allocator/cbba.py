"""
cbba.py
Consensus-Based Bundle Algorithm (CBBA) allocator.

Reference
---------
Choi, H.-L., Brunet, L., & How, J. P. (2009).
"Consensus-Based Decentralized Auctions for Robust Task Allocation."
IEEE Transactions on Robotics, 25(4), 912–926.

Adaptation notes
----------------
The original CBBA is a fully distributed protocol where each robot runs
its own instance and communicates over a network.  Here we run it in
centralised-simulation mode: a single coordinator object holds all robots'
bid tables, runs the bundle-building and consensus phases to convergence,
and returns the final assignment.  This is equivalent to the distributed
protocol with zero communication delay and is used as a *baseline* —
Phase 4 will add configurable comm_delay to measure robustness degradation.

Key concepts
~~~~~~~~~~~~
* Bundle  — ordered list of tasks a drone has decided to attempt.
* Path    — the same list in the order the drone will execute them
            (for point tasks this equals the bundle; for sweep it matters).
* Winning bid y[drone][task] — the highest bid value this drone has seen
  for that task from any drone (including itself).
* Winning drone z[drone][task] — which drone currently holds that winning bid
  from this drone's perspective.

Convergence: the bundle-build / consensus loop runs until no drone changes
its bundle in a full pass (guaranteed to converge in at most D×T rounds).

Shareable tasks
~~~~~~~~~~~~~~~
Standard CBBA assigns each task to exactly one drone.  We extend it:
after the main convergence loop, a second co-assignment pass mirrors the
GreedyAuction approach — idle drones bid on shareable tasks whose
remaining_work > SHARE_THRESHOLD.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np

from allocator.base_allocator import (
    BaseAllocator,
    WorldSnapshot,
    AllocationResult,
    Bid,
)
from envs.tasks.base_task import TaskStatus

_ACTIVE = {TaskStatus.PENDING, TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS}
SHARE_THRESHOLD = 0.4
_EPS = 1e-6
MAX_BUNDLE_SIZE = 3      # max tasks per drone bundle (keeps complexity bounded)
MAX_ITER = 200           # convergence guard


# ---------------------------------------------------------------------------
# Path-cost utility function
# ---------------------------------------------------------------------------

def _path_cost_utility(
    drone_pos: np.ndarray,
    bundle: list[int],
    candidate_idx: int,
    tasks: list,
    battery: float,
    gamma: float = 0.95,
) -> float:
    """
    Marginal utility of appending candidate_idx to the drone's current bundle.

    c(p) = battery_factor * gamma^|bundle| / travel_distance_to_candidate

    Using gamma^|bundle| discounts later tasks — CBBA's standard approach to
    prefer short bundles that can actually be completed this episode.
    """
    candidate = tasks[candidate_idx]
    # Approximate travel: from current position through all bundle tasks to candidate
    pos = drone_pos.copy()
    for tidx in bundle:
        pos = tasks[tidx].spec.target_position
    dist = float(np.linalg.norm(pos - candidate.spec.target_position))
    dist = max(dist, _EPS)
    battery_factor = 0.7 + 0.3 * battery  # [0.7, 1.0]
    return battery_factor * (gamma ** len(bundle)) / dist


# ---------------------------------------------------------------------------
# Per-drone CBBA state
# ---------------------------------------------------------------------------

@dataclass
class _DroneState:
    drone_id: str
    bundle: list[int] = field(default_factory=list)   # ordered task indices
    path: list[int] = field(default_factory=list)     # execution order
    # y[task_idx] = highest bid value this drone knows about for that task
    y: dict[int, float] = field(default_factory=dict)
    # z[task_idx] = drone_id this drone believes is the winner for that task
    z: dict[int, str | None] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# CBBA allocator
# ---------------------------------------------------------------------------

class CBBA(BaseAllocator):
    """
    Centralised-simulation CBBA.

    Phase A (bundle build): each drone greedily adds tasks to its bundle
    by selecting the task with the highest *marginal path-cost utility*
    until MAX_BUNDLE_SIZE is reached or no profitable task remains.

    Phase B (consensus): drones share bid tables; any drone whose winning
    bid has been outbid releases that task from its bundle.  Repeat until
    no bundle changes.

    After convergence: co-assignment pass for shareable tasks (same logic
    as GreedyAuction).
    """

    def __init__(self, comm_delay: int = 0):
        """
        comm_delay — number of steps a broadcast is delayed before it is
        received by other drones.  0 = instantaneous (default baseline).
        Non-zero delays are used in Phase 4 robustness experiments.
        """
        self.comm_delay = comm_delay
        # Pending outgoing messages: list of (deliver_at_step, sender_id, y, z)
        self._pending_msgs: list[tuple[int, str, dict, dict]] = []
        self._current_step: int = 0

    # ------------------------------------------------------------------
    # BaseAllocator interface
    # ------------------------------------------------------------------

    def allocate(self, snapshot: WorldSnapshot) -> AllocationResult:
        self._current_step = snapshot.step

        active_task_indices = [
            i for i, t in enumerate(snapshot.tasks)
            if t.status in _ACTIVE
        ]

        if not active_task_indices:
            return AllocationResult(
                assignments={d: None for d in snapshot.drone_positions},
                bids=[],
            )

        drone_ids = sorted(snapshot.drone_positions.keys())

        # Initialise per-drone state
        states: dict[str, _DroneState] = {
            d: _DroneState(
                drone_id=d,
                y={i: 0.0 for i in active_task_indices},
                z={i: None for i in active_task_indices},
            )
            for d in drone_ids
        }

        # Seed with any pending messages that have arrived by this step
        delivered = self._deliver_messages(snapshot.step, states, active_task_indices)

        # ---------------------------------------------------------------
        # Main CBBA loop: alternate Phase A and Phase B until convergence
        # ---------------------------------------------------------------
        for _iteration in range(MAX_ITER):
            changed = False

            # Phase A — bundle build
            for drone_id in drone_ids:
                s = states[drone_id]
                pos = snapshot.drone_positions[drone_id]
                batt = snapshot.drone_batteries.get(drone_id, 1.0)

                # Release tasks from bundle that have been outbid
                new_bundle = []
                for tidx in s.bundle:
                    if s.z[tidx] == drone_id:
                        new_bundle.append(tidx)
                    else:
                        # Outbid — reset this slot and stop (CBBA invariant)
                        break
                if new_bundle != s.bundle:
                    s.bundle = new_bundle
                    s.path = list(new_bundle)
                    changed = True

                # Greedily add tasks up to MAX_BUNDLE_SIZE
                while len(s.bundle) < MAX_BUNDLE_SIZE:
                    best_val = -math.inf
                    best_tidx = None
                    for tidx in active_task_indices:
                        if tidx in s.bundle:
                            continue
                        u = _path_cost_utility(
                            pos, s.bundle, tidx, snapshot.tasks, batt
                        )
                        # Only bid if we can outbid the current winner
                        if u > s.y[tidx] and u > best_val:
                            best_val = u
                            best_tidx = tidx
                    if best_tidx is None:
                        break
                    s.bundle.append(best_tidx)
                    s.path.append(best_tidx)
                    s.y[best_tidx] = best_val
                    s.z[best_tidx] = drone_id
                    changed = True

            # Phase B — consensus (centralised: share all tables simultaneously)
            for drone_id in drone_ids:
                s = states[drone_id]
                for other_id in drone_ids:
                    if other_id == drone_id:
                        continue
                    o = states[other_id]
                    for tidx in active_task_indices:
                        # If the other drone knows a higher bid, update
                        if o.y[tidx] > s.y[tidx]:
                            s.y[tidx] = o.y[tidx]
                            s.z[tidx] = o.z[tidx]
                            # If we thought we had this task but lost it:
                            if tidx in s.bundle and s.z[tidx] != drone_id:
                                idx = s.bundle.index(tidx)
                                s.bundle = s.bundle[:idx]
                                s.path = list(s.bundle)
                                changed = True

            if not changed:
                break

        # ---------------------------------------------------------------
        # Build assignments from converged bundles
        # ---------------------------------------------------------------
        assignments: dict[str, int | None] = {d: None for d in drone_ids}
        all_bids: list[Bid] = []

        # Collect all bids for logging
        for drone_id, s in states.items():
            for tidx in active_task_indices:
                if s.y[tidx] > 0:
                    all_bids.append(Bid(
                        drone_id=drone_id,
                        task_idx=tidx,
                        bid_value=s.y[tidx],
                        marginal=0.0,
                    ))

        # Winner for each task = highest-bidding *unassigned* drone that has
        # the task in its bundle.  Resolve most-contested tasks first so that
        # the globally best match is preserved when bids are tied.
        assigned_drones: set[str] = set()

        task_top_bid: dict[int, float] = {}
        for tidx in active_task_indices:
            task_top_bid[tidx] = max(
                (s.y[tidx] for s in states.values() if tidx in s.bundle),
                default=0.0,
            )

        for tidx in sorted(active_task_indices,
                           key=lambda i: task_top_bid[i], reverse=True):
            best_drone = None
            best_val = -math.inf
            for drone_id, s in states.items():
                if drone_id in assigned_drones:
                    continue
                if tidx in s.bundle and s.y[tidx] > best_val:
                    best_val = s.y[tidx]
                    best_drone = drone_id
            if best_drone is not None:
                assignments[best_drone] = tidx
                assigned_drones.add(best_drone)

        # Queue outgoing broadcasts (respects comm_delay)
        if self.comm_delay > 0:
            deliver_at = snapshot.step + self.comm_delay
            for drone_id, s in states.items():
                self._pending_msgs.append(
                    (deliver_at, drone_id, dict(s.y), dict(s.z))
                )

        # Co-assignment pass for shareable tasks (same as GreedyAuction)
        idle_drones = [d for d in drone_ids if d not in assigned_drones]
        if idle_drones:
            for tidx in active_task_indices:
                task = snapshot.tasks[tidx]
                if not task.spec.is_shareable:
                    continue
                if task.remaining_work() < SHARE_THRESHOLD:
                    continue
                pos_scores = [
                    (
                        drone_id,
                        _path_cost_utility(
                            snapshot.drone_positions[drone_id],
                            [],
                            tidx,
                            snapshot.tasks,
                            snapshot.drone_batteries.get(drone_id, 1.0),
                        ),
                    )
                    for drone_id in idle_drones
                ]
                pos_scores.sort(key=lambda x: x[1], reverse=True)
                if pos_scores:
                    best_drone, _ = pos_scores[0]
                    assignments[best_drone] = tidx
                    idle_drones.remove(best_drone)
                    assigned_drones.add(best_drone)
                if not idle_drones:
                    break

        return AllocationResult(assignments=assignments, bids=all_bids)

    # ------------------------------------------------------------------
    # Notification hooks
    # ------------------------------------------------------------------

    def on_task_complete(self, task_idx: int, step: int) -> None:
        """Purge stale messages referencing a completed task."""
        self._pending_msgs = [
            msg for msg in self._pending_msgs
            if task_idx not in msg[2]   # msg[2] = y dict
        ]

    def on_task_vanish(self, task_idx: int, step: int) -> None:
        """Purge stale messages referencing a vanished task."""
        self.on_task_complete(task_idx, step)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _deliver_messages(
        self,
        current_step: int,
        states: dict[str, _DroneState],
        active_task_indices: list[int],
    ) -> int:
        """Apply any pending messages whose deliver_at <= current_step."""
        delivered = 0
        remaining = []
        for msg in self._pending_msgs:
            deliver_at, sender_id, y, z = msg
            if deliver_at <= current_step:
                # Merge into all *other* drones' tables
                for drone_id, s in states.items():
                    if drone_id == sender_id:
                        continue
                    for tidx in active_task_indices:
                        if tidx in y and y[tidx] > s.y.get(tidx, 0.0):
                            s.y[tidx] = y[tidx]
                            s.z[tidx] = z.get(tidx)
                delivered += 1
            else:
                remaining.append(msg)
        self._pending_msgs = remaining
        return delivered
