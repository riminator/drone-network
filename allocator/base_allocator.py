"""
base_allocator.py
Abstract interface for all task-allocation strategies.

All allocators receive the same snapshot of world state and return
a mapping of drone_id → task_idx (or None if the drone should be idle).
Multiple drones may be mapped to the same task_idx when the task is
marked is_shareable=True.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from envs.tasks.base_task import BaseTask


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Bid:
    """
    A single drone's bid on a single task.

    bid_value  — higher is better; the allocator picks the highest bidder
                 per task (ties broken by drone_id lexicographically).
    marginal   — optional: the *extra* value of adding this drone to a task
                 that already has assignees. Used for co-assignment decisions.
    """
    drone_id: str
    task_idx: int
    bid_value: float
    marginal: float = 0.0


@dataclass
class WorldSnapshot:
    """
    Everything an allocator needs to make assignment decisions.
    Passed to allocate() each time a re-allocation is triggered.
    """
    # Per-drone state
    drone_positions: dict[str, np.ndarray]     # drone_id → xyz
    drone_batteries: dict[str, float]          # drone_id → [0, 1]
    drone_task_progress: dict[str, float]      # drone_id → [0, 1]
    current_assignments: dict[str, int | None] # drone_id → task_idx or None

    # Task state — only PENDING / ASSIGNED / IN_PROGRESS tasks are included
    tasks: list[BaseTask]                      # same indexing as HomeEnv._tasks

    # Simulation clock
    step: int
    max_steps: int


@dataclass
class AllocationResult:
    """
    Output of one allocation round.

    assignments — drone_id → task_idx (None = intentionally idle)
    bids        — all bids produced this round (for logging / analysis)
    """
    assignments: dict[str, int | None]
    bids: list[Bid] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseAllocator(ABC):
    """
    Contract:

    1. allocate(snapshot) → AllocationResult
       Called whenever the env decides a re-allocation is needed:
         - at episode start
         - when a task transitions to COMPLETE or VANISHED
         - when a new task is injected mid-episode
         - (optionally) on a periodic tick

    2. on_task_complete(task_idx, step)
       Notification hook so stateful allocators (e.g. CBBA) can update
       their internal bid tables without waiting for the next full round.

    3. on_task_vanish(task_idx, step)
       Same as above for tasks that disappear mid-episode.
    """

    @abstractmethod
    def allocate(self, snapshot: WorldSnapshot) -> AllocationResult:
        """
        Compute a new task-to-drone assignment.

        Must return an AllocationResult covering *all* drones in
        snapshot.drone_positions (idle drones map to None).
        """

    def on_task_complete(self, task_idx: int, step: int) -> None:
        """Optional: called when a task completes. Override in stateful allocators."""

    def on_task_vanish(self, task_idx: int, step: int) -> None:
        """Optional: called when a task is removed mid-episode."""
