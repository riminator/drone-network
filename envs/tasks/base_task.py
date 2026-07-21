"""
base_task.py
Abstract base class that every household task must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np


# ---------------------------------------------------------------------------
# Task lifecycle state machine
#
#   PENDING  ──► ASSIGNED  ──► IN_PROGRESS  ──► COMPLETE
#                   │                │
#                   └──► VANISHED ◄──┘   (task removed mid-episode)
# ---------------------------------------------------------------------------

class TaskStatus(Enum):
    PENDING     = auto()   # spawned, not yet assigned to any drone
    ASSIGNED    = auto()   # at least one drone is en-route
    IN_PROGRESS = auto()   # at least one drone is at target and engaging
    COMPLETE    = auto()   # task fully done
    VANISHED    = auto()   # removed from environment mid-episode


@dataclass
class TaskSpec:
    """Static description of a task instance placed in the environment."""
    task_id: str
    task_type: str                # "water_plant" | "sweep_floor" | "toggle_light"
    target_position: np.ndarray  # where the drone must go
    requires_tool: bool = True
    # How many consecutive timesteps the tool must stay engaged at target
    engage_steps_required: int = 10
    # Whether multiple drones can be assigned simultaneously
    # (e.g. large sweep areas can be shared; point tasks like toggle_light cannot)
    is_shareable: bool = False


class BaseTask(ABC):
    """
    A task has:
      - a spec  (static description)
      - mutable progress state tracked across all assigned drones

    Key changes vs original:
      - assigned_drone_ids  (list, not single string) — supports co-assignment
      - status              (TaskStatus enum) — full lifecycle visibility
      - remaining_work()    (abstract) — allocator uses this for marginal-value bids
      - shareable           (from spec) — allocator may co-assign when True
    """

    def __init__(self, spec: TaskSpec):
        self.spec = spec
        self.engage_steps_done: int = 0
        self.status: TaskStatus = TaskStatus.PENDING
        # Plural — may contain more than one drone_id when spec.is_shareable
        self.assigned_drone_ids: list[str] = []

    # ------------------------------------------------------------------
    # Convenience shim — keeps old single-drone code working
    # ------------------------------------------------------------------

    @property
    def completed(self) -> bool:
        return self.status == TaskStatus.COMPLETE

    @completed.setter
    def completed(self, value: bool) -> None:
        if value:
            self.status = TaskStatus.COMPLETE

    @property
    def assigned_drone_id(self) -> str | None:
        """Legacy single-drone accessor — returns first assignee or None."""
        return self.assigned_drone_ids[0] if self.assigned_drone_ids else None

    @assigned_drone_id.setter
    def assigned_drone_id(self, drone_id: str | None) -> None:
        """Legacy setter — replaces the full list."""
        self.assigned_drone_ids = [drone_id] if drone_id is not None else []

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @abstractmethod
    def step(self, drone_position: np.ndarray, tool_engaged: bool) -> float:
        """
        Called every env timestep by each assigned drone.
        Returns incremental shaped reward for this (drone, step) pair.
        Must set self.status = TaskStatus.COMPLETE when done.
        """

    @abstractmethod
    def completion_reward(self) -> float:
        """Sparse reward given once when task is fully completed."""

    @abstractmethod
    def remaining_work(self) -> float:
        """
        Normalised estimate of work left in [0, 1].
        Used by the auction allocator to compute marginal bid value.
        0.0 = done or vanished; 1.0 = not started.
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def distance_to_target(self, drone_position: np.ndarray) -> float:
        return float(np.linalg.norm(drone_position - self.spec.target_position))

    def is_at_target(self, drone_position: np.ndarray, tolerance: float = 1.0) -> bool:
        return self.distance_to_target(drone_position) <= tolerance

    def vanish(self) -> None:
        """Mark task as removed from the environment mid-episode."""
        self.status = TaskStatus.VANISHED
        self.assigned_drone_ids = []

    def reset(self):
        self.engage_steps_done = 0
        self.status = TaskStatus.PENDING
        self.assigned_drone_ids = []

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(id={self.spec.task_id} "
            f"status={self.status.name} progress={self.engage_steps_done}/"
            f"{self.spec.engage_steps_required} "
            f"assignees={self.assigned_drone_ids})"
        )
