"""
base_task.py
Abstract base class that every household task must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class TaskSpec:
    """Static description of a task instance placed in the environment."""
    task_id: str
    task_type: str            # "water_plant" | "sweep_floor" | "toggle_light"
    target_position: np.ndarray  # where the drone must go
    requires_tool: bool = True
    # How many consecutive timesteps the tool must stay engaged at target
    engage_steps_required: int = 10


class BaseTask(ABC):
    """
    A task has:
      - a spec (static description)
      - mutable progress state tracked per assigned drone
    """

    def __init__(self, spec: TaskSpec):
        self.spec = spec
        self.engage_steps_done: int = 0
        self.completed: bool = False
        self.assigned_drone_id: str | None = None

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @abstractmethod
    def step(self, drone_position: np.ndarray, tool_engaged: bool) -> float:
        """
        Called every env timestep by the assigned drone.
        Returns incremental progress (0-1 delta) this step.
        Must set self.completed = True when done.
        """

    @abstractmethod
    def completion_reward(self) -> float:
        """Sparse reward given once when task is fully completed."""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def distance_to_target(self, drone_position: np.ndarray) -> float:
        return float(np.linalg.norm(drone_position - self.spec.target_position))

    def is_at_target(self, drone_position: np.ndarray, tolerance: float = 0.5) -> bool:
        return self.distance_to_target(drone_position) <= tolerance

    def reset(self):
        self.engage_steps_done = 0
        self.completed = False
        self.assigned_drone_id = None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(id={self.spec.task_id} "
            f"completed={self.completed} progress={self.engage_steps_done}/"
            f"{self.spec.engage_steps_required})"
        )
