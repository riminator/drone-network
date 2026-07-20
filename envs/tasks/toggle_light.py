"""
toggle_light.py
Task: fly to a light switch position and tap it (brief tool engagement).
Simplest task — requires only one engage step at the target.
"""

import numpy as np
from .base_task import BaseTask, TaskSpec


class ToggleLightTask(BaseTask):
    """
    Completion: drone reaches the switch and engages the tool for
    `engage_steps_required` (default 1) timestep.
    """

    COMPLETION_REWARD = 8.0

    def __init__(self, spec: TaskSpec):
        # Override engage requirement to 1 — it's a momentary tap
        spec.engage_steps_required = max(1, spec.engage_steps_required)
        super().__init__(spec)
        self._prev_distance: float | None = None

    def step(self, drone_position: np.ndarray, tool_engaged: bool) -> float:
        dist = self.distance_to_target(drone_position)
        progress_delta = 0.0

        # Dense shaping: reward for closing in
        if self._prev_distance is not None:
            progress_delta += (self._prev_distance - dist) * 0.05
        self._prev_distance = dist

        if self.is_at_target(drone_position) and tool_engaged:
            self.engage_steps_done += 1
            if self.engage_steps_done >= self.spec.engage_steps_required:
                self.completed = True

        return float(progress_delta)

    def completion_reward(self) -> float:
        return self.COMPLETION_REWARD

    def reset(self):
        super().reset()
        self._prev_distance = None
