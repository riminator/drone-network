"""
water_plant.py
Task: fly to a plant's position and engage the watering tool for N steps.
"""

import numpy as np
from .base_task import BaseTask, TaskSpec


class WaterPlantTask(BaseTask):
    """
    Completion: drone hovers within tolerance of the plant pot and keeps
    tool_engaged=True for `engage_steps_required` consecutive timesteps.
    """

    COMPLETION_REWARD = 10.0
    PROXIMITY_SHAPING_SCALE = 0.05  # dense reward for getting closer

    def __init__(self, spec: TaskSpec):
        super().__init__(spec)
        self._prev_distance: float | None = None

    def step(self, drone_position: np.ndarray, tool_engaged: bool) -> float:
        """Returns shaped progress delta for this timestep."""
        dist = self.distance_to_target(drone_position)
        progress_delta = 0.0

        # Distance-based shaping reward (dense)
        if self._prev_distance is not None:
            progress_delta += (self._prev_distance - dist) * self.PROXIMITY_SHAPING_SCALE
        self._prev_distance = dist

        if self.is_at_target(drone_position):
            if tool_engaged:
                self.engage_steps_done += 1
                progress_delta += 0.1  # small bonus per engage step
            else:
                # Reset streak if drone stops engaging mid-task
                self.engage_steps_done = 0

            if self.engage_steps_done >= self.spec.engage_steps_required:
                self.completed = True

        return float(progress_delta)

    def completion_reward(self) -> float:
        return self.COMPLETION_REWARD

    def remaining_work(self) -> float:
        """Fraction of engage steps still needed, normalised to [0, 1]."""
        if self.completed:
            return 0.0
        done = min(self.engage_steps_done, self.spec.engage_steps_required)
        return 1.0 - done / self.spec.engage_steps_required

    def reset(self):
        super().reset()
        self._prev_distance = None
