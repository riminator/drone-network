"""
sweep_floor.py
Task: drone follows a sweeping path across a floor tile and engages
the sweeper tool to clean it.  Modelled as visiting a sequence of
waypoints in order.
"""

from __future__ import annotations

import numpy as np
from .base_task import BaseTask, TaskSpec


class SweepFloorTask(BaseTask):
    """
    The sweeping area is divided into a row of N waypoints.
    The drone must visit each one in sequence with tool_engaged.
    """

    COMPLETION_REWARD = 12.0
    WAYPOINT_TOLERANCE = 0.9

    def __init__(self, spec: TaskSpec, n_waypoints: int = 4):
        super().__init__(spec)
        self._waypoints = self._generate_waypoints(spec.target_position, n_waypoints)
        self._current_wp_idx = 0

    # ------------------------------------------------------------------

    def _generate_waypoints(
        self, centre: np.ndarray, n: int, spacing: float = 0.8
    ) -> list[np.ndarray]:
        """Generate a back-and-forth sweep pattern around the centre."""
        waypoints = []
        half = (n - 1) * spacing / 2
        for i in range(n):
            x_offset = -half + i * spacing
            waypoints.append(
                np.array(
                    [centre[0] + x_offset, centre[1], centre[2]],
                    dtype=np.float32,
                )
            )
        return waypoints

    @property
    def current_target(self) -> np.ndarray:
        return self._waypoints[min(self._current_wp_idx, len(self._waypoints) - 1)]

    # ------------------------------------------------------------------
    # BaseTask interface
    # ------------------------------------------------------------------

    def step(self, drone_position: np.ndarray, tool_engaged: bool) -> float:
        progress_delta = 0.0
        target = self.current_target

        dist = float(np.linalg.norm(drone_position - target))
        progress_delta += max(0.0, (1.0 - dist) * 0.02)  # shaped approach reward

        if dist < self.WAYPOINT_TOLERANCE and tool_engaged:
            self._current_wp_idx += 1
            progress_delta += 0.2  # waypoint reached bonus
            if self._current_wp_idx >= len(self._waypoints):
                self.completed = True

        return float(progress_delta)

    def completion_reward(self) -> float:
        return self.COMPLETION_REWARD

    def remaining_work(self) -> float:
        """Fraction of waypoints still unvisited, normalised to [0, 1]."""
        if self.completed:
            return 0.0
        n = len(self._waypoints)
        done = min(self._current_wp_idx, n)
        return 1.0 - done / n

    def reset(self):
        super().reset()
        self._current_wp_idx = 0
