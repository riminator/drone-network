"""
drone_agent.py
Represents the state and kinematics of a single drone in the simulation.
No hardware dependency — all physics is simplified grid/vector math.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_SPEED = 0.5          # metres per timestep
MAX_BATTERY = 100.0      # arbitrary units
BATTERY_DRAIN_MOVE = 0.1 # per timestep while moving
BATTERY_DRAIN_HOVER = 0.05
BATTERY_DRAIN_ENGAGE = 0.2  # per timestep while tool is engaged
COLLISION_RADIUS = 0.3   # metres — drones within this distance collide


@dataclass
class DroneState:
    """Mutable state snapshot for one drone."""
    drone_id: str
    position: np.ndarray          # shape (3,) — x, y, z in metres
    velocity: np.ndarray          # shape (3,)
    battery: float = MAX_BATTERY
    task_id: Optional[str] = None # ID of currently assigned task
    task_progress: float = 0.0    # 0.0 → 1.0
    is_done: bool = False          # episode termination flag
    tool_engaged: bool = False


class DroneAgent:
    """
    Lightweight drone model used inside HomeEnv.

    Observation vector layout (15 dims):
        [0:3]   position (x, y, z)
        [3]     battery (normalised 0-1)
        [4]     task_progress (0-1)
        [5:8]   delta to task target (dx, dy, dz), zeros if no task
        [8:11]  own velocity (vx, vy, vz)
        [11]    tool_engaged (0/1)
        [12:15] nearest neighbour relative position (dx, dy, dz), zeros if none
    """

    OBS_DIM = 15
    ACT_DIM = 4  # (dx, dy, dz, tool_engage) — tool_engage clipped to {0,1}

    def __init__(self, drone_id: str, init_position: np.ndarray):
        self.state = DroneState(
            drone_id=drone_id,
            position=np.array(init_position, dtype=np.float32),
            velocity=np.zeros(3, dtype=np.float32),
        )
        self._room_bounds: Optional[np.ndarray] = None  # set by HomeEnv

    # ------------------------------------------------------------------
    # Environment integration
    # ------------------------------------------------------------------

    def set_room_bounds(self, bounds: np.ndarray):
        """bounds shape (3,) — (max_x, max_y, max_z)."""
        self._room_bounds = np.array(bounds, dtype=np.float32)

    def apply_action(self, action: np.ndarray) -> float:
        """
        Apply a (4,) action vector. Returns battery consumed this step.
        action[:3] — desired delta position (clipped to MAX_SPEED)
        action[3]  — tool engage signal (>0.5 → True)
        """
        delta = np.clip(action[:3], -MAX_SPEED, MAX_SPEED).astype(np.float32)
        self.state.tool_engaged = float(action[3]) > 0.5

        # Move
        new_pos = self.state.position + delta
        if self._room_bounds is not None:
            new_pos = np.clip(new_pos, 0.0, self._room_bounds)
        self.state.position = new_pos
        self.state.velocity = delta

        # Battery
        moving = np.linalg.norm(delta) > 1e-4
        drain = BATTERY_DRAIN_MOVE if moving else BATTERY_DRAIN_HOVER
        if self.state.tool_engaged:
            drain += BATTERY_DRAIN_ENGAGE
        self.state.battery = max(0.0, self.state.battery - drain)

        if self.state.battery <= 0.0:
            self.state.is_done = True

        return drain

    def get_observation(
        self,
        task_target: Optional[np.ndarray],
        nearest_neighbour: Optional[np.ndarray],
    ) -> np.ndarray:
        """Build the 15-dim observation vector."""
        obs = np.zeros(self.OBS_DIM, dtype=np.float32)

        obs[0:3] = self.state.position
        obs[3] = self.state.battery / MAX_BATTERY
        obs[4] = self.state.task_progress

        if task_target is not None:
            obs[5:8] = task_target - self.state.position

        obs[8:11] = self.state.velocity
        obs[11] = float(self.state.tool_engaged)

        if nearest_neighbour is not None:
            obs[12:15] = nearest_neighbour - self.state.position

        return obs

    def reset(self, init_position: np.ndarray):
        self.state.position = np.array(init_position, dtype=np.float32)
        self.state.velocity = np.zeros(3, dtype=np.float32)
        self.state.battery = MAX_BATTERY
        self.state.task_id = None
        self.state.task_progress = 0.0
        self.state.is_done = False
        self.state.tool_engaged = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def distance_to(self, other: "DroneAgent") -> float:
        return float(np.linalg.norm(self.state.position - other.state.position))

    def is_colliding(self, other: "DroneAgent") -> bool:
        return self.distance_to(other) < COLLISION_RADIUS

    def __repr__(self) -> str:
        s = self.state
        return (
            f"Drone({s.drone_id} pos={s.position} "
            f"batt={s.battery:.1f} task={s.task_id} prog={s.task_progress:.2f})"
        )
