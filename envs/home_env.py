"""
home_env.py
Multi-agent Gymnasium environment simulating a home with a swarm of drones.

Implements the Ray RLlib MultiAgentEnv interface so it can be dropped
straight into RLlib's MAPPO trainer.

Observation space per drone: Box(15,)   — see DroneAgent.get_observation
Action space per drone:      Box(4,)    — (dx, dy, dz, tool_engage)
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
except ImportError:
    # Fallback base class so the env can be imported without RLlib installed
    MultiAgentEnv = object  # type: ignore

from envs.drone_agent import DroneAgent, MAX_BATTERY
from envs.tasks import WaterPlantTask, SweepFloorTask, ToggleLightTask
from envs.tasks.base_task import TaskSpec

# ---------------------------------------------------------------------------
# Reward constants (also used by reward_shaping.py)
# ---------------------------------------------------------------------------
REWARD_STEP_ALIVE = -0.01       # per-step cost to encourage efficiency
REWARD_COLLISION = -5.0         # drone–drone collision
REWARD_BATTERY_DEAD = -3.0      # drone runs out of battery mid-episode
REWARD_COOPERATIVE_BONUS = 2.0  # all tasks cleared faster than threshold


# ---------------------------------------------------------------------------
# Task factory
# ---------------------------------------------------------------------------
_TASK_REGISTRY = {
    "water_plant": WaterPlantTask,
    "sweep_floor": SweepFloorTask,
    "toggle_light": ToggleLightTask,
}

_DEFAULT_TASK_LAYOUTS = [
    # (task_type, target_position, engage_steps)
    ("water_plant",  [2.0, 3.0, 1.0], 10),
    ("water_plant",  [7.0, 1.5, 1.0], 10),
    ("sweep_floor",  [5.0, 5.0, 0.3], 15),
    ("toggle_light", [9.0, 0.5, 2.5],  1),
    ("toggle_light", [0.5, 9.0, 2.5],  1),
]


def _make_task(task_type: str, idx: int, target: list, engage_steps: int):
    spec = TaskSpec(
        task_id=f"{task_type}_{idx}",
        task_type=task_type,
        target_position=np.array(target, dtype=np.float32),
        engage_steps_required=engage_steps,
    )
    return _TASK_REGISTRY[task_type](spec)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class HomeEnv(MultiAgentEnv):
    """
    A simulated home room with n_drones cooperative agents.

    Episode ends when:
      - All tasks are completed, OR
      - max_steps is reached, OR
      - All drones are done (battery dead)
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config: dict | None = None):
        config = config or {}

        self.n_drones: int = config.get("n_drones", 3)
        self.room_size: tuple = tuple(config.get("room_size", [10.0, 10.0, 3.0]))
        self.max_steps: int = config.get("max_steps", 500)
        self.task_layouts: list = config.get("task_layouts", _DEFAULT_TASK_LAYOUTS)
        self.render_mode: str | None = config.get("render_mode", None)
        # Cooperative bonus: awarded if all tasks done before this fraction of
        # max_steps has elapsed
        self.coop_time_threshold: float = config.get("coop_time_threshold", 0.7)
        # Domain randomisation: Gaussian noise added to every observation during
        # training so the policy learns to be robust to position uncertainty.
        # Mimics the Crazyflie PID overshoot / sensor noise in PyBullet.
        # Set to 0 to disable (default off for backward compat).
        self.obs_noise_std: float = config.get("obs_noise_std", 0.0)

        self._room_bounds = np.array(self.room_size, dtype=np.float32)

        # Agent IDs (RLlib convention)
        self._agent_ids = {f"drone_{i}" for i in range(self.n_drones)}
        if hasattr(super(), "__init__"):
            super().__init__()

        # Spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(DroneAgent.OBS_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1, 0], dtype=np.float32),
            high=np.array([1,  1,  1, 1], dtype=np.float32),
            dtype=np.float32,
        )

        # Populated on reset()
        self._drones: dict[str, DroneAgent] = {}
        self._tasks: list = []
        self._step_count: int = 0
        # Map drone_id → task index (or None)
        self._drone_task_map: dict[str, int | None] = {}

    # ------------------------------------------------------------------
    # Gymnasium / RLlib interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self._step_count = 0

        # Spawn drones at spread-out positions near floor level
        self._drones = {}
        dock_positions = self._get_dock_positions()
        for i, agent_id in enumerate(sorted(self._agent_ids)):
            drone = DroneAgent(agent_id, dock_positions[i])
            drone.set_room_bounds(self._room_bounds)
            self._drones[agent_id] = drone

        # Instantiate tasks
        self._tasks = [
            _make_task(t, i, pos, steps)
            for i, (t, pos, steps) in enumerate(self.task_layouts)
        ]

        # Initial task assignment (round-robin)
        self._drone_task_map = {aid: None for aid in self._agent_ids}
        self._assign_tasks()

        obs = self._build_obs_dict()
        return obs, {}

    def step(
        self, action_dict: dict[str, np.ndarray]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, Any],
    ]:
        self._step_count += 1
        rewards = {aid: REWARD_STEP_ALIVE for aid in self._agent_ids}

        # 1. Apply actions and collect physics rewards
        for agent_id, action in action_dict.items():
            drone = self._drones[agent_id]
            if drone.state.is_done:
                continue
            drone.apply_action(action)
            if drone.state.battery <= 0.0:
                rewards[agent_id] += REWARD_BATTERY_DEAD

        # 2. Collision detection
        agent_list = list(self._drones.values())
        for i in range(len(agent_list)):
            for j in range(i + 1, len(agent_list)):
                if agent_list[i].is_colliding(agent_list[j]):
                    rewards[agent_list[i].state.drone_id] += REWARD_COLLISION
                    rewards[agent_list[j].state.drone_id] += REWARD_COLLISION

        # 3. Task progress
        for agent_id, drone in self._drones.items():
            task_idx = self._drone_task_map.get(agent_id)
            if task_idx is None:
                continue
            task = self._tasks[task_idx]
            if task.completed:
                # Re-assign to next open task
                self._drone_task_map[agent_id] = None
                self._assign_tasks()
                continue

            delta = task.step(drone.state.position, drone.state.tool_engaged)
            rewards[agent_id] += delta
            drone.state.task_progress = (
                task.engage_steps_done / task.spec.engage_steps_required
            )

            if task.completed:
                rewards[agent_id] += task.completion_reward()
                self._drone_task_map[agent_id] = None
                self._assign_tasks()

        # 4. Cooperative bonus
        if all(t.completed for t in self._tasks):
            if self._step_count < self.max_steps * self.coop_time_threshold:
                for aid in self._agent_ids:
                    rewards[aid] += REWARD_COOPERATIVE_BONUS

        # 5. Termination
        all_tasks_done = all(t.completed for t in self._tasks)
        all_drones_done = all(d.state.is_done for d in self._drones.values())
        time_limit = self._step_count >= self.max_steps

        terminated = {
            aid: all_tasks_done or all_drones_done
            for aid in self._agent_ids
        }
        truncated = {aid: time_limit for aid in self._agent_ids}
        terminated["__all__"] = all_tasks_done or all_drones_done
        truncated["__all__"] = time_limit

        obs = self._build_obs_dict()
        infos = self._build_info_dict()
        return obs, rewards, terminated, truncated, infos

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_dock_positions(self) -> list[np.ndarray]:
        """Evenly-spaced charging dock positions along one wall."""
        positions = []
        spacing = self._room_bounds[0] / (self.n_drones + 1)
        for i in range(self.n_drones):
            positions.append(
                np.array([(i + 1) * spacing, 0.5, 1.0], dtype=np.float32)
            )
        return positions

    def _assign_tasks(self):
        """Assign unassigned drones to incomplete, unclaimed tasks."""
        claimed = {
            idx
            for idx in self._drone_task_map.values()
            if idx is not None
        }
        pending = [
            i for i, t in enumerate(self._tasks)
            if not t.completed and i not in claimed
        ]

        for agent_id in self._agent_ids:
            if self._drone_task_map[agent_id] is None and pending:
                task_idx = pending.pop(0)
                self._drone_task_map[agent_id] = task_idx
                self._tasks[task_idx].assigned_drone_id = agent_id
                self._drones[agent_id].state.task_id = (
                    self._tasks[task_idx].spec.task_id
                )

    def _nearest_neighbour(self, agent_id: str) -> np.ndarray | None:
        """Return position of the closest other drone, or None."""
        own = self._drones[agent_id].state.position
        best_dist = float("inf")
        best_pos = None
        for other_id, other in self._drones.items():
            if other_id == agent_id:
                continue
            d = float(np.linalg.norm(own - other.state.position))
            if d < best_dist:
                best_dist = d
                best_pos = other.state.position
        return best_pos

    def _build_obs_dict(self) -> dict[str, np.ndarray]:
        obs = {}
        for agent_id, drone in self._drones.items():
            task_idx = self._drone_task_map.get(agent_id)
            task_target = (
                self._tasks[task_idx].spec.target_position
                if task_idx is not None and not self._tasks[task_idx].completed
                else None
            )
            neighbour = self._nearest_neighbour(agent_id)
            o = drone.get_observation(task_target, neighbour)
            if self.obs_noise_std > 0.0:
                o = o + np.random.normal(0.0, self.obs_noise_std, o.shape).astype(np.float32)
            obs[agent_id] = o
        return obs

    def _build_info_dict(self) -> dict[str, Any]:
        tasks_completed = sum(1 for t in self._tasks if t.completed)
        return {
            aid: {
                "tasks_completed": tasks_completed,
                "tasks_total": len(self._tasks),
                "step": self._step_count,
                "battery": self._drones[aid].state.battery,
            }
            for aid in self._agent_ids
        }

    # ------------------------------------------------------------------
    # Optional: human render (ASCII)
    # ------------------------------------------------------------------

    def render(self) -> str | None:
        if self.render_mode != "human":
            return None
        lines = [f"Step {self._step_count}/{self.max_steps}"]
        for aid, drone in self._drones.items():
            task_idx = self._drone_task_map.get(aid)
            task_str = (
                self._tasks[task_idx].spec.task_id
                if task_idx is not None
                else "idle"
            )
            lines.append(
                f"  {aid}: pos={drone.state.position.round(2)} "
                f"batt={drone.state.battery:.1f} task={task_str}"
            )
        tasks_done = sum(1 for t in self._tasks if t.completed)
        lines.append(f"  Tasks: {tasks_done}/{len(self._tasks)} completed")
        output = "\n".join(lines)
        print(output)
        return output
