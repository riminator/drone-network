"""
home_env.py
Multi-agent Gymnasium environment simulating a home with a swarm of drones.

Implements the Ray RLlib MultiAgentEnv interface so it can be dropped
straight into RLlib's MAPPO trainer.

Observation space per drone: Box(15,)   — see DroneAgent.get_observation
Action space per drone:      Box(4,)    — (dx, dy, dz, tool_engage)

Phase 1 additions
-----------------
* allocator   — optional BaseAllocator instance injected via config.
                Defaults to _GreedyFallbackAllocator (old behaviour).
* remove_task(task_id)   — vanish a task mid-episode; triggers re-allocation.
* add_task(task_spec)    — inject a new task mid-episode; triggers re-allocation.
* _apply_allocation()    — translates AllocationResult → _drone_task_map,
                           respects co-assignment for shareable tasks.
* step()                 — skips VANISHED tasks; calls task.step() for every
                           drone assigned to a shared task.
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
    MultiAgentEnv = object  # type: ignore

from envs.drone_agent import DroneAgent, MAX_BATTERY
from envs.tasks import WaterPlantTask, SweepFloorTask, ToggleLightTask
from envs.tasks.base_task import TaskSpec, TaskStatus
from allocator.base_allocator import BaseAllocator, WorldSnapshot, AllocationResult, Bid

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
    # (task_type, target_position, engage_steps, is_shareable)
    ("water_plant",  [2.0, 3.0, 1.0], 10, False),
    ("water_plant",  [7.0, 1.5, 1.0], 10, False),
    ("sweep_floor",  [5.0, 5.0, 0.3], 15, True),   # sweep is co-assignable
    ("toggle_light", [9.0, 0.5, 2.5],  1, False),
    ("toggle_light", [0.5, 9.0, 2.5],  1, False),
]


def _make_task(task_type: str, idx: int, target: list, engage_steps: int,
               is_shareable: bool = False):
    spec = TaskSpec(
        task_id=f"{task_type}_{idx}",
        task_type=task_type,
        target_position=np.array(target, dtype=np.float32),
        engage_steps_required=engage_steps,
        is_shareable=is_shareable,
    )
    return _TASK_REGISTRY[task_type](spec)


# ---------------------------------------------------------------------------
# Built-in fallback allocator (greedy first-available — old behaviour)
# ---------------------------------------------------------------------------

class _GreedyFallbackAllocator(BaseAllocator):
    """
    Mimics the original _assign_tasks() logic so HomeEnv behaviour is
    unchanged when no external allocator is provided.
    Assigns each idle drone to the first unclaimed PENDING/ASSIGNED task.
    """

    def allocate(self, snapshot: WorldSnapshot) -> AllocationResult:
        from envs.tasks.base_task import TaskStatus

        # Tasks eligible for assignment (not done, not vanished)
        active_statuses = {TaskStatus.PENDING, TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS}

        # Build set of task indices already claimed by a non-idle drone
        claimed: set[int] = set()
        for task_idx in snapshot.current_assignments.values():
            if task_idx is not None:
                claimed.add(task_idx)

        pending = [
            i for i, t in enumerate(snapshot.tasks)
            if t.status in active_statuses and i not in claimed
        ]

        assignments: dict[str, int | None] = {}
        for drone_id in sorted(snapshot.drone_positions):
            existing = snapshot.current_assignments.get(drone_id)
            # Keep current assignment if still valid
            if (existing is not None
                    and existing < len(snapshot.tasks)
                    and snapshot.tasks[existing].status in active_statuses):
                assignments[drone_id] = existing
            elif pending:
                assignments[drone_id] = pending.pop(0)
            else:
                assignments[drone_id] = None

        return AllocationResult(assignments=assignments)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class HomeEnv(MultiAgentEnv):
    """
    A simulated home room with n_drones cooperative agents.

    Episode ends when:
      - All tasks are completed (or vanished), OR
      - max_steps is reached, OR
      - All drones are done (battery dead)

    Phase 1 public API additions:
      env.remove_task(task_id)            — vanish task mid-episode
      env.add_task(task_spec_or_tuple)    — inject task mid-episode
      env.allocator                       — pluggable BaseAllocator
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config: dict | None = None):
        config = config or {}

        self.n_drones: int = config.get("n_drones", 3)
        self.room_size: tuple = tuple(config.get("room_size", [10.0, 10.0, 3.0]))
        self.max_steps: int = config.get("max_steps", 500)
        self.task_layouts: list = config.get("task_layouts", _DEFAULT_TASK_LAYOUTS)
        self.render_mode: str | None = config.get("render_mode", None)
        self.coop_time_threshold: float = config.get("coop_time_threshold", 0.7)
        self.obs_noise_std: float = config.get("obs_noise_std", 0.0)

        # Pluggable allocator — defaults to the greedy fallback
        self.allocator: BaseAllocator = config.get(
            "allocator", _GreedyFallbackAllocator()
        )

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

        # Instantiate tasks — support both old 3-tuple and new 4-tuple layouts
        self._tasks = []
        for i, layout in enumerate(self.task_layouts):
            if len(layout) == 4:
                t, pos, steps, shareable = layout
            else:
                t, pos, steps = layout
                shareable = False
            self._tasks.append(_make_task(t, i, pos, steps, shareable))

        # Initial task assignment via allocator
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

        # 3. Task progress — handle VANISHED + co-assignment
        realloc_needed = False

        for agent_id, drone in self._drones.items():
            task_idx = self._drone_task_map.get(agent_id)
            if task_idx is None:
                continue
            task = self._tasks[task_idx]

            # Drop reference to vanished or already-completed tasks
            if task.status in (TaskStatus.VANISHED, TaskStatus.COMPLETE):
                self._drone_task_map[agent_id] = None
                realloc_needed = True
                continue

            # Step the task
            delta = task.step(drone.state.position, drone.state.tool_engaged)
            rewards[agent_id] += delta

            # Update task status based on engagement
            if (task.status == TaskStatus.ASSIGNED
                    and task.is_at_target(drone.state.position)):
                task.status = TaskStatus.IN_PROGRESS

            # Update drone's local progress observation
            drone.state.task_progress = (
                task.engage_steps_done / task.spec.engage_steps_required
            )

            if task.status == TaskStatus.COMPLETE:
                # Split the completion reward among all assigned drones
                n_assigned = max(1, len(task.assigned_drone_ids))
                rewards[agent_id] += task.completion_reward() / n_assigned
                self._drone_task_map[agent_id] = None
                self.allocator.on_task_complete(task_idx, self._step_count)
                realloc_needed = True

        if realloc_needed:
            self._assign_tasks()

        # 4. Cooperative bonus — all active tasks done
        active_tasks = [
            t for t in self._tasks
            if t.status not in (TaskStatus.VANISHED,)
        ]
        if active_tasks and all(t.status == TaskStatus.COMPLETE for t in active_tasks):
            if self._step_count < self.max_steps * self.coop_time_threshold:
                for aid in self._agent_ids:
                    rewards[aid] += REWARD_COOPERATIVE_BONUS

        # 5. Termination
        all_tasks_done = all(
            t.status in (TaskStatus.COMPLETE, TaskStatus.VANISHED)
            for t in self._tasks
        )
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
    # Phase 1 disruption API
    # ------------------------------------------------------------------

    def remove_task(self, task_id: str) -> bool:
        """
        Vanish a task mid-episode (e.g. plant removed, light already off).
        Returns True if found and vanished, False if not found or already done.
        """
        for i, task in enumerate(self._tasks):
            if task.spec.task_id == task_id:
                if task.status in (TaskStatus.COMPLETE, TaskStatus.VANISHED):
                    return False
                task.vanish()
                self.allocator.on_task_vanish(i, self._step_count)
                # Free drones that were assigned to this task
                for aid, tidx in self._drone_task_map.items():
                    if tidx == i:
                        self._drone_task_map[aid] = None
                self._assign_tasks()
                return True
        return False

    def add_task(
        self,
        task_type: str,
        target: list[float],
        engage_steps: int = 10,
        is_shareable: bool = False,
    ) -> str:
        """
        Inject a new task mid-episode (e.g. a spill appears).
        Returns the new task_id.
        """
        idx = len(self._tasks)
        new_task = _make_task(task_type, idx, target, engage_steps, is_shareable)
        self._tasks.append(new_task)
        self._assign_tasks()
        return new_task.spec.task_id

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
        """
        Ask the allocator for a new assignment and apply it.
        Called at episode start and after any disruption event.
        """
        snapshot = WorldSnapshot(
            drone_positions={
                aid: d.state.position.copy()
                for aid, d in self._drones.items()
            },
            drone_batteries={
                aid: d.state.battery
                for aid, d in self._drones.items()
            },
            drone_task_progress={
                aid: d.state.task_progress
                for aid, d in self._drones.items()
            },
            current_assignments=dict(self._drone_task_map),
            tasks=self._tasks,
            step=self._step_count,
            max_steps=self.max_steps,
        )
        result = self.allocator.allocate(snapshot)
        self._apply_allocation(result)

    def _apply_allocation(self, result: AllocationResult) -> None:
        """
        Write an AllocationResult into _drone_task_map and update
        task.assigned_drone_ids to reflect the new assignment.
        Handles co-assignment for shareable tasks.
        """
        # Clear all task assignee lists first
        for task in self._tasks:
            task.assigned_drone_ids = []

        for agent_id, task_idx in result.assignments.items():
            self._drone_task_map[agent_id] = task_idx
            if task_idx is None:
                self._drones[agent_id].state.task_id = None
                continue

            task = self._tasks[task_idx]

            # Co-assignment: only allowed when task.spec.is_shareable
            if agent_id not in task.assigned_drone_ids:
                if task.spec.is_shareable or not task.assigned_drone_ids:
                    task.assigned_drone_ids.append(agent_id)
                else:
                    # Non-shareable task already has an assignee — keep drone idle
                    self._drone_task_map[agent_id] = None
                    self._drones[agent_id].state.task_id = None
                    continue

            # Advance status from PENDING → ASSIGNED
            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.ASSIGNED

            self._drones[agent_id].state.task_id = task.spec.task_id

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
            task = self._tasks[task_idx] if task_idx is not None else None
            task_target = (
                task.spec.target_position
                if task is not None and task.status not in (
                    TaskStatus.COMPLETE, TaskStatus.VANISHED
                )
                else None
            )
            neighbour = self._nearest_neighbour(agent_id)
            o = drone.get_observation(task_target, neighbour)
            if self.obs_noise_std > 0.0:
                o = o + np.random.normal(0.0, self.obs_noise_std, o.shape).astype(np.float32)
            obs[agent_id] = o
        return obs

    def _build_info_dict(self) -> dict[str, Any]:
        tasks_completed = sum(
            1 for t in self._tasks if t.status == TaskStatus.COMPLETE
        )
        tasks_active = sum(
            1 for t in self._tasks if t.status not in (
                TaskStatus.COMPLETE, TaskStatus.VANISHED
            )
        )
        return {
            aid: {
                "tasks_completed": tasks_completed,
                "tasks_total": len(self._tasks),
                "tasks_active": tasks_active,
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
            if task_idx is not None:
                task_str = self._tasks[task_idx].spec.task_id
            else:
                task_str = "idle"
            lines.append(
                f"  {aid}: pos={drone.state.position.round(2)} "
                f"batt={drone.state.battery:.1f} task={task_str}"
            )
        tasks_done = sum(1 for t in self._tasks if t.status == TaskStatus.COMPLETE)
        lines.append(f"  Tasks: {tasks_done}/{len(self._tasks)} completed")
        output = "\n".join(lines)
        print(output)
        return output
