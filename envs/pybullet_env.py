"""
pybullet_env.py
Physics-backed multi-drone environment for lab deployment and visual debugging.

Wraps gym-pybullet-drones' VelocityAviary with the exact same multi-agent
Gymnasium interface as HomeEnv — so any trained checkpoint loads without
modification.

Key differences from HomeEnv:
  - Real quadrotor aerodynamics (Crazyflie CF2X URDF via PyBullet)
  - PyBullet GUI window opens automatically (pass gui=False to suppress)
  - Household objects (plant pots, light switch, floor zone) loaded as URDF
  - Action conversion: policy (dx,dy,dz,tool) → VelocityAviary (vx,vy,vz,yaw_rate)

Install:
    pip install git+https://github.com/utiasDSL/gym-pybullet-drones.git

Usage:
    from envs.pybullet_env import PybulletHomeEnv
    env = PybulletHomeEnv(config={"n_drones": 3, "gui": True})
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
except ImportError:
    MultiAgentEnv = object  # type: ignore

try:
    import pybullet as p
    import pybullet_data
    from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
    from gym_pybullet_drones.utils.enums import DroneModel, Physics
    _PYBULLET_AVAILABLE = True
except ImportError:
    _PYBULLET_AVAILABLE = False

from gymnasium import spaces
from envs.drone_agent import DroneAgent, MAX_BATTERY
from envs.tasks import WaterPlantTask, SweepFloorTask, ToggleLightTask
from envs.tasks.base_task import TaskSpec

# ---------------------------------------------------------------------------
# Asset paths
# ---------------------------------------------------------------------------
_ASSETS_DIR = Path(__file__).parent.parent / "assets"

# ---------------------------------------------------------------------------
# Reward constants — mirror HomeEnv so policies transfer cleanly
# ---------------------------------------------------------------------------
REWARD_STEP_ALIVE   = -0.01
REWARD_COLLISION    = -5.0
REWARD_BATTERY_DEAD = -3.0
REWARD_COOP_BONUS   = 2.0

# ---------------------------------------------------------------------------
# Default task layout — positions in metres (x, y, z)
# Keep z low: plant pots on floor, light switch on wall at 1.5m
# ---------------------------------------------------------------------------
_DEFAULT_TASK_LAYOUTS = [
    ("water_plant",  [2.0, 3.0, 0.5], 10),
    ("water_plant",  [7.0, 1.5, 0.5], 10),
    ("sweep_floor",  [5.0, 5.0, 0.1], 15),
    ("toggle_light", [9.0, 0.5, 1.5],  1),
    ("toggle_light", [0.5, 9.0, 1.5],  1),
]

_TASK_REGISTRY = {
    "water_plant":  WaterPlantTask,
    "sweep_floor":  SweepFloorTask,
    "toggle_light": ToggleLightTask,
}

# URDF visual colours (r, g, b, a)
_TASK_COLOURS = {
    "water_plant":  [0.0, 0.6, 0.1, 1.0],  # green
    "sweep_floor":  [0.7, 0.5, 0.1, 1.0],  # brown
    "toggle_light": [1.0, 0.9, 0.0, 1.0],  # yellow
}


def _make_task(task_type: str, idx: int, target: list, engage_steps: int):
    spec = TaskSpec(
        task_id=f"{task_type}_{idx}",
        task_type=task_type,
        target_position=np.array(target, dtype=np.float32),
        engage_steps_required=engage_steps,
    )
    return _TASK_REGISTRY[task_type](spec)


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------

class PybulletHomeEnv(MultiAgentEnv):
    """
    Physics-accurate drone swarm environment backed by PyBullet.

    Observation space: identical to HomeEnv — Box(15,) per drone.
    Action space:      identical to HomeEnv — Box(4,)  per drone.

    Internally the (dx, dy, dz) action is converted to a target velocity
    command for VelocityAviary, which handles the low-level PID and motor
    mixing internally.
    """

    # Observation dim matches DroneAgent exactly — policy transfers without change
    OBS_DIM = DroneAgent.OBS_DIM   # 15
    ACT_DIM = DroneAgent.ACT_DIM   # 4

    def __init__(self, config: dict | None = None):
        if not _PYBULLET_AVAILABLE:
            raise ImportError(
                "gym-pybullet-drones is not installed.\n"
                "Run: pip install git+https://github.com/utiasDSL/gym-pybullet-drones.git"
            )

        config = config or {}
        self.n_drones: int        = config.get("n_drones", 3)
        self.room_size: list      = config.get("room_size", [10.0, 10.0, 3.0])
        self.max_steps: int       = config.get("max_steps", 500)
        self.task_layouts: list   = config.get("task_layouts", _DEFAULT_TASK_LAYOUTS)
        self.gui: bool            = config.get("gui", True)
        self.record: bool         = config.get("record", False)
        self.coop_time_threshold  = config.get("coop_time_threshold", 0.7)
        # Slow-motion factor — 1.0 = real-time, 0.1 = 10× slower
        self.time_scale: float    = config.get("time_scale", 1.0)

        self._agent_ids = {f"drone_{i}" for i in range(self.n_drones)}
        if hasattr(super(), "__init__"):
            super().__init__()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1, 0], dtype=np.float32),
            high=np.array([ 1,  1,  1, 1], dtype=np.float32),
            dtype=np.float32,
        )

        # Populated on reset()
        self._aviary: VelocityAviary | None = None
        self._tasks: list = []
        self._drone_task_map: dict[str, int | None] = {}
        self._step_count: int = 0
        self._tool_engaged: dict[str, bool] = {}
        self._task_object_ids: list[int] = []   # PyBullet body IDs for scene objects

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

        # Close previous aviary if one exists
        if self._aviary is not None:
            self._aviary.close()

        init_xyzs = self._get_init_positions()

        self._aviary = VelocityAviary(
            drone_model=DroneModel.CF2X,          # Crazyflie 2.X
            num_drones=self.n_drones,
            initial_xyzs=init_xyzs,
            physics=Physics.PYB,                  # standard PyBullet physics
            gui=self.gui,
            record=self.record,
            pyb_freq=240,                         # physics Hz (v2 API)
            ctrl_freq=48,                         # control Hz
            user_debug_gui=False,                 # hide propeller RPM sliders
        )

        # Position the camera to frame the whole 10×10m room from above-and-behind
        if self.gui:
            client = self._aviary.getPyBulletClient()
            cx = self.room_size[0] / 2   # room centre x  (5.0)
            cy = self.room_size[1] / 2   # room centre y  (5.0)
            p.resetDebugVisualizerCamera(
                cameraDistance=12,          # zoom: 12m back shows full 10×10 room
                cameraYaw=45,              # 45° side angle — diagonal overview
                cameraPitch=-40,           # looking slightly down
                cameraTargetPosition=[cx, cy, 1.0],
                physicsClientId=client,
            )

        # Instantiate task logic FIRST — _load_scene() iterates over self._tasks
        self._tasks = [
            _make_task(t, i, pos, steps)
            for i, (t, pos, steps) in enumerate(self.task_layouts)
        ]

        # Load household scene objects into the same PyBullet world
        self._task_object_ids = []
        self._load_scene()

        self._step_count = 0
        self._drone_task_map = {aid: None for aid in self._agent_ids}
        self._tool_engaged = {aid: False for aid in self._agent_ids}
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
        agent_ids_sorted = sorted(self._agent_ids)
        rewards = {aid: REWARD_STEP_ALIVE for aid in self._agent_ids}

        # --- Convert policy actions → VelocityAviary format ---
        # The policy outputs tanh-squashed values in [-1, 1] representing
        # displacement per step in HomeEnv (MAX_SPEED=0.5 m/step).
        # VelocityAviary expects velocity commands in m/s at ctrl_freq=48 Hz.
        # Scale factor: HomeEnv step ≈ 0.5m, ctrl period = 1/48s
        # → target_vel = action * MAX_SPEED * ctrl_freq = action * 0.5 * 48 = action * 24
        # Clamp to a reasonable physical max (3 m/s) so the Crazyflie doesn't overshoot.
        VEL_SCALE = 3.0   # m/s per unit action — matches Crazyflie's practical speed range
        vel_cmds = np.zeros((self.n_drones, 4), dtype=np.float32)
        for i, aid in enumerate(agent_ids_sorted):
            action = action_dict.get(aid, np.zeros(self.ACT_DIM))
            vel_cmds[i, :3] = np.clip(action[:3] * VEL_SCALE, -VEL_SCALE, VEL_SCALE)
            vel_cmds[i, 3]  = 0.0          # keep yaw fixed
            self._tool_engaged[aid] = float(action[3]) > 0.5

        # --- Step the physics ---
        _obs_pb, _rew_pb, terminated_pb, truncated_pb, _info_pb = self._aviary.step(vel_cmds)

        # --- Collision detection from PyBullet contact points ---
        drone_ids = [self._aviary.getDroneIds()[i] for i in range(self.n_drones)]
        for i in range(self.n_drones):
            for j in range(i + 1, self.n_drones):
                contacts = p.getContactPoints(
                    drone_ids[i], drone_ids[j],
                    physicsClientId=self._aviary.getPyBulletClient()
                )
                if contacts:
                    rewards[agent_ids_sorted[i]] += REWARD_COLLISION
                    rewards[agent_ids_sorted[j]] += REWARD_COLLISION

        # --- Task progress using real PyBullet positions ---
        real_positions = self._aviary.pos   # shape (n_drones, 3)
        for i, aid in enumerate(agent_ids_sorted):
            task_idx = self._drone_task_map.get(aid)
            if task_idx is None:
                continue
            task = self._tasks[task_idx]
            if task.completed:
                self._drone_task_map[aid] = None
                self._assign_tasks()
                continue

            delta = task.step(real_positions[i], self._tool_engaged[aid])
            rewards[aid] += delta

            if task.completed:
                rewards[aid] += task.completion_reward()
                self._update_task_visual(task_idx, completed=True)
                self._drone_task_map[aid] = None
                self._assign_tasks()

        # --- Cooperative bonus ---
        if all(t.completed for t in self._tasks):
            if self._step_count < self.max_steps * self.coop_time_threshold:
                for aid in self._agent_ids:
                    rewards[aid] += REWARD_COOP_BONUS

        # --- Termination ---
        all_tasks_done = all(t.completed for t in self._tasks)
        time_limit     = self._step_count >= self.max_steps
        # VelocityAviary returns numpy arrays — use .any() to safely collapse to bool
        pb_done = bool(np.asarray(terminated_pb).any()) or bool(np.asarray(truncated_pb).any())

        terminated = {aid: all_tasks_done or pb_done for aid in self._agent_ids}
        truncated  = {aid: time_limit for aid in self._agent_ids}
        terminated["__all__"] = all_tasks_done or pb_done
        truncated["__all__"]  = time_limit

        obs   = self._build_obs_dict()
        infos = self._build_info_dict()
        return obs, rewards, terminated, truncated, infos

    def close(self):
        if self._aviary is not None:
            self._aviary.close()
            self._aviary = None

    # ------------------------------------------------------------------
    # Scene loading
    # ------------------------------------------------------------------

    def _load_scene(self):
        """
        Load URDF objects for each task target into the PyBullet world.
        Falls back to a coloured sphere marker if the URDF file is missing.
        """
        client = self._aviary.getPyBulletClient()

        # Floor plane is already loaded by VelocityAviary.
        # Load room walls as thin boxes.
        self._load_walls(client)

        urdf_map = {
            "water_plant":  "plant_pot.urdf",
            "sweep_floor":  "floor_zone.urdf",
            "toggle_light": "light_switch.urdf",
        }

        for task in self._tasks:
            urdf_file = _ASSETS_DIR / urdf_map[task.spec.task_type]
            pos = task.spec.target_position.tolist()

            if urdf_file.exists():
                body_id = p.loadURDF(
                    str(urdf_file),
                    basePosition=pos,
                    useFixedBase=True,          # scene props don't fall or move
                    flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES,
                    physicsClientId=client,
                )
            else:
                # Fallback: visual-only sphere marker
                body_id = self._create_sphere_marker(
                    pos,
                    colour=_TASK_COLOURS[task.spec.task_type],
                    client=client,
                )

            self._task_object_ids.append(body_id)

    def _load_walls(self, client: int):
        """Load 4 thin box walls around the room perimeter."""
        w, d, h = self.room_size
        half_t = 0.05  # wall half-thickness
        walls = [
            # (position,              half-extents)
            ([w / 2, -half_t, h / 2], [w / 2, half_t, h / 2]),  # south
            ([w / 2, d + half_t, h / 2], [w / 2, half_t, h / 2]),  # north
            ([-half_t, d / 2, h / 2], [half_t, d / 2, h / 2]),  # west
            ([w + half_t, d / 2, h / 2], [half_t, d / 2, h / 2]),  # east
        ]
        for pos, half_ext in walls:
            col_id = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=half_ext, physicsClientId=client
            )
            vis_id = p.createVisualShape(
                p.GEOM_BOX, halfExtents=half_ext,
                rgbaColor=[0.85, 0.85, 0.85, 0.4],
                physicsClientId=client,
            )
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col_id,
                baseVisualShapeIndex=vis_id,
                basePosition=pos,
                physicsClientId=client,
            )

    def _create_sphere_marker(
        self, pos: list, colour: list, radius: float = 0.15, client: int = 0
    ) -> int:
        """Create a visual-only sphere (no collision) as a task target marker."""
        vis_id = p.createVisualShape(
            p.GEOM_SPHERE, radius=radius,
            rgbaColor=colour,
            physicsClientId=client,
        )
        return p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vis_id,
            basePosition=pos,
            physicsClientId=client,
        )

    def _update_task_visual(self, task_idx: int, completed: bool):
        """Turn completed task marker grey."""
        if task_idx >= len(self._task_object_ids):
            return
        body_id = self._task_object_ids[task_idx]
        client  = self._aviary.getPyBulletClient()
        p.changeVisualShape(
            body_id, -1,
            rgbaColor=[0.5, 0.5, 0.5, 0.4],
            physicsClientId=client,
        )

    # ------------------------------------------------------------------
    # Observation builder (matches HomeEnv's 15-dim layout exactly)
    # ------------------------------------------------------------------

    def _build_obs_dict(self) -> dict[str, np.ndarray]:
        agent_ids_sorted = sorted(self._agent_ids)
        real_positions = self._aviary.pos        # (n_drones, 3)
        real_velocities = self._aviary.vel       # (n_drones, 3)
        obs = {}

        for i, aid in enumerate(agent_ids_sorted):
            task_idx = self._drone_task_map.get(aid)
            task_target = (
                self._tasks[task_idx].spec.target_position
                if task_idx is not None and not self._tasks[task_idx].completed
                else None
            )

            # Find nearest neighbour position
            neighbour_pos = None
            best_dist = float("inf")
            for j, other_aid in enumerate(agent_ids_sorted):
                if j == i:
                    continue
                d = float(np.linalg.norm(real_positions[i] - real_positions[j]))
                if d < best_dist:
                    best_dist = d
                    neighbour_pos = real_positions[j]

            o = np.zeros(self.OBS_DIM, dtype=np.float32)
            o[0:3]  = real_positions[i]
            o[3]    = 1.0   # no battery model in pybullet env — always full
            task_progress = (
                self._tasks[task_idx].engage_steps_done
                / self._tasks[task_idx].spec.engage_steps_required
                if task_idx is not None else 0.0
            )
            o[4]    = task_progress
            if task_target is not None:
                o[5:8] = task_target - real_positions[i]
            o[8:11] = real_velocities[i]
            o[11]   = float(self._tool_engaged.get(aid, False))
            if neighbour_pos is not None:
                o[12:15] = neighbour_pos - real_positions[i]

            obs[aid] = o

        return obs

    def _build_info_dict(self) -> dict[str, Any]:
        tasks_completed = sum(1 for t in self._tasks if t.completed)
        return {
            aid: {
                "tasks_completed": tasks_completed,
                "tasks_total": len(self._tasks),
                "step": self._step_count,
            }
            for aid in self._agent_ids
        }

    # ------------------------------------------------------------------
    # Task assignment
    # ------------------------------------------------------------------

    def _assign_tasks(self):
        claimed = {idx for idx in self._drone_task_map.values() if idx is not None}
        pending = [
            i for i, t in enumerate(self._tasks)
            if not t.completed and i not in claimed
        ]
        for aid in sorted(self._agent_ids):
            if self._drone_task_map[aid] is None and pending:
                task_idx = pending.pop(0)
                self._drone_task_map[aid] = task_idx
                self._tasks[task_idx].assigned_drone_id = aid

    def _get_init_positions(self) -> np.ndarray:
        """Spread drones along one wall at hover height."""
        spacing = self.room_size[0] / (self.n_drones + 1)
        return np.array(
            [[(i + 1) * spacing, 0.5, 1.0] for i in range(self.n_drones)],
            dtype=np.float64,
        )
