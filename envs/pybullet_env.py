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

from allocator.base_allocator import BaseAllocator, WorldSnapshot, AllocationResult, Bid

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
from envs.tasks.base_task import TaskSpec, TaskStatus


# ---------------------------------------------------------------------------
# Built-in greedy fallback — mirrors HomeEnv._GreedyFallbackAllocator so
# PybulletHomeEnv works identically to the old hard-coded _assign_tasks when
# no external allocator is supplied.
# ---------------------------------------------------------------------------

class _GreedyFallbackAllocator(BaseAllocator):
    def allocate(self, snapshot: WorldSnapshot) -> AllocationResult:
        active = {TaskStatus.PENDING, TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS}
        claimed: set[int] = {
            idx for idx in snapshot.current_assignments.values()
            if idx is not None
        }
        pending = [
            i for i, t in enumerate(snapshot.tasks)
            if not t.completed and t.status in active and i not in claimed
        ]
        assignments: dict[str, int | None] = {}
        for drone_id in sorted(snapshot.drone_positions):
            existing = snapshot.current_assignments.get(drone_id)
            if (
                existing is not None
                and existing < len(snapshot.tasks)
                and not snapshot.tasks[existing].completed
            ):
                assignments[drone_id] = existing
            elif pending:
                assignments[drone_id] = pending.pop(0)
            else:
                assignments[drone_id] = None
        return AllocationResult(assignments=assignments)

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
    ("water_plant",  [2.0, 3.0, 1.0], 10),
    ("water_plant",  [7.0, 1.5, 1.0], 10),
    ("sweep_floor",  [5.0, 5.0, 1.0], 15),  # z=1.0: drone cruising altitude — no descent needed
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
    "water_plant":  [0.0, 0.7, 0.15, 1.0],  # green
    "sweep_floor":  [0.8, 0.5, 0.1,  1.0],  # brown
    "toggle_light": [1.0, 0.85, 0.0, 1.0],  # yellow
}

# Task short names shown in the floating label
_TASK_LABELS = {
    "water_plant":  "Water",
    "sweep_floor":  "Sweep",
    "toggle_light": "Light",
}

# Per-drone body colours — vivid, distinct, semi-transparent so the scene shows through
_DRONE_COLOURS = [
    [0.15, 0.65, 1.00, 0.85],   # 0  sky-blue
    [1.00, 0.45, 0.05, 0.85],   # 1  orange
    [0.20, 0.90, 0.35, 0.85],   # 2  lime
    [0.90, 0.15, 0.80, 0.85],   # 3  magenta
    [1.00, 0.95, 0.10, 0.85],   # 4  yellow
    [0.10, 0.85, 0.90, 0.85],   # 5  cyan
    [1.00, 0.25, 0.25, 0.85],   # 6  red
    [0.70, 0.40, 1.00, 0.85],   # 7  violet
]

# Disc radius (m) used for the visual-only drone body overlay.
# Sized for a 10×10 m room — large enough to see at a glance.
_DRONE_VISUAL_RADIUS = 0.45


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
        self.n_drones: int           = config.get("n_drones", 3)
        self.room_size: list         = config.get("room_size", [10.0, 10.0, 3.0])
        self.max_steps: int          = config.get("max_steps", 500)
        self.task_layouts: list      = config.get("task_layouts", _DEFAULT_TASK_LAYOUTS)
        self.gui: bool               = config.get("gui", True)
        self.record: bool            = config.get("record", False)
        self.coop_time_threshold     = config.get("coop_time_threshold", 0.7)
        self.time_scale: float       = config.get("time_scale", 1.0)
        # Pluggable auction allocator (same interface as HomeEnv)
        self.allocator: BaseAllocator = config.get(
            "allocator", _GreedyFallbackAllocator()
        )
        # Re-run the auction every this many steps (0 = only on task completion)
        self.auction_interval: int    = config.get("auction_interval", 0)

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

        # Episode state — populated/reset in reset()
        self._tasks: list = []
        self._drone_task_map: dict[str, int | None] = {}
        self._step_count: int = 0
        self._tool_engaged: dict[str, bool] = {}
        self._task_object_ids: list[int] = []
        self._init_xyzs = self._get_init_positions()

        # Debug overlay state (GUI only) — item IDs so we can move/replace them
        self._debug_drone_labels: dict[str, int] = {}  # aid → debug text item id
        self._debug_task_lines: dict[str, int] = {}    # aid → debug line item id
        self._debug_hud_id: int = -1                    # single HUD text item
        self._debug_key_hint_id: int = -1               # key-bindings reminder text
        # Bid visualisation — one line per (drone, task) pair, keyed by (aid, task_idx)
        self._debug_bid_lines: dict[tuple[str, int], int] = {}
        # Most recent bids from the last allocation round (empty until first alloc)
        self._last_bids: list[Bid] = []

        # Visual-override shape IDs — large coloured discs that replace the tiny
        # Crazyflie mesh so drones are visible at room scale.  Rebuilt after each
        # reset() because p.resetSimulation() wipes all visual shape registrations.
        self._drone_visual_shape_ids: list[int] = []    # one per drone

        # ── Camera state (GUI only) ───────────────────────────────────────────
        # These are updated each step by _handle_camera_keys() so keyboard/mouse
        # controls work without touching PyBullet's native mouse orbit.
        cx, cy = self.room_size[0] / 2, self.room_size[1] / 2
        self._cam_dist:   float = 13.0
        self._cam_yaw:    float = 30.0
        self._cam_pitch:  float = -40.0
        self._cam_target: list  = [cx, cy, 1.0]
        self._follow_drone_idx: int = -1   # -1 = free camera, 0–N = follow drone N

        # Create the aviary ONCE — opens the OS window a single time.
        # Each episode calls aviary.reset() which calls p.resetSimulation()
        # internally, then we reload the lightweight scene objects.
        self._aviary = VelocityAviary(
            drone_model=DroneModel.CF2X,
            num_drones=self.n_drones,
            initial_xyzs=self._init_xyzs,
            physics=Physics.PYB,
            gui=self.gui,
            record=self.record,
            pyb_freq=240,
            ctrl_freq=48,
            user_debug_gui=False,  # disable RPM sliders — they get wiped by
                                   # p.resetSimulation() on every reset() but are
                                   # never re-created, causing "Failed to read
                                   # parameter" crashes from episode 2 onward.
                                   # Our env adds its own debug overlays instead.
        )

        # VelocityAviary hard-codes SPEED_LIMIT = 0.03 × MAX_SPEED_KMH = 0.25 m/s.
        # 20% of max (1.67 m/s) is the sweet spot: fast enough to reach all targets
        # within 800 steps, slow enough that the Crazyflie PID maintains altitude
        # without the roll-induced descent that causes floor crashes at higher speeds.
        self._aviary.SPEED_LIMIT = 0.20 * self._aviary.MAX_SPEED_KMH * (1000 / 3600)

        # Camera is set after aviary.__init__ finishes its own GUI setup
        if self.gui:
            self._set_camera()

    # ------------------------------------------------------------------
    # Gymnasium / RLlib interface
    # ------------------------------------------------------------------

    def _set_camera(self):
        """Apply current camera state to the PyBullet visualiser and configure display."""
        client = self._aviary.getPyBulletClient()
        p.resetDebugVisualizerCamera(
            cameraDistance=self._cam_dist,
            cameraYaw=self._cam_yaw,
            cameraPitch=self._cam_pitch,
            cameraTargetPosition=self._cam_target,
            physicsClientId=client,
        )
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS,           1, physicsClientId=client)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=client)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING,      1, physicsClientId=client)

    # Preset camera views  (dist, yaw, pitch, target)
    _CAM_PRESETS = [
        (13.0,  30.0, -40.0, None),   # 1 — default isometric
        (16.0,   0.0, -90.0, None),   # 2 — top-down
        (14.0,   0.0, -20.0, None),   # 3 — north side
        (14.0,  90.0, -20.0, None),   # 4 — west side
        ( 8.0,  30.0, -15.0, None),   # 5 — low cinematic
    ]

    def _handle_camera_keys(self, real_positions: np.ndarray):
        """
        Read PyBullet keyboard events and update the camera each step.

        Controls
        ────────
        W / S       — pan camera forward / back (along target point)
        A / D       — pan camera left / right
        Q / E       — zoom in / out
        R / F       — tilt up / down (pitch)
        Z / X       — rotate left / right (yaw)
        1–5         — snap to preset view
        0           — follow next drone (cycles through drones, then free)
        """
        client = self._aviary.getPyBulletClient()
        keys = p.getKeyboardEvents(physicsClientId=client)

        # Key codes
        KEY_W = ord('w'); KEY_S = ord('s')
        KEY_A = ord('a'); KEY_D = ord('d')
        KEY_Q = ord('q'); KEY_E = ord('e')
        KEY_R = ord('r'); KEY_F = ord('f')
        KEY_Z = ord('z'); KEY_X = ord('x')
        KEY_0 = ord('0')

        PAN   = 0.15   # metres per step
        ZOOM  = 0.25   # metres per step
        ROT   = 1.5    # degrees per step
        PITCH = 1.5    # degrees per step

        pressed = lambda k: keys.get(k, 0) & p.KEY_IS_DOWN

        # ── Follow-drone mode ─────────────────────────────────────────────
        # Pressing 0 cycles: free → drone_0 → drone_1 → … → free
        if keys.get(KEY_0, 0) & p.KEY_WAS_TRIGGERED:
            self._follow_drone_idx = (self._follow_drone_idx + 1) % (self.n_drones + 1)
            if self._follow_drone_idx == self.n_drones:
                self._follow_drone_idx = -1  # back to free

        if self._follow_drone_idx >= 0 and self._follow_drone_idx < len(real_positions):
            # Lock target onto chosen drone, allow zoom/pitch while following
            pos = real_positions[self._follow_drone_idx]
            self._cam_target = [float(pos[0]), float(pos[1]), float(pos[2]) + 0.3]

        # ── Preset views ──────────────────────────────────────────────────
        cx, cy = self.room_size[0] / 2, self.room_size[1] / 2
        for i, preset in enumerate(self._CAM_PRESETS):
            if keys.get(ord(str(i + 1)), 0) & p.KEY_WAS_TRIGGERED:
                self._cam_dist, self._cam_yaw, self._cam_pitch, _ = preset
                self._cam_target = [cx, cy, 1.0]
                self._follow_drone_idx = -1

        # ── Free camera movement (skip if following a drone) ─────────────
        if self._follow_drone_idx < 0:
            # Pan: move target in the horizontal plane aligned with yaw
            yaw_r = np.deg2rad(self._cam_yaw)
            fwd   = np.array([ np.cos(yaw_r),  np.sin(yaw_r), 0.0])
            right = np.array([-np.sin(yaw_r),  np.cos(yaw_r), 0.0])
            if pressed(KEY_W): self._cam_target = (np.array(self._cam_target) + fwd   * PAN).tolist()
            if pressed(KEY_S): self._cam_target = (np.array(self._cam_target) - fwd   * PAN).tolist()
            if pressed(KEY_A): self._cam_target = (np.array(self._cam_target) - right * PAN).tolist()
            if pressed(KEY_D): self._cam_target = (np.array(self._cam_target) + right * PAN).tolist()

        # ── Zoom / rotate / pitch (always available) ──────────────────────
        if pressed(KEY_Q): self._cam_dist  = max(1.0,   self._cam_dist  - ZOOM)
        if pressed(KEY_E): self._cam_dist  = min(25.0,  self._cam_dist  + ZOOM)
        if pressed(KEY_Z): self._cam_yaw  -= ROT
        if pressed(KEY_X): self._cam_yaw  += ROT
        if pressed(KEY_R): self._cam_pitch = min(-5.0,  self._cam_pitch + PITCH)
        if pressed(KEY_F): self._cam_pitch = max(-89.0, self._cam_pitch - PITCH)

        # Apply
        p.resetDebugVisualizerCamera(
            cameraDistance=self._cam_dist,
            cameraYaw=self._cam_yaw,
            cameraPitch=self._cam_pitch,
            cameraTargetPosition=self._cam_target,
            physicsClientId=client,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)

        # aviary.reset() calls p.resetSimulation() which wipes the physics world
        # and respawns drones at initial_xyzs with zeroed velocities/state.
        self._aviary.reset()

        # p.resetSimulation() invalidates all debug item IDs — reset the caches
        # so _update_debug_visuals creates fresh items instead of trying to
        # replace items that no longer exist (which causes "User debug draw failed").
        self._debug_drone_labels = {}
        self._debug_task_lines   = {}
        self._debug_hud_id       = -1
        self._debug_key_hint_id  = -1
        self._debug_bid_lines    = {}
        self._last_bids          = []

        # resetSimulation removed our scene objects — reload them
        self._task_object_ids = []
        self._tasks = [
            _make_task(t, i, pos, steps)
            for i, (t, pos, steps) in enumerate(self.task_layouts)
        ]
        self._load_scene()

        # Re-apply camera (resetSimulation resets the debug visualiser too)
        if self.gui:
            self._set_camera()
            # Rebuild drone visual overrides (wipes happen with resetSimulation)
            self._apply_drone_visuals()

        self._step_count = 0
        self._drone_task_map = {aid: None for aid in self._agent_ids}
        self._tool_engaged   = {aid: False for aid in self._agent_ids}
        self._assign_tasks(self._aviary.pos)

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
        # VelocityAviary._preprocessAction() interprets action[k] as:
        #   action[k, 0:3] = desired velocity direction vector (will be normalised)
        #   action[k, 3]   = speed scalar in [0, 1] → target_vel = SPEED_LIMIT * speed * direction
        #
        # The policy outputs:
        #   action[:3] = tanh-squashed displacement direction in [-1, 1]
        #   action[3]  = sigmoid tool-engage signal in [0, 1]
        #
        # Mapping: pass policy[:3] as the direction, use its magnitude as speed.
        # When the drone is moving toward a target, ||action[:3]|| ≈ 1 → full speed.
        # tool_engage is tracked separately and NOT passed as speed.
        vel_cmds = np.zeros((self.n_drones, 4), dtype=np.float32)
        for i, aid in enumerate(agent_ids_sorted):
            action = action_dict.get(aid, np.zeros(self.ACT_DIM))
            direction = action[:3].astype(np.float32)
            norm = float(np.linalg.norm(direction))
            vel_cmds[i, :3] = direction                     # VelocityAviary normalises this internally
            # Scale speed proportionally to action magnitude so gentle corrections
            # don't snap to zero.  Dead-band is 0.01 to suppress pure noise.
            vel_cmds[i, 3]  = float(np.clip(norm, 0.0, 1.0)) if norm > 0.01 else 0.0
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

        # --- Periodic auction tick (runs before task progress so new assignments
        #     take effect this step, matching HomeEnv's allocate-then-step order) ---
        real_positions = self._aviary.pos   # shape (n_drones, 3)
        if self.auction_interval > 0 and self._step_count % self.auction_interval == 0:
            self._assign_tasks(real_positions)

        # --- Task progress using real PyBullet positions ---
        for i, aid in enumerate(agent_ids_sorted):
            task_idx = self._drone_task_map.get(aid)
            if task_idx is None:
                continue
            task = self._tasks[task_idx]
            if task.completed:
                self._drone_task_map[aid] = None
                self._assign_tasks(real_positions)
                continue

            delta = task.step(real_positions[i], self._tool_engaged[aid])
            rewards[aid] += delta

            if task.completed:
                rewards[aid] += task.completion_reward()
                self._update_task_visual(task_idx, completed=True)
                self._drone_task_map[aid] = None
                self._assign_tasks(real_positions)

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

        if self.gui:
            self._handle_camera_keys(real_positions)
            self._update_debug_visuals(real_positions, agent_ids_sorted)
            self._draw_bid_lines(real_positions, agent_ids_sorted)

        return obs, rewards, terminated, truncated, infos

    def _apply_drone_visuals(self):
        """
        Replace the tiny Crazyflie mesh on each drone body with a large coloured
        flat disc so drones are clearly visible in a 10 m room.

        Uses changeVisualShape(shapeIndex=…) which swaps the rendered geometry
        without touching any physics body, mass, or inertia.  Must be called after
        every aviary.reset() because p.resetSimulation() wipes the shape registry.
        """
        client = self._aviary.getPyBulletClient()
        drone_ids = self._aviary.getDroneIds()
        self._drone_visual_shape_ids = []

        for i in range(self.n_drones):
            colour = _DRONE_COLOURS[i % len(_DRONE_COLOURS)]
            # Flat horizontal disc — visually obvious from above/side
            vs = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=_DRONE_VISUAL_RADIUS,
                length=0.06,
                rgbaColor=colour,
                physicsClientId=client,
            )
            p.changeVisualShape(
                drone_ids[i], -1,
                shapeIndex=vs,
                physicsClientId=client,
            )
            self._drone_visual_shape_ids.append(vs)

    def _update_debug_visuals(
        self,
        real_positions: np.ndarray,
        agent_ids_sorted: list[str],
    ):
        """Refresh per-drone labels, task-target lines, and HUD text each step."""
        client = self._aviary.getPyBulletClient()

        # ── Per-drone floating label ──────────────────────────────────────────
        for i, aid in enumerate(agent_ids_sorted):
            pos = real_positions[i].tolist()
            label_pos = [pos[0], pos[1], pos[2] + 0.55]

            task_idx = self._drone_task_map.get(aid)
            drone_colour = _DRONE_COLOURS[i % len(_DRONE_COLOURS)][:3]

            if task_idx is not None and not self._tasks[task_idx].completed:
                task      = self._tasks[task_idx]
                task_type = task.spec.task_type
                task_name = _TASK_LABELS.get(task_type, task_type)

                if isinstance(task, SweepFloorTask):
                    filled = task._current_wp_idx
                    total  = len(task._waypoints)
                else:
                    filled = task.engage_steps_done
                    total  = task.spec.engage_steps_required

                # Unicode block progress bar: e.g. "████░░░░  4/10"
                bar_width = 8
                n_filled  = round(filled / max(total, 1) * bar_width)
                bar       = "\u2588" * n_filled + "\u2591" * (bar_width - n_filled)
                label     = f"Drone {i}  {task_name}\n{bar}  {filled}/{total}"
                text_colour = _TASK_COLOURS[task_type][:3]
            else:
                label       = f"Drone {i}  IDLE"
                text_colour = drone_colour

            prev = self._debug_drone_labels.get(aid, -1)
            item_id = p.addUserDebugText(
                label,
                label_pos,
                textColorRGB=text_colour,
                textSize=1.6,
                replaceItemUniqueId=prev if prev != -1 else -1,
                physicsClientId=client,
            )
            self._debug_drone_labels[aid] = item_id

        # ── Drone → target line ───────────────────────────────────────────────
        for i, aid in enumerate(agent_ids_sorted):
            task_idx = self._drone_task_map.get(aid)
            prev_line = self._debug_task_lines.get(aid, -1)

            if task_idx is not None and not self._tasks[task_idx].completed:
                task      = self._tasks[task_idx]
                task_type = task.spec.task_type
                # For sweep, point to the active waypoint not the zone centre
                if isinstance(task, SweepFloorTask):
                    target = task.current_target.tolist()
                else:
                    target = task.spec.target_position.tolist()
                drone_pos   = real_positions[i].tolist()
                line_colour = _TASK_COLOURS[task_type][:3]
                item_id = p.addUserDebugLine(
                    drone_pos, target,
                    lineColorRGB=line_colour,
                    lineWidth=3.0,
                    replaceItemUniqueId=prev_line if prev_line != -1 else -1,
                    physicsClientId=client,
                )
                self._debug_task_lines[aid] = item_id
            elif prev_line != -1:
                p.removeUserDebugItem(prev_line, physicsClientId=client)
                self._debug_task_lines[aid] = -1

        # ── HUD: step + task summary ──────────────────────────────────────────
        done_count  = sum(1 for t in self._tasks if t.completed)
        total_tasks = len(self._tasks)
        bar_str = "\u2588" * done_count + "\u2591" * (total_tasks - done_count)
        follow_str = (
            f"  |  Cam: drone_{self._follow_drone_idx}"
            if self._follow_drone_idx >= 0
            else ""
        )
        hud = (
            f"Step {self._step_count:4d} / {self.max_steps}{follow_str}\n"
            f"Tasks  {bar_str}  {done_count}/{total_tasks}"
        )
        self._debug_hud_id = p.addUserDebugText(
            hud,
            [0.2, 0.2, 2.85],
            textColorRGB=[1.0, 1.0, 0.2],
            textSize=2.0,
            replaceItemUniqueId=self._debug_hud_id if self._debug_hud_id != -1 else -1,
            physicsClientId=client,
        )

        # ── Key-bindings hint (bottom-left corner, small) ─────────────────────
        hint = (
            "WASD pan  Q/E zoom  Z/X rotate  R/F pitch\n"
            "1-5 preset views  0 follow drone  mouse drag orbits"
        )
        self._debug_key_hint_id = p.addUserDebugText(
            hint,
            [0.2, 0.2, 0.08],
            textColorRGB=[0.7, 0.7, 0.7],
            textSize=0.9,
            replaceItemUniqueId=self._debug_key_hint_id if self._debug_key_hint_id != -1 else -1,
            physicsClientId=client,
        )

    def _draw_bid_lines(
        self,
        real_positions: np.ndarray,
        agent_ids_sorted: list[str],
    ) -> None:
        """
        Draw (or update) one thin debug line per bid in ``self._last_bids``.

        Colour encodes normalised bid value:
          0.0  →  red   [1.0, 0.1, 0.1]
          0.5  →  yellow [1.0, 1.0, 0.1]
          1.0  →  green  [0.1, 1.0, 0.1]

        Only the top-N bids per drone are shown (N = BID_LINES_PER_DRONE) to
        avoid visual clutter when there are many tasks.  Lines for (drone, task)
        pairs that have fallen out of the current bid list are removed.
        """
        if not self._last_bids or not self.gui:
            return

        client = self._aviary.getPyBulletClient()

        # Normalise bid values across the current round so the colour span is full
        vals = [b.bid_value for b in self._last_bids]
        lo, hi = min(vals), max(vals)
        span = hi - lo if hi > lo else 1.0

        def _bid_colour(v: float) -> list[float]:
            """Linear red→yellow→green for t ∈ [0, 1]."""
            t = (v - lo) / span
            if t < 0.5:
                # red → yellow
                s = t * 2.0
                return [1.0, s, 0.1]
            else:
                # yellow → green
                s = (t - 0.5) * 2.0
                return [1.0 - s, 1.0, 0.1]

        # Keep only the highest-value bid per drone (one line = current winner)
        best_bid_per_drone: dict[str, "Bid"] = {}
        for b in self._last_bids:
            prev = best_bid_per_drone.get(b.drone_id)
            if prev is None or b.bid_value > prev.bid_value:
                best_bid_per_drone[b.drone_id] = b

        active_keys: set[tuple[str, int]] = set()

        for drone_id, b in best_bid_per_drone.items():
            task_idx = b.task_idx
            if task_idx >= len(self._tasks) or self._tasks[task_idx].completed:
                continue

            task = self._tasks[task_idx]
            task_target: list[float]
            if isinstance(task, SweepFloorTask):
                task_target = task.current_target.tolist()
            else:
                task_target = task.spec.target_position.tolist()

            # Drone position: use aviary real position when available
            try:
                di = agent_ids_sorted.index(drone_id)
                drone_pos = real_positions[di].tolist()
            except ValueError:
                continue

            colour = _bid_colour(b.bid_value)
            key = (drone_id, task_idx)
            prev_id = self._debug_bid_lines.get(key, -1)
            item_id = p.addUserDebugLine(
                drone_pos,
                task_target,
                lineColorRGB=colour,
                lineWidth=1.5,
                replaceItemUniqueId=prev_id if prev_id != -1 else -1,
                physicsClientId=client,
            )
            self._debug_bid_lines[key] = item_id
            active_keys.add(key)

        # Remove stale lines for (drone, task) pairs no longer in the top bids
        stale = [k for k in self._debug_bid_lines if k not in active_keys]
        for key in stale:
            item_id = self._debug_bid_lines.pop(key)
            if item_id != -1:
                try:
                    p.removeUserDebugItem(item_id, physicsClientId=client)
                except Exception:
                    pass  # item may already be gone after resetSimulation

    def close(self):
        if self._aviary is not None:
            self._aviary.close()
            self._aviary = None

    def _restore_task_visuals(self):
        """Restore task marker colours after a reset (completed tasks were greyed)."""
        if not self._task_object_ids:
            return
        client = self._aviary.getPyBulletClient()
        for i, task in enumerate(self._tasks):
            if i >= len(self._task_object_ids):
                break
            colour = _TASK_COLOURS.get(task.spec.task_type, [0.5, 0.5, 1.0, 1.0])
            p.changeVisualShape(
                self._task_object_ids[i], -1,
                rgbaColor=colour,
                physicsClientId=client,
            )

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
        # Overlay a coloured floor tile so the room looks like a real interior.
        self._load_floor_tile(client)
        # Ceiling
        self._load_ceiling(client)
        # Room walls as thin boxes
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

    def _load_floor_tile(self, client: int):
        """Warm cream floor tile overlaid on PyBullet's grey plane."""
        w, d = self.room_size[0], self.room_size[1]
        vis_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[w / 2, d / 2, 0.005],
            rgbaColor=[0.92, 0.88, 0.78, 1.0],   # warm cream
            physicsClientId=client,
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vis_id,
            basePosition=[w / 2, d / 2, 0.005],
            physicsClientId=client,
        )

    def _load_ceiling(self, client: int):
        """Translucent white ceiling so the room feels enclosed."""
        w, d, h = self.room_size
        vis_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[w / 2, d / 2, 0.02],
            rgbaColor=[1.0, 1.0, 1.0, 0.25],   # translucent white
            physicsClientId=client,
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vis_id,
            basePosition=[w / 2, d / 2, h],
            physicsClientId=client,
        )

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
                rgbaColor=[0.90, 0.88, 0.84, 0.70],   # warm off-white plaster
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
        self, pos: list, colour: list, radius: float = 0.30, client: int = 0
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

            # Use the task's *active* target — SweepFloorTask advances through
            # waypoints, so current_target differs from spec.target_position.
            task_target = None
            task_progress = 0.0
            if task_idx is not None and not self._tasks[task_idx].completed:
                task = self._tasks[task_idx]
                task_target = (
                    task.current_target
                    if isinstance(task, SweepFloorTask)
                    else task.spec.target_position
                )
                # Progress: waypoint fraction for sweep, engage fraction for others
                if isinstance(task, SweepFloorTask):
                    task_progress = task._current_wp_idx / len(task._waypoints)
                else:
                    task_progress = (
                        task.engage_steps_done / task.spec.engage_steps_required
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
    # Task assignment — auction-backed
    # ------------------------------------------------------------------

    def _assign_tasks(self, real_positions: np.ndarray | None = None) -> None:
        """
        Build a WorldSnapshot from current sim state, run the allocator,
        and apply the resulting assignments.

        ``real_positions`` is the aviary.pos array (n_drones × 3).  If None
        (only during the very first reset() call before the aviary exists), we
        fall back to the static initial positions.
        """
        agent_ids_sorted = sorted(self._agent_ids)

        # Drone positions: prefer live physics positions; fall back to init grid
        if real_positions is not None:
            positions = {
                aid: real_positions[i].astype(np.float32)
                for i, aid in enumerate(agent_ids_sorted)
            }
        else:
            positions = {
                aid: self._init_xyzs[i].astype(np.float32)
                for i, aid in enumerate(agent_ids_sorted)
            }

        # Battery: PybulletHomeEnv has no battery model — report full (1.0)
        batteries = {aid: 1.0 for aid in self._agent_ids}
        progress  = {aid: 0.0 for aid in self._agent_ids}

        # Task statuses: sync completed → COMPLETE so the allocator skips them
        for task in self._tasks:
            if task.completed and task.status != TaskStatus.COMPLETE:
                task.status = TaskStatus.COMPLETE

        snapshot = WorldSnapshot(
            drone_positions=positions,
            drone_batteries=batteries,
            drone_task_progress=progress,
            current_assignments=dict(self._drone_task_map),
            tasks=self._tasks,
            step=self._step_count,
            max_steps=self.max_steps,
        )

        result = self.allocator.allocate(snapshot)
        self._last_bids = result.bids  # stored for bid-line visualisation

        # Apply assignments
        # Clear all task assignee lists first (mirrors HomeEnv._apply_allocation)
        for task in self._tasks:
            task.assigned_drone_ids = []

        for aid, task_idx in result.assignments.items():
            self._drone_task_map[aid] = task_idx
            if task_idx is None:
                continue
            task = self._tasks[task_idx]
            if aid not in task.assigned_drone_ids:
                task.assigned_drone_ids.append(aid)
            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.ASSIGNED

    def _get_init_positions(self) -> np.ndarray:
        """Spread drones along one wall at hover height."""
        spacing = self.room_size[0] / (self.n_drones + 1)
        return np.array(
            [[(i + 1) * spacing, 0.5, 1.0] for i in range(self.n_drones)],
            dtype=np.float64,
        )
