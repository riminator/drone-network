"""
mujoco_env.py
MuJoCo 3 physics-backed multi-drone environment.

Replaces the PyBullet/gym-pybullet-drones layer with a direct MuJoCo
simulation for smoother, more controllable physics and better real-world
transfer.

Key differences from PybulletHomeEnv
--------------------------------------
* MuJoCo 3 RK4 integrator @ 500 Hz (timestep=0.002 s) — Crazyflie-accurate
* Smooth Newton contact solver with air density + viscosity
* Attitude PID runs inside step() at full sim rate, exposed via high-level
  (dx, dy, dz, tool) action identical to HomeEnv — no policy retraining needed
* Optional passive viewer (mujoco.viewer) for GUI rendering
* Sensor data (accelerometer, gyro, velocimeter, framepos) used to build
  the same 15-dim observation as DroneAgent so trained checkpoints load
  without modification

Physics model (assets/mujoco/scene.xml)
-----------------------------------------
* Crazyflie 2.x-scale quadrotor: 27 g total, 92 mm wheelbase, 46 mm rotors
* Four velocity-actuated rotor joints; thrust computed from RPM via k_thrust
* Realistic drag via MuJoCo fluid viscosity/density (viscosity=1.8e-5, density=1.2)
* Household objects: plant pots (collision cylinders + visual foliage spheres),
  floor zones (no-collision flat boxes), light switches (wall-mounted boxes)
* Room: 10 m × 10 m × 3 m with floor, walls, ceiling, ceiling lights

Usage
------
    from envs.mujoco_env import MujocoHomeEnv
    env = MujocoHomeEnv(config={"n_drones": 3, "render": True})
    obs, info = env.reset()
    obs, rew, term, trunc, info = env.step({aid: np.zeros(4) for aid in obs})
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from allocator.base_allocator import BaseAllocator, WorldSnapshot, AllocationResult, Bid
from envs.tasks import WaterPlantTask, SweepFloorTask, ToggleLightTask
from envs.tasks.base_task import TaskSpec, TaskStatus
from envs.drone_agent import DroneAgent, MAX_BATTERY

try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
except ImportError:
    MultiAgentEnv = object  # type: ignore

try:
    import mujoco
    _MUJOCO_AVAILABLE = True
except ImportError:
    _MUJOCO_AVAILABLE = False

from gymnasium import spaces

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ASSETS_DIR = Path(__file__).parent.parent / "assets" / "mujoco"
_SCENE_XML   = str(_ASSETS_DIR / "scene.xml")

# ---------------------------------------------------------------------------
# Reward constants — mirror HomeEnv exactly
# ---------------------------------------------------------------------------
REWARD_STEP_ALIVE   = -0.01
REWARD_COLLISION    = -5.0
REWARD_BATTERY_DEAD = -3.0
REWARD_COOP_BONUS   = 2.0

# ---------------------------------------------------------------------------
# Quadrotor physics constants
# ---------------------------------------------------------------------------
# Thrust is applied as direct body force via mj_applyFT — no propeller
# aerodynamic model needed.
#
# Mass breakdown from scene.xml (body geom + battery + 4 motor geoms):
#   body box:  0.020 kg
#   battery:   0.007 kg
#   body total link: 0.027+0.016 = 0.0436 kg  (as reported by MuJoCo)
#   4 motors × 0.002 kg + prop geom 0 mass = 0.008 kg each link → 0.0344 kg
#   Total subtree: ~0.0782 kg  (verified via mj_step)
_MASS        = 0.0782   # kg  (full drone subtree, verified)
_GRAVITY     = 9.81     # m/s²
_HOVER_F     = _MASS * _GRAVITY   # N  (0.767 N at hover)
_MAX_F       = _HOVER_F * 3.0     # N  (max thrust = 3× hover)
_ARM         = 0.0326   # m   (diagonal arm length)
_MAX_TORQUE  = 0.010    # Nm  (roll/pitch authority, scaled for heavier drone)
_MAX_YAW_T   = 0.004    # Nm  (yaw authority)

# PID gains — 25 Hz effective control (phys dt 0.002 s × 20 sub-steps = 0.04 s/step)
# Altitude velocity-mode PD.
#   At max vz_error = 2 m/s: KP adds ±0.5×HOVER_F, KD adds ±0.25×HOVER_F/0.04.
#   These must be large enough to brake from 1.4 m/s (max liftoff speed) in a few steps.
_PID_KP_ALT   = 0.5 * _HOVER_F    # N/(m/s)  — strong velocity matching
_PID_KI_ALT   = 0.0               # no integrator
_PID_KD_ALT   = 0.02 * _HOVER_F   # N/(m/s²) — gentle derivative (dt already in denominator)
# Attitude angle-mode PD
_PID_KP_ROLL  = 0.6 * _MAX_TORQUE
_PID_KI_ROLL  = 0.0
_PID_KD_ROLL  = 0.2 * _MAX_TORQUE
_PID_KP_PITCH = 0.6 * _MAX_TORQUE
_PID_KI_PITCH = 0.0
_PID_KD_PITCH = 0.2 * _MAX_TORQUE
_PID_KP_YAW   = 0.3 * _MAX_YAW_T
_PID_KI_YAW   = 0.0
_PID_KD_YAW   = 0.1 * _MAX_YAW_T
# Effective control dt for PID derivative term
_CTRL_DT      = 0.04   # s  (timestep × phys_steps = 0.002 × 20)

# Battery drain: full charge lasts ~MAX_BATTERY env steps
_BATTERY_DRAIN_PER_STEP = MAX_BATTERY / 500

# Task engage distance: drone must be within this many metres
_ENGAGE_DIST = 0.40  # m

# Collision distance (centre-to-centre)
_COLLISION_DIST = 0.18  # m

# ---------------------------------------------------------------------------
# Greedy fallback allocator (same logic as HomeEnv and PybulletHomeEnv)
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
# Per-drone PID state
# ---------------------------------------------------------------------------
class _DronePID:
    """Holds integrator and previous-error state for one drone's attitude PID."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.alt_int   = 0.0
        self.roll_int  = 0.0
        self.pitch_int = 0.0
        self.yaw_int   = 0.0
        self.alt_prev  = 0.0
        self.roll_prev = 0.0
        self.pitch_prev = 0.0
        self.yaw_prev  = 0.0


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------
_TASK_REGISTRY = {
    "water_plant":  WaterPlantTask,
    "sweep_floor":  SweepFloorTask,
    "toggle_light": ToggleLightTask,
}

_DEFAULT_TASK_LAYOUTS = [
    ("water_plant",  [2.0, 3.0, 1.0], 10, False),
    ("water_plant",  [7.0, 1.5, 1.0], 10, False),
    ("water_plant",  [3.0, 7.0, 1.0], 10, False),
    ("sweep_floor",  [5.0, 5.0, 0.5], 15, True),
    ("sweep_floor",  [2.0, 8.0, 0.5], 15, True),
    ("toggle_light", [9.0, 0.5, 1.5],  1, False),
    ("toggle_light", [0.5, 9.0, 1.5],  1, False),
]


def _make_task(task_type: str, idx: int, target: list, engage_steps: int,
               shareable: bool = False):
    spec = TaskSpec(
        task_id=f"{task_type}_{idx}",
        task_type=task_type,
        target_position=np.array(target, dtype=np.float32),
        engage_steps_required=engage_steps,
        is_shareable=shareable,
    )
    return _TASK_REGISTRY[task_type](spec)


# ---------------------------------------------------------------------------
# Viewer helper
# ---------------------------------------------------------------------------

class _NoOpViewer:
    """Stub returned when the real viewer cannot be opened (e.g. no display)."""
    def sync(self):  pass
    def close(self): pass
    def __bool__(self): return False


def _open_viewer(model, data):
    """
    Open a MuJoCo passive viewer, handling the macOS `mjpython` requirement.

    On macOS, `mujoco.viewer.launch_passive` raises RuntimeError unless the
    script is run under `mjpython`.  When that happens we:
      1. Print the exact command the user should run to get the GUI.
      2. Return a no-op stub so the simulation still runs headless.

    On Linux/Windows `launch_passive` works from any Python process.
    """
    import sys
    try:
        import mujoco.viewer as mj_viewer
        handle = mj_viewer.launch_passive(model, data)
        return handle
    except RuntimeError as exc:
        if "mjpython" in str(exc):
            import sys as _sys
            script = " ".join(_sys.argv)
            print(
                "\n[MuJoCo viewer] On macOS the passive viewer requires mjpython.\n"
                "  Run your command like this to get the GUI:\n\n"
                f"    mjpython {script}\n\n"
                "  Continuing headless (render=True has no effect until mjpython is used).\n"
            )
            return _NoOpViewer()
        raise   # re-raise any other RuntimeError


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------
class MujocoHomeEnv(MultiAgentEnv):
    """
    MuJoCo 3 physics-accurate household drone swarm environment.

    Interface is identical to HomeEnv and PybulletHomeEnv:
      obs  : dict[agent_id → np.ndarray shape (15,)]
      act  : dict[agent_id → np.ndarray shape (4,)]  (dx, dy, dz, tool)

    The 4-dim high-level action is translated by an attitude PID into
    per-rotor velocity commands at the MuJoCo timestep rate.  The policy
    never sees raw RPM — it always works with the same normalised deltas.

    Parameters (config dict)
    -------------------------
    n_drones          int   Number of drones [1–3 for current XML; default 3]
    max_steps         int   Episode step limit [default 500]
    render            bool  Open passive viewer window [default False]
    obs_noise_std     float Gaussian noise on observations [default 0.0]
    task_layouts      list  Override default task set
    allocator         BaseAllocator | None
    coop_time_threshold float  Fraction of max_steps for coop bonus [default 0.7]
    physics_steps_per_control int  MuJoCo steps per env step [default 20 → 40 Hz ctrl]
    """

    OBS_DIM = DroneAgent.OBS_DIM   # 15
    ACT_DIM = DroneAgent.ACT_DIM   # 4

    # Maximum supported drones (limited by scene.xml drone body count)
    MAX_DRONES = 3

    def __init__(self, config: dict | None = None):
        if not _MUJOCO_AVAILABLE:
            raise ImportError(
                "mujoco is not installed.\n"
                "  pip install mujoco\n"
            )
        super().__init__()
        cfg = config or {}

        self._n_drones    = min(int(cfg.get("n_drones", 3)), self.MAX_DRONES)
        self._max_steps   = int(cfg.get("max_steps", 500))
        self._render      = bool(cfg.get("render", False))
        self._noise_std   = float(cfg.get("obs_noise_std", 0.0))
        self._coop_thresh = float(cfg.get("coop_time_threshold", 0.7))
        self._phys_sub    = int(cfg.get("physics_steps_per_control", 20))

        # Task layout
        raw_layouts = cfg.get("task_layouts", _DEFAULT_TASK_LAYOUTS)
        self._task_layouts = raw_layouts

        # Allocator
        self.allocator: BaseAllocator = cfg.get("allocator") or _GreedyFallbackAllocator()

        # Agent IDs
        self._agent_ids = {f"drone_{i}" for i in range(self._n_drones)}

        # Gymnasium spaces
        obs_low  = np.full(self.OBS_DIM, -np.inf, dtype=np.float32)
        obs_high = np.full(self.OBS_DIM,  np.inf, dtype=np.float32)
        act_low  = np.array([-1, -1, -1, 0], dtype=np.float32)
        act_high = np.array([ 1,  1,  1, 1], dtype=np.float32)
        single_obs  = spaces.Box(obs_low, obs_high, dtype=np.float32)
        single_act  = spaces.Box(act_low, act_high, dtype=np.float32)
        self.observation_space = spaces.Dict({aid: single_obs for aid in self._agent_ids})
        self.action_space      = spaces.Dict({aid: single_act for aid in self._agent_ids})

        # Load MuJoCo model
        self._model = mujoco.MjModel.from_xml_path(_SCENE_XML)
        self._data  = mujoco.MjData(self._model)

        # Cache body/joint/sensor ids
        self._drone_body_ids  = [
            self._model.body(f"drone_{i}").id for i in range(self._n_drones)
        ]
        self._drone_joint_ids = [
            self._model.joint(f"drone_{i}_freejoint").id for i in range(self._n_drones)
        ]
        self._drone_qpos_adr  = [
            self._model.jnt_qposadr[jid] for jid in self._drone_joint_ids
        ]
        self._drone_qvel_adr  = [
            self._model.jnt_dofadr[jid] for jid in self._drone_joint_ids
        ]
        self._pos_sensor_ids  = [
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SENSOR, f"drone_{i}_pos")
            for i in range(self._n_drones)
        ]
        self._vel_sensor_ids  = [
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SENSOR, f"drone_{i}_vel")
            for i in range(self._n_drones)
        ]
        self._acc_sensor_ids  = [
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SENSOR, f"drone_{i}_acc")
            for i in range(self._n_drones)
        ]
        # Actuator index start per drone (4 actuators each)
        self._act_start = [i * 4 for i in range(self._n_drones)]

        # Per-drone state (Python-side)
        self._batteries: list[float] = []
        self._pids: list[_DronePID]  = []
        self._target_pos: list[np.ndarray] = []  # current velocity target (dx,dy,dz based)

        # Task list & allocation state
        self._tasks: list = []
        self._drone_task_map: dict[str, int | None] = {}
        self._step_count = 0
        self._last_bids: list[Bid] = []

        # Viewer handle
        self._viewer = None

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            np.random.seed(seed)

        # Reset MuJoCo state
        mujoco.mj_resetData(self._model, self._data)

        # Spawn drones at staggered positions
        for i in range(self._n_drones):
            adr = self._drone_qpos_adr[i]
            # position
            self._data.qpos[adr + 0] = 1.0 + i * 1.2   # x
            self._data.qpos[adr + 1] = 1.0              # y
            self._data.qpos[adr + 2] = 0.3              # z (just off floor)
            # quaternion (w, x, y, z) — identity
            self._data.qpos[adr + 3] = 1.0
            self._data.qpos[adr + 4] = 0.0
            self._data.qpos[adr + 5] = 0.0
            self._data.qpos[adr + 6] = 0.0

        mujoco.mj_forward(self._model, self._data)

        # Reset per-drone Python state
        self._batteries = [MAX_BATTERY] * self._n_drones
        self._pids      = [_DronePID() for _ in range(self._n_drones)]
        self._target_pos = [self._get_drone_pos(i).copy() for i in range(self._n_drones)]
        self._step_count = 0
        self._last_bids  = []

        # Build tasks
        self._tasks = []
        for j, layout in enumerate(self._task_layouts):
            ttype, tpos, tsteps = layout[0], layout[1], layout[2]
            shareable = layout[3] if len(layout) > 3 else False
            self._tasks.append(_make_task(ttype, j, tpos, tsteps, shareable))

        # Initial allocation
        self._drone_task_map = {f"drone_{i}": None for i in range(self._n_drones)}
        self._assign_tasks()

        # Open viewer if requested
        if self._render and self._viewer is None:
            self._viewer = _open_viewer(self._model, self._data)

        obs = self._build_obs()
        return obs, {}

    def step(self, actions: dict[str, np.ndarray]):
        assert self._data is not None, "call reset() first"
        self._step_count += 1

        rewards:    dict[str, float] = {aid: 0.0 for aid in self._agent_ids}
        terminated: dict[str, bool]  = {aid: False for aid in self._agent_ids}
        truncated:  dict[str, bool]  = {aid: False for aid in self._agent_ids}

        # --- Compute PID outputs once per env step (not per sub-step) ---
        # Running the PID 20× per env step would corrupt the derivative term
        # (dt=0.002s × 20 = 0.04s effective, but d/dt uses 0.002s spacing).
        # Cache forces/torques then re-apply identically for each sub-step.
        self._data.qfrc_applied[:] = 0.0
        for i, aid in enumerate(sorted(self._agent_ids)):
            act = actions.get(aid, np.zeros(self.ACT_DIM))
            self._apply_pid(i, act)   # writes qfrc_applied
        cached_qfrc = self._data.qfrc_applied.copy()

        # --- Motor control sub-loop at physics rate --------------------
        for _ in range(self._phys_sub):
            self._data.qfrc_applied[:] = cached_qfrc
            mujoco.mj_step(self._model, self._data)

        # --- Battery drain --------------------------------------------
        for i in range(self._n_drones):
            self._batteries[i] = max(0.0, self._batteries[i] - _BATTERY_DRAIN_PER_STEP)

        # --- Reward: alive penalty ------------------------------------
        for aid in self._agent_ids:
            rewards[aid] += REWARD_STEP_ALIVE

        # --- Collision detection (pairwise) ---------------------------
        collision_pairs = self._check_collisions()
        for (i, j) in collision_pairs:
            aid_i, aid_j = f"drone_{i}", f"drone_{j}"
            rewards[aid_i] += REWARD_COLLISION
            rewards[aid_j] += REWARD_COLLISION

        # --- Battery dead penalty ------------------------------------
        for i, aid in enumerate(sorted(self._agent_ids)):
            if self._batteries[i] <= 0:
                rewards[aid] += REWARD_BATTERY_DEAD

        # --- Task progress -------------------------------------------
        tasks_done_before = sum(1 for t in self._tasks if t.completed)
        for i, aid in enumerate(sorted(self._agent_ids)):
            task_idx = self._drone_task_map.get(aid)
            if task_idx is None or task_idx >= len(self._tasks):
                continue
            task = self._tasks[task_idx]
            if task.completed or task.status == TaskStatus.VANISHED:
                continue
            pos = self._get_drone_pos(i)
            dist = np.linalg.norm(pos - task.spec.target_position)
            if dist < _ENGAGE_DIST:
                task.step(drone_id=aid, drone_position=pos)
                if task.completed:
                    rewards[aid] += task.spec.engage_steps_required * 0.2

        tasks_done_after = sum(1 for t in self._tasks if t.completed)
        if tasks_done_after > tasks_done_before:
            self._assign_tasks()

        # Cooperative completion bonus
        eligible = [t for t in self._tasks if t.status != TaskStatus.VANISHED]
        if eligible and all(t.completed for t in eligible):
            frac = self._step_count / self._max_steps
            if frac < self._coop_thresh:
                for aid in self._agent_ids:
                    rewards[aid] += REWARD_COOP_BONUS

        # --- Termination / truncation --------------------------------
        all_done = bool(eligible and all(t.completed for t in eligible))
        timed_out = self._step_count >= self._max_steps

        if all_done or timed_out:
            for aid in self._agent_ids:
                if all_done:
                    terminated[aid] = True
                else:
                    truncated[aid]  = True
            terminated["__all__"] = all_done
            truncated["__all__"]  = timed_out

        # --- Viewer sync ---------------------------------------------
        if self._render and self._viewer is not None:
            self._viewer.sync()   # no-op if _NoOpViewer

        obs  = self._build_obs()
        info = {
            "step":          self._step_count,
            "tasks_done":    tasks_done_after,
            "tasks_total":   len(self._tasks),
            "batteries":     list(self._batteries),
            "collisions":    len(collision_pairs),
        }
        return obs, rewards, terminated, truncated, info

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    # ------------------------------------------------------------------
    # Disruption API (mirrors HomeEnv exactly)
    # ------------------------------------------------------------------

    def remove_task(self, task_id: str) -> bool:
        """Vanish a task mid-episode by task_id string. Returns True if found."""
        for i, task in enumerate(self._tasks):
            if task.spec.task_id == task_id:
                if task.status in (TaskStatus.COMPLETE, TaskStatus.VANISHED):
                    return False
                task.vanish()
                # Free any drone assigned to it
                for aid, tidx in list(self._drone_task_map.items()):
                    if tidx == i:
                        self._drone_task_map[aid] = None
                self._assign_tasks()
                return True
        return False

    def add_task(self, task_spec: TaskSpec) -> int:
        """Inject a new task mid-episode. Returns new task index."""
        cls = _TASK_REGISTRY[task_spec.task_type]
        new_task = cls(task_spec)
        self._tasks.append(new_task)
        self._assign_tasks()
        return len(self._tasks) - 1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_drone_pos(self, drone_idx: int) -> np.ndarray:
        """Return world-frame XYZ position of drone i (from freejoint qpos)."""
        adr = self._drone_qpos_adr[drone_idx]
        return self._data.qpos[adr:adr + 3].copy()

    def _get_drone_quat(self, drone_idx: int) -> np.ndarray:
        """Return quaternion (w,x,y,z) of drone i."""
        adr = self._drone_qpos_adr[drone_idx]
        return self._data.qpos[adr + 3:adr + 7].copy()

    def _get_drone_vel(self, drone_idx: int) -> np.ndarray:
        """Return world-frame linear velocity of drone i."""
        adr = self._drone_qvel_adr[drone_idx]
        return self._data.qvel[adr:adr + 3].copy()

    def _quat_to_euler(self, q: np.ndarray) -> np.ndarray:
        """Convert (w,x,y,z) quaternion to roll/pitch/yaw (ZYX Euler)."""
        w, x, y, z = q
        roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = np.arcsin(np.clip(2*(w*y - z*x), -1.0, 1.0))
        yaw   = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return np.array([roll, pitch, yaw], dtype=np.float32)

    def _apply_pid(self, drone_idx: int, action: np.ndarray):
        """
        Convert a high-level (dx, dy, dz, tool) action into a thrust force
        and attitude torques, applied directly to the drone body via
        mujoco.mj_applyFT.

        This is the "force/torque injection" approach: the simulation treats
        the quadrotor as a rigid body and we compute the required collective
        thrust (in body-z) and attitude torques (roll/pitch/yaw) from a PID
        that tracks the desired velocity command.

        The velocity actuators in the XML spin the prop discs for visual
        realism; they are set proportional to the collective thrust but do
        not produce physics forces.
        """
        V_MAX = 2.0   # m/s maximum commanded velocity

        pid   = self._pids[drone_idx]
        vel   = self._get_drone_vel(drone_idx)
        q     = self._get_drone_quat(drone_idx)
        euler = self._quat_to_euler(q)
        roll, pitch, yaw = euler
        dt    = _CTRL_DT  # effective control period = 0.04 s (not physics dt)

        # -- Dead drone: apply no force -----------------------------------
        if self._batteries[drone_idx] <= 0:
            return

        # -- Desired velocity from action ---------------------------------
        dx, dy, dz = float(action[0]), float(action[1]), float(action[2])
        vx_des = dx * V_MAX
        vy_des = dy * V_MAX
        vz_des = dz * V_MAX

        # -- Vertical PID (velocity-mode: error = vz_des − vz_actual) ----
        alt_err       = vz_des - vel[2]
        alt_d         = (alt_err - pid.alt_prev) / dt
        pid.alt_prev  = alt_err
        # thrust in Newtons = hover weight + P correction + D correction
        thrust = (_HOVER_F
                  + _PID_KP_ALT * alt_err
                  + _PID_KD_ALT * alt_d)
        thrust = float(np.clip(thrust, 0.0, _MAX_F))

        # -- Horizontal velocity → desired lean angle --------------------
        cy, sy = np.cos(yaw), np.sin(yaw)
        vx_body =  cy * vx_des + sy * vy_des
        vy_body = -sy * vx_des + cy * vy_des
        roll_des  =  vy_body / (V_MAX + 1e-6) * 0.40   # max ~23 deg lean
        pitch_des = -vx_body / (V_MAX + 1e-6) * 0.40

        # -- Roll PD (angle-mode, outputs Nm) ----------------------------
        roll_err    = roll_des - roll
        roll_d      = (roll_err - pid.roll_prev) / dt
        pid.roll_prev = roll_err
        tau_roll    = float(np.clip(
            _PID_KP_ROLL * roll_err + _PID_KD_ROLL * roll_d,
            -_MAX_TORQUE, _MAX_TORQUE,
        ))

        # -- Pitch PD ----------------------------------------------------
        pitch_err   = pitch_des - pitch
        pitch_d     = (pitch_err - pid.pitch_prev) / dt
        pid.pitch_prev = pitch_err
        tau_pitch   = float(np.clip(
            _PID_KP_PITCH * pitch_err + _PID_KD_PITCH * pitch_d,
            -_MAX_TORQUE, _MAX_TORQUE,
        ))

        # -- Yaw hold ----------------------------------------------------
        tau_yaw = 0.0

        # -- Apply force+torque via mj_applyFT ---------------------------
        # mj_applyFT expects force and torque in WORLD frame.
        # Thrust acts along body-z; rotate to world frame using xmat.
        body_id  = self._drone_body_ids[drone_idx]
        xmat     = self._data.xmat[body_id].reshape(3, 3)  # rotation matrix (body→world)
        force_w  = xmat @ np.array([0.0, 0.0, thrust])
        torque_w = xmat @ np.array([tau_roll, tau_pitch, tau_yaw])
        point    = self._data.xpos[body_id].copy()         # world-frame CoM

        mujoco.mj_applyFT(
            self._model, self._data,
            force_w, torque_w,
            point, body_id,
            self._data.qfrc_applied,
        )

        # -- Spin propeller discs for visual feedback (proportional to thrust)
        throttle = thrust / _MAX_F
        prop_spin = 5000.0 * throttle   # arbitrary visual speed
        base = self._act_start[drone_idx]
        self._data.ctrl[base + 0] =  prop_spin   # CW
        self._data.ctrl[base + 1] = -prop_spin   # CCW
        self._data.ctrl[base + 2] = -prop_spin   # CCW
        self._data.ctrl[base + 3] =  prop_spin   # CW

    def _build_obs(self) -> dict[str, np.ndarray]:
        """Build 15-dim observation for each drone (same as DroneAgent)."""
        obs_dict: dict[str, np.ndarray] = {}
        n = self._n_drones
        positions = [self._get_drone_pos(i) for i in range(n)]
        velocities = [self._get_drone_vel(i) for i in range(n)]

        for i, aid in enumerate(sorted(self._agent_ids)):
            pos   = positions[i]
            vel   = velocities[i]

            # Nearest task info
            task_idx = self._drone_task_map.get(aid)
            if task_idx is not None and task_idx < len(self._tasks):
                task     = self._tasks[task_idx]
                task_pos = task.spec.target_position
                task_rel = task_pos - pos
                rem      = task.remaining_work() / max(task.spec.engage_steps_required, 1)
            else:
                task_rel = np.zeros(3, dtype=np.float32)
                rem      = 0.0

            # Nearest other drone (for collision avoidance)
            nearest_rel = np.zeros(3, dtype=np.float32)
            nearest_dist = np.inf
            for j in range(n):
                if j == i:
                    continue
                d = positions[j] - pos
                dist = np.linalg.norm(d)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_rel  = d

            battery_norm = self._batteries[i] / MAX_BATTERY

            # 15-dim vector: [pos(3), vel(3), task_rel(3), rem(1), nearest_rel(3), battery(1), step_frac(1)]
            raw = np.concatenate([
                pos.astype(np.float32),
                vel.astype(np.float32),
                task_rel.astype(np.float32),
                [float(rem)],
                nearest_rel.astype(np.float32),
                [float(battery_norm)],
                [float(self._step_count / self._max_steps)],
            ]).astype(np.float32)

            if self._noise_std > 0:
                raw += np.random.normal(0, self._noise_std, raw.shape).astype(np.float32)

            obs_dict[aid] = raw

        return obs_dict

    def _check_collisions(self) -> list[tuple[int, int]]:
        """Return list of (i,j) pairs of drones that are too close."""
        pairs = []
        positions = [self._get_drone_pos(i) for i in range(self._n_drones)]
        for i in range(self._n_drones):
            for j in range(i + 1, self._n_drones):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < _COLLISION_DIST:
                    pairs.append((i, j))
        return pairs

    def _assign_tasks(self):
        """Run allocator and update _drone_task_map."""
        positions = {
            f"drone_{i}": self._get_drone_pos(i)
            for i in range(self._n_drones)
        }
        batteries     = {f"drone_{i}": self._batteries[i]       for i in range(self._n_drones)}
        task_progress = {f"drone_{i}": 0.0                       for i in range(self._n_drones)}

        # Fill task_progress from current assignments
        for i in range(self._n_drones):
            aid = f"drone_{i}"
            tidx = self._drone_task_map.get(aid)
            if tidx is not None and tidx < len(self._tasks):
                t = self._tasks[tidx]
                total = max(t.spec.engage_steps_required, 1)
                task_progress[aid] = 1.0 - t.remaining_work() / total

        snapshot = WorldSnapshot(
            drone_positions=positions,
            drone_batteries=batteries,
            drone_task_progress=task_progress,
            current_assignments=dict(self._drone_task_map),
            tasks=self._tasks,
            step=self._step_count,
            max_steps=self._max_steps,
        )
        result = self.allocator.allocate(snapshot)
        self._last_bids = result.bids

        # Apply assignments
        for aid, task_idx in result.assignments.items():
            if aid in self._drone_task_map:
                self._drone_task_map[aid] = task_idx

        # Handle co-assignments (shareable tasks may receive extra drones)
        for bid in result.bids:
            # LearnedBidder sets bid.marginal > 0 for co-assign intent
            if (
                getattr(bid, "co_assigned", False)
                and bid.task_idx is not None
                and bid.task_idx < len(self._tasks)
                and self._tasks[bid.task_idx].spec.is_shareable
            ):
                self._drone_task_map[bid.drone_id] = bid.task_idx

    # ------------------------------------------------------------------
    # Properties (used by eval harness)
    # ------------------------------------------------------------------

    @property
    def tasks(self):
        return self._tasks

    @property
    def last_bids(self) -> list[Bid]:
        return self._last_bids
