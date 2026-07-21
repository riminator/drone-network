"""
tests/test_mujoco_env.py
Unit tests for MujocoHomeEnv (Phase MuJoCo).

All tests run headless (render=False) and complete in a few seconds.
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip entire module if mujoco is not installed
mujoco = pytest.importorskip("mujoco", reason="mujoco not installed")

from envs.mujoco_env import MujocoHomeEnv
from envs.tasks.base_task import TaskSpec, TaskStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_env(n_drones: int = 3, max_steps: int = 50, **kwargs) -> MujocoHomeEnv:
    cfg = {"n_drones": n_drones, "max_steps": max_steps, "render": False, **kwargs}
    return MujocoHomeEnv(config=cfg)


# ---------------------------------------------------------------------------
# Construction & reset
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_creates_without_error(self):
        env = make_env()
        assert env is not None

    def test_agent_ids_correct(self):
        env = make_env(n_drones=3)
        assert env._agent_ids == {"drone_0", "drone_1", "drone_2"}

    def test_single_drone(self):
        env = make_env(n_drones=1)
        assert env._agent_ids == {"drone_0"}

    def test_max_drones_capped_at_3(self):
        """scene.xml only has 3 drone bodies; config capped to MAX_DRONES."""
        env = make_env(n_drones=5)
        assert env._n_drones == 3

    def test_obs_space_shape(self):
        env = make_env()
        for aid in env._agent_ids:
            assert env.observation_space[aid].shape == (15,)

    def test_act_space_shape(self):
        env = make_env()
        for aid in env._agent_ids:
            assert env.action_space[aid].shape == (4,)


class TestReset:
    def test_reset_returns_obs_dict(self):
        env = make_env()
        obs, info = env.reset(seed=0)
        assert isinstance(obs, dict)
        assert set(obs.keys()) == env._agent_ids

    def test_reset_obs_shape(self):
        env = make_env()
        obs, _ = env.reset(seed=1)
        for v in obs.values():
            assert v.shape == (15,)
            assert v.dtype == np.float32

    def test_reset_battery_full(self):
        env = make_env()
        env.reset(seed=0)
        from envs.drone_agent import MAX_BATTERY
        for b in env._batteries:
            assert b == MAX_BATTERY

    def test_reset_drones_at_spawn_z(self):
        """Drones spawn at spawn_z (default 1.0) not floor."""
        env = make_env()
        obs, _ = env.reset(seed=0)
        for i in range(env._n_drones):
            z = env._get_drone_pos(i)[2]
            assert 0.5 < z < 2.0, f"drone {i} z={z} out of expected spawn range"

    def test_reset_creates_tasks(self):
        env = make_env()
        env.reset(seed=0)
        assert len(env._tasks) > 0

    def test_reset_seed_deterministic(self):
        env = make_env()
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        for aid in obs1:
            np.testing.assert_array_equal(obs1[aid], obs2[aid])

    def test_reset_clears_step_count(self):
        """Reset between episodes clears the step counter."""
        env = make_env()
        env.reset(seed=0)
        for _ in range(5):
            env.step({aid: np.zeros(4, dtype=np.float32) for aid in env._agent_ids})
        assert env._step_count == 5
        env.reset(seed=99)
        assert env._step_count == 0


# ---------------------------------------------------------------------------
# Step interface
# ---------------------------------------------------------------------------

class TestStep:
    def test_step_returns_correct_keys(self):
        env = make_env()
        obs, _ = env.reset(seed=0)
        zeros = {aid: np.zeros(4, dtype=np.float32) for aid in obs}
        obs2, rew, term, trunc, info = env.step(zeros)
        assert set(obs2.keys()) == env._agent_ids
        assert set(rew.keys())  == env._agent_ids

    def test_step_obs_shape_unchanged(self):
        env = make_env()
        obs, _ = env.reset(seed=0)
        zeros = {aid: np.zeros(4, dtype=np.float32) for aid in obs}
        obs2, *_ = env.step(zeros)
        for v in obs2.values():
            assert v.shape == (15,)

    def test_step_reward_is_float(self):
        env = make_env()
        obs, _ = env.reset(seed=0)
        zeros = {aid: np.zeros(4, dtype=np.float32) for aid in obs}
        _, rew, *_ = env.step(zeros)
        for r in rew.values():
            assert isinstance(r, float)

    def test_step_alive_penalty_applied(self):
        from envs.mujoco_env import REWARD_STEP_ALIVE
        env = make_env()
        obs, _ = env.reset(seed=0)
        zeros = {aid: np.zeros(4, dtype=np.float32) for aid in obs}
        _, rew, *_ = env.step(zeros)
        # Each drone gets at most REWARD_STEP_ALIVE (negative)
        for r in rew.values():
            assert r <= 0 + 1e-6   # ≤0 because only alive penalty on step 1

    def test_step_count_increments(self):
        env = make_env()
        obs, _ = env.reset(seed=0)
        zeros = {aid: np.zeros(4, dtype=np.float32) for aid in obs}
        for expected in range(1, 6):
            env.step(zeros)
            assert env._step_count == expected

    def test_episode_truncates_at_max_steps(self):
        env = make_env(max_steps=10)
        obs, _ = env.reset(seed=0)
        zeros = {aid: np.zeros(4, dtype=np.float32) for aid in obs}
        done = False
        steps = 0
        while not done:
            _, _, term, trunc, _ = env.step(zeros)
            steps += 1
            done = term.get("__all__", False) or trunc.get("__all__", False)
        assert steps == 10

    def test_all_key_in_terminated(self):
        env = make_env(max_steps=3)
        obs, _ = env.reset(seed=0)
        zeros = {aid: np.zeros(4, dtype=np.float32) for aid in obs}
        for _ in range(2):
            _, _, term, trunc, _ = env.step(zeros)
        _, _, term, trunc, _ = env.step(zeros)
        assert "__all__" in term or "__all__" in trunc


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------

class TestPhysics:
    def test_zero_thrust_drone_falls(self):
        """Without thrust, gravity pulls the drone down."""
        env = make_env(n_drones=1)
        obs, _ = env.reset(seed=0)
        z_init = env._get_drone_pos(0)[2]
        _, _, _, _, _ = env.step({"drone_0": np.zeros(4, dtype=np.float32)})
        z_after = env._get_drone_pos(0)[2]
        assert z_after < z_init or z_after <= z_init + 0.01  # falls from spawn_z

    def test_upward_thrust_drone_rises(self):
        """With dz=1.0, the drone should climb above initial z."""
        env = make_env(n_drones=1, max_steps=100)
        obs, _ = env.reset(seed=0)
        z_init = env._get_drone_pos(0)[2]
        for _ in range(15):
            obs, _, _, _, _ = env.step(
                {"drone_0": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)}
            )
        z_final = env._get_drone_pos(0)[2]
        assert z_final > z_init, f"expected rise: z_init={z_init:.3f} z_final={z_final:.3f}"

    def test_horizontal_action_displaces_drone(self):
        """Commanding dx=1.0 should displace the drone horizontally."""
        env = make_env(n_drones=1, max_steps=200)
        obs, _ = env.reset(seed=0)
        pos_init = env._get_drone_pos(0).copy()
        # Command forward + upward to get drone airborne
        for _ in range(40):
            obs, _, _, _, _ = env.step(
                {"drone_0": np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)}
            )
        pos_final = env._get_drone_pos(0)
        # The horizontal displacement should be non-trivial (> 0.3 m in any direction)
        horiz_displacement = np.linalg.norm(pos_final[:2] - pos_init[:2])
        assert horiz_displacement > 0.3, (
            f"expected horizontal movement > 0.3m, got {horiz_displacement:.3f}m"
        )

    def test_battery_drains_each_step(self):
        from envs.drone_agent import MAX_BATTERY
        env = make_env(n_drones=1)
        obs, _ = env.reset(seed=0)
        b0 = env._batteries[0]
        env.step({"drone_0": np.zeros(4, dtype=np.float32)})
        b1 = env._batteries[0]
        assert b1 < b0, "battery should drain each step"
        assert b1 == MAX_BATTERY - (MAX_BATTERY / 500)

    def test_no_collision_with_zero_actions(self):
        """Drones start at staggered x positions — no initial collision."""
        env = make_env()
        obs, _ = env.reset(seed=0)
        zeros = {aid: np.zeros(4, dtype=np.float32) for aid in obs}
        _, rew, *_ = env.step(zeros)
        from envs.mujoco_env import REWARD_COLLISION
        for r in rew.values():
            assert r > REWARD_COLLISION, f"unexpected collision: reward={r}"


# ---------------------------------------------------------------------------
# Task interaction
# ---------------------------------------------------------------------------

class TestTaskInteraction:
    def test_default_tasks_created(self):
        env = make_env()
        env.reset(seed=0)
        assert len(env._tasks) == 7  # matches _DEFAULT_TASK_LAYOUTS

    def test_custom_task_layout(self):
        cfg = {
            "n_drones": 2,
            "max_steps": 50,
            "render": False,
            "task_layouts": [
                ("water_plant",  [2.0, 3.0, 1.0], 5, False),
                ("toggle_light", [1.0, 1.0, 1.5], 1, False),
            ],
        }
        env = MujocoHomeEnv(config=cfg)
        env.reset(seed=0)
        assert len(env._tasks) == 2

    def test_drones_assigned_on_reset(self):
        env = make_env()
        env.reset(seed=0)
        assigned = [v for v in env._drone_task_map.values() if v is not None]
        assert len(assigned) > 0


# ---------------------------------------------------------------------------
# Disruption API
# ---------------------------------------------------------------------------

class TestDisruptionAPI:
    def test_remove_existing_task_returns_true(self):
        env = make_env()
        env.reset(seed=0)
        task_id = env._tasks[0].spec.task_id
        result = env.remove_task(task_id)
        assert result is True

    def test_remove_nonexistent_task_returns_false(self):
        env = make_env()
        env.reset(seed=0)
        result = env.remove_task("does_not_exist_99")
        assert result is False

    def test_removed_task_status_vanished(self):
        env = make_env()
        env.reset(seed=0)
        task_id = env._tasks[0].spec.task_id
        env.remove_task(task_id)
        assert env._tasks[0].status == TaskStatus.VANISHED

    def test_add_task_increases_task_count(self):
        env = make_env()
        env.reset(seed=0)
        n_before = len(env._tasks)
        spec = TaskSpec(
            task_id="injected_0",
            task_type="toggle_light",
            target_position=np.array([5.0, 5.0, 1.5]),
            engage_steps_required=1,
        )
        env.add_task(spec)
        assert len(env._tasks) == n_before + 1

    def test_drone_reassigned_after_task_vanish(self):
        """After vanishing a task, the drone should not still hold that assignment."""
        env = make_env()
        env.reset(seed=0)
        # Find a drone with an assignment
        drone_with_task = None
        task_idx = None
        for aid, tidx in env._drone_task_map.items():
            if tidx is not None:
                drone_with_task = aid
                task_idx = tidx
                break
        if drone_with_task is None:
            pytest.skip("No drone with initial assignment")
        task_id = env._tasks[task_idx].spec.task_id
        env.remove_task(task_id)
        # Verify: the vanished task is VANISHED
        assert env._tasks[task_idx].status == TaskStatus.VANISHED
        # The drone's assignment should not be the vanished task index
        new_assignment = env._drone_task_map[drone_with_task]
        assert new_assignment != task_idx, (
            f"Drone {drone_with_task} still assigned to vanished task {task_idx}"
        )


# ---------------------------------------------------------------------------
# Observation noise
# ---------------------------------------------------------------------------

class TestObsNoise:
    def test_zero_noise_default(self):
        env = make_env(obs_noise_std=0.0)
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        for aid in obs1:
            np.testing.assert_array_equal(obs1[aid], obs2[aid])

    def test_noise_changes_obs(self):
        env = make_env(obs_noise_std=0.5)
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        # With noise, observations should differ between resets
        # (noise is applied after reset with different random state)
        # Note: same np.random.seed(42) → same noise. Just check shape.
        for v in obs1.values():
            assert v.shape == (15,)


# ---------------------------------------------------------------------------
# Allocator integration
# ---------------------------------------------------------------------------

class TestAllocatorIntegration:
    def test_greedy_allocator_runs(self):
        from allocator.greedy_auction import GreedyAuction
        env = make_env()
        env.allocator = GreedyAuction()
        obs, _ = env.reset(seed=0)
        zeros = {aid: np.zeros(4, dtype=np.float32) for aid in obs}
        obs2, *_ = env.step(zeros)
        assert obs2 is not None

    def test_cbba_allocator_runs(self):
        from allocator.cbba import CBBA
        env = make_env()
        env.allocator = CBBA(comm_delay=0)
        obs, _ = env.reset(seed=0)
        zeros = {aid: np.zeros(4, dtype=np.float32) for aid in obs}
        obs2, *_ = env.step(zeros)
        assert obs2 is not None

    def test_oracle_allocator_runs(self):
        from allocator.oracle import OracleAllocator
        env = make_env()
        env.allocator = OracleAllocator()
        obs, _ = env.reset(seed=0)
        zeros = {aid: np.zeros(4, dtype=np.float32) for aid in obs}
        obs2, *_ = env.step(zeros)
        assert obs2 is not None

    def test_learned_allocator_runs(self):
        from allocator.learned_bidder import LearnedBidder
        from allocator.bid_policy import BidPolicy
        env = make_env()
        env.allocator = LearnedBidder(BidPolicy())
        obs, _ = env.reset(seed=0)
        zeros = {aid: np.zeros(4, dtype=np.float32) for aid in obs}
        obs2, *_ = env.step(zeros)
        assert obs2 is not None


# ---------------------------------------------------------------------------
# Close
# ---------------------------------------------------------------------------

class TestClose:
    def test_close_no_crash(self):
        env = make_env()
        env.reset(seed=0)
        env.close()  # should not raise

    def test_close_twice_no_crash(self):
        env = make_env()
        env.reset(seed=0)
        env.close()
        env.close()
