"""
tests/test_phase1.py
Smoke tests for Phase 1 — allocation interface, TaskStatus lifecycle,
disruption API (remove_task / add_task), and co-assignment.

Run with:  python -m pytest tests/test_phase1.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from envs.tasks.base_task import TaskSpec, TaskStatus
from envs.tasks.water_plant import WaterPlantTask
from envs.tasks.sweep_floor import SweepFloorTask
from envs.tasks.toggle_light import ToggleLightTask
from envs.home_env import HomeEnv
from allocator.base_allocator import BaseAllocator, WorldSnapshot, AllocationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_spec(task_type="water_plant", shareable=False, engage=10):
    return TaskSpec(
        task_id=f"test_{task_type}",
        task_type=task_type,
        target_position=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        engage_steps_required=engage,
        is_shareable=shareable,
    )


def _small_env(**kwargs):
    cfg = {"n_drones": 3, "max_steps": 200}
    cfg.update(kwargs)
    return HomeEnv(cfg)


# ---------------------------------------------------------------------------
# TaskStatus / remaining_work
# ---------------------------------------------------------------------------

class TestTaskStatus:
    def test_initial_status_is_pending(self):
        task = WaterPlantTask(_make_spec())
        assert task.status == TaskStatus.PENDING

    def test_completed_property_shim(self):
        task = WaterPlantTask(_make_spec())
        task.completed = True
        assert task.status == TaskStatus.COMPLETE
        assert task.completed is True

    def test_vanish_clears_assignees(self):
        task = WaterPlantTask(_make_spec())
        task.assigned_drone_ids = ["drone_0", "drone_1"]
        task.vanish()
        assert task.status == TaskStatus.VANISHED
        assert task.assigned_drone_ids == []

    def test_reset_returns_to_pending(self):
        task = WaterPlantTask(_make_spec())
        task.completed = True
        task.assigned_drone_ids = ["drone_0"]
        task.reset()
        assert task.status == TaskStatus.PENDING
        assert task.assigned_drone_ids == []

    def test_assigned_drone_id_legacy_shim(self):
        task = WaterPlantTask(_make_spec())
        task.assigned_drone_id = "drone_0"
        assert task.assigned_drone_ids == ["drone_0"]
        task.assigned_drone_id = None
        assert task.assigned_drone_ids == []


class TestRemainingWork:
    def test_water_plant_starts_at_one(self):
        task = WaterPlantTask(_make_spec(engage=10))
        assert task.remaining_work() == pytest.approx(1.0)

    def test_water_plant_zero_when_complete(self):
        task = WaterPlantTask(_make_spec(engage=10))
        task.engage_steps_done = 10
        task.completed = True
        assert task.remaining_work() == pytest.approx(0.0)

    def test_water_plant_partial(self):
        task = WaterPlantTask(_make_spec(engage=10))
        task.engage_steps_done = 5
        assert task.remaining_work() == pytest.approx(0.5)

    def test_sweep_floor_waypoint_progress(self):
        spec = _make_spec(task_type="sweep_floor")
        task = SweepFloorTask(spec, n_waypoints=4)
        assert task.remaining_work() == pytest.approx(1.0)
        task._current_wp_idx = 2
        assert task.remaining_work() == pytest.approx(0.5)

    def test_toggle_light_binary(self):
        task = ToggleLightTask(_make_spec(task_type="toggle_light", engage=1))
        assert task.remaining_work() == pytest.approx(1.0)
        task.completed = True
        assert task.remaining_work() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TaskSpec.is_shareable
# ---------------------------------------------------------------------------

class TestShareable:
    def test_default_not_shareable(self):
        spec = _make_spec()
        assert spec.is_shareable is False

    def test_shareable_flag(self):
        spec = _make_spec(shareable=True)
        assert spec.is_shareable is True


# ---------------------------------------------------------------------------
# HomeEnv — basic episode still works
# ---------------------------------------------------------------------------

class TestHomeEnvBasic:
    def test_reset_returns_obs_for_all_drones(self):
        env = _small_env()
        obs, info = env.reset(seed=0)
        assert set(obs.keys()) == {f"drone_{i}" for i in range(3)}
        for v in obs.values():
            assert v.shape == (15,)

    def test_step_runs_without_error(self):
        env = _small_env()
        obs, _ = env.reset(seed=0)
        actions = {aid: env.action_space.sample() for aid in obs}
        obs2, rewards, terminated, truncated, infos = env.step(actions)
        assert set(rewards.keys()) == {f"drone_{i}" for i in range(3)}

    def test_tasks_have_pending_status_on_reset(self):
        env = _small_env()
        env.reset(seed=0)
        for task in env._tasks:
            # After reset + initial assignment, tasks should be PENDING or ASSIGNED
            assert task.status in (TaskStatus.PENDING, TaskStatus.ASSIGNED)


# ---------------------------------------------------------------------------
# Disruption API
# ---------------------------------------------------------------------------

class TestDisruptionAPI:
    def test_remove_task_vanishes_it(self):
        env = _small_env()
        env.reset(seed=0)
        task_id = env._tasks[0].spec.task_id
        result = env.remove_task(task_id)
        assert result is True
        assert env._tasks[0].status == TaskStatus.VANISHED

    def test_remove_task_frees_assigned_drone(self):
        env = _small_env()
        env.reset(seed=0)
        # Find the task_id assigned to drone_0
        task_idx = env._drone_task_map.get("drone_0")
        if task_idx is None:
            pytest.skip("drone_0 has no task assigned at reset")
        task_id = env._tasks[task_idx].spec.task_id
        env.remove_task(task_id)
        assert env._drone_task_map["drone_0"] != task_idx or \
               env._tasks[task_idx].status == TaskStatus.VANISHED

    def test_remove_nonexistent_task_returns_false(self):
        env = _small_env()
        env.reset(seed=0)
        assert env.remove_task("does_not_exist") is False

    def test_remove_completed_task_returns_false(self):
        env = _small_env()
        env.reset(seed=0)
        env._tasks[0].status = TaskStatus.COMPLETE
        result = env.remove_task(env._tasks[0].spec.task_id)
        assert result is False

    def test_add_task_appends_and_triggers_assign(self):
        env = _small_env()
        env.reset(seed=0)
        n_before = len(env._tasks)
        new_id = env.add_task("water_plant", [3.0, 3.0, 1.0], engage_steps=5)
        assert len(env._tasks) == n_before + 1
        assert env._tasks[-1].spec.task_id == new_id
        assert env._tasks[-1].status in (TaskStatus.PENDING, TaskStatus.ASSIGNED)

    def test_episode_terminates_when_all_tasks_vanish(self):
        env = _small_env()
        env.reset(seed=0)
        # Vanish every task
        for task in env._tasks:
            task.vanish()
        actions = {aid: env.action_space.sample() for aid in env._agent_ids}
        _, _, terminated, _, _ = env.step(actions)
        assert terminated["__all__"] is True


# ---------------------------------------------------------------------------
# Co-assignment
# ---------------------------------------------------------------------------

class TestCoAssignment:
    def test_shareable_task_accepts_two_drones(self):
        env = _small_env()
        env.reset(seed=0)
        # Find the sweep_floor task (index 2, shareable=True)
        sweep_idx = next(
            i for i, t in enumerate(env._tasks)
            if t.spec.task_type == "sweep_floor"
        )
        sweep_task = env._tasks[sweep_idx]
        assert sweep_task.spec.is_shareable is True
        # Manually force two drones onto it via _apply_allocation
        from allocator.base_allocator import AllocationResult
        result = AllocationResult(assignments={
            "drone_0": sweep_idx,
            "drone_1": sweep_idx,
            "drone_2": None,
        })
        env._apply_allocation(result)
        assert len(sweep_task.assigned_drone_ids) == 2
        assert "drone_0" in sweep_task.assigned_drone_ids
        assert "drone_1" in sweep_task.assigned_drone_ids

    def test_nonshareable_task_rejects_second_drone(self):
        env = _small_env()
        env.reset(seed=0)
        # water_plant_0 is non-shareable (index 0)
        from allocator.base_allocator import AllocationResult
        result = AllocationResult(assignments={
            "drone_0": 0,
            "drone_1": 0,   # should be rejected
            "drone_2": None,
        })
        env._apply_allocation(result)
        task = env._tasks[0]
        assert len(task.assigned_drone_ids) == 1
        # drone_1 should have been set idle
        assert env._drone_task_map["drone_1"] is None


# ---------------------------------------------------------------------------
# Allocator interface
# ---------------------------------------------------------------------------

class TestAllocatorInterface:
    def test_custom_allocator_is_called(self):
        calls = []

        class TrackingAllocator(BaseAllocator):
            def allocate(self, snapshot: WorldSnapshot) -> AllocationResult:
                calls.append(snapshot.step)
                # delegate to greedy
                from envs.home_env import _GreedyFallbackAllocator
                return _GreedyFallbackAllocator().allocate(snapshot)

        env = _small_env(allocator=TrackingAllocator())
        env.reset(seed=0)
        assert len(calls) >= 1, "allocate() should be called on reset"

    def test_world_snapshot_contains_all_drones(self):
        snapshots = []

        class CapturingAllocator(BaseAllocator):
            def allocate(self, snapshot: WorldSnapshot) -> AllocationResult:
                snapshots.append(snapshot)
                from envs.home_env import _GreedyFallbackAllocator
                return _GreedyFallbackAllocator().allocate(snapshot)

        env = _small_env(allocator=CapturingAllocator())
        env.reset(seed=0)
        snap = snapshots[-1]
        assert set(snap.drone_positions.keys()) == {f"drone_{i}" for i in range(3)}
        assert len(snap.tasks) == 5
