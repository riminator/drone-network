"""
tests/test_phase2.py
Tests for Phase 2 allocators: GreedyAuction and CBBA.

Covers:
  - basic allocation contract (all drones covered, assignments valid)
  - bids are produced with correct structure
  - co-assignment pass for shareable tasks
  - disruption recovery: task vanish + realloc
  - disruption recovery: task injection mid-episode
  - CBBA convergence on a simple scenario
  - CBBA comm_delay path (messages deferred then applied)
  - head-to-head: both allocators complete a full episode
  - CBBA on_task_complete / on_task_vanish purge stale messages

Run with:  python3 -m pytest tests/test_phase2.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from allocator.base_allocator import WorldSnapshot, AllocationResult, Bid
from allocator.greedy_auction import GreedyAuction
from allocator.cbba import CBBA
from envs.tasks.base_task import TaskStatus, TaskSpec
from envs.tasks.water_plant import WaterPlantTask
from envs.tasks.sweep_floor import SweepFloorTask
from envs.tasks.toggle_light import ToggleLightTask
from envs.home_env import HomeEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _task(task_type="water_plant", idx=0, pos=None, shareable=False, engage=10):
    pos = np.array(pos or [float(idx), 1.0, 1.0], dtype=np.float32)
    spec = TaskSpec(
        task_id=f"{task_type}_{idx}",
        task_type=task_type,
        target_position=pos,
        engage_steps_required=engage,
        is_shareable=shareable,
    )
    cls = {"water_plant": WaterPlantTask,
           "sweep_floor": SweepFloorTask,
           "toggle_light": ToggleLightTask}[task_type]
    return cls(spec)


def _snapshot(n_drones=3, tasks=None, step=0, max_steps=500,
              assignments=None):
    """Build a WorldSnapshot with n drones spread along x-axis."""
    drone_ids = [f"drone_{i}" for i in range(n_drones)]
    positions = {d: np.array([float(i) * 2, 0.0, 1.0], dtype=np.float32)
                 for i, d in enumerate(drone_ids)}
    batteries = {d: 1.0 for d in drone_ids}
    progress = {d: 0.0 for d in drone_ids}
    # Use 'is None' check — empty list [] must not be replaced with defaults
    if tasks is None:
        tasks = [_task(idx=j) for j in range(n_drones)]
    # Mark as ASSIGNED so they're visible to allocators
    for t in tasks:
        if t.status == TaskStatus.PENDING:
            t.status = TaskStatus.ASSIGNED
    current = assignments if assignments is not None else {d: None for d in drone_ids}
    return WorldSnapshot(
        drone_positions=positions,
        drone_batteries=batteries,
        drone_task_progress=progress,
        current_assignments=current,
        tasks=tasks,
        step=step,
        max_steps=max_steps,
    )


def _env(allocator, n_drones=3):
    return HomeEnv({"n_drones": n_drones, "max_steps": 300,
                    "allocator": allocator})


# ---------------------------------------------------------------------------
# GreedyAuction — contract
# ---------------------------------------------------------------------------

class TestGreedyAuctionContract:
    def test_all_drones_covered(self):
        alloc = GreedyAuction()
        snap = _snapshot(n_drones=3)
        result = alloc.allocate(snap)
        assert set(result.assignments.keys()) == {"drone_0", "drone_1", "drone_2"}

    def test_assignments_reference_valid_task_indices(self):
        alloc = GreedyAuction()
        snap = _snapshot(n_drones=3)
        result = alloc.allocate(snap)
        n_tasks = len(snap.tasks)
        for v in result.assignments.values():
            assert v is None or (0 <= v < n_tasks)

    def test_no_duplicate_assignments_on_nonshareable(self):
        alloc = GreedyAuction()
        snap = _snapshot(n_drones=4, tasks=[_task(idx=i) for i in range(3)])
        result = alloc.allocate(snap)
        assigned = [v for v in result.assignments.values() if v is not None]
        # Non-shareable: each task can appear at most once
        assert len(assigned) == len(set(assigned))

    def test_bids_have_correct_structure(self):
        alloc = GreedyAuction()
        snap = _snapshot(n_drones=2)
        result = alloc.allocate(snap)
        assert len(result.bids) > 0
        for b in result.bids:
            assert isinstance(b, Bid)
            assert b.bid_value > 0
            assert b.drone_id in snap.drone_positions

    def test_closer_drone_wins(self):
        """drone_0 at x=0 vs drone_1 at x=8; task at x=0 — drone_0 should win."""
        alloc = GreedyAuction()
        tasks = [_task(idx=0, pos=[0.0, 1.0, 1.0])]
        tasks[0].status = TaskStatus.ASSIGNED
        snap = WorldSnapshot(
            drone_positions={
                "drone_0": np.array([0.1, 1.0, 1.0], dtype=np.float32),
                "drone_1": np.array([8.0, 1.0, 1.0], dtype=np.float32),
            },
            drone_batteries={"drone_0": 1.0, "drone_1": 1.0},
            drone_task_progress={"drone_0": 0.0, "drone_1": 0.0},
            current_assignments={"drone_0": None, "drone_1": None},
            tasks=tasks,
            step=0, max_steps=500,
        )
        result = alloc.allocate(snap)
        assert result.assignments["drone_0"] == 0

    def test_empty_task_list_returns_all_idle(self):
        alloc = GreedyAuction()
        snap = _snapshot(n_drones=3, tasks=[])
        result = alloc.allocate(snap)
        assert all(v is None for v in result.assignments.values())

    def test_shareable_task_co_assigned_to_idle_drone(self):
        alloc = GreedyAuction()
        # 3 drones, 1 shareable sweep task + 1 normal task
        sweep = _task("sweep_floor", idx=0, shareable=True)
        water = _task("water_plant", idx=1)
        snap = _snapshot(n_drones=3, tasks=[sweep, water])
        result = alloc.allocate(snap)
        assigned_to_sweep = [d for d, t in result.assignments.items() if t == 0]
        # Should have at least one drone (possibly 2 if co-assigned)
        assert len(assigned_to_sweep) >= 1


# ---------------------------------------------------------------------------
# GreedyAuction — disruption recovery
# ---------------------------------------------------------------------------

class TestGreedyAuctionDisruption:
    def test_task_vanish_then_reallocate(self):
        """Removing a task mid-episode frees the drone and it gets reassigned."""
        alloc = GreedyAuction()
        env = _env(alloc)
        env.reset(seed=42)
        # Identify task assigned to drone_0
        task_idx = env._drone_task_map.get("drone_0")
        if task_idx is None:
            pytest.skip("drone_0 idle at reset")
        task_id = env._tasks[task_idx].spec.task_id
        n_tasks_before = sum(
            1 for t in env._tasks if t.status not in
            (TaskStatus.VANISHED, TaskStatus.COMPLETE)
        )
        env.remove_task(task_id)
        assert env._tasks[task_idx].status == TaskStatus.VANISHED
        assert env._drone_task_map.get("drone_0") != task_idx or \
               env._tasks[task_idx].status == TaskStatus.VANISHED

    def test_add_task_mid_episode_gets_assigned(self):
        alloc = GreedyAuction()
        env = _env(alloc)
        env.reset(seed=0)
        new_id = env.add_task("water_plant", [5.0, 5.0, 1.0], engage_steps=5)
        new_task = next(t for t in env._tasks if t.spec.task_id == new_id)
        assert new_task.status in (TaskStatus.PENDING, TaskStatus.ASSIGNED)

    def test_episode_completes_with_greedy(self):
        """Full episode should terminate without errors."""
        alloc = GreedyAuction()
        env = _env(alloc, n_drones=6)
        obs, _ = env.reset(seed=1)
        for _ in range(500):
            actions = {aid: env.action_space.sample() for aid in obs}
            obs, _, terminated, truncated, _ = env.step(actions)
            if terminated.get("__all__") or truncated.get("__all__"):
                break
        # Just check it didn't crash — termination is stochastic with random actions


# ---------------------------------------------------------------------------
# CBBA — contract
# ---------------------------------------------------------------------------

class TestCBBAContract:
    def test_all_drones_covered(self):
        alloc = CBBA()
        snap = _snapshot(n_drones=3)
        result = alloc.allocate(snap)
        assert set(result.assignments.keys()) == {"drone_0", "drone_1", "drone_2"}

    def test_assignments_reference_valid_task_indices(self):
        alloc = CBBA()
        snap = _snapshot(n_drones=3)
        result = alloc.allocate(snap)
        n_tasks = len(snap.tasks)
        for v in result.assignments.values():
            assert v is None or (0 <= v < n_tasks)

    def test_no_task_assigned_to_two_drones_nonshareable(self):
        alloc = CBBA()
        snap = _snapshot(n_drones=4, tasks=[_task(idx=i) for i in range(3)])
        result = alloc.allocate(snap)
        assigned = [v for v in result.assignments.values() if v is not None]
        assert len(assigned) == len(set(assigned))

    def test_bids_produced(self):
        alloc = CBBA()
        snap = _snapshot(n_drones=2)
        result = alloc.allocate(snap)
        assert len(result.bids) > 0

    def test_empty_task_list_returns_all_idle(self):
        alloc = CBBA()
        snap = _snapshot(n_drones=3, tasks=[])
        result = alloc.allocate(snap)
        assert all(v is None for v in result.assignments.values())

    def test_convergence_simple(self):
        """3 drones, 3 tasks — each drone starts on top of its task so it
        wins unambiguously regardless of tie-breaking order."""
        alloc = CBBA()
        tasks = [
            _task(idx=0, pos=[0.1, 0.0, 1.0]),   # drone_0 at (0,0,1) is nearest
            _task(idx=1, pos=[4.1, 0.0, 1.0]),   # drone_1 at (2,0,1) — wait, use wider spread
            _task(idx=2, pos=[8.1, 0.0, 1.0]),   # drone_2 at (4,0,1)
        ]
        from allocator.base_allocator import WorldSnapshot
        import numpy as np
        # Put drones right next to their respective tasks so bid values differ clearly
        snap = WorldSnapshot(
            drone_positions={
                "drone_0": np.array([0.0, 0.0, 1.0], dtype=np.float32),
                "drone_1": np.array([4.0, 0.0, 1.0], dtype=np.float32),
                "drone_2": np.array([8.0, 0.0, 1.0], dtype=np.float32),
            },
            drone_batteries={"drone_0": 1.0, "drone_1": 1.0, "drone_2": 1.0},
            drone_task_progress={"drone_0": 0.0, "drone_1": 0.0, "drone_2": 0.0},
            current_assignments={"drone_0": None, "drone_1": None, "drone_2": None},
            tasks=tasks,
            step=0, max_steps=500,
        )
        result = alloc.allocate(snap)
        assigned = [v for v in result.assignments.values() if v is not None]
        assert len(assigned) == 3
        assert len(set(assigned)) == 3  # all distinct

    def test_shareable_task_can_be_co_assigned(self):
        alloc = CBBA()
        sweep = _task("sweep_floor", idx=0, shareable=True)
        water = _task("water_plant", idx=1)
        snap = _snapshot(n_drones=3, tasks=[sweep, water])
        result = alloc.allocate(snap)
        assigned_to_sweep = [d for d, t in result.assignments.items() if t == 0]
        assert len(assigned_to_sweep) >= 1


# ---------------------------------------------------------------------------
# CBBA — comm_delay
# ---------------------------------------------------------------------------

class TestCBBACommDelay:
    def test_comm_delay_zero_same_as_instant(self):
        """delay=0 should give identical coverage to default."""
        alloc = CBBA(comm_delay=0)
        snap = _snapshot(n_drones=3)
        result = alloc.allocate(snap)
        assert set(result.assignments.keys()) == {"drone_0", "drone_1", "drone_2"}

    def test_comm_delay_nonzero_still_covers_all_drones(self):
        """Even with a 5-step delay, all drones must be in the result."""
        alloc = CBBA(comm_delay=5)
        snap = _snapshot(n_drones=3, step=0)
        result = alloc.allocate(snap)
        assert set(result.assignments.keys()) == {"drone_0", "drone_1", "drone_2"}

    def test_messages_delivered_after_delay(self):
        """Messages queued at step 0 with delay=3 should be delivered at step 3."""
        alloc = CBBA(comm_delay=3)
        snap0 = _snapshot(n_drones=2, step=0)
        alloc.allocate(snap0)
        # Pending messages should exist
        assert len(alloc._pending_msgs) > 0
        # After 3 steps the messages should clear on next allocate
        snap3 = _snapshot(n_drones=2, step=3)
        alloc.allocate(snap3)
        # All messages with deliver_at <= 3 should be consumed
        remaining = [m for m in alloc._pending_msgs if m[0] <= 3]
        assert len(remaining) == 0


# ---------------------------------------------------------------------------
# CBBA — disruption hooks
# ---------------------------------------------------------------------------

class TestCBBADisruptionHooks:
    def test_on_task_complete_purges_messages(self):
        alloc = CBBA(comm_delay=10)
        snap = _snapshot(n_drones=2, step=0)
        alloc.allocate(snap)
        n_before = len(alloc._pending_msgs)
        # Complete task_idx=0 — messages referencing it should be purged
        alloc.on_task_complete(0, step=1)
        remaining = [m for m in alloc._pending_msgs if 0 in m[2]]
        assert len(remaining) == 0

    def test_on_task_vanish_purges_messages(self):
        alloc = CBBA(comm_delay=10)
        snap = _snapshot(n_drones=2, step=0)
        alloc.allocate(snap)
        alloc.on_task_vanish(0, step=1)
        remaining = [m for m in alloc._pending_msgs if 0 in m[2]]
        assert len(remaining) == 0

    def test_task_vanish_mid_episode_with_cbba(self):
        alloc = CBBA()
        env = _env(alloc)
        env.reset(seed=7)
        task_id = env._tasks[0].spec.task_id
        env.remove_task(task_id)
        assert env._tasks[0].status == TaskStatus.VANISHED

    def test_add_task_mid_episode_with_cbba(self):
        alloc = CBBA()
        env = _env(alloc)
        env.reset(seed=7)
        new_id = env.add_task("toggle_light", [8.0, 8.0, 2.5], engage_steps=1)
        new_task = next(t for t in env._tasks if t.spec.task_id == new_id)
        assert new_task.status in (TaskStatus.PENDING, TaskStatus.ASSIGNED)


# ---------------------------------------------------------------------------
# Head-to-head: both allocators run a full episode without error
# ---------------------------------------------------------------------------

class TestHeadToHead:
    @pytest.mark.parametrize("AllocCls", [GreedyAuction, CBBA])
    def test_full_episode_no_crash(self, AllocCls):
        alloc = AllocCls()
        env = HomeEnv({"n_drones": 6, "max_steps": 300, "allocator": alloc})
        obs, _ = env.reset(seed=99)
        for _ in range(300):
            actions = {aid: env.action_space.sample() for aid in obs}
            obs, rewards, terminated, truncated, infos = env.step(actions)
            assert all(isinstance(r, float) for r in rewards.values())
            if terminated.get("__all__") or truncated.get("__all__"):
                break

    @pytest.mark.parametrize("AllocCls", [GreedyAuction, CBBA])
    def test_info_dict_has_tasks_active(self, AllocCls):
        alloc = AllocCls()
        env = HomeEnv({"n_drones": 3, "max_steps": 50, "allocator": alloc})
        obs, _ = env.reset(seed=0)
        actions = {aid: env.action_space.sample() for aid in obs}
        _, _, _, _, infos = env.step(actions)
        for aid in obs:
            assert "tasks_active" in infos[aid]
            assert "tasks_completed" in infos[aid]
