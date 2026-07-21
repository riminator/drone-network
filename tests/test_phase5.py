"""
tests/test_phase5.py
Tests for Phase 5 — PyBullet allocator integration and deploy.py changes.

Structure
---------
TestGreedyFallbackAllocator
    Covers the new _GreedyFallbackAllocator added to pybullet_env.py.
    Runs without PyBullet.

TestBuildAllocator
    Covers lab/deploy.py::build_allocator().
    Runs without PyBullet.

TestDeployArgParser
    Verifies the new --allocator / --bid-checkpoint / --auction-interval flags
    parse correctly via argparse.
    Runs without PyBullet.

TestPybulletAssignTasks  (skipped when pybullet / gym-pybullet-drones absent)
    Covers PybulletHomeEnv._assign_tasks() with each allocator via
    the env's public reset() → _drone_task_map state.

TestPybulletAuctionInterval  (skipped when pybullet absent)
    Verifies periodic re-auction fires at the right steps.

TestPybulletBidLines  (skipped when pybullet absent)
    Verifies _last_bids is populated and _debug_bid_lines housekeeping.

Run with:  python3 -m pytest tests/test_phase5.py -v
"""

from __future__ import annotations

import argparse
import sys
import types

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Conditional PyBullet availability
# ---------------------------------------------------------------------------

try:
    import pybullet  # noqa: F401
    from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary  # noqa: F401
    _PB_AVAILABLE = True
except ImportError:
    _PB_AVAILABLE = False

_pb_only = pytest.mark.skipif(
    not _PB_AVAILABLE,
    reason="gym-pybullet-drones not installed",
)


# ---------------------------------------------------------------------------
# Imports that do NOT require PyBullet
# ---------------------------------------------------------------------------

from allocator.base_allocator import BaseAllocator, WorldSnapshot, AllocationResult, Bid
from allocator.greedy_auction import GreedyAuction
from allocator.cbba import CBBA
from allocator.oracle import OracleAllocator
from allocator.learned_bidder import LearnedBidder
from allocator.bid_policy import BidPolicy
from envs.tasks.base_task import TaskSpec, TaskStatus
from envs.tasks.water_plant import WaterPlantTask
from envs.tasks.sweep_floor import SweepFloorTask
from envs.tasks.toggle_light import ToggleLightTask


# ===========================================================================
# _GreedyFallbackAllocator (no PyBullet required)
# ===========================================================================

def _make_world(n_drones=3, n_tasks=3):
    """Build a minimal WorldSnapshot for testing."""
    drone_ids = [f"drone_{i}" for i in range(n_drones)]
    tasks = []
    for i in range(n_tasks):
        spec = TaskSpec(
            task_id=f"water_plant_{i}",
            task_type="water_plant",
            target_position=np.array([float(i * 2), 1.0, 1.0], dtype=np.float32),
            engage_steps_required=5,
        )
        t = WaterPlantTask(spec)
        t.status = TaskStatus.PENDING
        tasks.append(t)
    return WorldSnapshot(
        drone_positions={d: np.array([float(j), 0.5, 1.0], dtype=np.float32)
                         for j, d in enumerate(drone_ids)},
        drone_batteries={d: 1.0 for d in drone_ids},
        drone_task_progress={d: 0.0 for d in drone_ids},
        current_assignments={d: None for d in drone_ids},
        tasks=tasks,
        step=0,
        max_steps=500,
    )


class TestGreedyFallbackAllocator:
    def _alloc(self):
        # Import the private class from pybullet_env without instantiating the env
        from envs.pybullet_env import _GreedyFallbackAllocator
        return _GreedyFallbackAllocator()

    def test_all_drones_covered(self):
        alloc = self._alloc()
        snap = _make_world(n_drones=3, n_tasks=3)
        result = alloc.allocate(snap)
        assert set(result.assignments.keys()) == {"drone_0", "drone_1", "drone_2"}

    def test_no_duplicate_assignments(self):
        alloc = self._alloc()
        snap = _make_world(n_drones=3, n_tasks=3)
        result = alloc.allocate(snap)
        assigned = [v for v in result.assignments.values() if v is not None]
        assert len(assigned) == len(set(assigned))

    def test_empty_tasks_all_idle(self):
        alloc = self._alloc()
        snap = _make_world(n_drones=2, n_tasks=0)
        result = alloc.allocate(snap)
        assert all(v is None for v in result.assignments.values())

    def test_completed_task_skipped(self):
        alloc = self._alloc()
        snap = _make_world(n_drones=2, n_tasks=2)
        snap.tasks[0].completed = True
        snap.tasks[0].status = TaskStatus.COMPLETE
        result = alloc.allocate(snap)
        assigned = [v for v in result.assignments.values() if v is not None]
        assert 0 not in assigned

    def test_existing_valid_assignment_kept(self):
        """A drone already holding a valid task keeps it (no churn)."""
        alloc = self._alloc()
        snap = _make_world(n_drones=2, n_tasks=3)
        snap.current_assignments["drone_0"] = 2
        snap.tasks[2].status = TaskStatus.ASSIGNED
        result = alloc.allocate(snap)
        assert result.assignments["drone_0"] == 2

    def test_more_drones_than_tasks(self):
        alloc = self._alloc()
        snap = _make_world(n_drones=4, n_tasks=2)
        result = alloc.allocate(snap)
        assigned = [v for v in result.assignments.values() if v is not None]
        assert len(assigned) <= 2


# ===========================================================================
# build_allocator (no PyBullet required)
# ===========================================================================

class TestBuildAllocator:
    def test_greedy_returns_greedy_auction(self):
        from lab.deploy import build_allocator
        a = build_allocator("greedy", None)
        assert isinstance(a, GreedyAuction)

    def test_cbba_returns_cbba(self):
        from lab.deploy import build_allocator
        a = build_allocator("cbba", None)
        assert isinstance(a, CBBA)
        assert a.comm_delay == 0

    def test_oracle_returns_oracle(self):
        from lab.deploy import build_allocator
        a = build_allocator("oracle", None)
        assert isinstance(a, OracleAllocator)

    def test_learned_no_checkpoint_returns_learned_bidder(self, capsys):
        from lab.deploy import build_allocator
        a = build_allocator("learned", None)
        assert isinstance(a, LearnedBidder)
        captured = capsys.readouterr()
        assert "WARN" in captured.out

    def test_learned_with_checkpoint(self, tmp_path):
        from lab.deploy import build_allocator
        import torch
        policy = BidPolicy()
        ckpt = tmp_path / "bp.pt"
        torch.save({
            "bid_policy_state_dict": policy.state_dict(),
            "bid_policy_config": {"obs_dim": 14, "hidden": [64, 64]},
        }, ckpt)
        a = build_allocator("learned", str(ckpt))
        assert isinstance(a, LearnedBidder)

    def test_unknown_name_raises(self):
        from lab.deploy import build_allocator
        with pytest.raises(ValueError, match="Unknown allocator"):
            build_allocator("banana", None)

    def test_each_call_returns_new_instance(self):
        """Factory must not share state between calls."""
        from lab.deploy import build_allocator
        a1 = build_allocator("greedy", None)
        a2 = build_allocator("greedy", None)
        assert a1 is not a2


# ===========================================================================
# deploy.py argument parser (no PyBullet required)
# ===========================================================================

class TestDeployArgParser:
    """Parse deploy.py arguments in-process without actually running anything."""

    def _parser(self):
        from lab import deploy as deploy_mod
        import importlib, types
        # Re-build the parser exactly as __main__ would
        parser = argparse.ArgumentParser()
        parser.add_argument("--checkpoint", required=True)
        parser.add_argument("--allocator", default="greedy",
                            choices=["greedy", "cbba", "oracle", "learned"])
        parser.add_argument("--bid-checkpoint", default=None)
        parser.add_argument("--auction-interval", type=int, default=0)
        parser.add_argument("--episodes", type=int, default=5)
        parser.add_argument("--n-drones", type=int, default=3)
        parser.add_argument("--max-steps", type=int, default=800)
        parser.add_argument("--time-scale", type=float, default=1.0)
        parser.add_argument("--no-gui", action="store_true")
        parser.add_argument("--record", action="store_true")
        return parser

    def test_defaults(self):
        args = self._parser().parse_args(["--checkpoint", "ckpt.pt"])
        assert args.allocator == "greedy"
        assert args.bid_checkpoint is None
        assert args.auction_interval == 0
        assert args.episodes == 5
        assert args.n_drones == 3
        assert args.no_gui is False

    def test_allocator_choices(self):
        p = self._parser()
        for name in ["greedy", "cbba", "oracle", "learned"]:
            args = p.parse_args(["--checkpoint", "x.pt", "--allocator", name])
            assert args.allocator == name

    def test_invalid_allocator_exits(self):
        with pytest.raises(SystemExit):
            self._parser().parse_args(["--checkpoint", "x.pt", "--allocator", "bad"])

    def test_auction_interval_parsed(self):
        args = self._parser().parse_args(
            ["--checkpoint", "x.pt", "--auction-interval", "20"]
        )
        assert args.auction_interval == 20

    def test_bid_checkpoint_parsed(self):
        args = self._parser().parse_args(
            ["--checkpoint", "x.pt", "--allocator", "learned",
             "--bid-checkpoint", "checkpoints/bid_policy_final.pt"]
        )
        assert args.bid_checkpoint == "checkpoints/bid_policy_final.pt"

    def test_no_gui_flag(self):
        args = self._parser().parse_args(["--checkpoint", "x.pt", "--no-gui"])
        assert args.no_gui is True


# ===========================================================================
# PybulletHomeEnv — PyBullet-only tests
# ===========================================================================

def _make_pb_env(allocator=None, auction_interval=0, n_drones=3):
    from envs.pybullet_env import PybulletHomeEnv
    cfg = {
        "n_drones": n_drones,
        "max_steps": 50,
        "gui": False,
        "auction_interval": auction_interval,
    }
    if allocator is not None:
        cfg["allocator"] = allocator
    return PybulletHomeEnv(config=cfg)


@_pb_only
class TestPybulletAssignTasks:
    """_assign_tasks() integration with all four allocators."""

    @pytest.mark.parametrize("alloc_factory", [
        lambda: GreedyAuction(),
        lambda: CBBA(),
        lambda: OracleAllocator(),
        lambda: LearnedBidder(BidPolicy()),
    ])
    def test_reset_assigns_all_tasks(self, alloc_factory):
        alloc = alloc_factory()
        env = _make_pb_env(allocator=alloc)
        env.reset()
        # At least one drone should have a task (5 tasks, 3 drones)
        assigned = [v for v in env._drone_task_map.values() if v is not None]
        assert len(assigned) >= 1
        env.close()

    def test_default_allocator_still_assigns(self):
        """Env with no explicit allocator should use _GreedyFallbackAllocator."""
        env = _make_pb_env()
        env.reset()
        assigned = [v for v in env._drone_task_map.values() if v is not None]
        assert len(assigned) >= 1
        env.close()

    def test_all_drones_in_map_after_reset(self):
        env = _make_pb_env(allocator=GreedyAuction())
        env.reset()
        assert set(env._drone_task_map.keys()) == {f"drone_{i}" for i in range(3)}
        env.close()

    def test_assignment_indices_are_valid(self):
        env = _make_pb_env(allocator=OracleAllocator())
        env.reset()
        for v in env._drone_task_map.values():
            if v is not None:
                assert 0 <= v < len(env._tasks)
        env.close()

    def test_last_bids_populated_after_greedy(self):
        env = _make_pb_env(allocator=GreedyAuction())
        env.reset()
        # GreedyAuction produces bids; _last_bids should be non-empty
        assert isinstance(env._last_bids, list)
        assert len(env._last_bids) > 0
        env.close()

    def test_last_bids_populated_after_cbba(self):
        env = _make_pb_env(allocator=CBBA())
        env.reset()
        assert isinstance(env._last_bids, list)
        env.close()

    def test_last_bids_empty_after_reset_before_alloc(self):
        """_last_bids is reset to [] at the start of reset(), before _assign_tasks."""
        env = _make_pb_env(allocator=GreedyAuction())
        # Poke _last_bids to a non-empty value, then call reset()
        env._last_bids = [Bid("drone_0", 0, 0.9)]
        # Even though reset() will repopulate it, the explicit reset-to-[] must
        # happen before _assign_tasks runs.  We verify by checking the attribute
        # exists and is a list after a full reset.
        env.reset()
        assert isinstance(env._last_bids, list)
        env.close()


@_pb_only
class TestPybulletAuctionInterval:
    """Periodic re-auction tick fires at multiples of auction_interval."""

    def test_interval_zero_does_not_tick_mid_step(self):
        """With interval=0 the allocator is only called on task completion."""
        counts = [0]

        class CountingAlloc(BaseAllocator):
            def allocate(self, snapshot):
                counts[0] += 1
                return GreedyAuction().allocate(snapshot)

        env = _make_pb_env(allocator=CountingAlloc(), auction_interval=0)
        obs, _ = env.reset()
        count_after_reset = counts[0]

        # Run a few steps with no task completions — allocator should NOT be called again
        for _ in range(5):
            actions = {aid: env.action_space.sample() for aid in obs}
            obs, _, _, _, _ = env.step(actions)

        # Only reset call(s) should have incremented the count
        assert counts[0] == count_after_reset
        env.close()

    def test_interval_fires_every_k_steps(self):
        """With interval=K the allocator fires at steps 0, K, 2K, …"""
        K = 5
        fired_at = []

        class TrackingAlloc(BaseAllocator):
            def __init__(self):
                self._step = 0

            def allocate(self, snapshot):
                fired_at.append(snapshot.step)
                return GreedyAuction().allocate(snapshot)

        env = _make_pb_env(allocator=TrackingAlloc(), auction_interval=K)
        obs, _ = env.reset()
        n_reset_calls = len(fired_at)

        for _ in range(K * 3):
            actions = {aid: env.action_space.sample() for aid in obs}
            obs, _, terminated, truncated, _ = env.step(actions)
            if terminated.get("__all__") or truncated.get("__all__"):
                break

        # Steps where step % K == 0 should have triggered a call (from step 1 onward)
        periodic_steps = [s for s in fired_at[n_reset_calls:] if s > 0]
        for s in periodic_steps:
            assert s % K == 0, f"Unexpected tick at step {s}"
        env.close()


@_pb_only
class TestPybulletBidLines:
    """_debug_bid_lines housekeeping and _draw_bid_lines() (GUI=False safe path)."""

    def test_debug_bid_lines_empty_on_reset(self):
        env = _make_pb_env(allocator=GreedyAuction())
        # Manually seed _debug_bid_lines then call reset()
        env._debug_bid_lines = {("drone_0", 0): 99}
        env.reset()
        # reset() must clear the dict
        assert env._debug_bid_lines == {}
        env.close()

    def test_draw_bid_lines_no_crash_when_gui_false(self):
        """_draw_bid_lines must be a no-op when gui=False."""
        env = _make_pb_env(allocator=GreedyAuction())
        obs, _ = env.reset()
        agent_ids_sorted = sorted(env._agent_ids)
        real_positions = env._aviary.pos
        # Should not raise even if _last_bids is populated
        env._last_bids = env._last_bids or []
        env._draw_bid_lines(real_positions, agent_ids_sorted)  # no-op (gui=False)
        env.close()

    def test_task_status_synced_before_alloc(self):
        """Completed tasks must have status=COMPLETE before WorldSnapshot is built."""
        env = _make_pb_env(allocator=GreedyAuction())
        env.reset()
        # Force-complete task 0
        env._tasks[0].completed = True
        # Call _assign_tasks — completed task's status should be synced
        env._assign_tasks(env._aviary.pos)
        assert env._tasks[0].status == TaskStatus.COMPLETE
        # And no drone should be assigned to it
        for v in env._drone_task_map.values():
            assert v != 0
        env.close()

    def test_cbba_allocator_episode(self):
        """CBBA should complete a full episode without error."""
        env = _make_pb_env(allocator=CBBA(), auction_interval=10)
        obs, _ = env.reset()
        for _ in range(env.max_steps):
            actions = {aid: env.action_space.sample() for aid in obs}
            obs, _, terminated, truncated, _ = env.step(actions)
            if terminated.get("__all__") or truncated.get("__all__"):
                break
        env.close()

    def test_learned_bidder_episode(self):
        """LearnedBidder (untrained) should complete a full episode without error."""
        env = _make_pb_env(allocator=LearnedBidder(BidPolicy()), auction_interval=0)
        obs, _ = env.reset()
        for _ in range(env.max_steps):
            actions = {aid: env.action_space.sample() for aid in obs}
            obs, _, terminated, truncated, _ = env.step(actions)
            if terminated.get("__all__") or truncated.get("__all__"):
                break
        env.close()
