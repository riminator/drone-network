"""
tests/test_phase4.py
Tests for Phase 4 — disruption-scenario evaluation harness.

Covers:
  - EpisodeMetrics dataclass: fields present, csv roundtrip
  - DisruptionHooks: each fires exactly once at the right step
  - _CountingAllocator: wraps allocator and increments count
  - run_episode: returns EpisodeMetrics with correct shapes/types
  - scenario_config: comm_delay swaps CBBA, others unchanged
  - benchmark: smoke-runs a 1-episode single-cell and returns metrics list
  - print_table / write_csv: no crash, CSV has correct columns

Run with:  python -m pytest tests/test_phase4.py -v
"""

from __future__ import annotations

import csv
import io
import sys
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import pytest

from evaluation.eval_allocation import (
    EpisodeMetrics,
    SIX_SCENARIOS,
    _ALLOCATOR_ORDER,
    _CountingAllocator,
    _make_hook_task_vanish,
    _make_hook_task_inject,
    _make_hook_drone_failure,
    _make_hook_surge,
    _null_hook,
    benchmark,
    build_allocators,
    print_table,
    run_episode,
    scenario_config,
    write_csv,
)
from allocator.greedy_auction import GreedyAuction
from allocator.cbba import CBBA
from allocator.oracle import OracleAllocator
from allocator.learned_bidder import LearnedBidder
from allocator.bid_policy import BidPolicy
from envs.home_env import HomeEnv
from envs.tasks.base_task import TaskStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_env(n_drones=3, max_steps=100) -> HomeEnv:
    return HomeEnv({"n_drones": n_drones, "max_steps": max_steps})


def _run_one(allocator=None, scenario="baseline", n_drones=3, max_steps=200, seed=0):
    """Run a single episode and return EpisodeMetrics."""
    if allocator is None:
        allocator = GreedyAuction()
    factories = build_allocators(None)
    hook_factory, allocs = scenario_config(scenario, factories)
    hook = hook_factory()
    return run_episode(
        allocator_name="greedy",
        allocator=allocator,
        scenario=scenario,
        hook=hook,
        n_drones=n_drones,
        max_steps=max_steps,
        seed=seed,
        episode=1,
    )


# ---------------------------------------------------------------------------
# EpisodeMetrics dataclass
# ---------------------------------------------------------------------------

class TestEpisodeMetrics:
    def test_all_expected_fields_present(self):
        field_names = {f.name for f in fields(EpisodeMetrics)}
        for expected in [
            "allocator", "scenario", "episode",
            "tasks_total", "tasks_completed", "tasks_vanished", "completion_rate",
            "makespan", "steps_taken",
            "total_reward", "mean_reward_step",
            "mean_battery_final", "min_battery_final",
            "collision_events", "realloc_count", "wall_secs",
        ]:
            assert expected in field_names, f"Missing field: {expected}"

    def test_asdict_round_trip(self):
        m = EpisodeMetrics(allocator="greedy", scenario="baseline", episode=1,
                           tasks_total=5, completion_rate=0.8, makespan=120)
        d = asdict(m)
        assert d["allocator"] == "greedy"
        assert d["completion_rate"] == pytest.approx(0.8)

    def test_defaults_are_zero_or_empty(self):
        m = EpisodeMetrics(allocator="x", scenario="y", episode=1)
        assert m.tasks_total == 0
        assert m.total_reward == pytest.approx(0.0)
        assert m.collision_events == 0


# ---------------------------------------------------------------------------
# Six scenarios enumerated
# ---------------------------------------------------------------------------

class TestSixScenarios:
    def test_six_scenarios_constant(self):
        assert len(SIX_SCENARIOS) == 6
        for name in ["baseline", "task_vanish", "task_inject",
                     "drone_failure", "comm_delay", "surge"]:
            assert name in SIX_SCENARIOS

    def test_four_allocators_constant(self):
        for name in ["greedy", "cbba", "oracle", "learned"]:
            assert name in _ALLOCATOR_ORDER


# ---------------------------------------------------------------------------
# Disruption hooks
# ---------------------------------------------------------------------------

class TestDisruptionHooks:
    def test_null_hook_no_side_effects(self):
        env = _small_env()
        env.reset(seed=0)
        n_before = len(env._tasks)
        _null_hook(env, 1)
        _null_hook(env, 50)
        assert len(env._tasks) == n_before

    def test_task_vanish_fires_once_at_step_50(self):
        env = _small_env()
        env.reset(seed=0)
        hook = _make_hook_task_vanish()
        n_before = len(env._tasks)
        # Steps before trigger — no vanish
        for s in range(1, 50):
            hook(env, s)
        assert all(t.status != TaskStatus.VANISHED for t in env._tasks[:2])
        # Step 50 — should vanish task index 1
        hook(env, 50)
        assert env._tasks[1].status == TaskStatus.VANISHED
        # Step 51 — should NOT vanish again (guard)
        hook(env, 51)
        vanished_count = sum(1 for t in env._tasks if t.status == TaskStatus.VANISHED)
        assert vanished_count == 1

    def test_task_inject_fires_once_at_step_60(self):
        env = _small_env()
        env.reset(seed=0)
        hook = _make_hook_task_inject()
        n_before = len(env._tasks)
        for s in range(1, 60):
            hook(env, s)
        assert len(env._tasks) == n_before
        hook(env, 60)
        assert len(env._tasks) == n_before + 1
        hook(env, 61)
        assert len(env._tasks) == n_before + 1  # no second injection

    def test_drone_failure_zeroes_battery_at_step_40(self):
        env = _small_env(n_drones=3)
        env.reset(seed=0)
        hook = _make_hook_drone_failure()
        for s in range(1, 40):
            hook(env, s)
        assert env._drones["drone_1"].state.battery > 0
        hook(env, 40)
        assert env._drones["drone_1"].state.battery == 0.0
        assert env._drones["drone_1"].state.is_done is True

    def test_surge_fires_twice(self):
        env = _small_env()
        env.reset(seed=0)
        hook = _make_hook_surge()
        n0 = len(env._tasks)
        hook(env, 30)
        assert len(env._tasks) == n0 + 1
        hook(env, 80)
        assert len(env._tasks) == n0 + 2
        # Third fire at step 80 again — no change
        hook(env, 80)
        assert len(env._tasks) == n0 + 2


# ---------------------------------------------------------------------------
# _CountingAllocator
# ---------------------------------------------------------------------------

class TestCountingAllocator:
    def test_count_starts_zero(self):
        ca = _CountingAllocator(GreedyAuction())
        assert ca.count == 0

    def test_count_increments_per_allocate(self):
        ca = _CountingAllocator(GreedyAuction())
        env = _small_env()
        env.reset(seed=0)
        snap = env._assign_tasks.__func__  # don't call; just test via run_episode
        # Use run_episode which wraps the allocator
        m = _run_one(allocator=GreedyAuction(), max_steps=50)
        assert m.realloc_count >= 1  # at least the initial reset call

    def test_delegates_to_inner(self):
        inner = OracleAllocator()
        ca = _CountingAllocator(inner)
        env = _small_env()
        env.reset(seed=0)
        from allocator.base_allocator import WorldSnapshot
        snap = WorldSnapshot(
            drone_positions={"d0": np.zeros(3)},
            drone_batteries={"d0": 1.0},
            drone_task_progress={"d0": 0.0},
            current_assignments={"d0": None},
            tasks=env._tasks,
            step=0, max_steps=500,
        )
        result = ca.allocate(snap)
        assert ca.count == 1
        assert "d0" in result.assignments

    def test_hooks_forwarded(self):
        inner = CBBA()
        ca = _CountingAllocator(inner)
        # Should not raise
        ca.on_task_complete(0, 1)
        ca.on_task_vanish(0, 1)


# ---------------------------------------------------------------------------
# run_episode
# ---------------------------------------------------------------------------

class TestRunEpisode:
    def test_returns_episode_metrics(self):
        m = _run_one()
        assert isinstance(m, EpisodeMetrics)

    def test_allocator_name_preserved(self):
        factories = build_allocators(None)
        hook_factory, allocs = scenario_config("baseline", factories)
        m = run_episode(
            allocator_name="oracle",
            allocator=OracleAllocator(),
            scenario="baseline",
            hook=hook_factory(),
            n_drones=3,
            max_steps=200,
            seed=1,
            episode=2,
        )
        assert m.allocator == "oracle"
        assert m.scenario == "baseline"
        assert m.episode == 2

    def test_steps_taken_positive(self):
        m = _run_one(max_steps=50)
        assert m.steps_taken > 0

    def test_completion_rate_in_range(self):
        m = _run_one(max_steps=200)
        assert 0.0 <= m.completion_rate <= 1.0

    def test_battery_final_normalised(self):
        m = _run_one(max_steps=50)
        assert 0.0 <= m.mean_battery_final <= 1.0
        assert 0.0 <= m.min_battery_final <= 1.0

    def test_realloc_count_positive(self):
        m = _run_one(max_steps=100)
        assert m.realloc_count >= 1

    def test_wall_secs_positive(self):
        m = _run_one(max_steps=30)
        assert m.wall_secs > 0.0

    def test_tasks_total_matches_env(self):
        # baseline has 5 tasks; run without disruption
        m = _run_one(scenario="baseline", max_steps=50)
        assert m.tasks_total == 5

    def test_task_inject_increases_tasks_total(self):
        # With task_inject scenario, a 6th task appears at step 60
        m = _run_one(scenario="task_inject", max_steps=200)
        assert m.tasks_total >= 6

    def test_task_vanish_reflected(self):
        # task_vanish removes one task
        m = _run_one(scenario="task_vanish", max_steps=200)
        assert m.tasks_vanished >= 1

    def test_drone_failure_scenario_runs(self):
        # Just check it doesn't crash
        m = _run_one(scenario="drone_failure", n_drones=3, max_steps=150)
        assert m.steps_taken > 0

    @pytest.mark.parametrize("scenario", SIX_SCENARIOS)
    def test_all_scenarios_run_without_crash(self, scenario):
        m = _run_one(scenario=scenario, max_steps=80)
        assert isinstance(m, EpisodeMetrics)


# ---------------------------------------------------------------------------
# scenario_config
# ---------------------------------------------------------------------------

class TestScenarioConfig:
    def test_baseline_has_all_four_allocators(self):
        factories = build_allocators(None)
        _, allocs = scenario_config("baseline", factories)
        for name in ["greedy", "cbba", "oracle", "learned"]:
            assert name in allocs

    def test_comm_delay_replaces_cbba(self):
        factories = build_allocators(None)
        _, allocs = scenario_config("comm_delay", factories)
        # cbba should still be present (swapped to delayed variant)
        assert "cbba" in allocs
        # _cbba_delayed should NOT be a separate key
        assert "_cbba_delayed" not in allocs

    def test_comm_delay_cbba_has_delay(self):
        factories = build_allocators(None, comm_delay=7)
        _, allocs = scenario_config("comm_delay", factories)
        cbba_instance = allocs["cbba"]()
        assert isinstance(cbba_instance, CBBA)
        assert cbba_instance.comm_delay == 7

    def test_non_comm_delay_cbba_has_no_delay(self):
        factories = build_allocators(None)
        _, allocs = scenario_config("baseline", factories)
        cbba_instance = allocs["cbba"]()
        assert cbba_instance.comm_delay == 0

    def test_private_key_not_exposed(self):
        factories = build_allocators(None)
        for scenario in SIX_SCENARIOS:
            _, allocs = scenario_config(scenario, factories)
            assert "_cbba_delayed" not in allocs


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------

class TestBenchmark:
    def test_returns_list_of_episode_metrics(self):
        factories = build_allocators(None)
        results = benchmark(
            scenarios=["baseline"],
            allocator_factories={k: v for k, v in factories.items()
                                 if k in {"greedy", "_cbba_delayed"} or k == "greedy"},
            n_episodes=1,
            n_drones=3,
            max_steps=80,
            base_seed=0,
        )
        assert isinstance(results, list)
        assert all(isinstance(m, EpisodeMetrics) for m in results)

    def test_cell_count_correct(self):
        """1 scenario × 2 allocators × 2 episodes = 4 rows."""
        factories = build_allocators(None)
        allocs = {k: factories[k] for k in ["greedy", "oracle"]}
        results = benchmark(
            scenarios=["baseline"],
            allocator_factories=allocs,
            n_episodes=2,
            n_drones=3,
            max_steps=60,
            base_seed=7,
        )
        assert len(results) == 4

    def test_episode_indices_correct(self):
        factories = build_allocators(None)
        allocs = {k: factories[k] for k in ["greedy"]}
        results = benchmark(
            scenarios=["baseline"],
            allocator_factories=allocs,
            n_episodes=3,
            n_drones=3,
            max_steps=50,
            base_seed=99,
        )
        episodes = [m.episode for m in results]
        assert sorted(episodes) == [1, 2, 3]


# ---------------------------------------------------------------------------
# print_table / write_csv
# ---------------------------------------------------------------------------

class TestReporting:
    def _make_metrics(self) -> list[EpisodeMetrics]:
        return [
            EpisodeMetrics(
                allocator=a, scenario=s, episode=1,
                tasks_total=5, tasks_completed=4, completion_rate=0.8,
                makespan=120, steps_taken=120, total_reward=-10.0,
                mean_battery_final=0.7, realloc_count=3,
            )
            for a in ["greedy", "cbba", "oracle", "learned"]
            for s in ["baseline", "task_vanish"]
        ]

    def test_print_table_no_crash(self, capsys):
        metrics = self._make_metrics()
        print_table(metrics, ["baseline", "task_vanish"])
        captured = capsys.readouterr()
        assert "baseline" in captured.out
        assert "CompRate" in captured.out
        assert "Makespan" in captured.out

    def test_print_table_shows_all_allocators(self, capsys):
        metrics = self._make_metrics()
        print_table(metrics, ["baseline"])
        captured = capsys.readouterr()
        for a in ["greedy", "cbba", "oracle", "learned"]:
            assert a in captured.out

    def test_write_csv_creates_file(self, tmp_path):
        metrics = self._make_metrics()
        out = str(tmp_path / "results.csv")
        write_csv(metrics, out)
        assert Path(out).exists()

    def test_write_csv_has_header_row(self, tmp_path):
        metrics = self._make_metrics()
        out = str(tmp_path / "out.csv")
        write_csv(metrics, out)
        with open(out) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == len(metrics)
        assert "allocator" in rows[0]
        assert "completion_rate" in rows[0]
        assert "makespan" in rows[0]

    def test_write_csv_creates_parent_dirs(self, tmp_path):
        metrics = self._make_metrics()
        deep = str(tmp_path / "a" / "b" / "c" / "results.csv")
        write_csv(metrics, deep)
        assert Path(deep).exists()

    def test_write_csv_empty_no_crash(self, tmp_path):
        out = str(tmp_path / "empty.csv")
        write_csv([], out)
        # File not created for empty list — just no exception
