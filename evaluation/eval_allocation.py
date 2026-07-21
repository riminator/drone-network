"""
eval_allocation.py
Phase 4 disruption-scenario evaluation harness.

Pass --exec-checkpoint to drive movement with the trained actor instead of
random actions.  Without it the harness still works — random actions are used
and completion rates will be near zero, which is only useful for measuring
reallocation counts and collision rates in isolation.

Benchmarks four allocators — Greedy, CBBA, Oracle, Learned — across six
disruption scenarios.  Each (allocator × scenario) cell is averaged over
``--episodes`` independent episodes; results are printed as an EpisodeMetrics
table and optionally saved to a CSV.

Six disruption scenarios
------------------------
  baseline      — no disruption; standard 5-task layout
  task_vanish   — task_1 is removed at step 50 (drone mid-transit)
  task_inject   — a new water_plant task is injected at step 60
  drone_failure — drone_1's battery is zeroed at step 40 (mid-flight)
  comm_delay    — CBBA runs with 5-step comm_delay; others unaffected
  surge         — two extra tasks injected at steps 30 and 80

Usage
-----
    # Quick smoke run (3 episodes per cell)
    python -m evaluation.eval_allocation --episodes 3

    # Full benchmark with a trained checkpoint
    python -m evaluation.eval_allocation \\
        --checkpoint checkpoints/bid_policy_final.pt \\
        --episodes 30 --csv results/phase4.csv

    # Single scenario, all allocators
    python -m evaluation.eval_allocation --scenarios baseline task_vanish \\
        --episodes 10
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from allocator.base_allocator import BaseAllocator
from allocator.cbba import CBBA
from allocator.greedy_auction import GreedyAuction
from allocator.oracle import OracleAllocator
from allocator.learned_bidder import LearnedBidder
from allocator.bid_policy import BidPolicy
from envs.home_env import HomeEnv
from envs.tasks.base_task import TaskStatus


# ---------------------------------------------------------------------------
# EpisodeMetrics — the canonical output row for one (allocator, scenario, ep)
# ---------------------------------------------------------------------------

@dataclass
class EpisodeMetrics:
    """
    All metrics recorded for a single episode run.

    Aggregated fields (mean ± std) are printed in the summary table;
    the raw per-episode rows can be written to CSV for further analysis.
    """
    allocator:          str   # "greedy" | "cbba" | "oracle" | "learned"
    scenario:           str   # one of the SIX_SCENARIOS keys
    episode:            int   # 1-indexed

    # Task outcomes
    tasks_total:        int   = 0
    tasks_completed:    int   = 0
    tasks_vanished:     int   = 0
    completion_rate:    float = 0.0   # tasks_completed / tasks_eligible  (excl. vanished)

    # Timing
    makespan:           int   = 0     # steps until all tasks done (or max_steps if not)
    steps_taken:        int   = 0     # actual episode length

    # Efficiency
    total_reward:       float = 0.0
    mean_reward_step:   float = 0.0   # total_reward / steps_taken

    # Battery
    mean_battery_final: float = 0.0   # mean remaining battery across all drones at end
    min_battery_final:  float = 0.0

    # Collisions
    collision_events:   int   = 0     # approximate: counted from step reward signal

    # Reallocation
    realloc_count:      int   = 0     # how many times allocator.allocate() was called

    # Wall-clock
    wall_secs:          float = 0.0   # real time for this episode


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

# Default 5-task layout, mirroring HomeEnv._DEFAULT_TASK_LAYOUTS
_BASE_LAYOUT = [
    ("water_plant",  [2.0, 3.0, 1.0], 10, False),
    ("water_plant",  [7.0, 1.5, 1.0], 10, False),
    ("sweep_floor",  [5.0, 5.0, 0.3], 15, True),
    ("toggle_light", [9.0, 0.5, 2.5],  1, False),
    ("toggle_light", [0.5, 9.0, 2.5],  1, False),
]

#: The six scenario names (ordered for display)
SIX_SCENARIOS = [
    "baseline",
    "task_vanish",
    "task_inject",
    "drone_failure",
    "comm_delay",
    "surge",
]

# A DisruptionHook is called once per step:
#   hook(env, step_number) → None
# It may call env.remove_task(), env.add_task(), or mutate drone state.
DisruptionHook = Callable[[HomeEnv, int], None]


def _make_hook_task_vanish() -> DisruptionHook:
    """Remove task index 1 at step 50."""
    fired = [False]

    def hook(env: HomeEnv, step: int) -> None:
        if not fired[0] and step == 50 and len(env._tasks) > 1:
            env.remove_task(env._tasks[1].spec.task_id)
            fired[0] = True

    return hook


def _make_hook_task_inject() -> DisruptionHook:
    """Inject a new water_plant at step 60."""
    fired = [False]

    def hook(env: HomeEnv, step: int) -> None:
        if not fired[0] and step == 60:
            env.add_task("water_plant", [4.0, 7.0, 1.0], engage_steps=10)
            fired[0] = True

    return hook


def _make_hook_drone_failure() -> DisruptionHook:
    """Zero drone_1's battery at step 40, forcing it out of service."""
    fired = [False]

    def hook(env: HomeEnv, step: int) -> None:
        if not fired[0] and step == 40 and "drone_1" in env._drones:
            env._drones["drone_1"].state.battery = 0.0
            env._drones["drone_1"].state.is_done = True
            # Re-trigger reallocation so the other drones pick up the orphaned task
            env._assign_tasks()
            fired[0] = True

    return hook


def _make_hook_surge() -> DisruptionHook:
    """Inject two extra tasks — one at step 30, another at step 80."""
    fired = [False, False]

    def hook(env: HomeEnv, step: int) -> None:
        if not fired[0] and step == 30:
            env.add_task("water_plant", [1.0, 8.0, 1.0], engage_steps=8)
            fired[0] = True
        if not fired[1] and step == 80:
            env.add_task("sweep_floor", [8.0, 8.0, 0.3], engage_steps=12,
                         is_shareable=True)
            fired[1] = True

    return hook


def _null_hook(env: HomeEnv, step: int) -> None:
    """No disruption."""


# ---------------------------------------------------------------------------
# Counting allocator — wraps any BaseAllocator to count allocate() calls
# ---------------------------------------------------------------------------

class _CountingAllocator(BaseAllocator):
    """Thin wrapper that counts allocate() invocations."""

    def __init__(self, inner: BaseAllocator):
        self._inner = inner
        self.count: int = 0

    def allocate(self, snapshot):
        self.count += 1
        return self._inner.allocate(snapshot)

    def on_task_complete(self, task_idx: int, step: int) -> None:
        self._inner.on_task_complete(task_idx, step)

    def on_task_vanish(self, task_idx: int, step: int) -> None:
        self._inner.on_task_vanish(task_idx, step)


# ---------------------------------------------------------------------------
# Single-episode runner
# ---------------------------------------------------------------------------

def run_episode(
    allocator_name: str,
    allocator: BaseAllocator,
    scenario: str,
    hook: DisruptionHook,
    n_drones: int,
    max_steps: int,
    seed: int,
    episode: int,
    actor=None,  # optional trained Actor; falls back to random actions if None
) -> EpisodeMetrics:
    """
    Run one episode with the given allocator + disruption hook and return
    a fully-populated EpisodeMetrics.

    The env is created fresh each call so allocator state is not shared
    across episodes (CBBA's _pending_msgs would otherwise bleed between runs).
    """
    counting = _CountingAllocator(allocator)

    env = HomeEnv({
        "n_drones": n_drones,
        "max_steps": max_steps,
        "task_layouts": _BASE_LAYOUT,
        "allocator": counting,
    })

    t0 = time.perf_counter()
    obs, _ = env.reset(seed=seed)

    agent_ids = sorted(env._agent_ids)
    total_reward = 0.0
    collision_reward_total = 0.0
    steps = 0
    makespan = max_steps  # defaults to max if never all-done

    from envs.home_env import REWARD_COLLISION  # -5.0 per collision per drone

    done = False
    while not done:
        steps += 1

        # Fire disruption hook before the step
        hook(env, steps)

        # Only dispatch actions for drones that are still active.
        # Done drones (e.g. battery=0 after drone_failure) must be excluded so
        # the actor doesn't keep flying them into neighbours causing collisions.
        active_ids = [
            aid for aid in agent_ids
            if not env._drones[aid].state.is_done
        ]

        if actor is not None:
            with torch.no_grad():
                obs_t = torch.tensor(
                    np.stack([obs.get(aid, np.zeros(15)) for aid in active_ids]),
                    dtype=torch.float32,
                )
                sq, _, _ = actor.get_action(obs_t, deterministic=True)
                actions = {aid: sq[i].cpu().numpy() for i, aid in enumerate(active_ids)}
        else:
            actions = {aid: env.action_space.sample() for aid in active_ids}
        obs, rewards, terminated, truncated, infos = env.step(actions)

        ep_reward = sum(rewards.values())
        total_reward += ep_reward

        # Count collision events: each collision penalty is REWARD_COLLISION per drone,
        # so a pairwise collision emits 2 × REWARD_COLLISION.  We detect the signature.
        step_collision_signal = sum(
            1 for r in rewards.values()
            if abs(r - REWARD_COLLISION) < 0.5  # only the exact penalty value
        )
        # Each pair produces 2 hits; integer-divide to get pairwise count
        collision_reward_total += step_collision_signal // 2

        done = (
            terminated.get("__all__", False)
            or truncated.get("__all__", False)
        )

        # Record makespan when all non-vanished tasks complete
        if terminated.get("__all__", False) and makespan == max_steps:
            makespan = steps

    wall = time.perf_counter() - t0

    # Final task counts
    tasks_completed = sum(1 for t in env._tasks if t.status == TaskStatus.COMPLETE)
    tasks_vanished  = sum(1 for t in env._tasks if t.status == TaskStatus.VANISHED)
    tasks_eligible  = len(env._tasks) - tasks_vanished
    completion_rate = tasks_completed / max(tasks_eligible, 1)

    # Final battery stats
    batteries = [d.state.battery for d in env._drones.values()]
    from envs.drone_agent import MAX_BATTERY
    batteries_norm = [b / MAX_BATTERY for b in batteries]

    return EpisodeMetrics(
        allocator=allocator_name,
        scenario=scenario,
        episode=episode,
        tasks_total=len(env._tasks),
        tasks_completed=tasks_completed,
        tasks_vanished=tasks_vanished,
        completion_rate=completion_rate,
        makespan=makespan,
        steps_taken=steps,
        total_reward=total_reward,
        mean_reward_step=total_reward / max(steps, 1),
        mean_battery_final=float(np.mean(batteries_norm)),
        min_battery_final=float(np.min(batteries_norm)),
        collision_events=int(collision_reward_total),
        realloc_count=counting.count,
        wall_secs=wall,
    )


# ---------------------------------------------------------------------------
# Allocator factory
# ---------------------------------------------------------------------------

def build_allocators(
    checkpoint: str | None,
    comm_delay: int = 5,
) -> dict[str, Callable[[], BaseAllocator]]:
    """
    Return a dict of name → factory (zero-arg callable).
    Factories are called fresh per episode so stateful allocators (CBBA)
    start clean each run.

    ``comm_delay`` is used only for CBBA in the *comm_delay* scenario;
    all other scenarios always instantiate CBBA with comm_delay=0.
    """
    def _greedy():
        return GreedyAuction()

    def _cbba_clean():
        return CBBA(comm_delay=0)

    def _cbba_delayed():
        return CBBA(comm_delay=comm_delay)

    def _oracle():
        return OracleAllocator()

    def _learned():
        if checkpoint:
            return LearnedBidder.from_checkpoint(checkpoint)
        # No checkpoint — use an untrained policy (random bids)
        return LearnedBidder(BidPolicy())

    return {
        "greedy":  _greedy,
        "cbba":    _cbba_clean,
        "oracle":  _oracle,
        "learned": _learned,
        # Special variant used only for comm_delay scenario
        "_cbba_delayed": _cbba_delayed,
    }


# ---------------------------------------------------------------------------
# Scenario → (hook factory, allocator overrides)
# ---------------------------------------------------------------------------

def scenario_config(scenario: str, allocator_factories: dict) -> tuple[
    Callable[[], DisruptionHook],
    dict[str, Callable[[], BaseAllocator]],
]:
    """
    Return (hook_factory, alloc_factories_for_this_scenario).

    For the comm_delay scenario CBBA is replaced with the delayed variant.
    All other scenarios use the standard allocators.
    """
    hook_map: dict[str, Callable[[], DisruptionHook]] = {
        "baseline":     lambda: _null_hook,
        "task_vanish":  _make_hook_task_vanish,
        "task_inject":  _make_hook_task_inject,
        "drone_failure": _make_hook_drone_failure,
        "comm_delay":   lambda: _null_hook,  # hook is no-op; CBBA delay is structural
        "surge":        _make_hook_surge,
    }

    hook_factory = hook_map[scenario]

    # For comm_delay, swap CBBA → delayed CBBA
    allocs = dict(allocator_factories)
    if scenario == "comm_delay":
        allocs = {k: (v if k != "cbba" else allocs["_cbba_delayed"])
                  for k, v in allocs.items()
                  if k != "_cbba_delayed"}
    else:
        allocs = {k: v for k, v in allocs.items() if k != "_cbba_delayed"}

    return hook_factory, allocs


# ---------------------------------------------------------------------------
# Benchmark driver
# ---------------------------------------------------------------------------

def benchmark(
    scenarios: list[str],
    allocator_factories: dict[str, Callable[[], BaseAllocator]],
    n_episodes: int,
    n_drones: int,
    max_steps: int,
    base_seed: int,
    actor=None,
) -> list[EpisodeMetrics]:
    """Run the full grid and return all EpisodeMetrics rows."""
    all_metrics: list[EpisodeMetrics] = []

    total_cells = len(scenarios) * len([k for k in allocator_factories if not k.startswith("_")])
    cell = 0

    for scenario in scenarios:
        hook_factory, allocs = scenario_config(scenario, allocator_factories)

        for alloc_name, alloc_factory in allocs.items():
            cell += 1
            cell_metrics: list[EpisodeMetrics] = []

            for ep in range(1, n_episodes + 1):
                seed = base_seed + (cell * 1000) + ep
                allocator = alloc_factory()          # fresh instance per episode
                hook = hook_factory()                # fresh hook state per episode

                m = run_episode(
                    allocator_name=alloc_name,
                    allocator=allocator,
                    scenario=scenario,
                    hook=hook,
                    n_drones=n_drones,
                    max_steps=max_steps,
                    seed=seed,
                    episode=ep,
                    actor=actor,
                )
                cell_metrics.append(m)

            all_metrics.extend(cell_metrics)

            # Progress line
            cr   = statistics.mean(m.completion_rate for m in cell_metrics)
            ms   = statistics.mean(m.makespan        for m in cell_metrics)
            rwd  = statistics.mean(m.total_reward     for m in cell_metrics)
            print(
                f"  [{cell:2d}/{total_cells}] {alloc_name:8s} | {scenario:14s} "
                f"| cr={cr:.2f} ms={ms:6.1f} rwd={rwd:+8.1f}"
            )

    return all_metrics


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _agg(rows: list[EpisodeMetrics], field_name: str) -> tuple[float, float]:
    """Return (mean, std) for a numeric field across rows."""
    vals = [getattr(r, field_name) for r in rows]
    if not vals:
        return 0.0, 0.0
    mean = statistics.mean(vals)
    std = statistics.pstdev(vals)
    return mean, std


_ALLOCATOR_ORDER = ["greedy", "cbba", "oracle", "learned"]
_METRIC_COLS = [
    ("completion_rate", "CompRate"),
    ("makespan",        "Makespan"),
    ("total_reward",    "TotReward"),
    ("mean_battery_final", "BattFinal"),
    ("realloc_count",   "Reallocs"),
    ("collision_events","Collisions"),
]


def print_table(all_metrics: list[EpisodeMetrics], scenarios: list[str]) -> None:
    """Print the full EpisodeMetrics comparison table to stdout."""
    # Group by (scenario, allocator)
    from collections import defaultdict
    groups: dict[tuple[str, str], list[EpisodeMetrics]] = defaultdict(list)
    for m in all_metrics:
        groups[(m.scenario, m.allocator)].append(m)

    col_w = 14
    hdr_alloc = "  ".join(f"{a:>{col_w}}" for a in _ALLOCATOR_ORDER)
    metric_lbl_w = 12

    for scenario in scenarios:
        print(f"\n{'=' * 80}")
        print(f"  Scenario: {scenario}")
        print(f"{'=' * 80}")
        print(f"  {'Metric':{metric_lbl_w}s}  {hdr_alloc}")
        print(f"  {'-' * (metric_lbl_w + 2 + len(hdr_alloc))}")

        for field_name, label in _METRIC_COLS:
            row_parts = []
            for alloc in _ALLOCATOR_ORDER:
                rows = groups.get((scenario, alloc), [])
                if rows:
                    mean, std = _agg(rows, field_name)
                    cell = f"{mean:7.2f}±{std:5.2f}"
                else:
                    cell = "     —      "
                row_parts.append(f"{cell:>{col_w}}")
            print(f"  {label:{metric_lbl_w}s}  {'  '.join(row_parts)}")

    print(f"\n{'=' * 80}")
    print("  Overall summary (mean across all scenarios)")
    print(f"{'=' * 80}")
    print(f"  {'Metric':{metric_lbl_w}s}  {hdr_alloc}")
    print(f"  {'-' * (metric_lbl_w + 2 + len(hdr_alloc))}")
    for field_name, label in _METRIC_COLS:
        row_parts = []
        for alloc in _ALLOCATOR_ORDER:
            rows = [m for m in all_metrics if m.allocator == alloc]
            if rows:
                mean, std = _agg(rows, field_name)
                cell = f"{mean:7.2f}±{std:5.2f}"
            else:
                cell = "     —      "
            row_parts.append(f"{cell:>{col_w}}")
        print(f"  {label:{metric_lbl_w}s}  {'  '.join(row_parts)}")
    print()


def write_csv(all_metrics: list[EpisodeMetrics], path: str) -> None:
    """Write all per-episode rows to a CSV file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(m) for m in all_metrics]
    if not rows:
        return
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results written to {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 4 — disruption-scenario allocator benchmark"
    )
    parser.add_argument(
        "--exec-checkpoint", default=None,
        help="Path to actor .pt checkpoint for policy-driven movement "
             "(omit to use random actions)",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to bid_policy .pt checkpoint (omit to use untrained policy)",
    )
    parser.add_argument(
        "--episodes", type=int, default=10,
        help="Episodes per (allocator × scenario) cell  [default: 10]",
    )
    parser.add_argument(
        "--scenarios", nargs="+", default=SIX_SCENARIOS,
        choices=SIX_SCENARIOS,
        help="Subset of scenarios to run  [default: all six]",
    )
    parser.add_argument(
        "--allocators", nargs="+", default=_ALLOCATOR_ORDER,
        choices=_ALLOCATOR_ORDER,
        help="Subset of allocators to run  [default: all four]",
    )
    parser.add_argument(
        "--n-drones", type=int, default=3,
        help="Drone count  [default: 3]",
    )
    parser.add_argument(
        "--max-steps", type=int, default=500,
        help="Episode step limit  [default: 500]",
    )
    parser.add_argument(
        "--comm-delay", type=int, default=5,
        help="CBBA comm delay (steps) for comm_delay scenario  [default: 5]",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed  [default: 42]",
    )
    parser.add_argument(
        "--csv", default=None,
        help="Optional path to write per-episode CSV results",
    )
    args = parser.parse_args()

    # Load the execution actor if supplied
    actor = None
    if args.exec_checkpoint:
        from models.actor import Actor
        ckpt = torch.load(args.exec_checkpoint, map_location="cpu", weights_only=False)
        actor = Actor(obs_dim=15, act_dim=4, hidden_sizes=[256, 256])
        actor.load_state_dict(ckpt["actor_state_dict"])
        actor.eval()

    print(f"\nPhase 4 Evaluation Harness")
    print(f"  scenarios  : {args.scenarios}")
    print(f"  allocators : {args.allocators}")
    print(f"  episodes   : {args.episodes} per cell")
    print(f"  n_drones   : {args.n_drones}")
    print(f"  max_steps  : {args.max_steps}")
    print(f"  execution  : {'actor ' + args.exec_checkpoint if actor else 'random actions'}")
    if args.checkpoint:
        print(f"  bid ckpt   : {args.checkpoint}")
    print()

    factories = build_allocators(args.checkpoint, comm_delay=args.comm_delay)
    # Filter to requested allocators (keep private _cbba_delayed if cbba requested)
    keep = set(args.allocators)
    if "cbba" in keep:
        keep.add("_cbba_delayed")
    factories = {k: v for k, v in factories.items() if k in keep}

    all_metrics = benchmark(
        scenarios=args.scenarios,
        allocator_factories=factories,
        n_episodes=args.episodes,
        n_drones=args.n_drones,
        max_steps=args.max_steps,
        base_seed=args.seed,
        actor=actor,
    )

    print_table(all_metrics, args.scenarios)

    if args.csv:
        write_csv(all_metrics, args.csv)
