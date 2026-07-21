"""
bid_env.py
Single-step "bidding environment" used to train the BidPolicy via PPO.

Design
------
One episode = one full HomeEnv episode run with the *execution* actors frozen.
At every reallocation trigger the BidEnv asks BidPolicy for bids, then applies
the resulting assignment and steps the underlying HomeEnv normally.

The BidPolicy observes one (drone, task) pair at a time.  The PPO agent is
a single shared network that handles every pair — the same parameter-sharing
trick used for the execution actors.

Observation  (14-dim per pair)  — see bid_policy.BID_OBS_DIM
Action       (1-dim)            — raw logit; bid = sigmoid(logit)

Reward signal (given to every bidding agent at episode end):
    R = - makespan_fraction                      (penalise slow completion)
      + alpha * tasks_completed / tasks_total    (reward task completion)
      - beta  * mean_reallocation_latency        (penalise slow reallocation)

The environment collects these components across the episode and returns them
as a single scalar per bid-step via a shaped per-step reward:
    per_step_reward = ALPHA_COMPLETE * tasks_just_completed
                    - BETA_LATENCY  * steps_since_last_realloc (if idle drone)
                    - GAMMA_MAKESPAN (constant alive penalty)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from allocator.bid_policy import BidPolicy, build_bid_obs, BID_OBS_DIM
from allocator.base_allocator import BaseAllocator, WorldSnapshot, AllocationResult, Bid
from envs.tasks.base_task import TaskStatus

# Reward shaping coefficients
ALPHA_COMPLETE  = 3.0   # reward per task completed this step
BETA_LATENCY    = 0.02  # penalty per idle-drone-step (unused drone)
GAMMA_MAKESPAN  = 0.005 # alive penalty per step (encourages speed)

_ACTIVE = {TaskStatus.PENDING, TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS}


@dataclass
class BidTransition:
    """One (obs, action, reward, logp) sample for the bid PPO buffer."""
    obs: np.ndarray        # (14,)
    action: float          # raw logit
    log_prob: float
    reward: float = 0.0
    value: float = 0.0


class BidBuffer:
    """Lightweight rollout buffer for bid-policy PPO."""

    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.transitions: list[BidTransition] = []

    def add(self, t: BidTransition):
        self.transitions.append(t)

    def reset(self):
        self.transitions = []

    def __len__(self):
        return len(self.transitions)

    def compute_returns(self, last_value: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Returns (returns, advantages) arrays, length = len(transitions)."""
        n = len(self.transitions)
        returns = np.zeros(n, dtype=np.float32)
        advantages = np.zeros(n, dtype=np.float32)
        last_ret = last_value
        last_adv = 0.0
        for i in reversed(range(n)):
            t = self.transitions[i]
            next_val = self.transitions[i + 1].value if i + 1 < n else last_value
            delta = t.reward + self.gamma * next_val - t.value
            last_adv = delta + self.gamma * self.gae_lambda * last_adv
            advantages[i] = last_adv
            returns[i] = advantages[i] + t.value
        return returns, advantages

    def get_arrays(self):
        obs = np.stack([t.obs for t in self.transitions])
        actions = np.array([t.action for t in self.transitions], dtype=np.float32)
        log_probs = np.array([t.log_prob for t in self.transitions], dtype=np.float32)
        values = np.array([t.value for t in self.transitions], dtype=np.float32)
        rewards = np.array([t.reward for t in self.transitions], dtype=np.float32)
        return obs, actions, log_probs, values, rewards


class BidEnv:
    """
    Wraps HomeEnv and frozen execution actors.  Exposes a
    collect_episode() method that runs one full HomeEnv episode,
    producing BidTransitions for every auction round.

    The BidPolicy is called once per (drone, task) pair per auction round.
    The resulting bids are assembled into an AllocationResult and applied
    to the environment.

    obs_delay
    ---------
    When obs_delay > 0 the position and battery components of the bid
    observation are taken from a circular history buffer instead of the
    current-step values.  This mimics the communication delay in the CBBA
    robustness experiment: the bidding policy must decide with stale sensor
    data, testing robustness to sensing/communication latency.
    """

    def __init__(
        self,
        home_env,           # HomeEnv instance (allocator will be overridden)
        exec_actor,         # Frozen execution Actor network
        bid_policy: BidPolicy,
        bid_value_net,      # Small critic for bid-policy PPO (BidValueNet)
        device: str = "cpu",
        obs_delay: int = 0, # steps of observation delay (0 = no delay)
    ):
        self.env = home_env
        self.exec_actor = exec_actor
        self.bid_policy = bid_policy
        self.bid_value_net = bid_value_net
        self.device = device
        self.obs_delay = obs_delay

        # Replace env's allocator with our learned one (injected after construction)
        self._learned_allocator: _InternalLearnedAllocator | None = None

    def collect_episode(self, seed: int | None = None) -> tuple[BidBuffer, dict]:
        """
        Run one HomeEnv episode.  Returns:
          - buffer: BidBuffer containing all bid transitions
          - info: episode summary (tasks_completed, makespan, n_reallocations)
        """
        buffer = BidBuffer()
        alloc = _InternalLearnedAllocator(
            self.bid_policy, self.bid_value_net, buffer, self.device,
            obs_delay=self.obs_delay,
        )
        self.env.allocator = alloc

        agent_ids = sorted(self.env._agent_ids)
        obs_dict, _ = self.env.reset(seed=seed)
        done = False
        step = 0
        prev_tasks_completed = 0

        while not done:
            # Execution step with frozen actor
            with torch.no_grad():
                obs_t = torch.tensor(
                    np.stack([obs_dict.get(aid, np.zeros(15)) for aid in agent_ids]),
                    dtype=torch.float32, device=self.device,
                )
                squashed, _, _ = self.exec_actor.get_action(obs_t, deterministic=True)
                action_dict = {
                    aid: squashed[i].cpu().numpy()
                    for i, aid in enumerate(agent_ids)
                }

            obs_dict, reward_dict, terminated, truncated, info = self.env.step(action_dict)
            done = terminated.get("__all__", False) or truncated.get("__all__", False)
            step += 1

            # Per-step shaped reward for any bid transitions produced this step
            tasks_completed_now = next(iter(info.values()))["tasks_completed"]
            delta_complete = tasks_completed_now - prev_tasks_completed
            prev_tasks_completed = tasks_completed_now

            # Distribute reward back to the last set of bid transitions
            n_idle = sum(
                1 for aid in agent_ids
                if self.env._drone_task_map.get(aid) is None
            )
            step_reward = (
                ALPHA_COMPLETE * delta_complete
                - BETA_LATENCY * n_idle
                - GAMMA_MAKESPAN
            )
            alloc.assign_reward(step_reward)

        ep_info = {
            "tasks_completed": prev_tasks_completed,
            "tasks_total": next(iter(info.values()))["tasks_total"],
            "makespan": step,
            "n_reallocations": alloc.n_reallocations,
            "n_bid_transitions": len(buffer),
        }
        return buffer, ep_info


# ---------------------------------------------------------------------------
# Internal allocator used by BidEnv
# ---------------------------------------------------------------------------

class _InternalLearnedAllocator(BaseAllocator):
    """
    Allocator used exclusively inside BidEnv.collect_episode().
    Calls BidPolicy for each (drone, task) pair and stores transitions.

    When obs_delay > 0, drone positions and batteries seen by the bid
    observation are taken from a history buffer that is obs_delay steps stale.
    This allows apples-to-apples robustness comparison with CBBA's comm_delay.
    """

    def __init__(
        self,
        policy: BidPolicy,
        value_net,
        buffer: BidBuffer,
        device: str,
        obs_delay: int = 0,
    ):
        self.policy = policy
        self.value_net = value_net
        self.buffer = buffer
        self.device = device
        self.obs_delay = obs_delay
        self.n_reallocations = 0
        self._last_bid_indices: list[int] = []  # indices into buffer for reward assignment
        # Circular history buffer: list of snapshots (positions, batteries)
        # at most obs_delay entries deep.
        self._obs_history: list[tuple[dict, dict]] = []

    def allocate(self, snapshot: WorldSnapshot) -> AllocationResult:
        self.n_reallocations += 1
        active_tasks = [
            (i, t) for i, t in enumerate(snapshot.tasks)
            if t.status in _ACTIVE
        ]

        if not active_tasks:
            return AllocationResult(
                assignments={d: None for d in snapshot.drone_positions},
            )

        # Update the obs history buffer with the current positions/batteries.
        # Keep only obs_delay entries so index 0 is the stalest available.
        self._obs_history.append(
            (dict(snapshot.drone_positions), dict(snapshot.drone_batteries))
        )
        if len(self._obs_history) > max(self.obs_delay, 1):
            self._obs_history.pop(0)

        # Use delayed observations when enough history has accumulated
        if self.obs_delay > 0 and len(self._obs_history) >= self.obs_delay:
            delayed_positions, delayed_batteries = self._obs_history[0]
        else:
            delayed_positions  = snapshot.drone_positions
            delayed_batteries  = snapshot.drone_batteries

        all_bids: list[Bid] = []
        self._last_bid_indices = []

        for drone_id, pos in snapshot.drone_positions.items():
            # Use stale pos/battery for bid obs; current progress is always fresh
            delayed_pos  = delayed_positions.get(drone_id, pos)
            batt = delayed_batteries.get(drone_id, 1.0)
            progress = snapshot.drone_task_progress.get(drone_id, 0.0)

            for task_idx, task in active_tasks:
                obs = build_bid_obs(
                    delayed_pos, batt, progress, task,
                    snapshot.step, snapshot.max_steps,
                )
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

                with torch.no_grad():
                    primary_logit, marginal_logit = self.policy.forward(obs_t)
                    primary_logit  = primary_logit.item()
                    marginal_logit = marginal_logit.item()
                    bid_val     = float(torch.sigmoid(torch.tensor(primary_logit)).item())
                    marginal_val = float(torch.sigmoid(torch.tensor(marginal_logit)).item())
                    log_prob = float(
                        torch.distributions.Bernoulli(logits=torch.tensor(primary_logit)).log_prob(
                            torch.tensor(1.0)
                        ).item()
                    )
                    value = float(self.value_net(obs_t).item()) if self.value_net else 0.0

                t = BidTransition(obs=obs, action=primary_logit, log_prob=log_prob, value=value)
                idx = len(self.buffer.transitions)
                self.buffer.add(t)
                self._last_bid_indices.append(idx)

                all_bids.append(Bid(
                    drone_id=drone_id,
                    task_idx=task_idx,
                    bid_value=bid_val,
                    marginal=marginal_val if task.spec.is_shareable else 0.0,
                ))

        # Resolve: highest bid per task, each drone wins at most once
        assignments: dict[str, int | None] = {d: None for d in snapshot.drone_positions}
        assigned_drones: set[str] = set()

        task_bids: dict[int, list[Bid]] = {}
        for b in all_bids:
            task_bids.setdefault(b.task_idx, []).append(b)

        task_order = sorted(
            task_bids.keys(),
            key=lambda i: max(b.bid_value for b in task_bids[i]),
            reverse=True,
        )
        for tidx in task_order:
            for b in sorted(task_bids[tidx], key=lambda b: (b.bid_value, b.drone_id), reverse=True):
                if b.drone_id not in assigned_drones:
                    assignments[b.drone_id] = tidx
                    assigned_drones.add(b.drone_id)
                    break

        # Co-assignment for shareable tasks — use the learned marginal bid
        idle_drones = [d for d in snapshot.drone_positions if d not in assigned_drones]
        for tidx, task in active_tasks:
            if not task.spec.is_shareable or task.remaining_work() < 0.4 or not idle_drones:
                continue
            candidates = sorted(
                [b for b in all_bids if b.task_idx == tidx and b.drone_id in idle_drones],
                key=lambda b: b.marginal, reverse=True,
            )
            if candidates and candidates[0].marginal > 0:
                best = candidates[0]
                assignments[best.drone_id] = tidx
                idle_drones.remove(best.drone_id)
                assigned_drones.add(best.drone_id)

        return AllocationResult(assignments=assignments, bids=all_bids)

    def assign_reward(self, reward: float):
        """Distribute a step reward back to the last batch of bid transitions."""
        for idx in self._last_bid_indices:
            if idx < len(self.buffer.transitions):
                self.buffer.transitions[idx].reward += reward
