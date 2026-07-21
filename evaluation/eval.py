"""
eval.py
Load a saved checkpoint and run the drone swarm in the HomeEnv.

Usage:
    # Watch 10 episodes with ASCII render
    python -m evaluation.eval --checkpoint checkpoints/actor_update500.pt --episodes 10 --render

    # Benchmark 100 episodes silently
    python -m evaluation.eval --checkpoint checkpoints/actor_update500.pt --episodes 100
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from envs.home_env import HomeEnv
from models.actor import Actor
from models.critic import CentralCritic


def load_checkpoint(path: str, cfg_override: dict | None = None) -> tuple[Actor, CentralCritic, dict]:
    """Load actor + critic weights from a training checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    obs_dim = (cfg_override or {}).get("obs_dim", 15)
    act_dim = (cfg_override or {}).get("act_dim", 4)

    # Infer n_drones from the critic's first-layer weight shape rather than
    # relying on the caller passing the right value.  The critic input dim is
    # n_drones * obs_dim, so: n_drones = input_dim / obs_dim.
    critic_input_dim = ckpt["critic_state_dict"]["net.0.weight"].shape[1]
    n_drones = critic_input_dim // obs_dim

    actor = Actor(obs_dim=obs_dim, act_dim=act_dim)
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()

    critic = CentralCritic(obs_dim=obs_dim, n_drones=n_drones)
    critic.load_state_dict(ckpt["critic_state_dict"])
    critic.eval()

    print(
        f"Loaded checkpoint: update={ckpt.get('update', '?')} "
        f"timesteps={ckpt.get('timesteps', '?'):,} "
        f"(n_drones={n_drones} inferred from checkpoint)"
    )
    return actor, critic, ckpt


def run_episode(
    env: HomeEnv,
    actor: Actor,
    agent_ids: list[str],
    deterministic: bool = True,
    render: bool = False,
    step_delay: float = 0.0,
    seed: int | None = None,
) -> dict:
    """Run one episode. Returns stats dict."""
    obs_dict, _ = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    done = False

    while not done:
        with torch.no_grad():
            obs_arrays = [obs_dict.get(aid, np.zeros(15)) for aid in agent_ids]
            obs_tensor = torch.tensor(
                np.stack(obs_arrays), dtype=torch.float32
            )
            squashed_tensor, _raw, _lp = actor.get_action(obs_tensor, deterministic=deterministic)
            action_dict = {
                aid: squashed_tensor[i].cpu().numpy()
                for i, aid in enumerate(agent_ids)
            }

        obs_dict, reward_dict, terminated, truncated, infos = env.step(action_dict)
        total_reward += sum(reward_dict.values())
        steps += 1

        if render:
            env.render()
            if step_delay > 0:
                time.sleep(step_delay)

        done = (
            terminated.get("__all__", False)
            or truncated.get("__all__", False)
        )

    # Grab info from any agent
    info = next(iter(infos.values())) if infos else {}
    return {
        "total_reward": total_reward,
        "steps": steps,
        "tasks_completed": info.get("tasks_completed", 0),
        "tasks_total": info.get("tasks_total", 0),
    }


def benchmark(
    env: HomeEnv,
    actor: Actor,
    n_episodes: int = 20,
    render: bool = False,
    step_delay: float = 0.0,
):
    """Run n_episodes and print a summary table."""
    agent_ids = sorted(env._agent_ids)
    results = []

    for ep in range(1, n_episodes + 1):
        stats = run_episode(env, actor, agent_ids, render=render,
                            step_delay=step_delay, seed=ep)
        results.append(stats)
        print(
            f"Episode {ep:3d} | "
            f"reward={stats['total_reward']:+8.2f} | "
            f"steps={stats['steps']:4d} | "
            f"tasks={stats['tasks_completed']}/{stats['tasks_total']}"
        )

    rewards = [r["total_reward"] for r in results]
    completions = [r["tasks_completed"] / max(r["tasks_total"], 1) for r in results]
    steps_list = [r["steps"] for r in results]

    print("\n--- Summary ---")
    print(f"Episodes          : {n_episodes}")
    print(f"Mean reward       : {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Task completion % : {np.mean(completions) * 100:.1f}%")
    print(f"Mean steps        : {np.mean(steps_list):.1f}")
    print(f"Min / Max reward  : {np.min(rewards):.2f} / {np.max(rewards):.2f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained drone swarm policy")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
    parser.add_argument("--episodes", type=int, default=10, help="Number of eval episodes")
    parser.add_argument("--render", action="store_true", help="Print ASCII render each step")
    parser.add_argument("--step-delay", type=float, default=0.0,
                        help="Seconds to sleep between steps when rendering")
    parser.add_argument("--n-drones", type=int, default=6,
                        help="Number of drones (default: 6)")
    parser.add_argument("--obs-noise-std", type=float, default=0.25,
                        help="Gaussian obs noise std — match training value (default: 0.25)")
    parser.add_argument("--allocator", default="greedy",
                        choices=["greedy", "cbba", "oracle", "learned"],
                        help="Allocator to use (default: greedy)")
    parser.add_argument("--bid-checkpoint", default=None,
                        help="Bid policy checkpoint (required when --allocator learned)")
    args = parser.parse_args()

    actor, _, _ = load_checkpoint(args.checkpoint)

    # Build allocator
    if args.allocator == "greedy":
        from allocator.greedy_auction import GreedyAuction
        allocator = GreedyAuction()
    elif args.allocator == "cbba":
        from allocator.cbba import CBBA
        allocator = CBBA()
    elif args.allocator == "oracle":
        from allocator.oracle import OracleAllocator
        allocator = OracleAllocator()
    else:  # learned
        from allocator.learned_bidder import LearnedBidder
        from allocator.bid_policy import BidPolicy
        if args.bid_checkpoint:
            allocator = LearnedBidder.from_checkpoint(args.bid_checkpoint)
        else:
            print("[WARN] --allocator learned requires --bid-checkpoint. Using untrained policy.")
            allocator = LearnedBidder(BidPolicy())

    env = HomeEnv(config={
        "n_drones":      args.n_drones,
        "allocator":     allocator,
        "obs_noise_std": args.obs_noise_std,
        "render_mode":   "human" if args.render else None,
    })

    benchmark(
        env,
        actor,
        n_episodes=args.episodes,
        render=args.render,
        step_delay=args.step_delay,
    )
