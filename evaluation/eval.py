"""
eval.py
Load a saved checkpoint and run the drone swarm in the HomeEnv.

Usage:
    # MLP actor — 50 episodes, 6 drones
    python -m evaluation.eval --checkpoint checkpoints/actor_update580_final.pt --n-drones 6 --episodes 50

    # CommActor — 50 episodes
    python -m evaluation.eval --checkpoint checkpoints_comm/comm_actor_update150.pt --actor-type comm --n-drones 6 --episodes 50

    # Head-to-head comparison
    python -m evaluation.eval --checkpoint checkpoints/actor_update580_final.pt     --n-drones 6 --episodes 50
    python -m evaluation.eval --checkpoint checkpoints_comm/comm_actor_update150.pt --actor-type comm --n-drones 6 --episodes 50

    # With ASCII render + slow-mo
    python -m evaluation.eval --checkpoint checkpoints_comm/comm_actor_update150.pt --actor-type comm --episodes 5 --render --step-delay 0.05
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from envs.home_env import HomeEnv
from models.actor import Actor
from models.comm_actor import CommActor
from models.critic import CentralCritic


def load_checkpoint(
    path: str,
    actor_type: str = "mlp",
    n_drones: int | None = None,
    obs_dim: int = 15,
    act_dim: int = 4,
) -> tuple[Actor | CommActor, int, dict]:
    """
    Load actor weights from a training checkpoint.

    Returns (actor, n_drones, raw_ckpt).

    n_drones resolution priority:
      1. --n-drones flag (explicit, always correct)
      2. Inferred from critic weight shape (only reliable when critic matches actor run)
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # --- Resolve n_drones ---
    if n_drones is None:
        # Infer from critic shape as a fallback; warn if ambiguous
        critic_input_dim = ckpt["critic_state_dict"]["net.0.weight"].shape[1]
        n_drones = critic_input_dim // obs_dim
        print(
            f"[WARN] --n-drones not specified; inferred n_drones={n_drones} from "
            f"critic weight shape. Pass --n-drones explicitly if this is wrong."
        )

    # --- Build actor ---
    if actor_type == "comm":
        # Read CommActor hyper-params stored in checkpoint if available,
        # otherwise fall back to training defaults.
        actor = CommActor(obs_dim=obs_dim, act_dim=act_dim)
        if "comm_actor_state_dict" in ckpt:
            actor.load_state_dict(ckpt["comm_actor_state_dict"])
        else:
            raise ValueError(
                f"Checkpoint {path} does not contain 'comm_actor_state_dict'. "
                "Use --actor-type mlp for plain Actor checkpoints."
            )
    else:
        actor = Actor(obs_dim=obs_dim, act_dim=act_dim)
        if "actor_state_dict" in ckpt:
            actor.load_state_dict(ckpt["actor_state_dict"])
        else:
            raise ValueError(
                f"Checkpoint {path} does not contain 'actor_state_dict'. "
                "Use --actor-type comm for CommActor checkpoints."
            )

    actor.eval()
    print(
        f"Loaded checkpoint : update={ckpt.get('update', '?')} "
        f"timesteps={ckpt.get('timesteps', 0):,}\n"
        f"Actor type        : {actor_type.upper()} "
        f"({'CommActor/GATv2' if actor_type == 'comm' else 'Actor/MLP'})\n"
        f"n_drones          : {n_drones}"
    )
    return actor, n_drones, ckpt


def run_episode(
    env: HomeEnv,
    actor: Actor | CommActor,
    agent_ids: list[str],
    deterministic: bool = True,
    render: bool = False,
    step_delay: float = 0.0,
    seed: int | None = None,
) -> dict:
    """Run one episode. Returns stats dict."""
    is_comm = isinstance(actor, CommActor)
    obs_dict, _ = env.reset(seed=seed)
    if is_comm:
        actor.reset_delay_buffer()

    total_reward = 0.0
    steps = 0
    done = False

    while not done:
        with torch.no_grad():
            obs_arrays = [obs_dict.get(aid, np.zeros(actor.obs_dim)) for aid in agent_ids]
            obs_tensor = torch.tensor(np.stack(obs_arrays), dtype=torch.float32)
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

    info = next(iter(infos.values())) if infos else {}
    return {
        "total_reward": total_reward,
        "steps": steps,
        "tasks_completed": info.get("tasks_completed", 0),
        "tasks_total":     info.get("tasks_total", 0),
    }


def benchmark(
    env: HomeEnv,
    actor: Actor | CommActor,
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
        pct = 100.0 * stats["tasks_completed"] / max(stats["tasks_total"], 1)
        print(
            f"Episode {ep:3d} | "
            f"reward={stats['total_reward']:+9.2f} | "
            f"steps={stats['steps']:4d} | "
            f"tasks={stats['tasks_completed']}/{stats['tasks_total']} ({pct:5.1f}%)"
        )

    rewards      = [r["total_reward"] for r in results]
    completions  = [r["tasks_completed"] / max(r["tasks_total"], 1) for r in results]
    steps_list   = [r["steps"] for r in results]

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
    parser.add_argument("--checkpoint",  required=True,
                        help="Path to .pt checkpoint file")
    parser.add_argument("--actor-type",  default="mlp", choices=["mlp", "comm"],
                        help="'mlp' for plain Actor, 'comm' for CommActor/GATv2 (default: mlp)")
    parser.add_argument("--n-drones",    type=int, default=None,
                        help="Number of drones. Should match training config (default: infer from checkpoint)")
    parser.add_argument("--episodes",    type=int, default=10,
                        help="Number of eval episodes (default: 10)")
    parser.add_argument("--render",      action="store_true",
                        help="Print ASCII render each step")
    parser.add_argument("--step-delay",  type=float, default=0.0,
                        help="Seconds to sleep between steps when rendering")
    parser.add_argument("--obs-noise-std", type=float, default=0.0,
                        help="Gaussian obs noise std (default: 0 — clean eval)")
    parser.add_argument("--allocator",   default="greedy",
                        choices=["greedy", "cbba", "oracle", "learned"],
                        help="Task allocator (default: greedy)")
    parser.add_argument("--bid-checkpoint", default=None,
                        help="Bid policy checkpoint (required when --allocator learned)")
    args = parser.parse_args()

    actor, n_drones, _ = load_checkpoint(
        args.checkpoint,
        actor_type=args.actor_type,
        n_drones=args.n_drones,
    )

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
        "n_drones":      n_drones,
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
