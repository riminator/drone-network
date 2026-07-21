"""
deploy.py
Deploy a trained policy checkpoint into the PyBullet physics lab.

Opens a GUI window showing the drone swarm navigating the household environment
with real quadrotor aerodynamics.  The task allocator is selectable at the
command line; bids are visualised in real-time as coloured lines from each
drone to its currently winning task.

Usage:
    # Greedy allocator (default), real-time speed
    python -m lab.deploy --checkpoint checkpoints/actor_update500_final.pt

    # CBBA allocator, slow-motion
    python -m lab.deploy --checkpoint checkpoints/actor_update500_final.pt \\
        --allocator cbba --time-scale 0.3

    # Learned bidder with a trained bid-policy checkpoint
    python -m lab.deploy --checkpoint checkpoints/actor_update500_final.pt \\
        --allocator learned --bid-checkpoint checkpoints/bid_policy_final.pt

    # Re-auction every 20 steps (rather than only on task completion)
    python -m lab.deploy --checkpoint checkpoints/actor_update500_final.pt \\
        --auction-interval 20

    # Headless benchmark, more drones, record video
    python -m lab.deploy --checkpoint checkpoints/actor_update500_final.pt \\
        --n-drones 4 --time-scale 0.3 --record --no-gui --episodes 20

Controls (when GUI is open):
    WASD pan  Q/E zoom  Z/X rotate  R/F pitch
    1-5  preset views   0  follow drone   mouse drag orbits
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from envs.pybullet_env import PybulletHomeEnv
from models.actor import Actor


# ---------------------------------------------------------------------------
# Actor loader (unchanged from original)
# ---------------------------------------------------------------------------

def load_actor(checkpoint_path: str, obs_dim: int = 15, act_dim: int = 4) -> Actor:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["actor_state_dict"]

    # Detect incompatible checkpoints saved with an old log_std_head linear layer
    if "log_std_head.weight" in state and "log_std" not in state:
        compatible = [
            "actor_update100.pt", "actor_update200.pt",
            "actor_update204_final.pt", "actor_update300.pt",
        ]
        raise RuntimeError(
            f"\nCheckpoint '{checkpoint_path}' was saved with an older Actor "
            f"architecture (log_std_head linear layer) that is incompatible "
            f"with the current model.\n"
            f"Use one of these compatible checkpoints instead:\n"
            + "\n".join(f"  checkpoints/{c}" for c in compatible)
        )

    actor = Actor(obs_dim=obs_dim, act_dim=act_dim)
    actor.load_state_dict(state)
    actor.eval()
    print(
        f"Loaded checkpoint: {checkpoint_path}\n"
        f"  update={ckpt.get('update', '?')}  "
        f"timesteps={ckpt.get('timesteps', 0):,}"
    )
    return actor


# ---------------------------------------------------------------------------
# Allocator factory
# ---------------------------------------------------------------------------

def build_allocator(name: str, bid_checkpoint: str | None):
    """
    Return a fresh BaseAllocator instance for the given name.

    name            — "greedy" | "cbba" | "oracle" | "learned"
    bid_checkpoint  — path to .pt file, required when name == "learned"
    """
    if name == "greedy":
        from allocator.greedy_auction import GreedyAuction
        return GreedyAuction()

    if name == "cbba":
        from allocator.cbba import CBBA
        return CBBA(comm_delay=0)

    if name == "oracle":
        from allocator.oracle import OracleAllocator
        return OracleAllocator()

    if name == "learned":
        from allocator.learned_bidder import LearnedBidder
        from allocator.bid_policy import BidPolicy
        if bid_checkpoint:
            return LearnedBidder.from_checkpoint(bid_checkpoint)
        # No checkpoint → untrained policy (produces random bids, for testing only)
        print(
            "[WARN] --allocator learned requires --bid-checkpoint. "
            "Using an untrained BidPolicy (random bids)."
        )
        return LearnedBidder(BidPolicy())

    raise ValueError(
        f"Unknown allocator '{name}'. "
        "Choose from: greedy, cbba, oracle, learned"
    )


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env: PybulletHomeEnv,
    actor: Actor,
    agent_ids: list[str],
    time_scale: float = 1.0,
    step_hz: int = 48,
) -> dict:
    """
    Run one episode.

    time_scale=1.0 → real-time (sleep between steps to match physics Hz).
    time_scale=0.3 → 0.3× speed (easier to watch).
    time_scale=0.0 → as fast as possible (benchmarking).
    """
    obs_dict, _ = env.reset()
    total_reward = 0.0
    steps = 0
    done = False
    step_dt = (1.0 / step_hz) / max(time_scale, 1e-6) if time_scale > 0 else 0.0

    while not done:
        t0 = time.perf_counter()

        with torch.no_grad():
            obs_arrays = [obs_dict.get(aid, np.zeros(15)) for aid in agent_ids]
            obs_tensor = torch.tensor(np.stack(obs_arrays), dtype=torch.float32)
            squashed_tensor, _raw, _lp = actor.get_action(obs_tensor, deterministic=True)
            action_dict = {
                aid: squashed_tensor[i].cpu().numpy()
                for i, aid in enumerate(agent_ids)
            }

        obs_dict, reward_dict, terminated, truncated, infos = env.step(action_dict)
        total_reward += sum(reward_dict.values())
        steps += 1

        done = (
            terminated.get("__all__", False)
            or truncated.get("__all__", False)
        )

        # Pace the loop to match time_scale
        if step_dt > 0:
            elapsed = time.perf_counter() - t0
            remaining = step_dt - elapsed
            if remaining > 0:
                time.sleep(remaining)

    info = next(iter(infos.values())) if infos else {}
    return {
        "total_reward": total_reward,
        "steps": steps,
        "tasks_completed": info.get("tasks_completed", 0),
        "tasks_total": info.get("tasks_total", 0),
    }


# ---------------------------------------------------------------------------
# Main deploy loop
# ---------------------------------------------------------------------------

def deploy(args: argparse.Namespace):
    actor = load_actor(args.checkpoint)

    allocator = build_allocator(args.allocator, args.bid_checkpoint)
    alloc_label = args.allocator
    if args.allocator == "learned" and args.bid_checkpoint:
        alloc_label = f"learned ({args.bid_checkpoint})"

    env_cfg = {
        "n_drones":        args.n_drones,
        "max_steps":       args.max_steps,
        "gui":             not args.no_gui,
        "record":          args.record,
        "time_scale":      args.time_scale,
        "allocator":       allocator,
        "auction_interval": args.auction_interval,
    }

    print(
        f"\nLaunching PyBullet env\n"
        f"  GUI        : {'on' if not args.no_gui else 'off'}\n"
        f"  drones     : {args.n_drones}\n"
        f"  allocator  : {alloc_label}\n"
        f"  auction_interval: {args.auction_interval} "
        f"({'periodic' if args.auction_interval > 0 else 'on-completion only'})\n"
        f"  time_scale : {args.time_scale}×\n"
    )

    env = PybulletHomeEnv(config=env_cfg)
    agent_ids = sorted(env._agent_ids)

    results = []
    try:
        for ep in range(1, args.episodes + 1):
            print(f"Episode {ep}/{args.episodes} starting...")
            stats = run_episode(
                env, actor, agent_ids,
                time_scale=args.time_scale,
            )
            results.append(stats)
            print(
                f"  Episode {ep:3d} | "
                f"reward={stats['total_reward']:+8.2f} | "
                f"steps={stats['steps']:4d} | "
                f"tasks={stats['tasks_completed']}/{stats['tasks_total']}"
            )

    except KeyboardInterrupt:
        print("\n[STOP] Deployment interrupted.")
    finally:
        env.close()

    if not results:
        return

    rewards      = [r["total_reward"] for r in results]
    completions  = [r["tasks_completed"] / max(r["tasks_total"], 1) for r in results]
    steps_list   = [r["steps"] for r in results]

    print("\n" + "─" * 50)
    print("Deployment Summary")
    print("─" * 50)
    print(f"Allocator          : {alloc_label}")
    print(f"Auction interval   : {args.auction_interval}")
    print(f"Episodes run       : {len(results)}")
    print(f"Mean reward        : {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Task completion %  : {np.mean(completions) * 100:.1f}%")
    print(f"Mean steps/episode : {np.mean(steps_list):.1f}")
    print(f"Min / Max reward   : {np.min(rewards):.2f} / {np.max(rewards):.2f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deploy a trained drone swarm policy in the PyBullet lab"
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to actor .pt checkpoint "
             "(e.g. checkpoints/actor_update204_final.pt)"
    )
    parser.add_argument(
        "--allocator", default="greedy",
        choices=["greedy", "cbba", "oracle", "learned"],
        help="Task-allocation strategy  [default: greedy]",
    )
    parser.add_argument(
        "--bid-checkpoint", default=None,
        help="Path to bid_policy .pt checkpoint "
             "(required when --allocator learned)",
    )
    parser.add_argument(
        "--auction-interval", type=int, default=0,
        help="Re-run the auction every N steps; 0 = on task completion only "
             "[default: 0]",
    )
    parser.add_argument("--episodes",    type=int,   default=5,
                        help="Number of episodes to run  [default: 5]")
    parser.add_argument("--n-drones",    type=int,   default=3,
                        help="Number of drones  [default: 3]")
    parser.add_argument("--max-steps",   type=int,   default=800,
                        help="Max steps per episode  [default: 800]")
    parser.add_argument("--time-scale",  type=float, default=1.0,
                        help="Playback speed: 1.0=real-time, 0.3=slow-mo, 0=max  "
                             "[default: 1.0]")
    parser.add_argument("--no-gui",      action="store_true",
                        help="Run headless — faster benchmarking")
    parser.add_argument("--record",      action="store_true",
                        help="Record video via PyBullet's built-in recorder")
    args = parser.parse_args()
    deploy(args)
