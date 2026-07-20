"""
deploy.py
Deploy a trained policy checkpoint into the PyBullet physics lab.

Opens a GUI window showing the drone swarm navigating the household environment
with real quadrotor aerodynamics.

Usage:
    # Run 5 episodes with the GUI open, 0.5× real-time speed
    python -m lab.deploy --checkpoint checkpoints/actor_update500_final.pt

    # More drones, slower, record video
    python -m lab.deploy --checkpoint checkpoints/actor_update500_final.pt \\
        --n-drones 4 --time-scale 0.3 --record

    # Headless benchmark (no GUI, fast)
    python -m lab.deploy --checkpoint checkpoints/actor_update500_final.pt \\
        --no-gui --episodes 20

Controls (when GUI is open):
    The PyBullet window supports mouse orbit / zoom.
    Press Ctrl+C in the terminal to stop between episodes.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from envs.pybullet_env import PybulletHomeEnv
from models.actor import Actor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_actor(checkpoint_path: str, obs_dim: int = 15, act_dim: int = 4) -> Actor:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actor = Actor(obs_dim=obs_dim, act_dim=act_dim)
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()
    print(
        f"Loaded checkpoint: {checkpoint_path}\n"
        f"  update={ckpt.get('update', '?')}  "
        f"timesteps={ckpt.get('timesteps', 0):,}"
    )
    return actor


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

    env_cfg = {
        "n_drones":   args.n_drones,
        "max_steps":  args.max_steps,
        "gui":        not args.no_gui,
        "record":     args.record,
        "time_scale": args.time_scale,
    }

    print(f"\nLaunching PyBullet env — GUI={'on' if not args.no_gui else 'off'}, "
          f"drones={args.n_drones}, time_scale={args.time_scale}×\n")

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
        help="Path to .pt checkpoint (e.g. checkpoints/actor_update500_final.pt)"
    )
    parser.add_argument("--episodes",   type=int,   default=5,
                        help="Number of episodes to run (default: 5)")
    parser.add_argument("--n-drones",   type=int,   default=3,
                        help="Number of drones (default: 3)")
    parser.add_argument("--max-steps",  type=int,   default=500,
                        help="Max steps per episode (default: 500)")
    parser.add_argument("--time-scale", type=float, default=1.0,
                        help="Playback speed: 1.0=real-time, 0.3=slow-mo, 0=max-speed")
    parser.add_argument("--no-gui",     action="store_true",
                        help="Run headless (no PyBullet window) — faster benchmarking")
    parser.add_argument("--record",     action="store_true",
                        help="Record video via PyBullet's built-in recorder")
    args = parser.parse_args()
    deploy(args)
