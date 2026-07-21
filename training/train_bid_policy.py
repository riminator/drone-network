"""
train_bid_policy.py
PPO trainer for the BidPolicy (Phase 3).

The execution Actor is loaded from a checkpoint and kept FROZEN.
Only the BidPolicy and its companion value network are trained.

Run
---
    python -m training.train_bid_policy
    python -m training.train_bid_policy --config training/config.yaml
    python -m training.train_bid_policy --exec-checkpoint checkpoints/actor_update204_final.pt

Algorithm
---------
Standard on-policy PPO over bid transitions collected by BidEnv.
One "rollout" = N HomeEnv episodes run with the current BidPolicy.
After each rollout the buffer is used for K PPO epochs, then discarded.

The bid action is a continuous scalar (raw logit from BidPolicy.forward).
We model it as a Gaussian with learned mean = policy(obs) and fixed
log_std (a separate parameter), so we can compute log-prob and entropy
analytically — exactly the same design as the execution Actor.

Reward shaping (from bid_env.py):
    per-step: +ALPHA * tasks_completed_this_step
             -BETA  * n_idle_drones
             -GAMMA (alive penalty)
"""

from __future__ import annotations

import argparse
import os
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from envs.home_env import HomeEnv
from models.actor import Actor
from allocator.bid_policy import BidPolicy, BID_OBS_DIM
from allocator.bid_env import BidEnv, BidBuffer
from allocator.oracle import OracleAllocator


# ---------------------------------------------------------------------------
# Small value network for bid-policy PPO
# ---------------------------------------------------------------------------

class BidValueNet(nn.Module):
    """Critic for the bid-policy PPO. Same input as BidPolicy."""
    def __init__(self, obs_dim: int = BID_OBS_DIM, hidden: list[int] | None = None):
        super().__init__()
        hidden = hidden or [64, 64]
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_exec_actor(checkpoint_path: str, device: str) -> Actor:
    """Load the frozen execution actor from a MAPPO checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    actor = Actor(obs_dim=15, act_dim=4, hidden_sizes=[256, 256]).to(device)
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()
    for p in actor.parameters():
        p.requires_grad_(False)
    print(f"  [EXEC] Loaded frozen actor from {checkpoint_path}")
    return actor


def _ppo_update(
    buffer: BidBuffer,
    policy: BidPolicy,
    value_net: BidValueNet,
    policy_optim: optim.Optimizer,
    value_optim: optim.Optimizer,
    cfg: dict,
    device: str,
) -> dict[str, float]:
    """One PPO update pass over a filled BidBuffer."""
    clip_eps      = cfg["bid_policy"]["clip_eps"]
    entropy_coef  = cfg["bid_policy"]["entropy_coef"]
    value_coef    = cfg["bid_policy"]["value_coef"]
    max_grad_norm = cfg["bid_policy"]["max_grad_norm"]
    n_epochs      = cfg["bid_policy"]["n_epochs"]
    batch_size    = cfg["bid_policy"]["mini_batch_size"]
    log_std_init  = cfg["bid_policy"].get("log_std_init", -1.0)

    if len(buffer) == 0:
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

    returns, advantages = buffer.compute_returns()
    obs_arr, act_arr, old_lp_arr, _, _ = buffer.get_arrays()

    obs_t     = torch.tensor(obs_arr, dtype=torch.float32, device=device)
    act_t     = torch.tensor(act_arr, dtype=torch.float32, device=device)
    old_lp_t  = torch.tensor(old_lp_arr, dtype=torch.float32, device=device)
    returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
    adv_t     = torch.tensor(advantages, dtype=torch.float32, device=device)
    adv_t     = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    n = len(obs_arr)
    stats = {"policy_loss": [], "value_loss": [], "entropy": []}

    # Fixed log_std for the bid Gaussian (separate from policy weights)
    log_std = torch.tensor(log_std_init, dtype=torch.float32, device=device)

    for _ in range(n_epochs):
        idx = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            b = idx[start: start + batch_size]
            obs_b    = obs_t[b]
            act_b    = act_t[b]
            old_lp_b = old_lp_t[b]
            ret_b    = returns_t[b]
            adv_b    = adv_t[b]

            # Policy logit → Gaussian with log_std
            mean = policy.forward(obs_b)   # (B,)
            std  = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            new_lp = dist.log_prob(act_b)
            entropy = dist.entropy().mean()

            ratio   = torch.exp(new_lp - old_lp_b)
            surr1   = ratio * adv_b
            surr2   = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_b
            p_loss  = -torch.min(surr1, surr2).mean() - entropy_coef * entropy

            values   = value_net(obs_b)
            v_loss   = 0.5 * ((values - ret_b) ** 2).mean()
            total    = p_loss + value_coef * v_loss

            policy_optim.zero_grad()
            value_optim.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_grad_norm)
            policy_optim.step()
            value_optim.step()

            stats["policy_loss"].append(p_loss.item())
            stats["value_loss"].append(v_loss.item())
            stats["entropy"].append(entropy.item())

    return {k: float(np.mean(v)) for k, v in stats.items()}


def _oracle_makespan(env: HomeEnv, n_eval: int, device: str) -> float:
    """
    Estimate oracle makespan: run n_eval episodes with frozen actor + oracle
    allocator and return mean steps-to-complete (or max_steps if incomplete).
    """
    from models.actor import Actor as _Actor
    oracle = OracleAllocator()
    env.allocator = oracle
    agent_ids = sorted(env._agent_ids)
    makespans = []
    for _ in range(n_eval):
        obs, _ = env.reset()
        done = False
        step = 0
        while not done:
            with torch.no_grad():
                obs_t = torch.tensor(
                    np.stack([obs.get(aid, np.zeros(15)) for aid in agent_ids]),
                    dtype=torch.float32, device=device,
                )
                squashed, _, _ = env.allocator.__class__  # won't be called
            # Oracle doesn't need execution — just step with zeros for oracle measurement
            actions = {aid: np.zeros(4, dtype=np.float32) for aid in agent_ids}
            obs, _, terminated, truncated, _ = env.step(actions)
            done = terminated.get("__all__", False) or truncated.get("__all__", False)
            step += 1
        makespans.append(step)
    return float(np.mean(makespans))


def _evaluate(
    env: HomeEnv,
    exec_actor: Actor,
    bid_policy: BidPolicy,
    value_net: BidValueNet,
    n_episodes: int,
    device: str,
) -> dict[str, float]:
    """Evaluate the current BidPolicy over n_episodes. Returns summary dict."""
    from allocator.bid_env import BidEnv
    bid_env = BidEnv(env, exec_actor, bid_policy, value_net, device)
    makespans, tasks_dones, tasks_totals = [], [], []
    for i in range(n_episodes):
        _, info = bid_env.collect_episode(seed=i)
        makespans.append(info["makespan"])
        tasks_dones.append(info["tasks_completed"])
        tasks_totals.append(info["tasks_total"])
    return {
        "mean_makespan": float(np.mean(makespans)),
        "mean_tasks_completed": float(np.mean(tasks_dones)),
        "tasks_total": float(np.mean(tasks_totals)),
    }


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

_stop = False


def _handle_signal(sig, frame):
    global _stop
    if not _stop:
        print("\n[STOP] Interrupt received — finishing current update then saving...")
        _stop = True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(cfg: dict, exec_checkpoint: str):
    global _stop
    _stop = False
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    device_str = cfg["training"].get("device", "auto")
    if device_str == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_str

    bp_cfg   = cfg["bid_policy"]
    n_drones = cfg["env"]["n_drones"]

    # --- Frozen execution actor ---
    exec_actor = load_exec_actor(exec_checkpoint, device)

    # --- Environment ---
    env = HomeEnv(config=cfg["env"])

    # --- Bid policy + value net ---
    policy    = BidPolicy(obs_dim=BID_OBS_DIM, hidden=bp_cfg.get("hidden", [64, 64])).to(device)
    value_net = BidValueNet(obs_dim=BID_OBS_DIM, hidden=bp_cfg.get("hidden", [64, 64])).to(device)

    policy_optim = optim.Adam(
        policy.parameters(),
        lr=bp_cfg["lr"],
        weight_decay=bp_cfg.get("weight_decay", 0.0),
    )
    value_optim  = optim.Adam(value_net.parameters(), lr=bp_cfg["lr_value"])

    ckpt_dir = Path(cfg["logging"]["checkpoint_dir"])
    ckpt_dir.mkdir(exist_ok=True)

    bid_env = BidEnv(env, exec_actor, policy, value_net, device)

    n_updates     = bp_cfg["n_updates"]
    episodes_per  = bp_cfg["episodes_per_update"]
    eval_every    = bp_cfg.get("eval_interval", 10)
    ckpt_every    = bp_cfg.get("checkpoint_interval", 50)

    print(f"Training BidPolicy — {n_updates} updates × {episodes_per} episodes each")
    print(f"Device: {device} | Bid obs dim: {BID_OBS_DIM} | Hidden: {bp_cfg.get('hidden', [64,64])}")
    print("Press Ctrl+C to stop and save.\n")

    start = time.time()
    for update in range(1, n_updates + 1):
        if _stop:
            break

        # Collect rollout
        combined = BidBuffer(
            gamma=bp_cfg.get("gamma", 0.99),
            gae_lambda=bp_cfg.get("gae_lambda", 0.95),
        )
        ep_makespans, ep_tasks = [], []
        for ep_idx in range(episodes_per):
            buf, info = bid_env.collect_episode(seed=update * 1000 + ep_idx)
            combined.transitions.extend(buf.transitions)
            ep_makespans.append(info["makespan"])
            ep_tasks.append(info["tasks_completed"])

        # PPO update
        stats = _ppo_update(combined, policy, value_net,
                            policy_optim, value_optim, cfg, device)

        if update % 1 == 0:  # log every update
            print(
                f"Update {update:4d}/{n_updates} | "
                f"Makespan {np.mean(ep_makespans):6.1f} | "
                f"TasksDone {np.mean(ep_tasks):.2f}/{ep_tasks[0] if ep_tasks else '?'} | "
                f"PL {stats['policy_loss']:+.4f} | "
                f"VL {stats['value_loss']:.4f} | "
                f"Ent {stats['entropy']:.4f} | "
                f"{time.time()-start:.0f}s"
            )

        if update % eval_every == 0:
            eval_stats = _evaluate(env, exec_actor, policy, value_net,
                                   n_episodes=5, device=device)
            print(
                f"  [EVAL] makespan={eval_stats['mean_makespan']:.1f} | "
                f"tasks={eval_stats['mean_tasks_completed']:.1f}/"
                f"{eval_stats['tasks_total']:.0f}"
            )

        if update % ckpt_every == 0 or _stop:
            path = ckpt_dir / f"bid_policy_update{update}.pt"
            torch.save({
                "update": update,
                "bid_policy_state_dict": policy.state_dict(),
                "bid_value_state_dict": value_net.state_dict(),
                "bid_policy_config": {
                    "obs_dim": BID_OBS_DIM,
                    "hidden": bp_cfg.get("hidden", [64, 64]),
                },
            }, path)
            print(f"  [CKPT] Saved {path}")

    # Final checkpoint
    final_path = ckpt_dir / "bid_policy_final.pt"
    torch.save({
        "update": update,
        "bid_policy_state_dict": policy.state_dict(),
        "bid_value_state_dict": value_net.state_dict(),
        "bid_policy_config": {
            "obs_dim": BID_OBS_DIM,
            "hidden": bp_cfg.get("hidden", [64, 64]),
        },
    }, final_path)
    print(f"\n[DONE] Final checkpoint: {final_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/config.yaml")
    parser.add_argument(
        "--exec-checkpoint",
        default="checkpoints/actor_update204_final.pt",
        help="Path to the frozen execution actor checkpoint",
    )
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg, args.exec_checkpoint)
