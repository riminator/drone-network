"""
train_mappo.py
Main training loop for MAPPO on the HomeEnv (or MujocoHomeEnv) drone swarm.

Run:
    python -m training.train_mappo
    python -m training.train_mappo --config training/config.yaml
    python -m training.train_mappo --config training/config_mujoco.yaml --sim mujoco
    python -m training.train_mappo --config training/config.yaml --wandb

Architecture:
    - One shared Actor (parameters shared across all drones — parameter sharing
      is the default MAPPO trick for homogeneous agents)
    - One CentralCritic that sees all drone observations concatenated
    - GAE-based rollout buffer
    - PPO clip update, n_epochs passes per rollout
"""

from __future__ import annotations

import argparse
import os
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import yaml

from envs.home_env import HomeEnv
from envs.mujoco_env import MujocoHomeEnv
from models.actor import Actor
from models.critic import CentralCritic
from utils.replay_buffer import RolloutBuffer
from utils.reward_shaping import RewardNormalizer, CurriculumScheduler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_global_obs(
    obs_dict: dict[str, np.ndarray],
    agent_ids: list[str],
    obs_dim: int,
) -> np.ndarray:
    """Concatenate all agent observations into a single vector for the critic."""
    return np.concatenate([obs_dict.get(aid, np.zeros(obs_dim)) for aid in agent_ids])


def dict_to_tensor(d: dict[str, np.ndarray], agent_ids: list[str], device: str):
    """Stack per-agent arrays into a (n_agents, ...) tensor."""
    arrays = [d.get(aid, np.zeros_like(list(d.values())[0])) for aid in agent_ids]
    return torch.tensor(np.stack(arrays), dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def ppo_update(
    buffer: RolloutBuffer,
    actor: Actor,
    critic: CentralCritic,
    actor_optim: optim.Optimizer,
    critic_optim: optim.Optimizer,
    cfg: dict,
) -> dict[str, float]:
    """Run n_epochs of PPO updates over the filled buffer. Returns loss stats."""
    clip_eps = cfg["training"]["clip_eps"]
    entropy_coef = cfg["training"]["entropy_coef"]
    value_coef = cfg["training"]["value_coef"]
    max_grad_norm = cfg["training"]["max_grad_norm"]
    n_epochs = cfg["training"]["n_epochs"]
    mini_batch_size = cfg["training"]["mini_batch_size"]

    stats = {"policy_loss": [], "value_loss": [], "entropy": [], "total_loss": []}

    for _ in range(n_epochs):
        for batch in buffer.get_batches(mini_batch_size):
            obs = batch["obs"]               # (B, obs_dim)
            actions = batch["actions"]       # (B, act_dim)
            old_log_probs = batch["log_probs"]  # (B,)
            returns = batch["returns"]       # (B,)
            advantages = batch["advantages"] # (B,)
            global_obs = batch["global_obs"] # (B, n_agents * obs_dim)

            # --- Actor loss (PPO clip) ---
            new_log_probs, entropy = actor.evaluate_actions(obs, actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # --- Critic loss ---
            values = critic(global_obs)
            value_loss = 0.5 * ((values - returns) ** 2).mean()

            # --- Entropy bonus ---
            entropy_loss = -entropy.mean()

            total_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

            actor_optim.zero_grad()
            critic_optim.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            actor_optim.step()
            critic_optim.step()

            stats["policy_loss"].append(policy_loss.item())
            stats["value_loss"].append(value_loss.item())
            stats["entropy"].append(-entropy_loss.item())
            stats["total_loss"].append(total_loss.item())

    return {k: float(np.mean(v)) for k, v in stats.items()}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(env: HomeEnv, actor: Actor, n_episodes: int, device: str) -> float:
    """Run n_episodes deterministically and return mean episode reward."""
    agent_ids = sorted(env._agent_ids)
    total_rewards = []

    for _ in range(n_episodes):
        obs_dict, _ = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            with torch.no_grad():
                obs_tensor = dict_to_tensor(obs_dict, agent_ids, device)
                squashed, _raw, _lp = actor.get_action(obs_tensor, deterministic=True)
                action_dict = {
                    aid: squashed[i].cpu().numpy()
                    for i, aid in enumerate(agent_ids)
                }

            obs_dict, rewards, terminated, truncated, _ = env.step(action_dict)
            ep_reward += sum(rewards.values())
            done = terminated.get("__all__", False) or truncated.get("__all__", False)

        total_rewards.append(ep_reward)

    return float(np.mean(total_rewards))


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

_stop_training = False


def _handle_signal(signum, frame):  # noqa: ARG001
    global _stop_training
    if not _stop_training:
        print("\n[STOP] Signal received — finishing current update then saving checkpoint...")
        _stop_training = True


# ---------------------------------------------------------------------------
# Checkpoint helper (shared by normal completion and early exit)
# ---------------------------------------------------------------------------

def _save_checkpoint(
    checkpoint_dir: Path,
    update_count: int,
    timesteps_collected: int,
    actor,
    critic,
    actor_optim,
    critic_optim,
    tag: str = "",
):
    suffix = f"_{tag}" if tag else ""
    ckpt_path = checkpoint_dir / f"actor_update{update_count}{suffix}.pt"
    torch.save(
        {
            "update": update_count,
            "timesteps": timesteps_collected,
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "actor_optim_state_dict": actor_optim.state_dict(),
            "critic_optim_state_dict": critic_optim.state_dict(),
        },
        ckpt_path,
    )
    print(f"  [CKPT] Saved {ckpt_path}")
    return ckpt_path


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: dict, sim: str = "home", resume_checkpoint: str | None = None):
    # Auto-detect best available device.
    # Priority: CUDA (Nvidia, Windows/Linux/Colab) → CPU
    # MPS (Apple Silicon) is intentionally skipped — benchmarked at only 1.05× CPU
    # for these small MLPs, not worth the overhead.
    import torch as _torch
    device = "cuda" if _torch.cuda.is_available() else "cpu"
    obs_dim = cfg["model"]["obs_dim"]
    act_dim = cfg["model"]["act_dim"]
    n_drones = cfg["env"]["n_drones"]
    rollout_steps = cfg["training"]["rollout_steps"]
    total_timesteps = cfg["training"]["total_timesteps"]

    # --- Environment ---
    if sim == "mujoco":
        env = MujocoHomeEnv(config=cfg["env"])
    else:
        env = HomeEnv(config=cfg["env"])
    obs_dict, _ = env.reset()
    agent_ids = sorted(env._agent_ids)

    # --- Models ---
    actor = Actor(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=cfg["model"]["actor_hidden"],
    ).to(device)

    critic = CentralCritic(
        obs_dim=obs_dim,
        n_drones=n_drones,
        hidden_sizes=cfg["model"]["critic_hidden"],
    ).to(device)

    actor_optim = optim.Adam(actor.parameters(), lr=cfg["training"]["lr_actor"])
    critic_optim = optim.Adam(critic.parameters(), lr=cfg["training"]["lr_critic"])

    # --- Resume from checkpoint (actor weights only) ---
    update_count = 0
    timesteps_collected = 0
    if resume_checkpoint:
        ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        actor.load_state_dict(ckpt["actor_state_dict"])
        # Reset log_std to -0.5 (std≈0.6) so the resumed policy acts more
        # deterministically. Checkpoints trained with broken rewards often have
        # log_std near the ceiling (+0.5) → entropy 7.65 → pure noise even
        # though the mean weights are good. Resetting lets PPO exploit the
        # navigation the checkpoint already learned.
        with torch.no_grad():
            actor.log_std.fill_(-0.5)
        # Critic starts fresh — old critic was trained on different reward scale
        update_count      = ckpt.get("update", 0)
        timesteps_collected = ckpt.get("timesteps", 0)
        print(
            f"Resumed actor from {resume_checkpoint}\n"
            f"  update={update_count}  timesteps={timesteps_collected:,}\n"
            f"  log_std reset to -0.5 (std≈0.6, entropy≈5.4)\n"
            f"  (critic re-initialised — reward scale changed)"
        )

    # --- Buffer ---
    buffer = RolloutBuffer(
        n_steps=rollout_steps,
        n_agents=n_drones,
        obs_dim=obs_dim,
        act_dim=act_dim,
        gamma=cfg["training"]["gamma"],
        gae_lambda=cfg["training"]["gae_lambda"],
        device=device,
    )
    buffer.set_agent_ids(agent_ids)

    # --- Utilities ---
    reward_normalizer = RewardNormalizer() if cfg["training"]["normalise_rewards"] else None
    curriculum = None
    if cfg["curriculum"]["enabled"]:
        curriculum = CurriculumScheduler(
            stage_thresholds=cfg["curriculum"]["stage_thresholds"],
            patience=cfg["curriculum"]["patience"],
        )

    checkpoint_dir = Path(cfg["logging"]["checkpoint_dir"])
    checkpoint_dir.mkdir(exist_ok=True)

    # --- W&B (optional) ---
    wandb_run = None
    if cfg["logging"]["use_wandb"]:
        try:
            import wandb
            wandb_run = wandb.init(
                project=cfg["logging"]["wandb_project"],
                entity=cfg["logging"]["wandb_entity"],
                config=cfg,
            )
        except ImportError:
            print("wandb not installed — skipping W&B logging.")

    # --- Signal handlers ---
    global _stop_training
    _stop_training = False
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # --- Training loop ---
    start_time = time.time()

    print(f"Starting MAPPO training — {total_timesteps:,} timesteps, {n_drones} drones")
    print(f"Device: {device} | Rollout steps: {rollout_steps} | Mini-batch: {cfg['training']['mini_batch_size']}")
    print("Press Ctrl+C at any time to stop safely and save a checkpoint.\n")

    while timesteps_collected < total_timesteps and not _stop_training:
        # ---- Rollout collection ----
        buffer.reset()
        ep_rewards: list[float] = []
        ep_reward = 0.0

        for _ in range(rollout_steps):
            with torch.no_grad():
                obs_tensor = dict_to_tensor(obs_dict, agent_ids, device)
                # get_action now returns (squashed_action, raw_action, log_prob)
                squashed_tensor, raw_tensor, log_probs_tensor = actor.get_action(obs_tensor)

                global_obs_np = build_global_obs(obs_dict, agent_ids, obs_dim)
                global_obs_tensor = torch.tensor(
                    global_obs_np, dtype=torch.float32, device=device
                ).unsqueeze(0)
                values_tensor = critic(global_obs_tensor)  # (1,) — one value for global state

            # Squashed actions go to the env
            action_dict = {
                aid: squashed_tensor[i].cpu().numpy() for i, aid in enumerate(agent_ids)
            }
            # RAW (pre-squash) actions go into the buffer for evaluate_actions()
            raw_action_dict = {
                aid: raw_tensor[i].cpu().numpy() for i, aid in enumerate(agent_ids)
            }
            log_prob_dict = {
                aid: log_probs_tensor[i].item() for i, aid in enumerate(agent_ids)
            }
            # All agents share the same global state value (CTDE: one critic for all)
            shared_value = values_tensor[0].item()
            value_dict = {aid: shared_value for aid in agent_ids}

            next_obs_dict, reward_dict, terminated, truncated, _ = env.step(action_dict)

            raw_rewards = np.array([reward_dict.get(aid, 0.0) for aid in agent_ids])
            if reward_normalizer is not None:
                raw_rewards = reward_normalizer.normalise(raw_rewards)
            norm_reward_dict = {aid: raw_rewards[i] for i, aid in enumerate(agent_ids)}

            done_dict = {
                aid: terminated.get(aid, False) or truncated.get(aid, False)
                for aid in agent_ids
            }

            # Store raw actions in buffer (not squashed) — required for correct PPO ratios
            buffer.add(obs_dict, raw_action_dict, norm_reward_dict, done_dict,
                       log_prob_dict, value_dict)

            ep_reward += sum(reward_dict.values())
            timesteps_collected += n_drones

            episode_done = (
                terminated.get("__all__", False) or truncated.get("__all__", False)
            )
            if episode_done:
                ep_rewards.append(ep_reward)
                ep_reward = 0.0
                obs_dict, _ = env.reset()
            else:
                obs_dict = next_obs_dict

        # Bootstrap last value
        with torch.no_grad():
            global_obs_np = build_global_obs(obs_dict, agent_ids, obs_dim)
            global_obs_tensor = torch.tensor(
                global_obs_np, dtype=torch.float32, device=device
            ).unsqueeze(0)
            last_val = critic(global_obs_tensor)[0].item()
        last_values = {aid: last_val for aid in agent_ids}
        buffer.compute_returns_and_advantages(last_values)

        # ---- PPO update ----
        loss_stats = ppo_update(buffer, actor, critic, actor_optim, critic_optim, cfg)
        update_count += 1

        # ---- Logging ----
        log_interval = cfg["logging"]["log_interval"]
        if update_count % log_interval == 0:
            mean_ep_reward = float(np.mean(ep_rewards)) if ep_rewards else 0.0
            elapsed = time.time() - start_time
            print(
                f"Update {update_count:5d} | "
                f"Steps {timesteps_collected:>10,} | "
                f"MeanReward {mean_ep_reward:+7.2f} | "
                f"PolicyLoss {loss_stats['policy_loss']:+.4f} | "
                f"ValueLoss {loss_stats['value_loss']:.4f} | "
                f"Entropy {loss_stats['entropy']:.4f} | "
                f"Elapsed {elapsed:.0f}s"
            )
            if wandb_run:
                wandb_run.log({
                    "mean_ep_reward": mean_ep_reward,
                    "timesteps": timesteps_collected,
                    **loss_stats,
                })

        # ---- Evaluation ----
        eval_interval = cfg["logging"]["eval_interval"]
        if update_count % eval_interval == 0:
            eval_reward = evaluate(
                env, actor, cfg["logging"]["eval_episodes"], device
            )
            print(f"  [EVAL] mean_reward={eval_reward:.2f}")

            # Curriculum advancement
            if curriculum is not None:
                advanced = curriculum.update(eval_reward)
                if advanced:
                    print(f"  [CURRICULUM] Advanced to stage {curriculum.stage}")

        # ---- Checkpoint ----
        ckpt_interval = cfg["logging"]["checkpoint_interval"]
        if update_count % ckpt_interval == 0:
            _save_checkpoint(
                checkpoint_dir, update_count, timesteps_collected,
                actor, critic, actor_optim, critic_optim,
            )

    # ---- Final save (normal completion or early stop) ----
    if _stop_training:
        print(f"[STOP] Training interrupted at update {update_count} ({timesteps_collected:,} steps).")
    else:
        print("Training complete.")

    _save_checkpoint(
        checkpoint_dir, update_count, timesteps_collected,
        actor, critic, actor_optim, critic_optim,
        tag="final" if not _stop_training else "interrupted",
    )

    if wandb_run:
        wandb_run.finish()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="training/config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--sim",
        default="home",
        choices=["home", "mujoco"],
        help="Environment backend: 'home' (default, fast teleport) or 'mujoco' (physics)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="CHECKPOINT",
        help="Resume actor weights from this .pt file (critic re-initialises). "
             "Use the best checkpoint from a previous run, e.g. "
             "checkpoints/actor_update345_interrupted.pt",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg, sim=args.sim, resume_checkpoint=args.resume)
