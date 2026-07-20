# 🚁 Drone Network — Household Task Swarm RL

A multi-agent reinforcement learning system that trains a swarm of mini-drones to cooperatively complete household tasks — watering plants, sweeping floors, and toggling lights.

Trains fast on a pure-Python simulation, then deploys into a **PyBullet physics lab** with real Crazyflie quadrotor aerodynamics.

---

## Architecture

```
MAPPO (Multi-Agent PPO) — CTDE pattern
  ├── Shared Actor       per-drone policy      15-dim obs → 4-dim action
  ├── Central Critic     global value network  (n_drones × 15)-dim obs → scalar
  └── GAE Rollout Buffer + PPO clip update + reward normalisation
```

**CTDE = Centralised Training, Decentralised Execution.**
During training the critic sees all drones at once. At deployment each drone only uses its own local observation — no communication required.

---

## Project Structure

```
drone-network/
├── envs/
│   ├── home_env.py          # Fast training env — pure Python/NumPy
│   ├── pybullet_env.py      # Physics deployment env — real quadrotor dynamics
│   ├── drone_agent.py       # Single-drone kinematics & observation builder
│   └── tasks/
│       ├── base_task.py     # Abstract task interface
│       ├── water_plant.py   # Hover at pot + engage tool for N steps
│       ├── sweep_floor.py   # Visit a sequence of floor waypoints
│       └── toggle_light.py  # Fly to switch + momentary tap
├── models/
│   ├── actor.py             # Stochastic Gaussian policy MLP
│   └── critic.py            # Centralised value MLP
├── training/
│   ├── train_mappo.py       # Main training loop (Ctrl+C saves checkpoint)
│   └── config.yaml          # All hyperparameters
├── evaluation/
│   └── eval.py              # Load checkpoint → benchmark episodes
├── lab/
│   └── deploy.py            # Load checkpoint → PyBullet GUI
├── assets/
│   ├── plant_pot.urdf       # 3D plant pot model
│   ├── light_switch.urdf    # 3D light switch model
│   └── floor_zone.urdf      # Sweeping zone marker
├── utils/
│   ├── replay_buffer.py     # On-policy GAE rollout buffer
│   └── reward_shaping.py    # Running reward normaliser + curriculum scheduler
├── install.py               # Cross-platform dependency installer
├── install.sh               # macOS / Linux launcher
└── install.bat              # Windows launcher
```

---

## Quick Start

### 1. Install dependencies

**macOS / Linux**
```bash
chmod +x install.sh && ./install.sh
```

**Windows**
```bat
install.bat
```

**Or directly:**
```bash
python install.py
```

> The installer auto-detects your platform. On macOS with clang 17+ / SDK 15+ (including macOS Sequoia/Tahoe) it patches PyBullet's source before compiling. On Windows and Linux it uses pre-built wheels — no compilation needed.

**Verify everything installed correctly:**
```bash
python install.py --check
```

### 2. Train

```bash
python -m training.train_mappo
```

Press **Ctrl+C** at any time — training stops cleanly after the current update and saves a checkpoint.

### 3. Evaluate a checkpoint

```bash
python -m evaluation.eval \
  --checkpoint checkpoints/actor_update100_final.pt \
  --episodes 20
```

### 4. Deploy in PyBullet physics lab

```bash
# Real-time, GUI open
python -m lab.deploy \
  --checkpoint checkpoints/actor_update100_final.pt

# Slow-motion (easier to watch)
python -m lab.deploy \
  --checkpoint checkpoints/actor_update100_final.pt \
  --time-scale 0.3

# Headless benchmark
python -m lab.deploy \
  --checkpoint checkpoints/actor_update100_final.pt \
  --no-gui --episodes 50
```

---

## Observation Space (per drone, 15 dims)

| Index | Meaning |
|---|---|
| 0–2 | Own position (x, y, z) metres |
| 3 | Battery level normalised 0–1 |
| 4 | Task progress 0–1 |
| 5–7 | Vector to task target (dx, dy, dz) |
| 8–10 | Own velocity (vx, vy, vz) |
| 11 | Tool engaged flag (0 / 1) |
| 12–14 | Nearest neighbour relative position |

## Action Space (per drone, 4 dims)

| Index | Meaning | Range |
|---|---|---|
| 0–2 | Δx, Δy, Δz movement | −1 to +1 (tanh squashed) |
| 3 | Tool engage signal | 0 to 1 (sigmoid, threshold 0.5) |

## Reward Structure

| Event | Value |
|---|---|
| Per-step alive penalty | −0.01 |
| Drone–drone collision | −5.0 |
| Battery depleted | −3.0 |
| Water plant complete | +10.0 |
| Sweep floor complete | +12.0 |
| Toggle light complete | +8.0 |
| Cooperative bonus (all done early) | +2.0 per drone |
| Dense proximity shaping | small positive gradient |

---

## Training Metrics — What to Expect

| Phase | Steps | Mean Reward | Entropy | Value Loss |
|---|---|---|---|---|
| Random | 0 | −5 to −15 | ~5.7 | ~0.5 |
| Early signal | ~100k | −2 to +5 | 5–8 | 0.5–2.0 |
| Learning | ~500k | +5 to +20 | 6–9 | 0.5–1.5 |
| Competent | ~1–2M | +20 to +40 | 4–7 (falling) | < 0.5 |
| Good | ~3–5M | +40 to +55 | 2–5 | < 0.3 |

**Healthy signs:**
- Policy loss: small and negative (−0.001 to −0.02)
- Entropy: starts ~5.7, rises slightly during exploration, then falls
- Eval reward should track training reward closely

**Red flags:**
- Entropy > 10 sustained → `log_std` saturating; lower `lr_actor`
- Policy loss turning positive → clip triggering too much; lower `lr_actor` or `n_epochs`
- Eval reward much worse than training reward → policy relying on noise; std too high

---

## Curriculum

Training automatically advances through 4 stages as eval reward crosses thresholds defined in [`training/config.yaml`](training/config.yaml):

| Stage | Threshold | Description |
|---|---|---|
| 0 | 5.0 | Default — all 5 tasks, 3 drones |
| 1 | 15.0 | Policy completing 1–2 tasks reliably |
| 2 | 30.0 | 3–4 tasks per episode |
| 3 | 50.0 | All tasks + cooperative bonus |

---

## Extending

**Add a new household task:**
1. Create `envs/tasks/my_task.py` subclassing [`BaseTask`](envs/tasks/base_task.py)
2. Implement `step(drone_position, tool_engaged) → float` and `completion_reward() → float`
3. Register in `envs/home_env.py` → `_TASK_REGISTRY` and `_DEFAULT_TASK_LAYOUTS`
4. Add a URDF asset to `assets/` and register it in `envs/pybullet_env.py` → `urdf_map`

**Scale up drones:**
Change `n_drones` in [`training/config.yaml`](training/config.yaml). The actor is parameter-shared so it works for any N without retraining from scratch. The critic input dim auto-scales.

---

## Windows Compatibility

| Component | Status | Notes |
|---|---|---|
| Training (`train_mappo.py`) | ✅ | Pure Python, no compilation |
| HomeEnv | ✅ | Pure Python/NumPy |
| PyBullet | ✅ | Pre-built wheel, no source build |
| PyBullet GUI | ✅ | Opens normally |
| gym-pybullet-drones | ✅ | Needs Git in PATH: `winget install Git.Git` |

---

## Dependencies

```
gymnasium >= 0.29.0
numpy     >= 1.24.0
torch     >= 2.0.0
pyyaml    >= 6.0
pybullet  3.2.7           (physics lab only)
gym-pybullet-drones 2.1.0 (physics lab only)
wandb                     (optional logging)
```

---

## License

MIT
