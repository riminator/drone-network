# 🚁 Drone Network — Market-Based Dynamic Task Allocation for RL Swarms

A research platform investigating **auction-based dynamic task allocation** for RL-controlled drone swarms performing household service tasks in continuously changing environments. The swarm must reassign work in real time as the task set itself shifts — tasks vanish mid-mission, new tasks appear, drones fail — without retraining the execution policies.

Trains fast on a pure-Python simulation, then deploys into a **PyBullet physics lab** with real Crazyflie quadrotor aerodynamics.

---

## Research Problem

Classical multi-robot task allocation (MRTA) assumes a fixed task set. This project addresses the harder problem where the task set is **non-stationary**: tasks appear and disappear at runtime, some tasks are divisible (multiple drones can collaborate on a single large task), and communication between drones is delayed or unreliable.

The core contribution is replacing hand-crafted bidding heuristics with a **learned bidding policy** (PPO-trained against optimal-assignment baselines) that can represent both:

- **Discrete reassignment** — a task vanishes; the freed drone must instantly re-enter the auction for remaining tasks.
- **Continuous task-sharing** — an idle drone can bid to *join* an in-progress task using a dedicated **learned marginal-value head**, rather than a hand-crafted formula.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  EXECUTION LAYER  (frozen after Phase 1 training)                    │
│                                                                      │
│   Actor (per drone)  ←  15-dim obs  →  4-dim action (dx,dy,dz,tool) │
│   CentralCritic      ←  n×15-dim global obs  →  V(s)                │
│   Training: MAPPO (CTDE) — shared Actor, centralised Critic          │
└──────────────────────────────────────────────────────────────────────┘
                              ↕  allocator.allocate(WorldSnapshot)
┌──────────────────────────────────────────────────────────────────────┐
│  ALLOCATION LAYER  (the research focus)                              │
│                                                                      │
│   BaseAllocator ← WorldSnapshot (positions, batteries, task states)  │
│        │                                                             │
│        ├── GreedyAuction   distance/battery bid, O(D×T) per round   │
│        ├── CBBA            Consensus-Based Bundle Algorithm          │
│        │                   configurable comm_delay for robustness    │
│        ├── OracleAllocator Hungarian algorithm upper bound           │
│        └── LearnedBidder   BidPolicy (PPO-trained 14-dim → sigmoid)  │
│                            dual-head: primary bid + marginal bid     │
│                            obs_delay parameter for robustness eval   │
│                                                                      │
│   BidPolicy obs (14 dims): drone pos, battery, progress,             │
│     vector-to-task, remaining_work, n_assigned, urgency, type-onehot │
└──────────────────────────────────────────────────────────────────────┘
                              ↕  AllocationResult (assignments + bids)
┌──────────────────────────────────────────────────────────────────────┐
│  ENVIRONMENT LAYER                                                   │
│                                                                      │
│   HomeEnv          fast Python/NumPy sim (training + eval)           │
│   MujocoHomeEnv    MuJoCo 3 physics sim  (recommended deploy)        │
│   PybulletHomeEnv  PyBullet physics sim  (legacy deploy)             │
│                                                                      │
│   Tasks: WaterPlant · SweepFloor (shareable) · ToggleLight           │
│   Disruptions: remove_task() · add_task() · drone battery failure    │
└──────────────────────────────────────────────────────────────────────┘
```

**CTDE = Centralised Training, Decentralised Execution.**  
During training the critic sees all drones at once. At deployment each drone uses only its own local observation — no communication required for execution. The bidding layer runs once per reallocation event (or periodically), not every step.

---

## Project Structure

```
drone-network/
├── envs/
│   ├── home_env.py           Fast training env — pure Python/NumPy
│   │                           allocator plug-in via config["allocator"]
│   │                           disruption API: remove_task() / add_task()
│   ├── mujoco_env.py         MuJoCo 3 physics env (recommended deploy)
│   │                           RK4 integrator @ 500 Hz, Newton contact solver
│   │                           Crazyflie-scale quadrotor, attitude PID
│   │                           render=True → passive viewer (no external deps)
│   ├── pybullet_env.py       PyBullet physics env (legacy deploy)
│   │                           auction tick every K steps (auction_interval)
│   │                           real-time bid visualisation (coloured lines)
│   ├── drone_agent.py        Single-drone kinematics & 15-dim obs builder
│   └── tasks/
│       ├── base_task.py      Abstract interface — TaskStatus FSM, remaining_work()
│       ├── water_plant.py    Hover at pot + engage tool for N steps
│       ├── sweep_floor.py    Visit N floor waypoints (shareable task)
│       └── toggle_light.py   Fly to switch + momentary tap
│
├── allocator/
│   ├── base_allocator.py     WorldSnapshot / AllocationResult / Bid dataclasses
│   │                           BaseAllocator ABC (allocate / on_task_complete / on_task_vanish)
│   ├── greedy_auction.py     Greedy distance×battery sealed-bid auction
│   │                           + co-assignment pass for shareable tasks
│   ├── cbba.py               Consensus-Based Bundle Algorithm
│   │                           configurable comm_delay, MAX_BUNDLE_SIZE=3
│   ├── oracle.py             Hungarian-algorithm optimal assignment (scipy)
│   ├── bid_policy.py         BidPolicy MLP: 14-dim obs → (primary_bid, marginal_bid)
│   │                           shared trunk + two independent output heads
│   │                           build_bid_obs() constructs per-pair observation
│   ├── learned_bidder.py     LearnedBidder wraps BidPolicy as a BaseAllocator
│   │                           from_checkpoint() with old-format migration
│   │                           obs_delay parameter for robustness experiments
│   └── bid_env.py            BidEnv: HomeEnv wrapper that collects bid transitions
│                               BidBuffer / BidTransition for bid-policy PPO
│                               obs_delay: stale observation simulation
│
├── models/
│   ├── actor.py              Gaussian policy MLP (tanh/sigmoid squashing, log-prob correction)
│   └── critic.py             Centralised value MLP (n_drones × 15 → scalar)
│
├── training/
│   ├── train_mappo.py        Phase 1: MAPPO loop — shared Actor + CentralCritic
│   ├── train_bid_policy.py   Phase 3: BidPolicy PPO — frozen exec actor + BidValueNet
│   ├── config.yaml           MAPPO hyperparameters + bid-policy section
│   └── config_bid.yaml       Dedicated bid-policy config (7-task harder layout)
│
├── evaluation/
│   ├── eval.py               Load checkpoint → benchmark episodes (execution eval)
│   └── eval_allocation.py    Phase 4: disruption-scenario harness
│                               4 allocators × 6 scenarios × N episodes
│                               EpisodeMetrics (incl. mean_realloc_latency) + CSV export
│
├── lab/
│   └── deploy.py             Load checkpoint → physics GUI
│                               --sim {pybullet,mujoco}  (backend selector)
│                               --allocator {greedy,cbba,oracle,learned}
│                               --auction-interval K  (periodic re-auction)
│                               --bid-checkpoint (LearnedBidder weights)
│
├── tests/
│   ├── test_phase1.py        TaskStatus FSM, disruption API, co-assignment (25 tests)
│   ├── test_phase2.py        GreedyAuction + CBBA contract + disruption (28 tests)
│   ├── test_phase3.py        BidPolicy dual-head, BidBuffer, OracleAllocator,
│   │                           LearnedBidder, obs_delay, marginal bids (35 tests)
│   ├── test_phase4.py        EpisodeMetrics, realloc latency, scenario hooks,
│   │                           _CountingAllocator, benchmark driver (52 tests)
│   └── test_phase5.py        PybulletEnv allocator integration, bid lines (36 tests)
│
├── utils/
│   ├── replay_buffer.py      On-policy GAE rollout buffer (MAPPO)
│   └── reward_shaping.py     Running reward normaliser + curriculum scheduler
│
├── assets/
│   ├── plant_pot.urdf
│   ├── light_switch.urdf
│   └── floor_zone.urdf
│
├── checkpoints/              (git-ignored — present locally)
│   ├── actor_update204_final.pt    Best execution actor (5M steps)
│   ├── bid_policy_final.pt         Best learned bidder (400 updates)
│   └── bid_policy_update{50,100,...,400}.pt
│
└── results/                  (git-ignored — present locally)
    └── phase4_results.csv    20-episode benchmark (480 rows)
```

---

## Testing Everything

All commands run from the repo root. Checkpoints are in `checkpoints/` (git-ignored).

### 0 — Unit tests (run first, always)

```bash
# Full suite — 176 tests, ~2 seconds
python3 -m pytest tests/ -v

# Per-phase
python3 -m pytest tests/test_phase1.py -v   # env, disruption API, co-assignment
python3 -m pytest tests/test_phase2.py -v   # GreedyAuction, CBBA
python3 -m pytest tests/test_phase3.py -v   # BidPolicy dual-head, BidEnv, LearnedBidder
python3 -m pytest tests/test_phase4.py -v   # eval harness, realloc latency metric
python3 -m pytest tests/test_phase5.py -v   # PyBullet integration

# Specific feature groups
python3 -m pytest tests/test_phase3.py -k "obs_delay" -v          # robustness parameter
python3 -m pytest tests/test_phase3.py -k "marginal" -v           # marginal-value head
python3 -m pytest tests/test_phase4.py -k "latency" -v            # realloc latency metric
python3 -m pytest tests/test_phase4.py -k "disruption" -v         # disruption hooks
python3 -m pytest tests/ -k "shareable" -v                        # co-assignment
```

Expected output: `217 passed` (176 existing + 41 new MuJoCo tests)

---

### 1 — Verify checkpoints load correctly

```bash
# Execution actor
python3 -c "
import torch
ckpt = torch.load('checkpoints/actor_update204_final.pt', map_location='cpu', weights_only=False)
print('Actor  — update:', ckpt['update'], '| timesteps:', f\"{ckpt['timesteps']:,}\")
"

# Bid policy (includes backward-compat migration for old single-head format)
python3 -c "
from allocator.learned_bidder import LearnedBidder
lb = LearnedBidder.from_checkpoint('checkpoints/bid_policy_final.pt')
import numpy as np
from allocator.bid_policy import build_bid_obs, BID_OBS_DIM
from envs.tasks.base_task import TaskSpec, TaskStatus
from envs.tasks.water_plant import WaterPlantTask
spec = TaskSpec('t', 'water_plant', np.array([3.,3.,1.], dtype=np.float32), engage_steps_required=10)
task = WaterPlantTask(spec); task.status = TaskStatus.ASSIGNED
obs = build_bid_obs(np.zeros(3, dtype=np.float32), 1.0, 0.0, task, 0, 500)
print('primary bid :', lb.policy.bid_numpy(obs))
print('marginal bid:', lb.policy.marginal_bid_numpy(obs))
"
```

---

### 2 — Evaluate execution quality

```bash
# 20 deterministic episodes with the trained actor
python3 -m evaluation.eval \
    --checkpoint checkpoints/actor_update204_final.pt \
    --episodes 20

# Render ASCII output step-by-step (slow)
python3 -m evaluation.eval \
    --checkpoint checkpoints/actor_update204_final.pt \
    --episodes 3 \
    --render
```

Expected: task completion ~80–100%, mean reward +40 to +60.

---

### 3 — Disruption-scenario benchmark (the main paper results)

**Important:** `--exec-checkpoint` takes the **actor** file; `--checkpoint` takes the **bid policy** file.

```bash
# Full run — 4 allocators × 6 scenarios × 20 episodes = 480 episodes (~3 min)
python3 -m evaluation.eval_allocation \
    --exec-checkpoint checkpoints/actor_update204_final.pt \
    --checkpoint     checkpoints/bid_policy_final.pt \
    --episodes 20 \
    --csv results/phase4_results.csv

# Quick smoke — random actions, no checkpoints needed (~15 seconds)
python3 -m evaluation.eval_allocation --episodes 3

# Single scenario
python3 -m evaluation.eval_allocation \
    --exec-checkpoint checkpoints/actor_update204_final.pt \
    --checkpoint     checkpoints/bid_policy_final.pt \
    --scenarios task_vanish \
    --episodes 10

# Two allocators head-to-head on all scenarios
python3 -m evaluation.eval_allocation \
    --exec-checkpoint checkpoints/actor_update204_final.pt \
    --checkpoint     checkpoints/bid_policy_final.pt \
    --allocators cbba learned \
    --episodes 20

# Robustness: CBBA (comm_delay=10) vs Learned (obs_delay has no CLI flag —
# controlled at library level; use the comm_delay scenario for the paper comparison)
python3 -m evaluation.eval_allocation \
    --exec-checkpoint checkpoints/actor_update204_final.pt \
    --checkpoint     checkpoints/bid_policy_final.pt \
    --scenarios comm_delay \
    --allocators greedy cbba oracle learned \
    --comm-delay 10 \
    --episodes 20

# Surge scenario only (tests co-assignment + task injection handling)
python3 -m evaluation.eval_allocation \
    --exec-checkpoint checkpoints/actor_update204_final.pt \
    --checkpoint     checkpoints/bid_policy_final.pt \
    --scenarios surge \
    --episodes 20

# Drone failure scenario only
python3 -m evaluation.eval_allocation \
    --exec-checkpoint checkpoints/actor_update204_final.pt \
    --checkpoint     checkpoints/bid_policy_final.pt \
    --scenarios drone_failure \
    --episodes 20
```

**Metrics in the output table:**

| Column | Meaning |
|---|---|
| `CompRate` | Fraction of eligible tasks completed (excl. vanished) |
| `Makespan` | Steps until all tasks done, or `max_steps` if not |
| `TotReward` | Total episode reward |
| `BattFinal` | Mean normalised drone battery at episode end |
| `Reallocs` | Number of `allocate()` calls |
| `ReallocLat` | Mean steps from disruption to drone reassignment (−1 = no disruption) |
| `Collisions` | Pairwise drone collision count |

---

### 4 — MuJoCo physics lab

> **Checkpoint note:** use `actor_update500.pt` for MuJoCo (trained end-to-end in
> the physics sim). `actor_update204_final.pt` is the HomeEnv policy — it will
> navigate but won't reliably complete tasks due to different kinematics.

```bash
# MuJoCo headless (recommended — no mjpython required)
python3 -m lab.deploy \
    --sim mujoco --no-gui \
    --checkpoint checkpoints/actor_update500.pt \
    --allocator greedy --episodes 5

# MuJoCo — learned bidder, headless
python3 -m lab.deploy \
    --sim mujoco --no-gui \
    --checkpoint     checkpoints/actor_update500.pt \
    --allocator      learned \
    --bid-checkpoint checkpoints/bid_policy_final.pt \
    --episodes 5

# MuJoCo GUI (macOS: must use mjpython, not python3)
mjpython -m lab.deploy \
    --sim mujoco \
    --checkpoint checkpoints/actor_update500.pt
```

**MuJoCo viewer controls:** Right-click + drag to rotate · scroll to zoom · double-click a body to track it.

**On macOS** running `python3 -m lab.deploy --sim mujoco` without `mjpython` silently
falls back to headless and prints the correct `mjpython` command.

---

### 5 — PyBullet physics lab (legacy)

```bash
# Greedy allocator, real-time GUI (default)
python3 -m lab.deploy \
    --checkpoint checkpoints/actor_update204_final.pt

# Learned bidder with trained weights, slow-motion
python3 -m lab.deploy \
    --checkpoint     checkpoints/actor_update204_final.pt \
    --allocator      learned \
    --bid-checkpoint checkpoints/bid_policy_final.pt \
    --time-scale 0.3

# CBBA, 6 drones, slow-motion
python3 -m lab.deploy \
    --checkpoint checkpoints/actor_update204_final.pt \
    --allocator  cbba \
    --n-drones   6 \
    --time-scale 0.3

# Oracle allocator (Hungarian, upper bound)
python3 -m lab.deploy \
    --checkpoint checkpoints/actor_update204_final.pt \
    --allocator  oracle \
    --time-scale 0.3

# Periodic re-auction every 20 steps (bid lines update visibly)
python3 -m lab.deploy \
    --checkpoint      checkpoints/actor_update204_final.pt \
    --allocator       greedy \
    --auction-interval 20

# Headless benchmark — no GUI, max speed
python3 -m lab.deploy \
    --checkpoint     checkpoints/actor_update204_final.pt \
    --allocator      learned \
    --bid-checkpoint checkpoints/bid_policy_final.pt \
    --no-gui \
    --episodes 20
```

**Camera controls** (click the PyBullet window first):

| Key | Action |
|---|---|
| `W` / `S` | Pan forward / back |
| `A` / `D` | Pan left / right |
| `Q` / `E` | Zoom in / out |
| `Z` / `X` | Rotate left / right (yaw) |
| `R` / `F` | Tilt up / down (pitch) |
| `1`–`5` | Preset views (isometric / top-down / sides / cinematic) |
| `0` | Cycle follow-drone mode |
| Mouse drag | Orbit (left), pan (right), scroll zoom |

**Bid visualisation:** coloured lines from each drone to its winning task. Red = low bid, yellow = mid, green = high.

---

### 5 — Re-train from scratch (optional)

Skip these if you already have the checkpoints.

**Phase 1 — HomeEnv execution actor (MAPPO, fast teleport sim):**

```bash
# ~5M steps, ~40 min on CPU  (this produced actor_update204_final.pt)
python3 -m training.train_mappo --config training/config.yaml

# Press Ctrl+C at any time — saves checkpoint cleanly
# Output: checkpoints/actor_update<N>_final.pt
```

**Phase 1b — MuJoCo execution actor (physics sim, ~40 min on CPU):**

```bash
# Train from scratch in MuJoCo physics
python3 -m training.train_mappo \
    --config training/config_mujoco.yaml \
    --sim mujoco

# Fine-tune from an existing MuJoCo checkpoint (recommended)
python3 -m training.train_mappo \
    --config training/config_mujoco.yaml \
    --sim mujoco \
    --resume checkpoints/actor_update500.pt
# Output: checkpoints/actor_update<N>_final.pt
```

**Phase 3 — bid policy:**

```bash
# Uses the harder 7-task layout in config_bid.yaml (~400 updates, ~2–4h on GPU)
python3 -m training.train_bid_policy \
    --config training/config_bid.yaml \
    --exec-checkpoint checkpoints/actor_update204_final.pt

# Press Ctrl+C — saves checkpoint cleanly
# Output: checkpoints/bid_policy_final.pt
```

---

### 6 — Programmatic usage examples

```python
# Run one episode with any allocator
from envs.home_env import HomeEnv
from allocator.greedy_auction import GreedyAuction

env = HomeEnv({"n_drones": 3, "allocator": GreedyAuction()})
obs, _ = env.reset(seed=42)
done = False
while not done:
    actions = {aid: env.action_space.sample() for aid in obs}
    obs, rewards, terminated, truncated, infos = env.step(actions)
    done = terminated["__all__"] or truncated["__all__"]

# Load the learned bidder from checkpoint
from allocator.learned_bidder import LearnedBidder
lb = LearnedBidder.from_checkpoint("checkpoints/bid_policy_final.pt")
env = HomeEnv({"n_drones": 3, "allocator": lb})

# Load with obs_delay for robustness experiment (5-step stale observations)
lb_delayed = LearnedBidder.from_checkpoint(
    "checkpoints/bid_policy_final.pt",
    obs_delay=5,
)

# Disruption API
env = HomeEnv({"n_drones": 3, "allocator": GreedyAuction()})
obs, _ = env.reset()
env.remove_task("water_plant_1")           # vanish task mid-episode → re-auction
env.add_task("water_plant", [4., 7., 1.])  # inject task mid-episode → re-auction

# BidPolicy dual-head (primary + marginal)
import numpy as np, torch
from allocator.bid_policy import BidPolicy, build_bid_obs, BID_OBS_DIM
policy = BidPolicy()                       # 14-dim → (primary_logit, marginal_logit)
obs_vec = np.zeros(BID_OBS_DIM, dtype=np.float32)
print(policy.bid_numpy(obs_vec))           # primary bid ∈ (0, 1)
print(policy.marginal_bid_numpy(obs_vec))  # marginal co-assignment bid ∈ (0, 1)
```

---

## Quick Reference — Checkpoint Map

| File | Env | Updates | Tasks/ep | Use for |
|---|---|---|---|---|
| `actor_update204_final.pt` | HomeEnv | 204 (5M steps) | ~7/7 | allocation benchmark, PyBullet deploy |
| `actor_update500.pt` | MuJoCo | 500 (4M steps) | ~1/7 | MuJoCo physics demo |
| `bid_policy_final.pt` | HomeEnv | 400 | — | `--allocator learned` in any env |

**Do not mix environments:** `actor_update204_final.pt` was trained with teleport
kinematics (HomeEnv) — it works in PyBullet/HomeEnv but is unreliable in MuJoCo.
`actor_update500.pt` was trained in MuJoCo physics and only deploys correctly there.

**Flag reference:**

| Flag | File | Purpose |
|---|---|---|
| `--exec-checkpoint` | `actor_update204_final.pt` | Execution actor (eval_allocation.py) |
| `--checkpoint` | `bid_policy_final.pt` | Bid policy (eval_allocation.py, train_bid_policy.py) |
| `--bid-checkpoint` | `bid_policy_final.pt` | Same file (lab/deploy.py) |
| `--resume` | any actor `.pt` | Warm-start train_mappo.py actor weights |

**Common mistake:** passing the bid policy to `--exec-checkpoint` (or vice versa) gives
a `KeyError: 'actor_state_dict'` — the keys in each checkpoint file are different.

---

## Install

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
python3 install.py
```

> On macOS with clang 17+ / SDK 15+ (Sequoia / Tahoe) the installer patches PyBullet's source before compiling. On Windows and Linux it uses pre-built wheels.

**Verify install:**
```bash
python3 install.py --check
```

---

## Observations, Actions, and Rewards

### Observation space per drone (15 dims)

| Index | Meaning |
|---|---|
| 0–2 | Own position (x, y, z) metres |
| 3 | Battery level normalised 0–1 |
| 4 | Task progress 0–1 |
| 5–7 | Vector to task target (dx, dy, dz) — zeros if idle |
| 8–10 | Own velocity (vx, vy, vz) |
| 11 | Tool-engaged flag (0 / 1) |
| 12–14 | Nearest neighbour relative position — zeros if no neighbour |

### BidPolicy observation per (drone, task) pair (14 dims)

| Index | Meaning |
|---|---|
| 0–2 | Drone position (x, y, z) |
| 3 | Battery level |
| 4 | Current task progress (0 if idle) |
| 5–7 | Vector to candidate task (dx, dy, dz) |
| 8 | `remaining_work()` of candidate task ∈ [0, 1] |
| 9 | Number of drones already assigned to candidate task |
| 10 | Step / max_steps (urgency) |
| 11–13 | Task type one-hot [water_plant, sweep_floor, toggle_light] |

### Action space per drone (4 dims)

| Index | Meaning | Range |
|---|---|---|
| 0–2 | Δx, Δy, Δz movement | −1 to +1 (tanh squashed) |
| 3 | Tool engage signal | 0 to 1 (sigmoid, threshold 0.5) |

### Reward structure

| Event | Value |
|---|---|
| Per-step alive penalty | −0.01 |
| Drone–drone collision | −5.0 |
| Battery depleted | −3.0 |
| Water plant completed | +10.0 |
| Sweep floor completed | +12.0 |
| Toggle light completed | +8.0 |
| Cooperative bonus (all done early) | +2.0 per drone |
| Dense proximity shaping | small positive gradient toward target |

---

## Six Disruption Scenarios (Phase 4)

| Scenario | Disruption | Hook fires at |
|---|---|---|
| `baseline` | None — standard 5-task layout | — |
| `task_vanish` | Task 1 removed (drone mid-transit) | step 50 |
| `task_inject` | New water_plant injected | step 60 |
| `drone_failure` | drone_1 battery zeroed, re-auction triggered | step 40 |
| `comm_delay` | CBBA uses `--comm-delay` broadcast delay; others unaffected | structural |
| `surge` | Two extra tasks injected | steps 30 and 80 |

---

## Allocator Interface

All allocators implement `BaseAllocator`:

```python
from allocator.base_allocator import BaseAllocator, WorldSnapshot, AllocationResult, Bid

class MyAllocator(BaseAllocator):
    def allocate(self, snapshot: WorldSnapshot) -> AllocationResult:
        # snapshot.drone_positions     — dict[str, np.ndarray(3,)]
        # snapshot.drone_batteries     — dict[str, float]  ∈ [0, 1]
        # snapshot.drone_task_progress — dict[str, float]  ∈ [0, 1]
        # snapshot.current_assignments — dict[str, int | None]
        # snapshot.tasks               — list[BaseTask]
        # snapshot.step, .max_steps
        return AllocationResult(
            assignments={"drone_0": 2, "drone_1": None, ...},
            bids=[Bid("drone_0", task_idx=2, bid_value=0.87), ...],
        )

    def on_task_complete(self, task_idx: int, step: int) -> None: ...
    def on_task_vanish(self, task_idx: int, step: int) -> None:   ...
```

Plug into either environment:

```python
from envs.home_env import HomeEnv
env = HomeEnv({"n_drones": 4, "allocator": MyAllocator()})

from envs.pybullet_env import PybulletHomeEnv
env = PybulletHomeEnv({
    "n_drones": 4,
    "gui": True,
    "allocator": MyAllocator(),
    "auction_interval": 20,   # re-auction every 20 steps
})
```

---

## Training Metrics — What to Expect

### Execution policy (MAPPO)

| Phase | Steps | Mean Reward | Entropy | Value Loss |
|---|---|---|---|---|
| Random | 0 | −5 to −15 | ~5.7 | ~0.5 |
| Early signal | ~100k | −2 to +5 | 5–8 | 0.5–2.0 |
| Learning | ~500k | +5 to +20 | 6–9 | 0.5–1.5 |
| Competent | ~1–2M | +20 to +40 | 4–7 (falling) | < 0.5 |
| Good | ~3–5M | +40 to +55 | 2–5 | < 0.3 |

**Healthy signs:** policy loss small and negative (−0.001 to −0.02); entropy falls as policy specialises; eval reward tracks training reward.

**Red flags:** entropy > 10 sustained → lower `lr_actor`; policy loss positive → lower `lr_actor` or `n_epochs`.

### Bid policy (PPO on BidPolicy)

| Update | Makespan | Tasks done | Note |
|---|---|---|---|
| 0 | ~480 | 2–3 / 5 | Random bids |
| 50 | ~350 | 3–4 / 5 | Learns proximity signal |
| 100 | ~280 | 4 / 5 | Co-assignment emerging |
| 200 | ~220 | 4–5 / 5 | Near-greedy quality |
| 400 | ~30–60 | 5 / 5 | Exceeds greedy on all scenarios |

---

## Curriculum

Training automatically advances through 4 stages as eval reward crosses thresholds in [`training/config.yaml`](training/config.yaml):

| Stage | Threshold | Description |
|---|---|---|
| 0 | 5.0 | Default — all 5 tasks, 3 drones |
| 1 | 15.0 | Policy completing 1–2 tasks reliably |
| 2 | 30.0 | 3–4 tasks per episode |
| 3 | 50.0 | All tasks + cooperative bonus |

---

## Extending

### Add a new household task

1. Create `envs/tasks/my_task.py` subclassing [`BaseTask`](envs/tasks/base_task.py)
2. Implement `step(drone_position, tool_engaged) → float`, `completion_reward() → float`, `remaining_work() → float`
3. Register in [`envs/home_env.py`](envs/home_env.py) → `_TASK_REGISTRY` and `_DEFAULT_TASK_LAYOUTS`
4. Register in [`envs/pybullet_env.py`](envs/pybullet_env.py) → `_TASK_REGISTRY`
5. Add a URDF asset to `assets/`

### Implement a new allocator

Subclass [`BaseAllocator`](allocator/base_allocator.py) and pass it via config — see [Allocator Interface](#allocator-interface) above.

### Scale up drones

Change `n_drones` in [`training/config.yaml`](training/config.yaml). The Actor is parameter-shared so it generalises to any N without retraining from scratch. The Critic input dim auto-scales.

---

## Windows Compatibility

| Component | Status | Notes |
|---|---|---|
| Training (`train_mappo.py`) | ✅ | Pure Python, no compilation |
| Bid policy training (`train_bid_policy.py`) | ✅ | Pure Python/PyTorch |
| HomeEnv + allocators | ✅ | Pure Python/NumPy |
| Phase 4 eval harness | ✅ | No PyBullet required |
| PyBullet | ✅ | Pre-built wheel, no source build |
| PyBullet GUI | ✅ | Opens normally |
| gym-pybullet-drones | ✅ | Needs Git in PATH: `winget install Git.Git` |

---

## Dependencies

```
gymnasium           >= 0.29.0
numpy               >= 1.24.0
torch               >= 2.0.0
pyyaml              >= 6.0
scipy                          OracleAllocator Hungarian algorithm (optional — falls back to greedy)
pybullet            3.2.7      physics lab only
gym-pybullet-drones 2.1.0      physics lab only
wandb                          optional W&B logging
```

---

## License

MIT
