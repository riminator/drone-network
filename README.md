# 🚁 Drone Network — Market-Based Dynamic Task Allocation for RL Swarms

A research platform investigating **auction-based dynamic task allocation** for RL-controlled drone swarms performing household service tasks in continuously changing environments. The swarm must reassign work in real time as the task set itself shifts — tasks vanish mid-mission, new tasks appear, drones fail — without retraining the execution policies.

Trains fast on a pure-Python simulation, then deploys into a **PyBullet physics lab** with real Crazyflie quadrotor aerodynamics.

---

## Research Problem

Classical multi-robot task allocation (MRTA) assumes a fixed task set. This project addresses the harder problem where the task set is **non-stationary**: tasks appear and disappear at runtime, some tasks are divisible (multiple drones can collaborate on a single large task), and communication between drones is delayed or unreliable.

The core contribution is replacing hand-crafted bidding heuristics with a **learned bidding policy** (PPO-trained against optimal-assignment baselines) that can represent both:

- **Discrete reassignment** — a task vanishes; the freed drone must instantly re-enter the auction for remaining tasks.
- **Continuous task-sharing** — an idle drone can bid to *join* an in-progress task (marginal-value co-assignment), rather than only bidding on unclaimed tasks.

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
│                                                                      │
│   BidPolicy obs (14 dims): drone pos, battery, progress,             │
│     vector-to-task, remaining_work, n_assigned, urgency, type-onehot │
└──────────────────────────────────────────────────────────────────────┘
                              ↕  AllocationResult (assignments + bids)
┌──────────────────────────────────────────────────────────────────────┐
│  ENVIRONMENT LAYER                                                   │
│                                                                      │
│   HomeEnv          fast Python/NumPy sim (training + eval)           │
│   PybulletHomeEnv  physics-backed sim (deployment + vis)             │
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
│   ├── pybullet_env.py       Physics deployment env — real quadrotor dynamics
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
│   ├── bid_policy.py         BidPolicy MLP: 14-dim obs → sigmoid bid ∈ (0,1)
│   │                           build_bid_obs() helper constructs per-pair obs
│   ├── learned_bidder.py     LearnedBidder wraps BidPolicy as a BaseAllocator
│   │                           from_checkpoint() factory
│   └── bid_env.py            BidEnv: HomeEnv wrapper that collects bid transitions
│                               BidBuffer / BidTransition for bid-policy PPO
│
├── models/
│   ├── actor.py              Gaussian policy MLP (tanh/sigmoid squashing, log-prob correction)
│   └── critic.py             Centralised value MLP (n_drones × 15 → scalar)
│
├── training/
│   ├── train_mappo.py        Phase 1: MAPPO loop — shared Actor + CentralCritic
│   ├── train_bid_policy.py   Phase 3: BidPolicy PPO — frozen exec actor + BidValueNet
│   └── config.yaml           All hyperparameters (MAPPO + bid-policy sections)
│
├── evaluation/
│   ├── eval.py               Load checkpoint → benchmark episodes (execution eval)
│   └── eval_allocation.py    Phase 4: disruption-scenario harness
│                               4 allocators × 6 scenarios × N episodes
│                               EpisodeMetrics table + optional CSV export
│
├── lab/
│   └── deploy.py             Load checkpoint → PyBullet GUI
│                               --allocator {greedy,cbba,oracle,learned}
│                               --auction-interval K  (periodic re-auction)
│                               --bid-checkpoint (LearnedBidder weights)
│
├── tests/
│   ├── test_phase1.py        TaskStatus FSM, disruption API, co-assignment (25 tests)
│   ├── test_phase2.py        GreedyAuction + CBBA contract + disruption (28 tests)
│   ├── test_phase3.py        BidPolicy, BidBuffer, OracleAllocator, LearnedBidder (29 tests)
│   ├── test_phase4.py        EpisodeMetrics, scenario hooks, benchmark driver (45 tests)
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
└── checkpoints/
    ├── actor_update204_final.pt      Best execution actor
    ├── bid_policy_final.pt           Best learned bidder
    └── bid_policy_update{50,100,150,200}.pt
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
python3 install.py
```

> The installer auto-detects your platform. On macOS with clang 17+ / SDK 15+ (including Sequoia / Tahoe) it patches PyBullet's source before compiling. On Windows and Linux it uses pre-built wheels — no compilation needed.

**Verify:**
```bash
python3 install.py --check
```

**Run the test suite:**
```bash
python3 -m pytest tests/ -v
# Expected: 163 passed
```

---

### 2. Train the execution policy (Phase 1)

```bash
python -m training.train_mappo
# or with custom config:
python -m training.train_mappo --config training/config.yaml
# or with W&B logging:
python -m training.train_mappo --config training/config.yaml --wandb
```

Press **Ctrl+C** at any time — training stops cleanly after the current PPO update and saves a checkpoint.

---

### 3. Train the learned bidder (Phase 3)

Requires a frozen execution actor checkpoint:

```bash
python -m training.train_bid_policy \
    --exec-checkpoint checkpoints/actor_update204_final.pt
```

The bid policy is trained via PPO against a shaped reward that penalises slow completion (makespan), idle drones (reallocation latency), and uses task completions as positive signal. The oracle allocator provides a theoretical upper bound for comparison.

---

### 4. Evaluate execution quality

```bash
python -m evaluation.eval \
    --checkpoint checkpoints/actor_update204_final.pt \
    --episodes 20
```

---

### 5. Run the disruption-scenario benchmark (Phase 4)

Benchmarks all four allocators across all six disruption scenarios:

```bash
# Quick smoke run (3 episodes per cell)
python -m evaluation.eval_allocation --episodes 3

# Full benchmark with trained bid-policy checkpoint
python -m evaluation.eval_allocation \
    --checkpoint checkpoints/bid_policy_final.pt \
    --episodes 30 \
    --csv results/phase4.csv

# Single scenario, two allocators
python -m evaluation.eval_allocation \
    --scenarios surge comm_delay \
    --allocators greedy cbba \
    --episodes 10
```

**Output** (mean ± std per cell):

```
================================================================================
  Scenario: task_vanish
================================================================================
  Metric          greedy          cbba        oracle       learned
  -----------------------------------------------------------------------
  CompRate    0.80± 0.10    0.82± 0.09    0.95± 0.04    0.78± 0.12
  Makespan  310.20±42.10  295.60±38.70  201.30±28.10  320.40±45.20
  TotReward  -8.40± 3.20   -7.10± 2.80    2.40± 1.90   -9.80± 3.60
  ...
```

---

### 6. Deploy in the PyBullet physics lab

```bash
# Default (greedy allocator), real-time
python -m lab.deploy \
    --checkpoint checkpoints/actor_update204_final.pt

# CBBA, slow-motion, 6 drones
python -m lab.deploy \
    --checkpoint checkpoints/actor_update204_final.pt \
    --allocator cbba \
    --n-drones 6 \
    --time-scale 0.3

# Learned bidder with trained weights
python -m lab.deploy \
    --checkpoint checkpoints/actor_update204_final.pt \
    --allocator learned \
    --bid-checkpoint checkpoints/bid_policy_final.pt \
    --time-scale 0.3

# Periodic re-auction every 20 steps (visible as line colour updates)
python -m lab.deploy \
    --checkpoint checkpoints/actor_update204_final.pt \
    --allocator greedy \
    --auction-interval 20

# Headless benchmark, fast
python -m lab.deploy \
    --checkpoint checkpoints/actor_update204_final.pt \
    --no-gui --episodes 20
```

**Camera controls** (click the PyBullet window first):

| Key | Action |
|---|---|
| `W` / `S` | Pan camera forward / back |
| `A` / `D` | Pan camera left / right |
| `Q` / `E` | Zoom in / out |
| `Z` / `X` | Rotate left / right (yaw) |
| `R` / `F` | Tilt up / down (pitch) |
| `1`–`5` | Preset views (isometric / top-down / sides / cinematic) |
| `0` | Cycle follow-drone (locks on each drone in turn, then free) |
| Mouse drag | Orbit (left), pan (right), scroll zoom |

**Bid visualisation** (GUI mode): a thin coloured line is drawn from each drone to its currently winning task after every allocation round. Colour encodes normalised bid value — red (low) → yellow (mid) → green (high) — so you can see which drones are competing hardest for which targets.

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

## Allocator Interface

All allocators implement `BaseAllocator`:

```python
from allocator.base_allocator import BaseAllocator, WorldSnapshot, AllocationResult

class MyAllocator(BaseAllocator):
    def allocate(self, snapshot: WorldSnapshot) -> AllocationResult:
        # snapshot.drone_positions  — dict[str, np.ndarray(3,)]
        # snapshot.drone_batteries  — dict[str, float]  ∈ [0, 1]
        # snapshot.drone_task_progress — dict[str, float] ∈ [0, 1]
        # snapshot.current_assignments — dict[str, int | None]
        # snapshot.tasks            — list[BaseTask]
        # snapshot.step, .max_steps
        ...
        return AllocationResult(
            assignments={"drone_0": 2, "drone_1": None, ...},
            bids=[Bid("drone_0", task_idx=2, bid_value=0.87), ...],
        )

    def on_task_complete(self, task_idx: int, step: int) -> None:
        ...  # optional: update stateful tables (used by CBBA)

    def on_task_vanish(self, task_idx: int, step: int) -> None:
        ...  # optional
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
    "auction_interval": 20,  # re-auction every 20 steps
})
```

---

## Disruption API

```python
env = HomeEnv({"n_drones": 3, "allocator": GreedyAuction()})
obs, _ = env.reset()

# Remove a task mid-episode (e.g., plant already watered by human)
removed = env.remove_task("water_plant_1")   # triggers immediate re-auction

# Inject a new task mid-episode (e.g., spill appears)
new_id = env.add_task(
    "water_plant",
    target=[4.0, 7.0, 1.0],
    engage_steps=10,
    is_shareable=False,
)                                             # triggers immediate re-auction
```

---

## Six Disruption Scenarios (Phase 4)

| Scenario | Description |
|---|---|
| `baseline` | No disruption — standard 5-task layout |
| `task_vanish` | Task 1 removed at step 50 (drone mid-transit) |
| `task_inject` | New water_plant injected at step 60 |
| `drone_failure` | drone_1 battery zeroed at step 40, re-auction triggered |
| `comm_delay` | CBBA runs with 5-step broadcast delay; others unaffected |
| `surge` | Two extra tasks injected at steps 30 and 80 |

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

**Healthy signs:**
- Policy loss: small and negative (−0.001 to −0.02)
- Entropy starts ~5.7, rises during exploration, then falls as policy specialises
- Eval reward should track training reward closely

**Red flags:**
- Entropy > 10 sustained → `log_std` saturating; lower `lr_actor`
- Policy loss turning positive → clip triggering too much; lower `lr_actor` or `n_epochs`
- Eval reward much worse than training → policy relying on noise; std too high

### Bid policy (PPO on BidPolicy)

| Update | Makespan | Tasks done | Note |
|---|---|---|---|
| 0 | ~480 | 2–3 / 5 | Random bids |
| 50 | ~350 | 3–4 / 5 | Learns proximity signal |
| 100 | ~280 | 4 / 5 | Co-assignment emerging |
| 200 | ~220 | 4–5 / 5 | Near-greedy quality |

The learned bidder starts matching GreedyAuction around update 100 and can exceed it after 200 updates on disruption scenarios where greedy's myopic distance heuristic fails.

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

### Add a new household task

1. Create `envs/tasks/my_task.py` subclassing [`BaseTask`](envs/tasks/base_task.py)
2. Implement `step(drone_position, tool_engaged) → float`, `completion_reward() → float`, `remaining_work() → float`
3. Register in [`envs/home_env.py`](envs/home_env.py) → `_TASK_REGISTRY` and `_DEFAULT_TASK_LAYOUTS`
4. Register in [`envs/pybullet_env.py`](envs/pybullet_env.py) → `_TASK_REGISTRY` and `urdf_map`
5. Add a URDF asset to `assets/`

### Implement a new allocator

Subclass [`BaseAllocator`](allocator/base_allocator.py) and pass it via config — see the [Allocator Interface](#allocator-interface) section above.

### Scale up drones

Change `n_drones` in [`training/config.yaml`](training/config.yaml). The Actor is parameter-shared so it works for any N without retraining from scratch. The Critic input dim auto-scales.

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
gymnasium   >= 0.29.0
numpy       >= 1.24.0
torch       >= 2.0.0
pyyaml      >= 6.0
scipy                    (OracleAllocator Hungarian algorithm — optional, falls back to greedy)
pybullet    3.2.7        (physics lab only)
gym-pybullet-drones 2.1.0  (physics lab only)
wandb                    (optional W&B logging)
```

---

## License

MIT
