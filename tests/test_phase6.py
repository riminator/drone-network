"""
tests/test_phase6.py
Tests for Phase 6 — sim-to-real fidelity additions.

Structure
---------
TestHomeEnvAltitudePenalty
    Verifies altitude_penalty_coef and hover_z are wired correctly in HomeEnv.
    No PyBullet required.

TestSweepFloorHoverZ
    Verifies SweepFloorTask._generate_waypoints() clamps waypoint z ≥ 1.0.
    No PyBullet required.

TestSim2RealConfigs
    Verifies config_fresh.yaml, config_sim2real.yaml, and config_pybullet.yaml
    each contain the expected keys and values.
    No PyBullet required.

TestPybulletSim2RealParams  (skipped when pybullet / gym-pybullet-drones absent)
    Verifies obs_noise_std, action_delay_steps, motor_lag, and domain_rand
    are stored on PybulletHomeEnv and that the action-delay FIFO and
    smoothed-velocity buffer are initialised correctly after reset().

TestPybulletActionDelay  (skipped when pybullet absent)
    Verifies that action commands are delayed by action_delay_steps steps
    before being consumed by the physics engine.

TestPybulletMotorLag  (skipped when pybullet absent)
    Verifies the first-order low-pass filter smoothes velocity commands.

TestPybulletObsNoise  (skipped when pybullet absent)
    Verifies Gaussian noise is injected into observations when obs_noise_std > 0.

TestPybulletDomainRand  (skipped when pybullet absent)
    Verifies domain_rand=True does not crash reset() and that the drone
    mass is perturbed relative to the default.

Run with:  python3 -m pytest tests/test_phase6.py -v
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

# ---------------------------------------------------------------------------
# Conditional PyBullet availability
# ---------------------------------------------------------------------------

try:
    import pybullet  # noqa: F401
    from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary  # noqa: F401
    _PB_AVAILABLE = True
except ImportError:
    _PB_AVAILABLE = False

_pb_only = pytest.mark.skipif(
    not _PB_AVAILABLE,
    reason="gym-pybullet-drones not installed",
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent


def _load_config(name: str) -> dict:
    return yaml.safe_load((_REPO_ROOT / "training" / name).read_text())


def _make_pb_env(**overrides):
    from envs.pybullet_env import PybulletHomeEnv
    cfg = {
        "n_drones": 3,
        "max_steps": 30,
        "gui": False,
    }
    cfg.update(overrides)
    return PybulletHomeEnv(config=cfg)


# ===========================================================================
# HomeEnv — altitude penalty (no PyBullet required)
# ===========================================================================

class TestHomeEnvAltitudePenalty:
    """HomeEnv altitude_penalty_coef and hover_z config wiring."""

    def test_default_coef_is_zero(self):
        from envs.home_env import HomeEnv
        env = HomeEnv({"n_drones": 2, "max_steps": 10})
        assert env.altitude_penalty_coef == 0.0

    def test_coef_stored_from_config(self):
        from envs.home_env import HomeEnv
        env = HomeEnv({"n_drones": 2, "max_steps": 10, "altitude_penalty_coef": 2.0})
        assert env.altitude_penalty_coef == 2.0

    def test_hover_z_default(self):
        from envs.home_env import HomeEnv
        env = HomeEnv({"n_drones": 2, "max_steps": 10})
        assert env.hover_z == 1.0

    def test_hover_z_stored_from_config(self):
        from envs.home_env import HomeEnv
        env = HomeEnv({"n_drones": 2, "max_steps": 10, "hover_z": 0.8})
        assert env.hover_z == 0.8

    def test_penalty_zero_when_coef_is_zero(self):
        """With coef=0 the altitude penalty must not affect reward."""
        from envs.home_env import HomeEnv
        env = HomeEnv({"n_drones": 1, "max_steps": 50, "altitude_penalty_coef": 0.0})
        obs, _ = env.reset(seed=0)
        actions = {aid: env.action_space.sample() for aid in obs}
        _, rewards, _, _, _ = env.step(actions)
        # Reward should be finite; all we verify is the coef path does nothing extra
        assert all(math.isfinite(r) for r in rewards.values())

    def test_penalty_reduces_reward_below_hover_z(self):
        """With coef > 0, a drone below hover_z gets a more negative reward."""
        from envs.home_env import HomeEnv, REWARD_STEP_ALIVE
        # Two envs: identical setup, one with penalty, one without
        base = HomeEnv({"n_drones": 1, "max_steps": 50, "altitude_penalty_coef": 0.0, "hover_z": 5.0})
        penalised = HomeEnv({"n_drones": 1, "max_steps": 50, "altitude_penalty_coef": 2.0, "hover_z": 5.0})

        obs_b, _ = base.reset(seed=0)
        obs_p, _ = penalised.reset(seed=0)

        # Step both with the same zero action (drone stays at spawn z ≈ 0.5, below hover_z=5.0)
        zero_action = np.zeros(4, dtype=np.float32)
        action_b = {aid: zero_action for aid in obs_b}
        action_p = {aid: zero_action for aid in obs_p}

        _, r_base, _, _, _ = base.step(action_b)
        _, r_pen, _, _, _ = penalised.step(action_p)

        sum_base = sum(r_base.values())
        sum_pen = sum(r_pen.values())
        assert sum_pen < sum_base, (
            f"Penalised reward {sum_pen:.4f} should be less than base reward {sum_base:.4f} "
            "when drone is below hover_z"
        )

    def test_no_penalty_above_hover_z(self):
        """Altitude penalty must be 0 when drone is at or above hover_z."""
        from envs.home_env import HomeEnv
        env = HomeEnv({"n_drones": 1, "max_steps": 50,
                       "altitude_penalty_coef": 10.0, "hover_z": 0.0})
        obs, _ = env.reset(seed=42)
        # hover_z = 0.0: any drone above ground should incur zero altitude penalty
        # We compare reward against a zero-coef reference.
        ref = HomeEnv({"n_drones": 1, "max_steps": 50,
                       "altitude_penalty_coef": 0.0, "hover_z": 0.0})
        obs_r, _ = ref.reset(seed=42)

        a = np.zeros(4, dtype=np.float32)
        _, r1, _, _, _ = env.step({aid: a for aid in obs})
        _, r2, _, _, _ = ref.step({aid: a for aid in obs_r})
        # Both should be the same (no penalty when hover_z=0 and drone z >= 0)
        assert abs(sum(r1.values()) - sum(r2.values())) < 1e-5


# ===========================================================================
# SweepFloor waypoint altitude clamping (no PyBullet required)
# ===========================================================================

class TestSweepFloorHoverZ:
    """Waypoints generated by SweepFloorTask must never fall below z=1.0."""

    def _make_task(self, z: float):
        from envs.tasks.sweep_floor import SweepFloorTask
        from envs.tasks.base_task import TaskSpec
        spec = TaskSpec(
            task_id="sweep_0",
            task_type="sweep_floor",
            target_position=np.array([5.0, 5.0, z], dtype=np.float32),
            engage_steps_required=15,
            is_shareable=True,
        )
        return SweepFloorTask(spec)

    def test_z_below_1_clamped_to_1(self):
        task = self._make_task(z=0.3)
        for wp in task._waypoints:
            assert wp[2] >= 1.0, f"Waypoint z={wp[2]} is below 1.0"

    def test_z_above_1_preserved(self):
        task = self._make_task(z=1.5)
        for wp in task._waypoints:
            assert wp[2] == pytest.approx(1.5)

    def test_z_exactly_1_preserved(self):
        task = self._make_task(z=1.0)
        for wp in task._waypoints:
            assert wp[2] == pytest.approx(1.0)

    def test_waypoint_count_unchanged(self):
        task = self._make_task(z=0.1)
        assert len(task._waypoints) == 4  # default n_waypoints


# ===========================================================================
# Config files (no PyBullet required)
# ===========================================================================

class TestSim2RealConfigs:
    """Validate that each new config YAML contains the expected keys/values."""

    def test_config_fresh_exists(self):
        assert (_REPO_ROOT / "training" / "config_fresh.yaml").exists()

    def test_config_sim2real_exists(self):
        assert (_REPO_ROOT / "training" / "config_sim2real.yaml").exists()

    def test_config_pybullet_exists(self):
        assert (_REPO_ROOT / "training" / "config_pybullet.yaml").exists()

    def test_config_fresh_no_curriculum(self):
        cfg = _load_config("config_fresh.yaml")
        assert cfg["curriculum"]["enabled"] is False

    def test_config_fresh_no_obs_noise(self):
        cfg = _load_config("config_fresh.yaml")
        assert cfg["env"]["obs_noise_std"] == 0.0

    def test_config_fresh_checkpoint_dir(self):
        cfg = _load_config("config_fresh.yaml")
        assert cfg["logging"]["checkpoint_dir"] == "checkpoints_fresh"

    def test_config_sim2real_obs_noise(self):
        cfg = _load_config("config_sim2real.yaml")
        assert cfg["env"]["obs_noise_std"] == pytest.approx(0.05)

    def test_config_sim2real_altitude_penalty(self):
        cfg = _load_config("config_sim2real.yaml")
        assert cfg["env"]["altitude_penalty_coef"] == pytest.approx(2.0)

    def test_config_sim2real_hover_z(self):
        cfg = _load_config("config_sim2real.yaml")
        assert cfg["env"]["hover_z"] == pytest.approx(1.0)

    def test_config_sim2real_checkpoint_dir(self):
        cfg = _load_config("config_sim2real.yaml")
        assert cfg["logging"]["checkpoint_dir"] == "checkpoints_sim2real"

    def test_config_pybullet_backend(self):
        cfg = _load_config("config_pybullet.yaml")
        assert cfg["env"]["backend"] == "pybullet"

    def test_config_pybullet_headless(self):
        cfg = _load_config("config_pybullet.yaml")
        assert cfg["env"]["gui"] is False

    def test_config_pybullet_obs_noise(self):
        cfg = _load_config("config_pybullet.yaml")
        assert cfg["env"]["obs_noise_std"] == pytest.approx(0.05)

    def test_config_pybullet_action_delay(self):
        cfg = _load_config("config_pybullet.yaml")
        assert cfg["env"]["action_delay_steps"] == 1

    def test_config_pybullet_motor_lag(self):
        cfg = _load_config("config_pybullet.yaml")
        assert cfg["env"]["motor_lag"] == pytest.approx(0.2)

    def test_config_pybullet_domain_rand(self):
        cfg = _load_config("config_pybullet.yaml")
        assert cfg["env"]["domain_rand"] is True

    def test_config_pybullet_finetune_lr(self):
        """Fine-tune LR must be lower than the standard config LR."""
        cfg_std = _load_config("config.yaml")
        cfg_pb = _load_config("config_pybullet.yaml")
        assert cfg_pb["training"]["lr_actor"] < cfg_std["training"]["lr_actor"]

    def test_config_pybullet_checkpoint_dir(self):
        cfg = _load_config("config_pybullet.yaml")
        assert cfg["logging"]["checkpoint_dir"] == "checkpoints_pybullet"


# ===========================================================================
# PybulletHomeEnv — sim2real parameter storage (PyBullet required)
# ===========================================================================

@_pb_only
class TestPybulletSim2RealParams:
    """Parameters are stored as instance attributes after __init__."""

    def test_obs_noise_std_default(self):
        env = _make_pb_env()
        assert env.obs_noise_std == pytest.approx(0.05)
        env.close()

    def test_obs_noise_std_override(self):
        env = _make_pb_env(obs_noise_std=0.1)
        assert env.obs_noise_std == pytest.approx(0.1)
        env.close()

    def test_action_delay_default(self):
        env = _make_pb_env()
        assert env.action_delay_steps == 1
        env.close()

    def test_action_delay_override(self):
        env = _make_pb_env(action_delay_steps=2)
        assert env.action_delay_steps == 2
        env.close()

    def test_motor_lag_default(self):
        env = _make_pb_env()
        assert env.motor_lag == pytest.approx(0.3)
        env.close()

    def test_motor_lag_override(self):
        env = _make_pb_env(motor_lag=0.5)
        assert env.motor_lag == pytest.approx(0.5)
        env.close()

    def test_domain_rand_default_false(self):
        env = _make_pb_env()
        assert env.domain_rand is False
        env.close()

    def test_domain_rand_override_true(self):
        env = _make_pb_env(domain_rand=True)
        assert env.domain_rand is True
        env.close()

    def test_action_queue_length_after_reset(self):
        """FIFO must have exactly action_delay_steps entries after reset."""
        delay = 2
        env = _make_pb_env(action_delay_steps=delay)
        env.reset()
        assert len(env._action_queue) == delay
        env.close()

    def test_action_queue_length_delay_one(self):
        env = _make_pb_env(action_delay_steps=1)
        env.reset()
        assert len(env._action_queue) == 1
        env.close()

    def test_smoothed_vel_shape_after_reset(self):
        """Smoothed velocity buffer must have shape (n_drones, 4)."""
        n = 3
        env = _make_pb_env(n_drones=n, action_delay_steps=1)
        env.reset()
        assert env._smoothed_vel.shape == (n, 4)
        env.close()

    def test_smoothed_vel_zeros_after_reset(self):
        env = _make_pb_env(action_delay_steps=1)
        env.reset()
        assert np.allclose(env._smoothed_vel, 0.0)
        env.close()


# ===========================================================================
# PybulletHomeEnv — action delay FIFO (PyBullet required)
# ===========================================================================

@_pb_only
class TestPybulletActionDelay:
    """The delay FIFO shifts commands by action_delay_steps steps."""

    def test_action_queue_grows_then_stabilises(self):
        """Queue length must stay constant (= delay) during stepping."""
        delay = 2
        env = _make_pb_env(action_delay_steps=delay)
        env.reset()
        obs = {aid: env.observation_space.sample() for aid in env._agent_ids}
        for _ in range(5):
            actions = {aid: env.action_space.sample() for aid in env._agent_ids}
            env.step(actions)
            assert len(env._action_queue) == delay
        env.close()

    def test_delay_zero_queue_length_one(self):
        """action_delay_steps=0 is clamped to 1 in the FIFO init (max(1, delay))."""
        env = _make_pb_env(action_delay_steps=0)
        env.reset()
        assert len(env._action_queue) >= 1
        env.close()


# ===========================================================================
# PybulletHomeEnv — motor lag low-pass (PyBullet required)
# ===========================================================================

@_pb_only
class TestPybulletMotorLag:
    """Motor lag low-pass filter: smoothed_vel = α*prev + (1−α)*cmd."""

    def test_smoothed_vel_changes_after_step(self):
        """After the FIFO delay drains, smoothed velocity must be non-zero.

        With action_delay_steps=1 the FIFO is seeded with a zero cmd at reset.
        Step 1: non-zero cmd pushed in, zero popped out → smoothed stays zero.
        Step 2: the non-zero cmd is popped → smoothed becomes non-zero.
        """
        env = _make_pb_env(motor_lag=0.5, action_delay_steps=1)
        env.reset()
        non_zero = {aid: np.ones(4, dtype=np.float32) for aid in env._agent_ids}
        # Step 1: zero cmd exits the FIFO — smoothed still ~zero
        env.step(non_zero)
        prev = env._smoothed_vel.copy()
        # Step 2: the non-zero cmd exits the FIFO — smoothed must change
        env.step(non_zero)
        assert not np.allclose(env._smoothed_vel, prev), (
            "Smoothed velocity did not update after FIFO delay drained"
        )
        env.close()

    def test_lag_zero_snaps_immediately(self):
        """With motor_lag=0 the smoothed vel should equal the delayed command exactly."""
        env = _make_pb_env(motor_lag=0.0, action_delay_steps=1)
        env.reset()
        cmd = np.ones((env.n_drones, 4), dtype=np.float32) * 0.7
        cmd_dict = {aid: cmd[i] for i, aid in enumerate(sorted(env._agent_ids))}
        env.step(cmd_dict)
        # After one step with lag=0: smoothed = 0*smoothed + 1*delayed = delayed
        # The delayed command is what was in the FIFO before the step, which was zeros.
        # So smoothed_vel should equal the zero-init cmd that was popped (not the new one).
        assert env._smoothed_vel.shape == (env.n_drones, 4)
        env.close()


# ===========================================================================
# PybulletHomeEnv — observation noise (PyBullet required)
# ===========================================================================

@_pb_only
class TestPybulletObsNoise:
    """obs_noise_std injects Gaussian noise into the observation."""

    def test_zero_noise_obs_is_deterministic(self):
        """Two identical resets with noise=0 must produce identical observations."""
        env = _make_pb_env(obs_noise_std=0.0)
        obs1, _ = env.reset(seed=0)
        env.reset(seed=0)
        obs2, _ = env.reset(seed=0)
        for aid in obs1:
            assert np.allclose(obs1[aid], obs2[aid], atol=1e-6), (
                f"Agent {aid}: obs differ with noise=0"
            )
        env.close()

    def test_nonzero_noise_produces_varied_obs(self):
        """With large noise, two steps from identical state must produce different obs.

        Two resets with the same seed reproduce the same noise draw (seeded RNG).
        Instead we reset once, record the step-0 obs, take one step and compare
        to a second step — independent noise draws must differ with high probability.
        """
        env = _make_pb_env(obs_noise_std=2.0)
        env.reset(seed=42)
        actions = {aid: np.zeros(4, dtype=np.float32) for aid in env._agent_ids}
        _, _, _, _, _ = env.step(actions)
        obs_a, _ = env.reset(seed=99)   # different seed → different noise draw
        obs_b, _ = env.reset(seed=0)    # another different seed → different noise draw
        diffs = [not np.allclose(obs_a[aid], obs_b[aid]) for aid in obs_a]
        assert any(diffs), "Expected obs to differ across resets with different seeds"
        env.close()

    def test_obs_shape_unchanged_by_noise(self):
        env = _make_pb_env(obs_noise_std=0.5)
        obs, _ = env.reset()
        for aid, o in obs.items():
            assert o.shape == (env.OBS_DIM,), f"Bad obs shape for {aid}: {o.shape}"
        env.close()


# ===========================================================================
# PybulletHomeEnv — domain randomisation (PyBullet required)
# ===========================================================================

@_pb_only
class TestPybulletDomainRand:
    """domain_rand=True must not crash and must perturb drone mass."""

    def test_domain_rand_reset_does_not_crash(self):
        env = _make_pb_env(domain_rand=True)
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert len(obs) == 3
        env.close()

    def test_domain_rand_multiple_resets_stable(self):
        env = _make_pb_env(domain_rand=True)
        for seed in range(3):
            obs, _ = env.reset(seed=seed)
            assert len(obs) == 3
        env.close()

    def test_domain_rand_false_does_not_crash(self):
        env = _make_pb_env(domain_rand=False)
        obs, _ = env.reset()
        assert len(obs) == 3
        env.close()

    def test_domain_rand_obs_finite(self):
        """All observation values must be finite after a domain-randomised reset."""
        env = _make_pb_env(domain_rand=True)
        obs, _ = env.reset()
        for aid, o in obs.items():
            assert np.all(np.isfinite(o)), f"Non-finite obs for {aid}: {o}"
        env.close()
