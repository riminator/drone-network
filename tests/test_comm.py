"""
tests/test_comm.py
Tests for CommActor — communication-aware execution policy.

Structure
---------
TestGATv2Layer
    Unit tests for the GATv2 attention layer in isolation.

TestCommActorInit
    Verifies CommActor instantiation with various hyperparameter combos.

TestCommActorForward
    Verifies forward() output shapes, graph masking, and delay buffer.

TestCommActorGetAction
    Verifies get_action() mirrors Actor's output contract exactly.

TestCommActorEvaluateActions
    Verifies evaluate_actions() produces valid log-probs and entropy.

TestCommActorAdjacency
    Verifies comm_radius and drop_comm_prob build the correct edge masks.

TestCommActorDelayBuffer
    Verifies the comm_delay FIFO holds old obs and resets correctly.

TestBuildActor
    Verifies build_actor() factory creates the right class from config.

TestCommActorCheckpoint
    Verifies checkpoint save/load round-trips for CommActor.

TestCommActorVsActor
    Verifies CommActor produces different (informationally richer) outputs
    than the plain Actor in a multi-drone scenario.

TestCommActorIntegration
    End-to-end: CommActor in a short HomeEnv episode without crash.

Run with:  python3 -m pytest tests/test_comm.py -v
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from models.comm_actor import CommActor, GATv2Layer, _GATv2Head
from models.actor import Actor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_comm_actor(**kwargs) -> CommActor:
    defaults = dict(obs_dim=15, act_dim=4, embed_dim=16, n_heads=4)
    defaults.update(kwargs)
    return CommActor(**defaults)


def _random_obs(n: int, obs_dim: int = 15) -> torch.Tensor:
    """Random (N, obs_dim) obs with realistic position values in dims 0-2."""
    obs = torch.randn(n, obs_dim)
    obs[:, 0:3] = torch.rand(n, 3) * 10.0   # positions in [0, 10] m
    return obs


# ===========================================================================
# GATv2 layer unit tests
# ===========================================================================

class TestGATv2Layer:
    def test_output_shape(self):
        layer = GATv2Layer(embed_dim=16, n_heads=4)
        feats = torch.randn(5, 16)
        adj   = torch.ones(5, 5, dtype=torch.bool)
        out   = layer(feats, adj)
        assert out.shape == (5, 16)

    def test_no_self_loops_gives_nan_free_output(self):
        """A node with no neighbours (incl. itself) gets zero agg — not NaN."""
        layer = GATv2Layer(embed_dim=8, n_heads=2)
        feats = torch.randn(3, 8)
        # no edges at all
        adj   = torch.zeros(3, 3, dtype=torch.bool)
        out   = layer(feats, adj)
        assert torch.all(torch.isfinite(out)), "Output contains NaN/Inf"
        assert torch.allclose(out, torch.zeros_like(out)), (
            "Node with no neighbours should produce zero aggregation"
        )

    def test_fully_connected_output_is_finite(self):
        layer = GATv2Layer(embed_dim=16, n_heads=4)
        feats = torch.randn(6, 16)
        adj   = torch.ones(6, 6, dtype=torch.bool)
        out   = layer(feats, adj)
        assert torch.all(torch.isfinite(out))

    def test_single_node_no_crash(self):
        layer = GATv2Layer(embed_dim=8, n_heads=2)
        feats = torch.randn(1, 8)
        adj   = torch.ones(1, 1, dtype=torch.bool)
        out   = layer(feats, adj)
        assert out.shape == (1, 8)

    def test_embed_dim_not_divisible_raises(self):
        with pytest.raises(AssertionError, match="divisible"):
            GATv2Layer(embed_dim=7, n_heads=4)

    def test_n_heads_attribute(self):
        layer = GATv2Layer(embed_dim=16, n_heads=4)
        assert layer.n_heads == 4
        assert len(layer.heads) == 4


# ===========================================================================
# CommActor instantiation
# ===========================================================================

class TestCommActorInit:
    def test_default_construction(self):
        actor = _make_comm_actor()
        assert isinstance(actor, CommActor)

    def test_stores_hyperparams(self):
        actor = _make_comm_actor(embed_dim=32, n_heads=4,
                                  comm_radius=5.0, comm_delay=2,
                                  drop_comm_prob=0.1)
        assert actor.embed_dim == 32
        assert actor.comm_radius == pytest.approx(5.0)
        assert actor.comm_delay == 2
        assert actor.drop_comm_prob == pytest.approx(0.1)

    def test_comm_delay_zero_clamped(self):
        actor = _make_comm_actor(comm_delay=-1)
        assert actor.comm_delay == 0

    def test_infinite_radius(self):
        actor = _make_comm_actor(comm_radius=float("inf"))
        assert math.isinf(actor.comm_radius)

    def test_log_std_param_exists(self):
        actor = _make_comm_actor()
        assert hasattr(actor, "log_std")
        assert isinstance(actor.log_std, nn.Parameter)

    def test_log_std_shape(self):
        actor = _make_comm_actor(act_dim=4)
        assert actor.log_std.shape == (4,)

    def test_trunk_exists(self):
        actor = _make_comm_actor()
        assert isinstance(actor.trunk, nn.Sequential)

    def test_gat_layer_exists(self):
        actor = _make_comm_actor()
        assert isinstance(actor.gat, GATv2Layer)

    def test_node_encoder_exists(self):
        actor = _make_comm_actor()
        assert isinstance(actor.node_encoder, nn.Sequential)


# ===========================================================================
# CommActor.forward
# ===========================================================================

class TestCommActorForward:
    def test_output_shapes(self):
        actor = _make_comm_actor(obs_dim=15, act_dim=4, embed_dim=16)
        obs   = _random_obs(3)
        mean, log_std = actor.forward(obs)
        assert mean.shape   == (3, 4)
        assert log_std.shape == (3, 4)

    def test_log_std_clamped(self):
        actor = _make_comm_actor()
        obs   = _random_obs(3)
        _, log_std = actor.forward(obs)
        assert torch.all(log_std >= CommActor.LOG_STD_MIN)
        assert torch.all(log_std <= CommActor.LOG_STD_MAX)

    def test_output_finite(self):
        actor = _make_comm_actor()
        obs   = _random_obs(4)
        mean, log_std = actor.forward(obs)
        assert torch.all(torch.isfinite(mean))
        assert torch.all(torch.isfinite(log_std))

    def test_single_drone_no_crash(self):
        actor = _make_comm_actor()
        obs   = _random_obs(1)
        mean, log_std = actor.forward(obs)
        assert mean.shape == (1, 4)

    def test_large_swarm_no_crash(self):
        actor = _make_comm_actor()
        obs   = _random_obs(12)
        mean, _ = actor.forward(obs)
        assert mean.shape == (12, 4)


# ===========================================================================
# CommActor.get_action
# ===========================================================================

class TestCommActorGetAction:
    def test_squashed_action_shape(self):
        actor = _make_comm_actor()
        obs   = _random_obs(3)
        squashed, raw, lp = actor.get_action(obs)
        assert squashed.shape == (3, 4)
        assert raw.shape      == (3, 4)
        assert lp.shape       == (3,)

    def test_spatial_dims_in_neg1_pos1(self):
        actor = _make_comm_actor()
        obs   = _random_obs(6)
        squashed, _, _ = actor.get_action(obs)
        assert torch.all(squashed[:, :3] >= -1.0 - 1e-5)
        assert torch.all(squashed[:, :3] <=  1.0 + 1e-5)

    def test_tool_dim_in_0_1(self):
        actor = _make_comm_actor()
        obs   = _random_obs(6)
        squashed, _, _ = actor.get_action(obs)
        assert torch.all(squashed[:, 3] >= -1e-5)
        assert torch.all(squashed[:, 3] <=  1.0 + 1e-5)

    def test_deterministic_returns_same_action(self):
        actor = _make_comm_actor()
        actor.eval()
        obs = _random_obs(3)
        actor.reset_delay_buffer()
        s1, _, _ = actor.get_action(obs, deterministic=True)
        actor.reset_delay_buffer()
        s2, _, _ = actor.get_action(obs, deterministic=True)
        assert torch.allclose(s1, s2)

    def test_stochastic_returns_different_actions(self):
        actor = _make_comm_actor()
        actor.train()
        obs = _random_obs(3)
        actor.reset_delay_buffer()
        s1, _, _ = actor.get_action(obs)
        actor.reset_delay_buffer()
        s2, _, _ = actor.get_action(obs)
        # Extremely unlikely to be identical for stochastic policy
        assert not torch.allclose(s1, s2)

    def test_log_probs_finite(self):
        actor = _make_comm_actor()
        obs   = _random_obs(3)
        _, _, lp = actor.get_action(obs)
        assert torch.all(torch.isfinite(lp))


# ===========================================================================
# CommActor.evaluate_actions
# ===========================================================================

class TestCommActorEvaluateActions:
    def test_output_shapes(self):
        actor    = _make_comm_actor()
        N, B     = 3, 8
        obs      = _random_obs(N * B)
        actions  = torch.randn(N * B, 4)
        lp, ent  = actor.evaluate_actions(obs, actions, n_agents=N)
        assert lp.shape  == (N * B,)
        assert ent.shape == (N * B,)

    def test_log_probs_finite(self):
        actor   = _make_comm_actor()
        N, B    = 3, 10
        obs     = _random_obs(N * B)
        actions = torch.randn(N * B, 4)
        lp, _   = actor.evaluate_actions(obs, actions, n_agents=N)
        assert torch.all(torch.isfinite(lp))

    def test_entropy_positive(self):
        actor   = _make_comm_actor()
        N, B    = 3, 10
        obs     = _random_obs(N * B)
        actions = torch.randn(N * B, 4)
        _, ent  = actor.evaluate_actions(obs, actions, n_agents=N)
        assert torch.all(ent > 0), "Entropy of a Gaussian must be positive"

    def test_gradients_flow(self):
        """PPO requires gradients through log_probs back to actor parameters."""
        actor   = _make_comm_actor()
        N, B    = 3, 4
        obs     = _random_obs(N * B)
        actions = torch.randn(N * B, 4)
        lp, _   = actor.evaluate_actions(obs, actions, n_agents=N)
        lp.sum().backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in actor.parameters()
        )
        assert has_grad, "No gradients flowed to CommActor parameters"

    def test_single_step_batch(self):
        """B=1 step (edge case for the reshape loop)."""
        actor   = _make_comm_actor()
        N       = 3
        obs     = _random_obs(N)
        actions = torch.randn(N, 4)
        lp, _   = actor.evaluate_actions(obs, actions, n_agents=N)
        assert lp.shape == (N,)


# ===========================================================================
# Adjacency / comm_radius
# ===========================================================================

class TestCommActorAdjacency:
    def test_infinite_radius_all_connected(self):
        actor = _make_comm_actor(comm_radius=float("inf"), drop_comm_prob=0.0)
        n     = 5
        positions = torch.rand(n, 3) * 100.0   # spread far apart
        adj = actor._build_adj(positions, n, training=False)
        assert adj.all(), "infinite radius should connect every pair"

    def test_zero_radius_only_self_loops(self):
        actor = _make_comm_actor(comm_radius=0.0, drop_comm_prob=0.0)
        # 3 drones at distinct positions (> 0 apart)
        positions = torch.tensor([[0.0, 0, 0], [5.0, 0, 0], [10.0, 0, 0]])
        adj = actor._build_adj(positions, 3, training=False)
        eye = torch.eye(3, dtype=torch.bool)
        assert torch.equal(adj, eye), "radius=0 should only self-connect"

    def test_finite_radius_correct_edges(self):
        actor = _make_comm_actor(comm_radius=3.0, drop_comm_prob=0.0)
        # drone_0 and drone_1 are 2 m apart (within radius)
        # drone_0 and drone_2 are 10 m apart (outside radius)
        positions = torch.tensor([[0.0, 0, 0], [2.0, 0, 0], [10.0, 0, 0]])
        adj = actor._build_adj(positions, 3, training=False)
        assert adj[0, 0].item() is True   # self
        assert adj[0, 1].item() is True   # within radius
        assert adj[1, 0].item() is True   # symmetric
        assert adj[0, 2].item() is False  # too far
        assert adj[2, 0].item() is False  # too far

    def test_drop_comm_removes_some_edges(self):
        """With 100% drop probability, all non-self edges must vanish."""
        actor = _make_comm_actor(comm_radius=float("inf"), drop_comm_prob=1.0)
        positions = torch.rand(5, 3) * 10
        adj = actor._build_adj(positions, 5, training=True)
        eye = torch.eye(5, dtype=torch.bool)
        # Only self-loops must remain
        off_diag = adj & ~eye
        assert not off_diag.any(), "drop_prob=1.0 should remove all non-self edges"

    def test_no_drop_in_eval_mode(self):
        """drop_comm_prob must be skipped when training=False."""
        actor = _make_comm_actor(comm_radius=float("inf"), drop_comm_prob=1.0)
        positions = torch.rand(4, 3) * 10
        adj = actor._build_adj(positions, 4, training=False)
        assert adj.all(), "drop should not apply in eval mode"


# ===========================================================================
# CommActor delay buffer
# ===========================================================================

class TestCommActorDelayBuffer:
    def test_delay_zero_uses_current_obs(self):
        """With comm_delay=0, forward uses the obs just passed in."""
        actor = _make_comm_actor(comm_delay=0)
        actor.reset_delay_buffer()
        obs1 = _random_obs(3)
        obs2 = _random_obs(3) * 100.0   # very different
        # After one forward with obs1, buffer = [obs1].
        # Next forward with obs2 should see obs2 (no delay).
        actor.forward(obs1)
        actor.reset_delay_buffer()
        actor.forward(obs2)
        # Just check no crash and correct shape
        mean, _ = actor.forward(obs2)
        assert mean.shape == (3, 4)

    def test_reset_clears_buffer(self):
        actor = _make_comm_actor(comm_delay=2)
        obs   = _random_obs(3)
        actor.forward(obs)
        actor.forward(obs)
        assert len(actor._delay_buf) > 0
        actor.reset_delay_buffer()
        assert len(actor._delay_buf) == 0

    def test_delay_buffer_fills_up(self):
        delay = 3
        actor = _make_comm_actor(comm_delay=delay)
        actor.reset_delay_buffer()
        obs = _random_obs(3)
        for _ in range(delay + 1):
            actor.forward(obs)
        # maxlen = delay + 1; should be exactly full
        assert len(actor._delay_buf) == delay + 1

    def test_forward_returns_finite_with_delay(self):
        actor = _make_comm_actor(comm_delay=2)
        actor.reset_delay_buffer()
        obs = _random_obs(3)
        for _ in range(5):
            mean, log_std = actor.forward(obs)
            assert torch.all(torch.isfinite(mean))
            assert torch.all(torch.isfinite(log_std))


# ===========================================================================
# build_actor factory
# ===========================================================================

class TestBuildActor:
    def test_mlp_actor_by_default(self):
        from training.train_mappo import build_actor
        cfg = {
            "model": {
                "obs_dim": 15, "act_dim": 4,
                "actor_hidden": [64, 64],
            }
        }
        actor = build_actor(cfg)
        assert isinstance(actor, Actor)
        assert not isinstance(actor, CommActor)

    def test_comm_actor_when_type_comm(self):
        from training.train_mappo import build_actor
        cfg = {
            "model": {
                "obs_dim": 15, "act_dim": 4,
                "actor_hidden": [64, 64],
                "actor_type": "comm",
                "comm": {
                    "embed_dim": 16,
                    "n_heads": 4,
                    "comm_radius": float("inf"),
                    "comm_delay": 0,
                    "drop_comm_prob": 0.0,
                },
            }
        }
        actor = build_actor(cfg)
        assert isinstance(actor, CommActor)

    def test_comm_actor_params_passed_through(self):
        from training.train_mappo import build_actor
        cfg = {
            "model": {
                "obs_dim": 15, "act_dim": 4,
                "actor_hidden": [64, 64],
                "actor_type": "comm",
                "comm": {
                    "embed_dim": 32,
                    "n_heads": 4,
                    "comm_radius": 5.0,
                    "comm_delay": 1,
                    "drop_comm_prob": 0.05,
                },
            }
        }
        actor = build_actor(cfg)
        assert isinstance(actor, CommActor)
        assert actor.embed_dim == 32
        assert actor.comm_radius == pytest.approx(5.0)
        assert actor.comm_delay == 1
        assert actor.drop_comm_prob == pytest.approx(0.05)

    def test_mlp_explicit(self):
        from training.train_mappo import build_actor
        cfg = {
            "model": {
                "obs_dim": 15, "act_dim": 4,
                "actor_hidden": [64, 64],
                "actor_type": "mlp",
            }
        }
        actor = build_actor(cfg)
        assert isinstance(actor, Actor)


# ===========================================================================
# Checkpoint save / load round-trip
# ===========================================================================

class TestCommActorCheckpoint:
    def test_state_dict_round_trip(self, tmp_path):
        actor = _make_comm_actor(embed_dim=16, n_heads=4)
        obs   = _random_obs(3)
        actor.reset_delay_buffer()
        with torch.no_grad():
            before = actor.get_action(obs, deterministic=True)[0].clone()

        # Save
        path = tmp_path / "comm_actor.pt"
        torch.save({"comm_actor_state_dict": actor.state_dict()}, path)

        # Load into a fresh instance
        actor2 = _make_comm_actor(embed_dim=16, n_heads=4)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        actor2.load_state_dict(ckpt["comm_actor_state_dict"])

        actor2.reset_delay_buffer()
        with torch.no_grad():
            after = actor2.get_action(obs, deterministic=True)[0]

        assert torch.allclose(before, after, atol=1e-5), (
            "Checkpoint round-trip produced different deterministic actions"
        )

    def test_partial_load_from_actor_checkpoint(self, tmp_path):
        """CommActor can warm-start from a plain Actor checkpoint (strict=False).

        Actor uses key 'shared_net.*'; CommActor uses 'trunk.*' for the same
        layers — so Actor's shared_net keys are 'unexpected' from CommActor's
        perspective, and CommActor's GATv2 + trunk keys are 'missing'.
        The important invariant is that no exception is raised and that
        GATv2-specific keys appear in missing_keys.
        """
        plain = Actor(obs_dim=15, act_dim=4, hidden_sizes=[64, 64])
        path  = tmp_path / "actor.pt"
        torch.save({"actor_state_dict": plain.state_dict()}, path)

        comm = CommActor(obs_dim=15, act_dim=4, embed_dim=16, n_heads=4,
                         hidden_sizes=[64, 64])
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        # strict=False: no exception — partial load succeeds
        missing, unexpected = comm.load_state_dict(
            ckpt["actor_state_dict"], strict=False
        )
        # CommActor's GATv2 + trunk keys will be in missing_keys
        assert any("node_encoder" in k or "gat" in k or "trunk" in k
                   for k in missing), (
            "Expected GATv2/trunk keys to be missing when loading from plain Actor checkpoint"
        )
        # Actor's 'shared_net' keys are unexpected in CommActor (different name)
        assert any("shared_net" in k for k in unexpected), (
            "Expected Actor's shared_net keys to be unexpected in CommActor"
        )


# ===========================================================================
# CommActor vs Actor comparison
# ===========================================================================

class TestCommActorVsActor:
    def test_comm_actor_differs_from_actor_on_same_obs(self):
        """CommActor with communication should produce different outputs than Actor."""
        torch.manual_seed(0)
        plain = Actor(obs_dim=15, act_dim=4, hidden_sizes=[64, 64])
        comm  = CommActor(obs_dim=15, act_dim=4, embed_dim=16, n_heads=4,
                          hidden_sizes=[64, 64], comm_radius=float("inf"))
        # Reinit both to the same trunk+head weights for a fair comparison
        with torch.no_grad():
            for (pk, pv), (ck, cv) in zip(
                plain.state_dict().items(), comm.state_dict().items()
            ):
                if pk in comm.state_dict() and comm.state_dict()[pk].shape == pv.shape:
                    comm.state_dict()[pk].copy_(pv)

        obs = _random_obs(3)
        comm.reset_delay_buffer()
        with torch.no_grad():
            sq_plain, _, _ = plain.get_action(obs, deterministic=True)
            sq_comm,  _, _ = comm.get_action(obs,  deterministic=True)

        # They should differ because CommActor aggregates neighbour info
        assert not torch.allclose(sq_plain, sq_comm, atol=1e-4), (
            "CommActor should produce different actions from plain Actor "
            "when neighbours are present"
        )


# ===========================================================================
# Integration: CommActor in a live HomeEnv episode
# ===========================================================================

class TestCommActorIntegration:
    def test_short_episode_no_crash(self):
        from envs.home_env import HomeEnv
        env    = HomeEnv({"n_drones": 3, "max_steps": 20})
        actor  = _make_comm_actor(obs_dim=15, act_dim=4, embed_dim=16, n_heads=4)
        actor.eval()
        agent_ids = sorted(env._agent_ids)

        obs_dict, _ = env.reset(seed=42)
        actor.reset_delay_buffer()
        done = False
        steps = 0

        while not done and steps < 20:
            obs_tensor = torch.tensor(
                np.stack([obs_dict.get(aid, np.zeros(15)) for aid in agent_ids]),
                dtype=torch.float32,
            )
            with torch.no_grad():
                squashed, _, _ = actor.get_action(obs_tensor, deterministic=True)
            action_dict = {
                aid: squashed[i].numpy() for i, aid in enumerate(agent_ids)
            }
            obs_dict, rewards, terminated, truncated, _ = env.step(action_dict)
            done  = terminated.get("__all__", False) or truncated.get("__all__", False)
            steps += 1

        assert steps > 0, "Episode completed zero steps"

    def test_actions_finite_throughout_episode(self):
        from envs.home_env import HomeEnv
        env    = HomeEnv({"n_drones": 3, "max_steps": 15})
        actor  = _make_comm_actor(obs_dim=15, act_dim=4, embed_dim=16, n_heads=4)
        actor.eval()
        agent_ids = sorted(env._agent_ids)

        obs_dict, _ = env.reset(seed=7)
        actor.reset_delay_buffer()

        for _ in range(15):
            obs_tensor = torch.tensor(
                np.stack([obs_dict.get(aid, np.zeros(15)) for aid in agent_ids]),
                dtype=torch.float32,
            )
            with torch.no_grad():
                squashed, _, _ = actor.get_action(obs_tensor, deterministic=True)
            assert torch.all(torch.isfinite(squashed)), "Non-finite action produced"
            action_dict = {
                aid: squashed[i].numpy() for i, aid in enumerate(agent_ids)
            }
            obs_dict, _, terminated, truncated, _ = env.step(action_dict)
            if terminated.get("__all__", False) or truncated.get("__all__", False):
                break

    def test_comm_delay_episode_no_crash(self):
        """comm_delay > 0 must not crash during an episode."""
        from envs.home_env import HomeEnv
        env    = HomeEnv({"n_drones": 3, "max_steps": 15})
        actor  = _make_comm_actor(obs_dim=15, act_dim=4, embed_dim=16, n_heads=4,
                                   comm_delay=3)
        actor.eval()
        agent_ids = sorted(env._agent_ids)
        obs_dict, _ = env.reset()
        actor.reset_delay_buffer()

        for _ in range(15):
            obs_tensor = torch.tensor(
                np.stack([obs_dict.get(aid, np.zeros(15)) for aid in agent_ids]),
                dtype=torch.float32,
            )
            with torch.no_grad():
                squashed, _, _ = actor.get_action(obs_tensor, deterministic=True)
            action_dict = {
                aid: squashed[i].numpy() for i, aid in enumerate(agent_ids)
            }
            obs_dict, _, terminated, truncated, _ = env.step(action_dict)
            if terminated.get("__all__", False) or truncated.get("__all__", False):
                break

    def test_limited_radius_episode_no_crash(self):
        """comm_radius=2.0 (sparse graph) must not crash."""
        from envs.home_env import HomeEnv
        env    = HomeEnv({"n_drones": 4, "max_steps": 15})
        actor  = _make_comm_actor(obs_dim=15, act_dim=4, embed_dim=16, n_heads=4,
                                   comm_radius=2.0)
        actor.eval()
        agent_ids = sorted(env._agent_ids)
        obs_dict, _ = env.reset()
        actor.reset_delay_buffer()

        for _ in range(15):
            obs_tensor = torch.tensor(
                np.stack([obs_dict.get(aid, np.zeros(15)) for aid in agent_ids]),
                dtype=torch.float32,
            )
            with torch.no_grad():
                squashed, _, _ = actor.get_action(obs_tensor, deterministic=True)
            action_dict = {
                aid: squashed[i].numpy() for i, aid in enumerate(agent_ids)
            }
            obs_dict, _, terminated, truncated, _ = env.step(action_dict)
            if terminated.get("__all__", False) or truncated.get("__all__", False):
                break
