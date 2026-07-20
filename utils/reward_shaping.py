"""
reward_shaping.py
Utility functions for reward normalisation and curriculum-based
reward schedule adjustments.

Import these in train_mappo.py to augment the raw env rewards.
"""

from __future__ import annotations

import numpy as np


class RunningMeanStd:
    """
    Welford online algorithm for running mean and variance.
    Used to normalise rewards across the training run.
    """

    def __init__(self, shape: tuple = (), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var + 1e-8)


class RewardNormalizer:
    """
    Normalises rewards to roughly unit variance using a running estimate.
    Clip prevents extreme values from destabilising training.
    """

    def __init__(self, clip: float = 10.0):
        self.rms = RunningMeanStd()
        self.clip = clip

    def normalise(self, rewards: np.ndarray) -> np.ndarray:
        self.rms.update(rewards)
        normalised = (rewards - self.rms.mean) / self.rms.std
        return np.clip(normalised, -self.clip, self.clip)


class CurriculumScheduler:
    """
    Controls when to advance to the next curriculum stage.
    
    Stages are defined as a list of thresholds on mean episode reward.
    When the rolling mean reward exceeds the threshold for `patience`
    consecutive evaluations, stage is incremented and the env config
    can be updated accordingly.

    Example:
        scheduler = CurriculumScheduler(
            stage_thresholds=[5.0, 15.0, 30.0],
            patience=3,
        )
        stage_changed = scheduler.update(mean_reward)
        if stage_changed:
            env.update_curriculum(scheduler.stage)
    """

    def __init__(self, stage_thresholds: list[float], patience: int = 3):
        self.thresholds = stage_thresholds
        self.patience = patience
        self.stage = 0
        self._patience_count = 0

    def update(self, mean_reward: float) -> bool:
        """Returns True if stage advanced."""
        if self.stage >= len(self.thresholds):
            return False  # already at max stage

        if mean_reward >= self.thresholds[self.stage]:
            self._patience_count += 1
        else:
            self._patience_count = 0

        if self._patience_count >= self.patience:
            self.stage += 1
            self._patience_count = 0
            return True

        return False

    @property
    def at_max_stage(self) -> bool:
        return self.stage >= len(self.thresholds)
