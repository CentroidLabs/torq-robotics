"""Unit tests for torq.quality.smoothness (Story 3.1, Task 4)."""

from __future__ import annotations

import numpy as np
import pytest

from torq.quality import smoothness


def test_smooth_trajectory_scores_high(make_quality_episode):
    """A sine-wave trajectory (low jerk) should score close to 1.0."""
    t = np.linspace(0, 2 * np.pi, 50)
    result = smoothness.score(make_quality_episode(np.sin(t).reshape(-1, 1)))
    assert result is not None
    assert result > 0.7, f"Expected > 0.7, got {result}"


def test_jerky_trajectory_scores_low(make_quality_episode):
    """Random noise (high jerk) should score low."""
    rng = np.random.RandomState(42)
    actions = (rng.randn(50, 1) * 10).astype(np.float64)
    result = smoothness.score(make_quality_episode(actions))
    assert result is not None
    assert result < 0.5, f"Expected < 0.5, got {result}"


def test_constant_velocity_scores_near_one(make_quality_episode):
    """A ramp (constant velocity) has zero jerk and should score 1.0."""
    actions = np.linspace(0, 1, 50).reshape(-1, 1)
    result = smoothness.score(make_quality_episode(actions))
    assert result is not None
    assert result == pytest.approx(1.0), f"Expected 1.0, got {result}"


def test_single_jerk_spike(make_quality_episode):
    """A single spike in an otherwise smooth trajectory should score < 1.0 but > 0."""
    t = np.linspace(0, 2 * np.pi, 50)
    actions = np.sin(t).reshape(-1, 1).copy()
    actions[25] += 5.0  # inject a spike
    result = smoothness.score(make_quality_episode(actions))
    assert result is not None
    assert 0.0 < result < 1.0, f"Expected score between 0 and 1, got {result}"


def test_short_episode_returns_none(make_quality_episode):
    """Episodes with < 10 timesteps must return None (not NaN, not 0.0)."""
    result = smoothness.score(make_quality_episode(n_timesteps=5))
    assert result is None


def test_nan_episode_returns_none(make_quality_episode):
    """Episodes with NaN in actions must return None."""
    actions = np.ones((20, 1))
    actions[10] = float("nan")
    result = smoothness.score(make_quality_episode(actions))
    assert result is None


def test_score_always_in_range(make_quality_episode):
    """Score must always be in [0.0, 1.0] for any valid episode."""
    rng = np.random.RandomState(0)
    for _ in range(20):
        actions = rng.randn(30, 6).astype(np.float64)
        result = smoothness.score(make_quality_episode(actions))
        assert result is not None
        assert 0.0 <= result <= 1.0, f"Score out of range: {result}"


def test_single_vs_multi_dimension(make_quality_episode):
    """Score should be in [0, 1] for both 1-D and multi-D action arrays."""
    t = np.linspace(0, 2 * np.pi, 50)
    result_1d = smoothness.score(make_quality_episode(np.sin(t).reshape(-1, 1)))
    result_md = smoothness.score(
        make_quality_episode(np.column_stack([np.sin(t), np.cos(t), t / (2 * np.pi)]))
    )
    assert result_1d is not None and 0.0 <= result_1d <= 1.0
    assert result_md is not None and 0.0 <= result_md <= 1.0
