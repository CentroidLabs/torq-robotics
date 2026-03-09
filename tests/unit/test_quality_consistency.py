"""Unit tests for torq.quality.consistency (Story 3.1, Task 5)."""

from __future__ import annotations

import numpy as np

from torq.quality import consistency


def test_consistent_unidirectional_scores_high(make_quality_episode):
    """A monotonically increasing trajectory should score high (> 0.8)."""
    actions = np.linspace(0, 10, 50).reshape(-1, 1)
    result = consistency.score(make_quality_episode(actions))
    assert result is not None
    assert result > 0.8, f"Expected > 0.8, got {result}"


def test_oscillating_trajectory_scores_low(make_quality_episode):
    """Alternating +1 / -1 steps (maximum oscillation) should score low."""
    actions = np.array([(-1.0) ** i for i in range(50)]).reshape(-1, 1)
    result = consistency.score(make_quality_episode(actions))
    assert result is not None
    assert result < 0.5, f"Expected < 0.5, got {result}"


def test_short_episode_returns_none(make_quality_episode):
    """Episodes with < 10 timesteps must return None."""
    result = consistency.score(make_quality_episode(n_timesteps=5))
    assert result is None


def test_nan_episode_returns_none(make_quality_episode):
    """Episodes with NaN in actions must return None."""
    actions = np.ones((20, 1))
    actions[10] = float("nan")
    result = consistency.score(make_quality_episode(actions))
    assert result is None


def test_score_always_in_range(make_quality_episode):
    """Score must always be in [0.0, 1.0] for any valid episode."""
    rng = np.random.RandomState(1)
    for _ in range(20):
        actions = rng.randn(30, 4).astype(np.float64)
        result = consistency.score(make_quality_episode(actions))
        assert result is not None
        assert 0.0 <= result <= 1.0, f"Score out of range: {result}"
