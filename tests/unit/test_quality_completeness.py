"""Unit tests for torq.quality.completeness (Story 3.1, Task 6)."""

from __future__ import annotations

import numpy as np
import pytest

from torq.quality import completeness


def test_success_true_returns_near_one(make_quality_episode):
    """metadata.success = True must return 1.0."""
    result = completeness.score(make_quality_episode(metadata={"success": True}))
    assert result == pytest.approx(1.0)


def test_success_false_returns_near_zero(make_quality_episode):
    """metadata.success = False must return 0.0."""
    result = completeness.score(make_quality_episode(metadata={"success": False}))
    assert result == pytest.approx(0.0)


def test_no_success_flag_uses_duration_heuristic(make_quality_episode):
    """Without a success flag, score should scale with episode duration."""
    short = completeness.score(make_quality_episode(duration_seconds=5.0))
    long_ = completeness.score(make_quality_episode(duration_seconds=30.0))

    assert short is not None and long_ is not None
    assert short < long_, "Longer episodes should score higher"
    assert long_ == pytest.approx(1.0), "30s episode should reach the plateau"


def test_short_episode_returns_none(make_quality_episode):
    """Episodes with < 10 timesteps must return None."""
    result = completeness.score(make_quality_episode(n_timesteps=5))
    assert result is None


def test_nan_episode_returns_none(make_quality_episode):
    """Episodes with NaN in actions must return None."""
    actions = np.ones((20, 1))
    actions[10] = float("nan")
    result = completeness.score(make_quality_episode(actions))
    assert result is None


def test_score_always_in_range(make_quality_episode):
    """Score must always be in [0.0, 1.0] for valid episodes of varying duration."""
    for dur in [0.1, 5.0, 15.0, 30.0, 60.0]:
        result = completeness.score(make_quality_episode(duration_seconds=dur))
        assert result is not None
        assert 0.0 <= result <= 1.0, f"Score out of range for duration={dur}s: {result}"
