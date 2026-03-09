"""Unit tests for torq.quality.feasibility R1 stub (Story 3.5, Task 4, AC #4)."""

from __future__ import annotations

import numpy as np

import torq.quality as q


def test_feasibility_score_returns_one_for_normal_episode(make_quality_episode):
    """score() returns 1.0 for any valid episode (AC #4)."""
    ep = make_quality_episode()
    result = q.feasibility.score(ep)
    assert result == 1.0


def test_feasibility_score_return_type_is_float(make_quality_episode):
    """Return type is float, not int or None."""
    ep = make_quality_episode()
    result = q.feasibility.score(ep)
    assert isinstance(result, float)


def test_feasibility_score_returns_one_for_short_episode(make_quality_episode):
    """Returns 1.0 even for episodes with < 10 timesteps (unlike other scorers)."""
    ep = make_quality_episode(n_timesteps=3)
    result = q.feasibility.score(ep)
    assert result == 1.0


def test_feasibility_score_returns_one_for_episode_with_nan_actions(make_quality_episode):
    """Returns 1.0 even when actions contain NaN values."""
    actions = np.full((20, 3), np.nan)
    ep = make_quality_episode(actions=actions)
    result = q.feasibility.score(ep)
    assert result == 1.0


def test_feasibility_score_returns_one_for_minimal_episode(make_quality_episode):
    """Returns 1.0 for an episode with a single timestep."""
    ep = make_quality_episode(n_timesteps=1)
    result = q.feasibility.score(ep)
    assert result == 1.0


def test_feasibility_accessible_as_tq_quality_feasibility_score(make_quality_episode):
    """tq.quality.feasibility.score() is accessible via module attribute path."""
    ep = make_quality_episode()
    result = q.feasibility.score(ep)
    assert result == 1.0
