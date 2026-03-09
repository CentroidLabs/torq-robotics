"""Unit tests for tq.quality.filter() — quality gate filtering.

Tests cover:
- Basic threshold filtering
- All-filtered edge case (empty return, no exception, warning logged)
- Unscored episodes excluded with warning
- None overall excluded with warning
- Boundary min_score values (0.0 and 1.0)
- Validation errors for bad min_score and non-list input
- Empty input list
- Return is a new list (not same object)
- Info log message format
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

import torq as tq
from torq.errors import TorqQualityError
from torq.quality.report import QualityReport


# ── Helpers ───────────────────────────────────────────────────────────────────

def _scored_episode(make_quality_episode, overall: float | None, *, n_timesteps: int = 30):
    """Create an episode and manually attach a QualityReport with given overall."""
    ep = make_quality_episode(n_timesteps=n_timesteps)
    ep.quality = QualityReport(
        smoothness=overall,
        consistency=overall,
        completeness=overall,
        weights={"smoothness": 0.40, "consistency": 0.35, "completeness": 0.25},
    )
    # If overall is None, force it (QualityReport computes overall automatically
    # when all three sub-scores are provided, so we override it for test control)
    if overall is None:
        object.__setattr__(ep.quality, "overall", None)
    return ep


# ── Basic filtering ───────────────────────────────────────────────────────────

def test_basic_filtering_returns_episodes_above_threshold(make_quality_episode):
    """5 episodes with varying scores: only those >= 0.5 are returned."""
    eps = [_scored_episode(make_quality_episode, s) for s in [0.3, 0.5, 0.6, 0.2, 0.8]]
    result = tq.quality.filter(eps, min_score=0.5)
    scores = [ep.quality.overall for ep in result]
    assert scores == [0.5, 0.6, 0.8]


def test_threshold_is_inclusive(make_quality_episode):
    """Episode scoring exactly min_score passes the gate."""
    ep = _scored_episode(make_quality_episode, 0.75)
    result = tq.quality.filter([ep], min_score=0.75)
    assert len(result) == 1
    assert result[0] is ep


def test_episode_just_below_threshold_excluded(make_quality_episode):
    """Episode just below threshold is excluded."""
    ep = _scored_episode(make_quality_episode, 0.749)
    result = tq.quality.filter([ep], min_score=0.75)
    assert result == []


# ── All-filtered edge case ────────────────────────────────────────────────────

def test_all_filtered_returns_empty_list_no_exception(make_quality_episode):
    """When no episodes pass, returns empty list without raising."""
    eps = [_scored_episode(make_quality_episode, s) for s in [0.1, 0.2, 0.3]]
    result = tq.quality.filter(eps, min_score=0.9)
    assert result == []


def test_warning_logged_when_all_episodes_filtered(make_quality_episode, caplog):
    """A warning is logged when 0 episodes pass."""
    eps = [_scored_episode(make_quality_episode, 0.1)]
    with caplog.at_level(logging.WARNING, logger="torq.quality.filters"):
        tq.quality.filter(eps, min_score=0.9)
    assert any("0/1 episodes passed quality gate" in r.message for r in caplog.records)
    assert any("Consider lowering the threshold" in r.message for r in caplog.records)


# ── Unscored episodes ─────────────────────────────────────────────────────────

def test_unscored_episode_excluded_with_warning(make_quality_episode, caplog):
    """Episodes with episode.quality is None are excluded and a warning is logged."""
    ep = make_quality_episode()
    # ep.quality is None — never scored
    assert ep.quality is None

    with caplog.at_level(logging.WARNING, logger="torq.quality.filters"):
        result = tq.quality.filter([ep], min_score=0.0)

    assert result == []
    assert any(
        "ep_test" in r.message and "never scored" in r.message for r in caplog.records
    )


def test_none_overall_excluded_with_warning(make_quality_episode, caplog):
    """Episodes where episode.quality.overall is None are excluded with warning."""
    ep = _scored_episode(make_quality_episode, overall=None)
    assert ep.quality is not None
    assert ep.quality.overall is None

    with caplog.at_level(logging.WARNING, logger="torq.quality.filters"):
        result = tq.quality.filter([ep], min_score=0.0)

    assert result == []
    assert any(
        "ep_test" in r.message and "no overall score" in r.message for r in caplog.records
    )


# ── Boundary min_score values ─────────────────────────────────────────────────

def test_min_score_zero_passes_all_scored_episodes(make_quality_episode):
    """min_score=0.0 passes all episodes that have a non-None overall score."""
    eps = [_scored_episode(make_quality_episode, s) for s in [0.0, 0.1, 0.5, 1.0]]
    result = tq.quality.filter(eps, min_score=0.0)
    assert len(result) == 4


def test_min_score_one_only_passes_perfect_scores(make_quality_episode):
    """min_score=1.0 only passes episodes with overall == 1.0."""
    eps = [_scored_episode(make_quality_episode, s) for s in [0.5, 0.99, 1.0]]
    result = tq.quality.filter(eps, min_score=1.0)
    assert len(result) == 1
    assert result[0].quality.overall == 1.0


# ── Validation errors ─────────────────────────────────────────────────────────

def test_min_score_below_zero_raises():
    """min_score < 0 raises TorqQualityError."""
    with pytest.raises(TorqQualityError, match="min_score must be in"):
        tq.quality.filter([], min_score=-0.1)


def test_min_score_above_one_raises():
    """min_score > 1 raises TorqQualityError."""
    with pytest.raises(TorqQualityError, match="min_score must be in"):
        tq.quality.filter([], min_score=1.1)


def test_min_score_bool_true_raises():
    """min_score=True (bool) is rejected even though bool is a subclass of int."""
    with pytest.raises(TorqQualityError, match="min_score must be"):
        tq.quality.filter([], min_score=True)  # type: ignore[arg-type]


def test_min_score_bool_false_raises():
    """min_score=False (bool) is rejected even though bool is a subclass of int."""
    with pytest.raises(TorqQualityError, match="min_score must be"):
        tq.quality.filter([], min_score=False)  # type: ignore[arg-type]


def test_min_score_int_accepted_and_coerced(make_quality_episode):
    """min_score=1 (int) is accepted and coerced to 1.0."""
    ep = _scored_episode(make_quality_episode, 1.0)
    result = tq.quality.filter([ep], min_score=1)
    assert len(result) == 1


def test_mixed_episodes_correct_count_logging(make_quality_episode, caplog):
    """Mixed episode types: scored-above, scored-below, unscored, None-overall.

    Verify the info log shows the correct count for realistic usage.
    """
    ep_above = _scored_episode(make_quality_episode, 0.9)
    ep_below = _scored_episode(make_quality_episode, 0.3)
    ep_unscored = make_quality_episode()          # quality is None
    ep_none_overall = _scored_episode(make_quality_episode, overall=None)

    eps = [ep_above, ep_below, ep_unscored, ep_none_overall]

    with caplog.at_level(logging.INFO, logger="torq.quality.filters"):
        result = tq.quality.filter(eps, min_score=0.5)

    assert len(result) == 1
    assert result[0] is ep_above

    # Info log should report 1 passed out of 4 total
    info_msgs = [r.message for r in caplog.records if r.levelno == logging.INFO]
    assert any("1/4 episodes passed" in m and "min_score=0.50" in m for m in info_msgs)


def test_non_list_episodes_raises():
    """Passing a non-list for episodes raises TorqQualityError."""
    with pytest.raises(TorqQualityError, match="episodes must be a list"):
        tq.quality.filter("not_a_list", min_score=0.5)  # type: ignore[arg-type]


def test_tuple_episodes_raises(make_quality_episode):
    """Passing a tuple instead of list raises TorqQualityError."""
    ep = _scored_episode(make_quality_episode, 0.8)
    with pytest.raises(TorqQualityError, match="episodes must be a list"):
        tq.quality.filter((ep,), min_score=0.5)  # type: ignore[arg-type]


# ── Empty input list ──────────────────────────────────────────────────────────

def test_empty_input_list_returns_empty_list():
    """An empty input list returns an empty list immediately."""
    result = tq.quality.filter([], min_score=0.5)
    assert result == []


# ── Return is a new list ──────────────────────────────────────────────────────

def test_return_is_new_list_not_input(make_quality_episode):
    """The returned list is a new object, not the same as the input list."""
    ep = _scored_episode(make_quality_episode, 0.8)
    eps = [ep]
    result = tq.quality.filter(eps, min_score=0.5)
    assert result is not eps
    assert result[0] is ep  # but the episodes themselves are not copied


# ── Info log message ──────────────────────────────────────────────────────────

def test_info_log_message_format(make_quality_episode, caplog):
    """Info log includes count and threshold."""
    eps = [_scored_episode(make_quality_episode, s) for s in [0.3, 0.8, 0.9]]
    with caplog.at_level(logging.INFO, logger="torq.quality.filters"):
        tq.quality.filter(eps, min_score=0.5)

    info_records = [r for r in caplog.records if r.levelno == logging.INFO]
    assert any("2/3 episodes passed" in r.message and "min_score=0.50" in r.message
               for r in info_records)


# ── Input order preserved ─────────────────────────────────────────────────────

def test_input_order_preserved(make_quality_episode):
    """Returned episodes are in the same order as the input."""
    scores = [0.9, 0.8, 0.7, 0.6]
    eps = [_scored_episode(make_quality_episode, s) for s in scores]
    result = tq.quality.filter(eps, min_score=0.5)
    assert [ep.quality.overall for ep in result] == scores
