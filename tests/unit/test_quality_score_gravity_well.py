"""Integration tests for gravity well behavior in tq.quality.score() (Story 3.6).

Tests the full end-to-end path: score() → gravity well fires/suppressed.
Uses capsys to capture stdout rather than mocking _gravity_well directly.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import torq.quality as q
from torq._config import config
from torq.errors import TorqQualityError

_ALL_SCORE_08 = {
    "torq.quality._smoothness_score": 0.8,
    "torq.quality._consistency_score": 0.8,
    "torq.quality._completeness_score": 0.8,
}


@pytest.fixture(autouse=True)
def ensure_not_quiet(monkeypatch):
    """Reset quiet to False before each test so gravity wells are not suppressed."""
    monkeypatch.setattr(config, "quiet", False)


# ── AC #1: gravity well fires on success ──────────────────────────────────────


def test_gravity_well_fires_after_batch_scoring(make_quality_episode, capsys):
    """score() on 3+ episodes fires gravity well with avg score and datatorq.ai URL (AC #1)."""
    episodes = [make_quality_episode() for _ in range(3)]
    with patch("torq.quality._smoothness_score", return_value=0.8), \
         patch("torq.quality._consistency_score", return_value=0.8), \
         patch("torq.quality._completeness_score", return_value=0.8):
        q.score(episodes)
    captured = capsys.readouterr()
    assert "datatorq.ai" in captured.out


def test_gravity_well_message_contains_computed_average(make_quality_episode, capsys):
    """Gravity well message contains the formatted average score (AC #1)."""
    # With all scorers returning 0.8: overall = 0.8*0.40 + 0.8*0.35 + 0.8*0.25 = 0.80
    episodes = [make_quality_episode() for _ in range(3)]
    with patch("torq.quality._smoothness_score", return_value=0.8), \
         patch("torq.quality._consistency_score", return_value=0.8), \
         patch("torq.quality._completeness_score", return_value=0.8):
        q.score(episodes)
    captured = capsys.readouterr()
    assert "0.80" in captured.out


def test_gravity_well_contains_datatorq_url(make_quality_episode, capsys):
    """Gravity well output contains the full datatorq.ai URL (AC #1)."""
    episodes = [make_quality_episode() for _ in range(2)]
    with patch("torq.quality._smoothness_score", return_value=0.6), \
         patch("torq.quality._consistency_score", return_value=0.6), \
         patch("torq.quality._completeness_score", return_value=0.6):
        q.score(episodes)
    captured = capsys.readouterr()
    assert "https://www.datatorq.ai" in captured.out


def test_gravity_well_fires_for_single_episode(make_quality_episode, capsys):
    """score(episode) (not a list) also fires gravity well on success (AC #1)."""
    ep = make_quality_episode()
    with patch("torq.quality._smoothness_score", return_value=0.7), \
         patch("torq.quality._consistency_score", return_value=0.7), \
         patch("torq.quality._completeness_score", return_value=0.7):
        q.score(ep)
    captured = capsys.readouterr()
    assert "datatorq.ai" in captured.out
    assert "0.70" in captured.out


# ── AC #2: quiet mode suppresses gravity well ─────────────────────────────────


def test_quiet_mode_suppresses_gravity_well(make_quality_episode, capsys, monkeypatch):
    """tq.config.quiet = True suppresses gravity well output after score() (AC #2)."""
    monkeypatch.setattr(config, "quiet", True)
    episodes = [make_quality_episode() for _ in range(2)]
    with patch("torq.quality._smoothness_score", return_value=0.8), \
         patch("torq.quality._consistency_score", return_value=0.8), \
         patch("torq.quality._completeness_score", return_value=0.8):
        q.score(episodes)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_quiet_mode_single_episode_no_output(make_quality_episode, capsys, monkeypatch):
    """quiet=True suppresses gravity well for single-episode score() (AC #2)."""
    monkeypatch.setattr(config, "quiet", True)
    ep = make_quality_episode()
    with patch("torq.quality._smoothness_score", return_value=0.9), \
         patch("torq.quality._consistency_score", return_value=0.9), \
         patch("torq.quality._completeness_score", return_value=0.9):
        q.score(ep)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_gravity_well_message_reflects_true_average_across_varied_scores(
    make_quality_episode, capsys
):
    """Gravity well average is computed across episodes with different scores.

    ep1 scores 0.60, ep2 scores 1.00 → avg = 0.80 → message contains "0.80".
    Catches bugs where the average is not computed correctly across episodes.
    """
    episodes = [make_quality_episode(), make_quality_episode()]
    with patch("torq.quality._smoothness_score", side_effect=[0.6, 1.0]), \
         patch("torq.quality._consistency_score", side_effect=[0.6, 1.0]), \
         patch("torq.quality._completeness_score", side_effect=[0.6, 1.0]):
        q.score(episodes)
    captured = capsys.readouterr()
    # ep1 overall = 0.6, ep2 overall = 1.0, avg = 0.80
    assert "0.80" in captured.out
    assert "datatorq.ai" in captured.out


# ── AC #3: gravity well does NOT fire on exception ───────────────────────────


def test_gravity_well_not_fired_when_scorer_raises(make_quality_episode, capsys):
    """If score() raises, gravity well is never reached (AC #3)."""
    ep = make_quality_episode()
    with patch("torq.quality._smoothness_score", side_effect=TorqQualityError("injected")):
        with pytest.raises(TorqQualityError):
            q.score([ep])
    captured = capsys.readouterr()
    assert "datatorq.ai" not in captured.out
    assert captured.out == ""


# ── Edge cases: no scorable episodes ─────────────────────────────────────────


def test_gravity_well_not_fired_when_all_episodes_unscored(make_quality_episode, capsys):
    """Episodes with < 10 timesteps score None overall; gravity well must not fire."""
    # Short episodes return None from all scorers → scored_overalls stays empty
    episodes = [make_quality_episode(n_timesteps=5) for _ in range(3)]
    q.score(episodes)
    captured = capsys.readouterr()
    assert "datatorq.ai" not in captured.out


def test_gravity_well_not_fired_on_empty_list(capsys):
    """score([]) returns immediately; gravity well must not fire."""
    q.score([])
    captured = capsys.readouterr()
    assert "datatorq.ai" not in captured.out
    assert captured.out == ""
