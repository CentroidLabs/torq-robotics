"""Unit tests for tq.quality.report() — quality distribution reporting.

Tests cover:
- Basic distribution output (min, max, mean, median, std present + exact values)
- Outlier detection (episode ID appears + sigma formatting verified)
- Zero variance (no exception, notes zero variance)
- < 3 episodes (stats shown + warning logged)
- 0 scored episodes (prints message, no exception)
- Unscored episodes silently excluded from stats
- Return value is None
- Output goes to stdout (captured via capsys)
- Validation: tuple and single Episode input rejected
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

import torq as tq
from torq.errors import TorqQualityError
from torq.quality.report import QualityReport

_DEFAULT_WEIGHTS = {"smoothness": 0.40, "consistency": 0.35, "completeness": 0.25}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _attach_score(ep, score: float | None):
    """Attach a QualityReport to an episode, optionally with a None overall."""
    ep.quality = QualityReport(
        smoothness=score if score is not None else 0.5,
        consistency=score if score is not None else 0.5,
        completeness=score if score is not None else 0.5,
        weights=_DEFAULT_WEIGHTS,
    )
    if score is None:
        object.__setattr__(ep.quality, "overall", None)
    return ep


def _scored(make_quality_episode, score: float, episode_id: str = "ep_test"):
    """Create a scored episode using the conftest make_quality_episode fixture."""
    ep = make_quality_episode()
    # Override episode_id via object.__setattr__ since it may be locked
    object.__setattr__(ep, "episode_id", episode_id)
    _attach_score(ep, score)
    return ep


# ── Basic output ──────────────────────────────────────────────────────────────

def test_basic_distribution_output_exact_values(make_quality_episode, capsys):
    """10 episodes with deterministic scores — verify exact computed values in output."""
    scores = [0.5, 0.6, 0.65, 0.7, 0.72, 0.75, 0.78, 0.8, 0.85, 0.9]
    eps = [_scored(make_quality_episode, s, f"ep_{i:02d}") for i, s in enumerate(scores)]
    tq.quality.report(eps)
    out = capsys.readouterr().out

    arr = np.array(scores)
    assert f"{np.min(arr):.3f}" in out        # Min : 0.500
    assert f"{np.max(arr):.3f}" in out        # Max : 0.900
    assert f"{np.mean(arr):.3f}" in out       # Mean
    assert f"{np.median(arr):.3f}" in out     # Median
    assert f"{np.std(arr, ddof=0):.3f}" in out  # Std Dev
    assert "10 / 10" in out
    assert "Min" in out
    assert "Max" in out
    assert "Mean" in out
    assert "Median" in out
    assert "Std Dev" in out


def test_output_goes_to_stdout_not_stderr(make_quality_episode, capsys):
    """The report is printed to stdout, not stderr."""
    ep = _scored(make_quality_episode, 0.8, "ep_0")
    tq.quality.report([ep])
    captured = capsys.readouterr()
    assert "Quality Distribution Report" in captured.out
    assert captured.err == ""


# ── Outlier detection ─────────────────────────────────────────────────────────

def test_outlier_episode_appears_in_output(make_quality_episode, capsys):
    """A low-score outlier in a high-scoring group appears by episode ID."""
    eps = [_scored(make_quality_episode, 0.9, f"ep_{i:02d}") for i in range(9)]
    eps.append(_scored(make_quality_episode, 0.01, "outlier_episode"))
    tq.quality.report(eps)
    out = capsys.readouterr().out
    assert "outlier_episode" in out


def test_outlier_sigma_formatting(make_quality_episode, capsys):
    """Outlier lines include the sigma deviation string with sign and σ symbol."""
    eps = [_scored(make_quality_episode, 0.9, f"ep_{i:02d}") for i in range(9)]
    eps.append(_scored(make_quality_episode, 0.01, "outlier_episode"))
    tq.quality.report(eps)
    out = capsys.readouterr().out

    # The outlier line must contain the sigma character and a deviation value
    outlier_lines = [line for line in out.splitlines() if "outlier_episode" in line]
    assert len(outlier_lines) == 1, f"Expected 1 outlier line, got: {outlier_lines}"
    line = outlier_lines[0]
    assert "σ" in line, f"Expected sigma symbol in outlier line: {line!r}"
    # Should contain either + or − prefix before the sigma value
    assert ("+" in line or "\u2212" in line), f"Expected sign prefix in outlier line: {line!r}"


def test_no_outliers_reported_when_scores_clustered(make_quality_episode, capsys):
    """When no episode deviates > 2σ, report shows no outliers."""
    eps = [_scored(make_quality_episode, s, f"ep_{i}") for i, s in
           enumerate([0.75, 0.76, 0.77, 0.78, 0.79])]
    tq.quality.report(eps)
    out = capsys.readouterr().out
    assert "none" in out.lower()


# ── Zero variance ─────────────────────────────────────────────────────────────

def test_zero_variance_no_exception(make_quality_episode, capsys):
    """All episodes with identical scores: no exception AND zero variance noted."""
    eps = [_scored(make_quality_episode, 0.8, f"ep_{i}") for i in range(5)]
    tq.quality.report(eps)
    out = capsys.readouterr().out
    # Must contain BOTH the zero variance message AND 0.000 std — use 'and' not 'or'
    assert "zero variance" in out.lower() and "0.000" in out


def test_zero_variance_std_shown_as_zero(make_quality_episode, capsys):
    """Std Dev is 0.000 when all scores identical."""
    eps = [_scored(make_quality_episode, 0.8, f"ep_{i}") for i in range(5)]
    tq.quality.report(eps)
    out = capsys.readouterr().out
    assert "0.000" in out


# ── < 3 episodes ──────────────────────────────────────────────────────────────

def test_one_episode_shows_stats_and_logs_warning(make_quality_episode, capsys, caplog):
    """1 scored episode: stats shown AND warning logged about unreliability."""
    ep = _scored(make_quality_episode, 0.75, "ep_solo")
    with caplog.at_level(logging.WARNING, logger="torq.quality"):
        tq.quality.report([ep])
    out = capsys.readouterr().out
    assert "0.750" in out  # min/max/mean/median all equal score
    assert any("unreliable" in r.message or "fewer than 3" in r.message for r in caplog.records)


def test_two_episodes_shows_stats_and_logs_warning(make_quality_episode, capsys, caplog):
    """2 scored episodes: stats shown AND warning logged about unreliability."""
    eps = [_scored(make_quality_episode, 0.6, "ep_a"), _scored(make_quality_episode, 0.8, "ep_b")]
    with caplog.at_level(logging.WARNING, logger="torq.quality"):
        tq.quality.report(eps)
    out = capsys.readouterr().out
    assert "2 / 2" in out
    assert any("unreliable" in r.message or "fewer than 3" in r.message for r in caplog.records)


# ── 0 scored episodes ─────────────────────────────────────────────────────────

def test_zero_scored_episodes_prints_message_no_exception(make_quality_episode, capsys):
    """0 scored episodes: prints a message and returns without exception."""
    eps = [make_quality_episode(), make_quality_episode()]
    tq.quality.report(eps)  # must not raise
    out = capsys.readouterr().out
    assert "0 / 2" in out


def test_empty_list_no_exception(capsys):
    """Empty list: no exception, prints header."""
    tq.quality.report([])
    out = capsys.readouterr().out
    assert "Quality Distribution Report" in out


# ── Unscored episodes silently excluded ───────────────────────────────────────

def test_unscored_episodes_excluded_from_stats(make_quality_episode, capsys):
    """Episodes with quality=None are excluded from stats but counted in total."""
    ep_good = _scored(make_quality_episode, 0.9, "ep_good")
    ep_bad = make_quality_episode()  # quality is None
    tq.quality.report([ep_good, ep_bad])
    out = capsys.readouterr().out
    assert "1 / 2" in out
    assert "0.900" in out


def test_none_overall_excluded_from_stats(make_quality_episode, capsys):
    """Episodes with quality.overall=None are excluded from statistics."""
    ep_scored = _scored(make_quality_episode, 0.8, "ep_good")
    ep_none = _scored(make_quality_episode, 0.5, "ep_none_overall")
    object.__setattr__(ep_none.quality, "overall", None)

    tq.quality.report([ep_scored, ep_none])
    out = capsys.readouterr().out
    assert "1 / 2" in out


# ── Return value ──────────────────────────────────────────────────────────────

def test_return_value_is_none(make_quality_episode):
    """report() returns None."""
    ep = _scored(make_quality_episode, 0.8, "ep_0")
    result = tq.quality.report([ep])
    assert result is None


# ── Validation ────────────────────────────────────────────────────────────────

def test_non_list_string_raises():
    """Passing a string raises TorqQualityError."""
    with pytest.raises(TorqQualityError, match="episodes must be a list"):
        tq.quality.report("not_a_list")  # type: ignore[arg-type]


def test_tuple_raises(make_quality_episode):
    """Passing a tuple instead of a list raises TorqQualityError."""
    ep = _scored(make_quality_episode, 0.8, "ep_0")
    with pytest.raises(TorqQualityError, match="episodes must be a list"):
        tq.quality.report((ep,))  # type: ignore[arg-type]


def test_single_episode_raises(make_quality_episode):
    """Passing a single Episode (common user mistake) raises TorqQualityError."""
    ep = _scored(make_quality_episode, 0.8, "ep_0")
    with pytest.raises(TorqQualityError, match="episodes must be a list"):
        tq.quality.report(ep)  # type: ignore[arg-type]
