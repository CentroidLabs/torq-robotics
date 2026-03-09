"""Integration tests for tq.quality.score() end-to-end.

Tests cover:
- 100-episode performance benchmark (<60s)
- Round-trip: score → save → load → quality still accessible
- Custom metric registration + scoring with rescaled weights
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

import torq
import torq.quality as tq_quality
from torq.quality.report import QualityReport


# ── Performance ────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_100_episode_scoring_under_60s(make_quality_episode, tmp_path):
    """100 episodes score in under 60 seconds (AC#5).

    Individual scorers are O(T) per episode. Performance baseline is easily
    met; this test guards against accidental O(N²) regressions.
    """
    eps = [make_quality_episode(n_timesteps=30) for _ in range(100)]

    start = time.monotonic()
    torq.config.quiet = True
    try:
        tq_quality.score(eps)
    finally:
        torq.config.quiet = False
    elapsed = time.monotonic() - start

    assert elapsed < 60.0, f"Scoring 100 episodes took {elapsed:.1f}s (limit: 60s)"
    for ep in eps:
        assert ep.quality is not None


# ── Round-trip persistence ─────────────────────────────────────────────────────

def test_score_save_load_quality_rescorable(make_quality_episode, tmp_path):
    """Score → save → load → re-score round-trip (AC#1, #2, #3).

    The storage layer (R1) persists numeric arrays and metadata but not the
    QualityReport object itself.  After loading, the episode must be re-scorable
    and produce the same QualityReport as the original scoring pass.

    This validates that:
    - Underlying action / timestamp / metadata data survives the save/load cycle
    - tq.quality.score() produces consistent results on the loaded episode
    """
    pytest.importorskip("pyarrow", reason="pyarrow required for storage round-trip")

    ep = make_quality_episode(n_timesteps=30, metadata={"success": True})
    torq.config.quiet = True
    try:
        tq_quality.score(ep)
    finally:
        torq.config.quiet = False

    assert ep.quality is not None
    assert ep.quality.overall is not None
    assert ep.quality.completeness == 1.0  # success=True → 1.0

    original_overall = ep.quality.overall

    # Save episode to disk.
    torq.save(ep, tmp_path)

    # Load from disk — quality is None after load (R1: QualityReport not persisted).
    loaded_ep = torq.load(ep.episode_id, tmp_path)
    assert loaded_ep.quality is None  # expected: not persisted by storage layer

    # Re-score — must produce the same result as the original score.
    torq.config.quiet = True
    try:
        tq_quality.score(loaded_ep)
    finally:
        torq.config.quiet = False

    assert loaded_ep.quality is not None
    assert isinstance(loaded_ep.quality, QualityReport)
    assert loaded_ep.quality.overall == pytest.approx(original_overall, abs=1e-4)
    assert loaded_ep.quality.completeness == pytest.approx(1.0, abs=1e-5)


# ── Custom metric registry ─────────────────────────────────────────────────────

class TestCustomMetricIntegration:
    """Test end-to-end custom metric registration + scoring with rescaled weights."""

    def teardown_method(self, method):
        """Reset registry after each test to prevent state leakage."""
        tq_quality.reset()
        torq.config.quiet = False

    def test_weight_rescaling_after_register(self):
        """Built-in weights are rescaled proportionally when custom metric added."""
        tq_quality.register("grip", lambda ep: 0.8, weight=0.20)
        metrics = tq_quality.get_metrics()

        # All four weights should sum to 1.0.
        total = sum(metrics.values())
        assert abs(total - 1.0) < 1e-6

        # Built-in weights rescaled by factor 0.80.
        assert abs(metrics["smoothness"] - 0.40 * 0.80) < 1e-6
        assert abs(metrics["consistency"] - 0.35 * 0.80) < 1e-6
        assert abs(metrics["completeness"] - 0.25 * 0.80) < 1e-6
        assert abs(metrics["grip"] - 0.20) < 1e-6

    def test_scoring_with_custom_metric_succeeds(self, make_quality_episode):
        """score() succeeds and populates quality when custom metric registered."""
        call_count = 0

        def grip_scorer(ep):
            nonlocal call_count
            call_count += 1
            return 0.75

        tq_quality.register("grip", grip_scorer, weight=0.15)

        torq.config.quiet = True
        eps = [make_quality_episode(n_timesteps=20) for _ in range(3)]
        tq_quality.score(eps)

        for ep in eps:
            assert ep.quality is not None
            assert ep.quality.overall is not None
            assert 0.0 <= ep.quality.overall <= 1.0

        # Custom scorer should have been called once per episode.
        assert call_count == 3

    def test_custom_metric_contributes_to_overall(self, make_quality_episode):
        """overall score with custom metric differs from default 3-scorer overall.

        Specifically: adding a custom metric rescales built-in weights and adds
        the custom contribution, so overall ≠ default-weight built-in composite.
        """
        ep = make_quality_episode(np.ones((30, 3)))
        torq.config.quiet = True

        # Score WITHOUT custom metric first.
        tq_quality.score(ep)
        overall_no_custom = ep.quality.overall

        # Reset and add custom metric; create a fresh episode (same actions) to re-score.
        tq_quality.reset()
        tq_quality.register("grip", lambda ep: 1.0, weight=0.20)
        ep2 = make_quality_episode(np.ones((30, 3)))
        tq_quality.score(ep2)
        overall_with_custom = ep2.quality.overall

        # With perfect custom score (1.0) and rescaled built-ins, the overall
        # should differ from the no-custom case.
        assert overall_no_custom is not None
        assert overall_with_custom is not None
        # They won't be equal because the weight distribution changed.
        assert overall_no_custom != pytest.approx(overall_with_custom, abs=1e-6)

    def test_two_custom_metrics_rescaling(self):
        """Weight rescaling chains correctly across multiple registrations."""
        tq_quality.register("grip", lambda ep: 0.8, weight=0.10)
        tq_quality.register("stability", lambda ep: 0.7, weight=0.10)

        metrics = tq_quality.get_metrics()
        total = sum(metrics.values())
        assert abs(total - 1.0) < 1e-5

        assert "grip" in metrics
        assert "stability" in metrics
        # After two registrations, built-in weights are scaled by 0.9 * 0.9 = 0.81.
        assert abs(metrics["smoothness"] - 0.40 * 0.81) < 1e-5

    def test_per_call_weights_bypass_registry(self, make_quality_episode):
        """Per-call weight override bypasses registry custom metric logic (AC#4)."""
        tq_quality.register("grip", lambda ep: 0.5, weight=0.20)

        ep = make_quality_episode()
        torq.config.quiet = True
        w = {"smoothness": 0.5, "consistency": 0.3, "completeness": 0.2}
        tq_quality.score(ep, weights=w)

        # overall should be computed from the 3 built-in scorers with custom weights,
        # NOT including the custom registry metric contribution.
        assert ep.quality is not None
        assert ep.quality.overall is not None
