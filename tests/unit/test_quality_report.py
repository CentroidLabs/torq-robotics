"""Unit tests for QualityReport dataclass and tq.quality.score() entry point.

Tests cover:
- Composite scoring formula
- None propagation
- Single Episode and list[Episode] scoring
- Per-call weight override validation
- Empty list fast path
- tqdm progress bar behaviour
- Gravity well fires / does not fire
- Custom metric registration and weight rescaling

Uses shared ``make_quality_episode`` fixture from tests/conftest.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import torq
import torq.quality as tq_quality
from torq._config import DEFAULT_QUALITY_WEIGHTS
from torq.errors import TorqQualityError
from torq.quality.report import QualityReport


# ── QualityReport dataclass ────────────────────────────────────────────────────

class TestQualityReportComposite:
    """Test the weighted composite formula in __post_init__."""

    def test_composite_default_weights(self):
        """overall = s×0.40 + c×0.35 + co×0.25 with DEFAULT_QUALITY_WEIGHTS."""
        report = QualityReport(smoothness=0.8, consistency=0.6, completeness=1.0)
        expected = 0.8 * 0.40 + 0.6 * 0.35 + 1.0 * 0.25
        assert report.overall == pytest.approx(expected, abs=1e-6)

    def test_composite_custom_weights(self):
        """Weights passed to QualityReport override DEFAULT_QUALITY_WEIGHTS."""
        w = {"smoothness": 0.5, "consistency": 0.3, "completeness": 0.2}
        report = QualityReport(smoothness=1.0, consistency=1.0, completeness=1.0, weights=w)
        assert report.overall == pytest.approx(1.0, abs=1e-6)

    def test_composite_extreme_values(self):
        """Boundary scores 0.0 and 1.0 are handled correctly."""
        report = QualityReport(smoothness=0.0, consistency=0.0, completeness=0.0)
        assert report.overall == pytest.approx(0.0, abs=1e-6)

        report2 = QualityReport(smoothness=1.0, consistency=1.0, completeness=1.0)
        assert report2.overall == pytest.approx(1.0, abs=1e-6)

    def test_overall_rounded_to_6_decimals(self):
        """overall is rounded to 6 decimal places."""
        report = QualityReport(smoothness=1 / 3, consistency=1 / 3, completeness=1 / 3)
        # Should be rounded, not have more than 6 sig figs after decimal.
        assert report.overall is not None
        assert report.overall == round(
            (1 / 3) * 0.40 + (1 / 3) * 0.35 + (1 / 3) * 0.25, 6
        )

    def test_frozen_immutable(self):
        """QualityReport fields cannot be mutated after creation."""
        report = QualityReport(smoothness=0.5, consistency=0.5, completeness=0.5)
        with pytest.raises(Exception):  # FrozenInstanceError
            report.smoothness = 0.9  # type: ignore[misc]


class TestQualityReportNonePropagation:
    """Test that None in any component produces overall=None."""

    def test_smoothness_none(self):
        report = QualityReport(smoothness=None, consistency=0.8, completeness=1.0)
        assert report.overall is None

    def test_consistency_none(self):
        report = QualityReport(smoothness=0.8, consistency=None, completeness=1.0)
        assert report.overall is None

    def test_completeness_none(self):
        report = QualityReport(smoothness=0.8, consistency=0.8, completeness=None)
        assert report.overall is None

    def test_all_none(self):
        report = QualityReport(smoothness=None, consistency=None, completeness=None)
        assert report.overall is None

    def test_partial_none_no_partial_composite(self):
        """Even when two components are valid, overall is None if one is None."""
        report = QualityReport(smoothness=1.0, consistency=1.0, completeness=None)
        # Must be None, not 1.0*0.40 + 1.0*0.35 = 0.75
        assert report.overall is None


# ── tq.quality.score() entry point ────────────────────────────────────────────

class TestScoreSingleEpisode:
    """Test scoring a single Episode (not in a list)."""

    def test_single_episode_in_out_same_identity(self, make_quality_episode):
        ep = make_quality_episode()
        result = tq_quality.score(ep)
        assert result is ep

    def test_single_episode_quality_populated(self, make_quality_episode):
        ep = make_quality_episode()
        tq_quality.score(ep)
        assert ep.quality is not None
        assert isinstance(ep.quality, QualityReport)

    def test_single_episode_overall_is_float(self, make_quality_episode):
        ep = make_quality_episode()
        tq_quality.score(ep)
        assert isinstance(ep.quality.overall, float)


class TestScoreList:
    """Test scoring a list of Episodes."""

    def test_list_same_identity(self, make_quality_episode):
        eps = [make_quality_episode() for _ in range(3)]
        result = tq_quality.score(eps)
        assert result is eps

    def test_all_episodes_populated(self, make_quality_episode):
        eps = [make_quality_episode() for _ in range(5)]
        tq_quality.score(eps)
        for ep in eps:
            assert ep.quality is not None
            assert ep.quality.overall is not None

    def test_episode_object_identity_preserved(self, make_quality_episode):
        """Confirm in-place mutation — same Episode objects, not copies."""
        eps = [make_quality_episode() for _ in range(2)]
        ids_before = [id(ep) for ep in eps]
        tq_quality.score(eps)
        ids_after = [id(ep) for ep in eps]
        assert ids_before == ids_after


class TestScoreEmptyList:
    """Test that score([]) returns immediately with no side effects."""

    def test_empty_list_returns_empty(self):
        result = tq_quality.score([])
        assert result == []

    def test_empty_list_no_gravity_well(self):
        with patch("torq.quality._gravity_well") as mock_gw:
            tq_quality.score([])
            mock_gw.assert_not_called()

    def test_empty_list_no_tqdm(self):
        """Empty list short-circuits before tqdm is ever called."""
        with patch("torq.quality.tqdm") as mock_tqdm:
            result = tq_quality.score([])
            assert result == []
            mock_tqdm.assert_not_called()


class TestInputTypeValidation:
    """Test early type-checking for the episodes parameter."""

    def test_string_input_raises_quality_error(self):
        with pytest.raises(TorqQualityError, match="Episode or list"):
            tq_quality.score("not_an_episode")  # type: ignore[arg-type]

    def test_int_input_raises_quality_error(self):
        with pytest.raises(TorqQualityError, match="Episode or list"):
            tq_quality.score(42)  # type: ignore[arg-type]

    def test_none_input_raises_quality_error(self):
        with pytest.raises(TorqQualityError, match="Episode or list"):
            tq_quality.score(None)  # type: ignore[arg-type]


class TestWeightOverride:
    """Test per-call weight override behaviour."""

    def test_valid_weights_used_for_call(self, make_quality_episode):
        ep = make_quality_episode(np.ones((30, 3)))
        w = {"smoothness": 0.5, "consistency": 0.3, "completeness": 0.2}
        tq_quality.score(ep, weights=w)
        assert ep.quality is not None
        assert ep.quality.overall is not None
        # With all-ones actions (constant trajectory), smoothness ≈ 1.0,
        # consistency ≈ 1.0 — we just check overall is a valid float.
        assert 0.0 <= ep.quality.overall <= 1.0

    def test_global_config_unchanged_after_per_call(self, make_quality_episode):
        import torq as tq
        original_weights = tq.config.quality_weights.copy()
        ep = make_quality_episode()
        w = {"smoothness": 0.5, "consistency": 0.3, "completeness": 0.2}
        tq_quality.score(ep, weights=w)
        assert tq.config.quality_weights == original_weights

    def test_invalid_weights_raise_before_scoring(self, make_quality_episode):
        ep = make_quality_episode()
        bad_weights = {"smoothness": 0.5, "consistency": 0.3, "completeness": 0.5}
        with pytest.raises(TorqQualityError, match="sum to 1.0"):
            tq_quality.score(ep, weights=bad_weights)
        # Episode should NOT be scored (error raised before any scoring).
        assert ep.quality is None

    def test_weights_not_summing_raises_exact_message(self, make_quality_episode):
        ep = make_quality_episode()
        bad_weights = {"smoothness": 0.6, "consistency": 0.6, "completeness": 0.6}
        with pytest.raises(TorqQualityError) as exc_info:
            tq_quality.score(ep, weights=bad_weights)
        assert "1.0" in str(exc_info.value)

    def test_weights_summing_within_tolerance_accepted(self, make_quality_episode):
        ep = make_quality_episode()
        # Exactly 1.001 — within tolerance.
        w = {"smoothness": 0.401, "consistency": 0.35, "completeness": 0.25}
        # Should not raise (sum = 1.001 which is ±0.001 boundary).
        tq_quality.score(ep, weights=w)

    def test_weights_at_boundary_raises(self, make_quality_episode):
        ep = make_quality_episode()
        # 1.002 — just outside tolerance.
        w = {"smoothness": 0.402, "consistency": 0.35, "completeness": 0.25}
        with pytest.raises(TorqQualityError):
            tq_quality.score(ep, weights=w)

    def test_missing_weight_keys_raises(self, make_quality_episode):
        """Weights dict missing required keys raises TorqQualityError with clear message."""
        ep = make_quality_episode()
        bad_weights = {"foo": 0.5, "bar": 0.5}
        with pytest.raises(TorqQualityError, match="missing required keys"):
            tq_quality.score(ep, weights=bad_weights)  # type: ignore[arg-type]

    def test_unknown_extra_keys_allowed(self, make_quality_episode):
        """Extra keys beyond the three required are tolerated (ignored by report)."""
        ep = make_quality_episode()
        # Extra keys don't block scoring as long as required keys are present and sum OK.
        w = {"smoothness": 0.4, "consistency": 0.35, "completeness": 0.25, "custom": 0.0}
        # sum = 1.0 — should not raise (extra key 'custom' is harmless for built-ins)
        tq_quality.score(ep, weights=w)


class TestGravityWell:
    """Test gravity well fires on success and not on failure."""

    def test_gravity_well_fires_on_success(self, make_quality_episode):
        ep = make_quality_episode()
        with patch("torq.quality._gravity_well") as mock_gw:
            tq_quality.score([ep])
            mock_gw.assert_called_once()
            call_args = mock_gw.call_args
            assert "Average quality" in call_args[0][0]
            assert call_args[0][1] == "GW-SDK-01"

    def test_gravity_well_not_fired_on_exception(self, make_quality_episode):
        ep = make_quality_episode()
        with patch("torq.quality._gravity_well") as mock_gw:
            with patch("torq.quality._smoothness_score", side_effect=RuntimeError("oops")):
                with pytest.raises(TorqQualityError):
                    tq_quality.score([ep])
            mock_gw.assert_not_called()

    def test_gravity_well_message_contains_avg_score(self, make_quality_episode):
        ep = make_quality_episode()
        messages = []
        with patch("torq.quality._gravity_well", side_effect=lambda msg, feat: messages.append(msg)):
            tq_quality.score([ep])
        assert len(messages) == 1
        assert "Average quality:" in messages[0]

    def test_gravity_well_not_fired_for_empty_list(self):
        with patch("torq.quality._gravity_well") as mock_gw:
            tq_quality.score([])
            mock_gw.assert_not_called()


class TestTqdmProgressBar:
    """Test tqdm progress bar behaviour."""

    def test_progress_bar_disabled_for_single_episode(self, make_quality_episode):
        ep = make_quality_episode()
        with patch("torq.quality.tqdm", side_effect=lambda it, **kw: it) as mock_tqdm:
            tq_quality.score(ep)
            # tqdm must be called — with disable=True for a single episode.
            assert mock_tqdm.called, "tqdm should be called even for single episode"
            _, kwargs = mock_tqdm.call_args
            assert kwargs.get("disable") is True

    def test_progress_bar_uses_quiet_config(self, make_quality_episode):
        import torq as tq
        eps = [make_quality_episode() for _ in range(3)]
        tq.config.quiet = True
        try:
            with patch("torq.quality.tqdm", side_effect=lambda it, **kw: it) as mock_tqdm:
                tq_quality.score(eps)
                assert mock_tqdm.called
                _, kwargs = mock_tqdm.call_args
                assert kwargs.get("disable") is True
        finally:
            tq.config.quiet = False


class TestRegistryIntegration:
    """Test custom metric registration and weight rescaling via score()."""

    def teardown_method(self, method):
        """Always reset registry after each test to prevent state leakage."""
        tq_quality.reset()

    def test_register_and_get_metrics(self):
        tq_quality.register("grip", lambda ep: 0.9, weight=0.20)
        metrics = tq_quality.get_metrics()
        assert abs(metrics["smoothness"] - 0.32) < 1e-6
        assert abs(metrics["consistency"] - 0.28) < 1e-6
        assert abs(metrics["completeness"] - 0.20) < 1e-6
        assert abs(metrics["grip"] - 0.20) < 1e-6

    def test_reset_restores_defaults(self):
        tq_quality.register("grip", lambda ep: 0.9, weight=0.20)
        tq_quality.reset()
        metrics = tq_quality.get_metrics()
        assert metrics == DEFAULT_QUALITY_WEIGHTS

    def test_builtin_name_collision_raises(self):
        with pytest.raises(TorqQualityError, match="built-in"):
            tq_quality.register("smoothness", lambda ep: 0.5, weight=0.10)

    def test_invalid_weight_raises(self):
        with pytest.raises(TorqQualityError, match="weight must be in"):
            tq_quality.register("grip", lambda ep: 0.5, weight=1.5)

    def test_score_with_custom_metric(self, make_quality_episode):
        """Scoring with a registered custom metric completes without error."""
        tq_quality.register("grip", lambda ep: 0.8, weight=0.10)
        ep = make_quality_episode()
        tq_quality.score([ep])
        assert ep.quality is not None

    def test_custom_metric_out_of_range_raises(self, make_quality_episode):
        """Custom scorer returning value outside [0,1] raises TorqQualityError."""
        tq_quality.register("bad_metric", lambda ep: 1.5, weight=0.10)
        ep = make_quality_episode()
        with pytest.raises(TorqQualityError, match="\\[0.0, 1.0\\]"):
            tq_quality.score([ep])

    def test_custom_metric_returning_none_propagates_to_overall(self, make_quality_episode):
        """Custom scorer returning None must set overall to None."""
        tq_quality.register("grip", lambda ep: None, weight=0.10)
        ep = make_quality_episode()
        tq_quality.score(ep)
        assert ep.quality is not None
        assert ep.quality.overall is None  # None propagation from custom metric

    def test_reregistration_does_not_double_scale(self):
        """Re-registering a metric reverses old contribution before applying new one."""
        tq_quality.register("grip", lambda ep: 0.8, weight=0.20)
        metrics_first = tq_quality.get_metrics()

        # Re-register same metric with same weight.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            tq_quality.register("grip", lambda ep: 0.9, weight=0.20)
        metrics_second = tq_quality.get_metrics()

        # Built-in weights should be same after re-registration with same weight.
        assert abs(metrics_first["smoothness"] - metrics_second["smoothness"]) < 1e-5
        assert abs(metrics_first["consistency"] - metrics_second["consistency"]) < 1e-5
        assert abs(metrics_first["completeness"] - metrics_second["completeness"]) < 1e-5
        # Total must still sum to 1.0.
        assert abs(sum(metrics_second.values()) - 1.0) < 1e-5
