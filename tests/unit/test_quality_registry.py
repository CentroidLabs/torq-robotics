"""Unit tests for the custom metric registry (Story 3.5, Tasks 1-2, AC #1-3)."""

from __future__ import annotations

import warnings

import pytest

import torq as tq
import torq.quality as q
from torq._config import DEFAULT_QUALITY_WEIGHTS
from torq.errors import TorqQualityError
from torq.quality.registry import _registry


def _dummy_scorer(episode):
    """A minimal scorer that always returns 0.5."""
    return 0.5


@pytest.fixture(autouse=True)
def reset_registry():
    """Ensure registry is reset after each test to prevent cross-test pollution."""
    yield
    q.reset()


# ── AC #1: Weight rescaling ───────────────────────────────────────────────────


def test_register_one_metric_rescales_built_ins_proportionally():
    """Register grip_force at 0.20 → built-in weights scaled by 0.80 (AC #1)."""
    q.register("grip_force", _dummy_scorer, weight=0.20)
    metrics = q.get_metrics()
    assert metrics["smoothness"] == pytest.approx(0.32, rel=1e-6)
    assert metrics["consistency"] == pytest.approx(0.28, rel=1e-6)
    assert metrics["completeness"] == pytest.approx(0.20, rel=1e-6)
    assert metrics["grip_force"] == pytest.approx(0.20, rel=1e-6)


def test_register_one_metric_all_weights_sum_to_one():
    """All weights sum to 1.0 after registering one custom metric."""
    q.register("grip_force", _dummy_scorer, weight=0.20)
    assert sum(q.get_metrics().values()) == pytest.approx(1.0, rel=1e-6)


def test_register_two_custom_metrics_all_weights_sum_to_one():
    """All weights sum to 1.0 after registering two custom metrics."""
    q.register("grip_force", _dummy_scorer, weight=0.20)
    q.register("torque_quality", _dummy_scorer, weight=0.10)
    metrics = q.get_metrics()
    assert len(metrics) == 5
    assert sum(metrics.values()) == pytest.approx(1.0, rel=1e-6)


def test_register_two_custom_metrics_exact_rescaled_values():
    """Verify exact rescaled values after two sequential registrations."""
    q.register("grip_force", _dummy_scorer, weight=0.20)
    q.register("torque_quality", _dummy_scorer, weight=0.10)
    metrics = q.get_metrics()
    assert metrics["smoothness"] == pytest.approx(0.288, rel=1e-5)
    assert metrics["consistency"] == pytest.approx(0.252, rel=1e-5)
    assert metrics["completeness"] == pytest.approx(0.180, rel=1e-5)
    assert metrics["grip_force"] == pytest.approx(0.180, rel=1e-5)
    assert metrics["torque_quality"] == pytest.approx(0.10, rel=1e-5)


# ── AC #2: Re-registration ────────────────────────────────────────────────────


def test_re_registration_emits_user_warning():
    """Re-registering an existing metric emits UserWarning (AC #2)."""
    q.register("grip_force", _dummy_scorer, weight=0.20)
    with pytest.warns(UserWarning, match="Re-registering"):
        q.register("grip_force", _dummy_scorer, weight=0.15)


def test_re_registration_replaces_scorer_without_double_scaling():
    """Re-registration at the same weight does not double-scale existing weights (AC #2)."""
    q.register("grip_force", _dummy_scorer, weight=0.20)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        q.register("grip_force", _dummy_scorer, weight=0.20)
    metrics = q.get_metrics()
    assert sum(metrics.values()) == pytest.approx(1.0, rel=1e-6)
    assert metrics["grip_force"] == pytest.approx(0.20, rel=1e-6)


def test_re_registration_with_different_weight_rescales_correctly():
    """Re-registration at a new weight (0.20 → 0.15) reverses old scale then applies new (AC #2)."""
    q.register("grip_force", _dummy_scorer, weight=0.20)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        q.register("grip_force", _dummy_scorer, weight=0.15)
    metrics = q.get_metrics()
    assert sum(metrics.values()) == pytest.approx(1.0, rel=1e-6)
    assert metrics["grip_force"] == pytest.approx(0.15, rel=1e-6)
    # Built-ins scaled by 0.85 from defaults
    assert metrics["smoothness"] == pytest.approx(0.40 * 0.85, rel=1e-5)
    assert metrics["consistency"] == pytest.approx(0.35 * 0.85, rel=1e-5)
    assert metrics["completeness"] == pytest.approx(0.25 * 0.85, rel=1e-5)


# ── Validation: built-in name collision ──────────────────────────────────────


@pytest.mark.parametrize("builtin_name", ["smoothness", "consistency", "completeness"])
def test_builtin_name_collision_raises_quality_error(builtin_name):
    """Registering with a built-in name raises TorqQualityError."""
    with pytest.raises(TorqQualityError, match="built-in"):
        q.register(builtin_name, _dummy_scorer, weight=0.20)


# ── Validation: callable check ────────────────────────────────────────────────


def test_non_callable_fn_raises_quality_error():
    """Passing a non-callable as fn raises TorqQualityError at registration time."""
    with pytest.raises(TorqQualityError, match="callable"):
        q.register("custom", 42, weight=0.20)  # type: ignore[arg-type]


def test_non_callable_string_raises_quality_error():
    """Passing a string as fn raises TorqQualityError."""
    with pytest.raises(TorqQualityError, match="callable"):
        q.register("custom", "not_a_fn", weight=0.20)  # type: ignore[arg-type]


# ── Validation: weight bounds ─────────────────────────────────────────────────


def test_weight_zero_raises_quality_error():
    """weight=0.0 is not strictly in (0, 1) — must raise TorqQualityError."""
    with pytest.raises(TorqQualityError):
        q.register("custom", _dummy_scorer, weight=0.0)


def test_weight_one_raises_quality_error():
    """weight=1.0 is not strictly in (0, 1) — must raise TorqQualityError."""
    with pytest.raises(TorqQualityError):
        q.register("custom", _dummy_scorer, weight=1.0)


def test_weight_greater_than_one_raises_quality_error():
    """weight > 1.0 raises TorqQualityError."""
    with pytest.raises(TorqQualityError):
        q.register("custom", _dummy_scorer, weight=1.5)


def test_weight_negative_raises_quality_error():
    """Negative weight raises TorqQualityError."""
    with pytest.raises(TorqQualityError):
        q.register("custom", _dummy_scorer, weight=-0.1)


# ── get_metrics() ────────────────────────────────────────────────────────────


def test_get_metrics_returns_all_built_ins_initially():
    """get_metrics() returns the three built-in metrics before any registration."""
    metrics = q.get_metrics()
    assert set(metrics.keys()) == {"smoothness", "consistency", "completeness"}


def test_get_metrics_includes_custom_after_register():
    """get_metrics() includes custom metrics and their weights after register."""
    q.register("grip_force", _dummy_scorer, weight=0.20)
    metrics = q.get_metrics()
    assert "grip_force" in metrics
    assert metrics["grip_force"] == pytest.approx(0.20, rel=1e-6)


# ── reset() ──────────────────────────────────────────────────────────────────


def test_quality_reset_restores_default_weights():
    """q.reset() restores DEFAULT_QUALITY_WEIGHTS exactly."""
    q.register("grip_force", _dummy_scorer, weight=0.20)
    q.reset()
    metrics = q.get_metrics()
    assert metrics == pytest.approx(DEFAULT_QUALITY_WEIGHTS, rel=1e-6)


def test_quality_reset_removes_custom_metrics():
    """q.reset() clears all custom metrics."""
    q.register("grip_force", _dummy_scorer, weight=0.20)
    q.reset()
    metrics = q.get_metrics()
    assert len(metrics) == 3
    assert "grip_force" not in metrics


# ── AC #3: config.reset_quality_weights() also clears registry ───────────────


def test_config_reset_quality_weights_also_clears_registry():
    """tq.config.reset_quality_weights() resets both config weights and registry (AC #3)."""
    q.register("grip_force", _dummy_scorer, weight=0.20)
    assert _registry.has_custom_metrics()
    tq.config.reset_quality_weights()
    assert not _registry.has_custom_metrics()
    metrics = q.get_metrics()
    assert metrics == pytest.approx(DEFAULT_QUALITY_WEIGHTS, rel=1e-6)


def test_config_reset_quality_weights_restores_config_weights():
    """tq.config.reset_quality_weights() restores tq.config.quality_weights to defaults (AC #3)."""
    q.register("grip_force", _dummy_scorer, weight=0.20)
    tq.config.reset_quality_weights()
    assert tq.config.quality_weights == pytest.approx(DEFAULT_QUALITY_WEIGHTS, rel=1e-6)


# ── has_custom_metrics() ──────────────────────────────────────────────────────


def test_has_custom_metrics_false_initially():
    """has_custom_metrics() returns False before any registration."""
    assert not _registry.has_custom_metrics()


def test_has_custom_metrics_true_after_register():
    """has_custom_metrics() returns True after registration."""
    q.register("grip_force", _dummy_scorer, weight=0.20)
    assert _registry.has_custom_metrics()


def test_has_custom_metrics_false_after_reset():
    """has_custom_metrics() returns False after reset."""
    q.register("grip_force", _dummy_scorer, weight=0.20)
    q.reset()
    assert not _registry.has_custom_metrics()
