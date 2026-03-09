"""Unit tests for src/torq/_config.py — Config singleton.

Tests follow the Given/When/Then pattern from TESTING.md.
All tests are fast (< 1s). No @pytest.mark.slow needed.
"""

import os
from unittest.mock import patch

import pytest

from torq._config import Config, DEFAULT_QUALITY_WEIGHTS, config
from torq.errors import TorqConfigError


# ── AC #2 & #1: quiet property ──


def test_quiet_defaults_to_false():
    """Given TORQ_QUIET not set, config.quiet is False."""
    with patch.dict(os.environ, {}, clear=True):
        fresh = Config()
        assert fresh.quiet is False


def test_quiet_settable():
    """Given config.quiet = True, reading it returns True."""
    with patch.dict(os.environ, {}, clear=True):
        fresh = Config()
        fresh.quiet = True
        assert fresh.quiet is True


def test_torq_quiet_env_var():
    """Given TORQ_QUIET=1 set before import, config.quiet is True."""
    with patch.dict(os.environ, {"TORQ_QUIET": "1"}):
        fresh = Config()
        assert fresh.quiet is True


def test_torq_quiet_env_var_zero():
    """Given TORQ_QUIET=0 set, config.quiet is False."""
    with patch.dict(os.environ, {"TORQ_QUIET": "0"}):
        fresh = Config()
        assert fresh.quiet is False


def test_quiet_set_false_after_true():
    """Setting quiet back to False works."""
    with patch.dict(os.environ, {}, clear=True):
        fresh = Config()
        fresh.quiet = True
        fresh.quiet = False
        assert fresh.quiet is False


# ── AC #4: default quality_weights ──


def test_default_quality_weights():
    """config.quality_weights returns DEFAULT_QUALITY_WEIGHTS by default."""
    with patch.dict(os.environ, {}, clear=True):
        fresh = Config()
        assert fresh.quality_weights == DEFAULT_QUALITY_WEIGHTS


def test_default_quality_weights_values():
    """DEFAULT_QUALITY_WEIGHTS has the exact values from architecture."""
    assert DEFAULT_QUALITY_WEIGHTS["smoothness"] == pytest.approx(0.40)
    assert DEFAULT_QUALITY_WEIGHTS["consistency"] == pytest.approx(0.35)
    assert DEFAULT_QUALITY_WEIGHTS["completeness"] == pytest.approx(0.25)


# ── AC #3: quality_weights validation ──


def test_set_valid_quality_weights():
    """Setting weights that sum exactly to 1.0 succeeds."""
    with patch.dict(os.environ, {}, clear=True):
        fresh = Config()
        valid = {"smoothness": 0.5, "consistency": 0.3, "completeness": 0.2}
        fresh.quality_weights = valid
        assert fresh.quality_weights == valid


def test_set_weights_near_tolerance_low():
    """Weights summing to 0.999 (within -0.001 tolerance) are accepted."""
    with patch.dict(os.environ, {}, clear=True):
        fresh = Config()
        near = {"smoothness": 0.333, "consistency": 0.333, "completeness": 0.333}  # sum = 0.999
        fresh.quality_weights = near  # must not raise


def test_set_weights_near_tolerance_high():
    """Weights summing to ~1.0005 (within +0.001 tolerance) are accepted."""
    with patch.dict(os.environ, {}, clear=True):
        fresh = Config()
        near = {"smoothness": 0.5005, "consistency": 0.5}  # sum ≈ 1.0005
        fresh.quality_weights = near  # must not raise


def test_set_invalid_quality_weights():
    """Weights not summing to 1.0 raise TorqConfigError."""
    with patch.dict(os.environ, {}, clear=True):
        fresh = Config()
        bad = {"smoothness": 0.5, "consistency": 0.5, "completeness": 0.5}  # sum = 1.5
        with pytest.raises(TorqConfigError):
            fresh.quality_weights = bad


def test_invalid_weights_error_message():
    """Error message contains actual sum and correction hint."""
    with patch.dict(os.environ, {}, clear=True):
        fresh = Config()
        bad = {"smoothness": 0.3, "consistency": 0.3}  # sum = 0.6
        with pytest.raises(TorqConfigError, match="0.6") as exc_info:
            fresh.quality_weights = bad
        msg = str(exc_info.value)
        # Must include [what] + [why] + [what to try]
        assert "sum to 1.0" in msg
        assert "Adjust" in msg or "adjust" in msg


# ── AC #4: reset_quality_weights ──


def test_reset_quality_weights():
    """After custom set + reset, weights equal DEFAULT_QUALITY_WEIGHTS."""
    with patch.dict(os.environ, {}, clear=True):
        fresh = Config()
        fresh.quality_weights = {"smoothness": 0.6, "consistency": 0.4}
        fresh.reset_quality_weights()
        assert fresh.quality_weights == DEFAULT_QUALITY_WEIGHTS


# ── Encapsulation: getter returns copy ──


def test_quality_weights_returns_copy():
    """Modifying the returned dict does not change internal state."""
    with patch.dict(os.environ, {}, clear=True):
        fresh = Config()
        weights = fresh.quality_weights
        weights["smoothness"] = 0.99
        assert fresh.quality_weights["smoothness"] == pytest.approx(0.40)


# ── Singleton behaviour ──


def test_config_singleton():
    """module-level config is the same object across imports."""
    from torq._config import config as c1
    from torq._config import config as c2

    assert c1 is c2


def test_config_accessible_via_tq():
    """tq.config refers to the same singleton as torq._config.config."""
    import torq as tq

    assert tq.config is config


# ── Dependency rule ──


def test_config_module_imports():
    """_config.py imports only from torq.errors (dependency rule enforced)."""
    import ast
    import importlib.util

    spec = importlib.util.find_spec("torq._config")
    assert spec is not None and spec.origin is not None, "Cannot locate torq._config source"
    with open(spec.origin) as f:
        source = f.read()

    # Use AST to find real import statements (not docstring content)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("torq"), (
                    f"_config.py must not use bare 'import torq.*', found: import {alias.name}"
                )
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("torq"):
                assert node.module == "torq.errors", (
                    f"_config.py must only import from torq.errors, found: from {node.module} import ..."
                )


# ── M2: unknown key validation ──


def test_set_unknown_weight_key_raises_config_error():
    """Setting weights with an unrecognised dimension key raises TorqConfigError."""
    with patch.dict(os.environ, {}, clear=True):
        fresh = Config()
        with pytest.raises(TorqConfigError, match="Unknown quality weight dimension"):
            fresh.quality_weights = {"smoothness": 0.5, "unknown_dim": 0.5}


def test_set_all_unknown_keys_raises_config_error():
    """Weights with entirely arbitrary keys raise TorqConfigError."""
    with patch.dict(os.environ, {}, clear=True):
        fresh = Config()
        with pytest.raises(TorqConfigError, match="Valid dimensions are"):
            fresh.quality_weights = {"foo": 0.6, "bar": 0.4}


# ── M3: non-numeric weight values ──


def test_set_non_numeric_weights_raises_config_error():
    """Non-numeric weight values raise TorqConfigError (not bare TypeError)."""
    with patch.dict(os.environ, {}, clear=True):
        fresh = Config()
        with pytest.raises(TorqConfigError, match="must be numeric"):
            fresh.quality_weights = {"smoothness": "high", "consistency": 0.5}


def test_set_none_weight_raises_config_error():
    """None weight value raises TorqConfigError (not bare TypeError)."""
    with patch.dict(os.environ, {}, clear=True):
        fresh = Config()
        with pytest.raises(TorqConfigError):
            fresh.quality_weights = {"smoothness": None, "consistency": 0.5}


# ── L1: __repr__ ──


def test_config_repr_contains_quiet_and_weights():
    """repr(config) includes quiet status and current weights."""
    with patch.dict(os.environ, {}, clear=True):
        fresh = Config()
        r = repr(fresh)
        assert "quiet" in r
        assert "quality_weights" in r or "smoothness" in r


def test_config_repr_reflects_state():
    """repr(config) reflects the current quiet value."""
    with patch.dict(os.environ, {}, clear=True):
        fresh = Config()
        fresh.quiet = True
        assert "True" in repr(fresh)


# ── L4: boundary-rejection ──


def test_weights_just_below_tolerance_rejected():
    """Weights summing to 0.998 (outside -0.001 tolerance) raise TorqConfigError."""
    with patch.dict(os.environ, {}, clear=True):
        fresh = Config()
        with pytest.raises(TorqConfigError, match="sum to 1.0"):
            fresh.quality_weights = {"smoothness": 0.499, "consistency": 0.499}  # sum = 0.998


def test_weights_just_above_tolerance_rejected():
    """Weights summing to 1.002 (outside +0.001 tolerance) raise TorqConfigError."""
    with patch.dict(os.environ, {}, clear=True):
        fresh = Config()
        with pytest.raises(TorqConfigError, match="sum to 1.0"):
            # sum = 1.002, clearly outside tolerance
            fresh.quality_weights = {"smoothness": 0.502, "consistency": 0.5}  # sum = 1.002
