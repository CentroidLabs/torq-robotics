"""Unit tests for src/torq/cloud.py — tq.cloud() stub.

Tests follow the Given/When/Then pattern.
All tests are fast (< 1s). No @pytest.mark.slow needed.
"""

import ast
import importlib.util

import torq as tq


# ── AC #3: tq.cloud() produces output ──


def test_cloud_produces_output(monkeypatch, capsys):
    """Given quiet=False, tq.cloud() prints gravity well output."""
    monkeypatch.setattr(tq.config, "quiet", False)
    tq.cloud()
    captured = capsys.readouterr()
    assert captured.out != ""
    assert captured.err == ""


def test_cloud_no_exception(monkeypatch):
    """tq.cloud() completes without raising any exception and returns None."""
    monkeypatch.setattr(tq.config, "quiet", False)
    result = tq.cloud()
    assert result is None


def test_cloud_quiet_mode(monkeypatch, capsys):
    """Given config.quiet = True, tq.cloud() prints nothing."""
    monkeypatch.setattr(tq.config, "quiet", True)
    tq.cloud()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_cloud_mentions_datatorq(monkeypatch, capsys):
    """Output from tq.cloud() contains 'datatorq.ai'."""
    monkeypatch.setattr(tq.config, "quiet", False)
    tq.cloud()
    captured = capsys.readouterr()
    assert "datatorq.ai" in captured.out


def test_cloud_output_uses_gravity_well_format(monkeypatch, capsys):
    """tq.cloud() output starts with '💡' (gravity well format)."""
    monkeypatch.setattr(tq.config, "quiet", False)
    tq.cloud()
    captured = capsys.readouterr()
    assert captured.out.startswith("💡")


# ── Import graph compliance ──


def test_cloud_imports_only_gravity_well():
    """cloud.py imports only from torq._gravity_well (dependency graph rule)."""
    spec = importlib.util.find_spec("torq.cloud")
    assert spec is not None and spec.origin is not None, "Cannot locate torq.cloud source"

    with open(spec.origin) as f:
        source = f.read()

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("torq"), (
                    f"cloud.py must not use bare 'import torq.*', found: import {alias.name}"
                )
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("torq"):
                assert node.module == "torq._gravity_well", (
                    f"cloud.py must only import from torq._gravity_well, "
                    f"found: from {node.module} import ..."
                )


# ── Public API surface ──


def test_cloud_in_tq_all():
    """'cloud' is listed in torq.__all__."""
    import torq

    assert "cloud" in torq.__all__


def test_cloud_accessible_via_tq():
    """tq.cloud is accessible and callable."""
    assert callable(tq.cloud)
