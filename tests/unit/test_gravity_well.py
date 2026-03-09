"""Unit tests for src/torq/_gravity_well.py — gravity well infrastructure.

Tests follow the Given/When/Then pattern.
All tests are fast (< 1s). No @pytest.mark.slow needed.
"""

import ast
import importlib.util

from torq._config import config
from torq._gravity_well import DATATORQ_URL, _gravity_well


# ── AC #1: output format ──


def test_output_format(monkeypatch, capsys):
    """Given quiet=False, _gravity_well() prints the exact format."""
    monkeypatch.setattr(config, "quiet", False)
    _gravity_well(message="Hello world", feature="GW-TEST-01")
    captured = capsys.readouterr()
    assert captured.out == f"💡 Hello world\n   → {DATATORQ_URL}\n"
    assert captured.err == ""


def test_output_contains_datatorq_url(monkeypatch, capsys):
    """Output always contains the datatorq.ai URL."""
    monkeypatch.setattr(config, "quiet", False)
    _gravity_well(message="Test message", feature="GW-TEST-02")
    captured = capsys.readouterr()
    assert "https://www.datatorq.ai" in captured.out


def test_datatorq_url_constant():
    """DATATORQ_URL constant matches the expected URL."""
    assert DATATORQ_URL == "https://www.datatorq.ai"


def test_message_appears_in_output(monkeypatch, capsys):
    """The message argument appears verbatim in the output."""
    monkeypatch.setattr(config, "quiet", False)
    _gravity_well(message="Custom gravity well message", feature="GW-TEST-03")
    captured = capsys.readouterr()
    assert "Custom gravity well message" in captured.out


# ── AC #2: quiet mode ──


def test_quiet_mode_suppresses_output(monkeypatch, capsys):
    """Given config.quiet = True, _gravity_well() prints nothing."""
    monkeypatch.setattr(config, "quiet", True)
    _gravity_well(message="Should not appear", feature="GW-TEST-04")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_quiet_mode_suppresses_url(monkeypatch, capsys):
    """Given quiet=True, the datatorq.ai URL is NOT printed."""
    monkeypatch.setattr(config, "quiet", True)
    _gravity_well(message="Any message", feature="GW-TEST-05")
    captured = capsys.readouterr()
    assert "datatorq.ai" not in captured.out


# ── Feature parameter acceptance ──


def test_feature_parameter_accepted(monkeypatch):
    """feature parameter is accepted without error for various values."""
    monkeypatch.setattr(config, "quiet", True)  # suppress output; only testing no exception
    _gravity_well(message="msg", feature="GW-SDK-01")
    _gravity_well(message="msg", feature="GW-SDK-04")
    _gravity_well(message="msg", feature="GW-CUSTOM-99")
    _gravity_well(message="msg", feature="")
    # No assertion needed — reaching here means no exception raised


# ── message=None behaviour (documented contract) ──


def test_gravity_well_message_none_documented(monkeypatch, capsys):
    """_gravity_well(message=None) outputs the string 'None' — no runtime type guard in R1.

    This test documents accepted behaviour: Python's f-string silently coerces None to
    the string "None". A runtime TypeError is NOT raised. Future stories may add a guard;
    for R1 this is the observed and accepted contract.
    """
    monkeypatch.setattr(config, "quiet", False)
    _gravity_well(message=None, feature="GW-TEST-NONE")  # type: ignore[arg-type]
    captured = capsys.readouterr()
    assert captured.out == f"💡 None\n   → {DATATORQ_URL}\n"


# ── No network calls ──


def test_no_network_imports():
    """_gravity_well.py does not import socket, urllib, requests, or httpx."""
    spec = importlib.util.find_spec("torq._gravity_well")
    assert spec is not None and spec.origin is not None, "Cannot locate torq._gravity_well source"

    with open(spec.origin) as f:
        source = f.read()

    forbidden = {"socket", "urllib", "requests", "httpx", "http.client"}
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name not in forbidden, (
                    f"_gravity_well.py must not import network modules, found: import {alias.name}"
                )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                assert top not in forbidden, (
                    f"_gravity_well.py must not import network modules, found: from {node.module} ..."
                )


# ── Import graph compliance ──


def test_gravity_well_imports_only_config():
    """_gravity_well.py imports only from torq._config (dependency graph rule)."""
    spec = importlib.util.find_spec("torq._gravity_well")
    assert spec is not None and spec.origin is not None, "Cannot locate torq._gravity_well source"

    with open(spec.origin) as f:
        source = f.read()

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("torq"), (
                    f"_gravity_well.py must not use bare 'import torq.*', found: import {alias.name}"
                )
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("torq"):
                assert node.module == "torq._config", (
                    f"_gravity_well.py must only import from torq._config, "
                    f"found: from {node.module} import ..."
                )
