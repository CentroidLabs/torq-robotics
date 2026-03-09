"""Tests for torq.errors — exception hierarchy and helpers."""

from unittest.mock import patch

import pytest


class TestAllExceptionsImportable:
    def test_all_exceptions_importable(self):
        from torq.errors import (  # noqa: F401
            EpisodeImmutableFieldError,
            TorqComposeError,
            TorqConfigError,
            TorqError,
            TorqImportError,
            TorqIngestError,
            TorqQualityError,
            TorqStorageError,
        )


class TestInheritanceChain:
    def test_torq_error_is_base_exception(self):
        from torq.errors import TorqError

        assert issubclass(TorqError, Exception)

    def test_all_subclasses_inherit_torq_error(self):
        from torq.errors import (
            EpisodeImmutableFieldError,
            TorqComposeError,
            TorqConfigError,
            TorqError,
            TorqImportError,
            TorqIngestError,
            TorqQualityError,
            TorqStorageError,
        )

        subclasses = [
            TorqIngestError,
            TorqStorageError,
            TorqQualityError,
            TorqComposeError,
            TorqConfigError,
            TorqImportError,
            EpisodeImmutableFieldError,
        ]
        for cls in subclasses:
            assert issubclass(cls, TorqError), f"{cls.__name__} is not a subclass of TorqError"

    def test_isinstance_check_for_each_subclass(self):
        from torq.errors import (
            EpisodeImmutableFieldError,
            TorqComposeError,
            TorqConfigError,
            TorqError,
            TorqImportError,
            TorqIngestError,
            TorqQualityError,
            TorqStorageError,
        )

        pairs = [
            (TorqIngestError, "ingest error"),
            (TorqStorageError, "storage error"),
            (TorqQualityError, "quality error"),
            (TorqComposeError, "compose error"),
            (TorqConfigError, "config error"),
            (TorqImportError, "import error"),
            (EpisodeImmutableFieldError, "immutable field error"),
        ]
        for cls, msg in pairs:
            instance = cls(msg)
            assert isinstance(instance, TorqError), (
                f"{cls.__name__} instance not isinstance TorqError"
            )


class TestErrorMessagePreservation:
    def test_error_message_preserved(self):
        from torq.errors import TorqError

        msg = "Episode 'ep_0001' failed to load: file not found. Try re-ingesting the dataset."
        err = TorqError(msg)
        assert str(err) == msg

    def test_subclass_message_preserved(self):
        from torq.errors import TorqIngestError

        msg = "MCAP parse failed at byte 1024: unexpected EOF. Check that file is not truncated."
        err = TorqIngestError(msg)
        assert str(err) == msg


class TestRequireTorch:
    def test_require_torch_raises_import_error_when_torch_missing(self):
        """_require_torch() raises TorqImportError when torch is not installed."""
        from torq.errors import TorqImportError, _require_torch

        with patch.dict("sys.modules", {"torch": None}):
            with pytest.raises(TorqImportError):
                _require_torch()

    def test_require_torch_message_contains_install_hint(self):
        """Error message includes pip install hint."""
        from torq.errors import _require_torch

        with patch.dict("sys.modules", {"torch": None}):
            with pytest.raises(Exception) as exc_info:
                _require_torch()
            assert "pip install torq-robotics[torch]" in str(exc_info.value)

    def test_require_torch_returns_torch_when_available(self):
        """_require_torch() returns torch module when available."""
        import types

        fake_torch = types.ModuleType("torch")
        with patch.dict("sys.modules", {"torch": fake_torch}):
            from torq.errors import _require_torch

            result = _require_torch()
            assert result is fake_torch


class TestDunderAll:
    def test_all_contains_exactly_expected_public_names(self):
        """__all__ lists the 8 public exception classes and nothing else."""
        import torq.errors as errors_module

        expected = {
            "TorqError",
            "TorqIngestError",
            "TorqStorageError",
            "TorqQualityError",
            "TorqComposeError",
            "TorqConfigError",
            "TorqImportError",
            "EpisodeImmutableFieldError",
        }
        assert set(errors_module.__all__) == expected

    def test_all_does_not_contain_private_names(self):
        """Private helpers (underscore-prefixed) must not appear in __all__."""
        import torq.errors as errors_module

        private_in_all = [name for name in errors_module.__all__ if name.startswith("_")]
        assert private_in_all == [], f"Private names found in __all__: {private_in_all}"


class TestNoTorqImports:
    def test_errors_module_has_no_torq_imports(self):
        """errors.py must not import from torq.* (dependency leaf)."""
        import ast
        from pathlib import Path

        errors_path = Path(__file__).parents[2] / "src" / "torq" / "errors.py"
        source = errors_path.read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("torq"), (
                        f"errors.py imports from torq: 'import {alias.name}'"
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("torq"):
                    pytest.fail(f"errors.py imports from torq: 'from {node.module} import ...'")
