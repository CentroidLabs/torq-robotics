"""Torq exception hierarchy.

All exceptions raised by the Torq SDK are subclasses of TorqError.
No module raises bare Python exceptions (ValueError, Exception, etc.).

Error message format — mandatory:
    [what failed] + [why] + [what the user should try next]
"""

__all__ = [
    "TorqError",
    "TorqIngestError",
    "TorqStorageError",
    "TorqQualityError",
    "TorqComposeError",
    "TorqConfigError",
    "TorqImportError",
    "EpisodeImmutableFieldError",
]


class TorqError(Exception):
    """Base exception for all Torq SDK errors."""


class TorqIngestError(TorqError):
    """Raised on file parsing or episode boundary detection failures."""


class TorqStorageError(TorqError):
    """Raised on read/write/index failures."""


class TorqQualityError(TorqError):
    """Raised on scoring configuration or computation failures."""


class TorqComposeError(TorqError):
    """Raised on dataset composition or query failures."""


class TorqConfigError(TorqError):
    """Raised on invalid configuration values."""


class TorqImportError(TorqError):
    """Raised when an optional dependency is not installed."""


class EpisodeImmutableFieldError(TorqError):
    """Raised when code attempts to mutate a locked Episode field."""


def _require_torch():
    """Import torch or raise TorqImportError with install instructions.

    Returns:
        The ``torch`` module.

    Raises:
        TorqImportError: When torch is not installed in the current environment.
    """
    try:
        import torch

        return torch
    except ImportError:
        raise TorqImportError(
            "PyTorch is required for tq.DataLoader(). "
            "Install it with: pip install torq-robotics[torch]"
        ) from None
