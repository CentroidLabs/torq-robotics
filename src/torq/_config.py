"""Torq SDK configuration singleton.

Provides SDK-wide settings accessible via ``tq.config``.
Reads ``TORQ_QUIET`` environment variable on import.

Usage::

    import torq as tq
    tq.config.quiet = True              # suppress gravity wells and tqdm
    tq.config.quality_weights = {...}   # override scoring weights
    tq.config.reset_quality_weights()   # restore defaults

Note:
    ``TORQ_QUIET`` only recognises the value ``"1"`` (sets quiet=True).
    Any other value (``"0"``, ``"true"``, ``"yes"``) leaves quiet=False.
    After import, use ``tq.config.quiet = True`` to change the setting.
"""

import os

from torq.errors import TorqConfigError

__all__ = ["Config", "DEFAULT_QUALITY_WEIGHTS", "config"]

# External modules append callables here to be invoked by reset_quality_weights().
# This avoids circular imports: registry.py → _config.py → registry.py.
# Not part of the public API — internal extension point only.
from collections.abc import Callable

_quality_reset_hooks: list[Callable[[], None]] = []

DEFAULT_QUALITY_WEIGHTS: dict[str, float] = {
    "smoothness": 0.40,
    "consistency": 0.35,
    "completeness": 0.25,
}


class Config:
    """SDK-wide configuration singleton.

    Attributes:
        quiet: When True, suppresses gravity well prompts and tqdm progress bars.
            Initialised from ``TORQ_QUIET`` environment variable (``"1"`` = quiet).
        quality_weights: Dict mapping dimension names to float weights.
            Keys must be a subset of ``DEFAULT_QUALITY_WEIGHTS`` keys.
            Values must be numeric and sum to 1.0 +/- 0.001.
    """

    def __init__(self) -> None:
        self._quiet: bool = os.environ.get("TORQ_QUIET", "0") == "1"
        self._quality_weights: dict[str, float] = dict(DEFAULT_QUALITY_WEIGHTS)

    @property
    def quiet(self) -> bool:
        """Whether to suppress gravity wells and tqdm progress bars."""
        return self._quiet

    @quiet.setter
    def quiet(self, value: bool) -> None:
        self._quiet = bool(value)

    @property
    def quality_weights(self) -> dict[str, float]:
        """Current quality scoring weights. Must sum to 1.0 +/- 0.001."""
        return dict(self._quality_weights)

    @quality_weights.setter
    def quality_weights(self, weights: dict[str, float]) -> None:
        valid_keys = set(DEFAULT_QUALITY_WEIGHTS.keys())
        unknown_keys = set(weights.keys()) - valid_keys
        if unknown_keys:
            raise TorqConfigError(
                f"Unknown quality weight dimension(s): {sorted(unknown_keys)}. "
                f"Valid dimensions are: {sorted(valid_keys)}. "
                f"Use reset_quality_weights() to restore defaults."
            )
        try:
            total = sum(weights.values())
        except TypeError as exc:
            raise TorqConfigError(
                f"Quality weight values must be numeric (float or int), got a non-numeric value. "
                f"Ensure all weight values are numbers. "
                f"Received: {weights}"
            ) from exc
        if abs(total - 1.0) > 0.001:
            raise TorqConfigError(
                f"Quality weights must sum to 1.0 (+/- 0.001), got {total:.4f}. "
                f"Adjust weights so they sum to 1.0. "
                f"Current weights: {weights}"
            )
        self._quality_weights = dict(weights)

    def reset_quality_weights(self) -> None:
        """Restore quality weights to DEFAULT_QUALITY_WEIGHTS and clear custom metrics.

        Also invokes any registered reset hooks (e.g. the custom metric registry)
        via ``_quality_reset_hooks`` to avoid circular imports.
        """
        self._quality_weights = dict(DEFAULT_QUALITY_WEIGHTS)
        for hook in _quality_reset_hooks:
            hook()

    def __repr__(self) -> str:
        """Return a human-readable representation of the current config state."""
        return f"Config(quiet={self._quiet}, quality_weights={self._quality_weights})"


config = Config()
