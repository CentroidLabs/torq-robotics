"""Custom metric registry for the Torq quality scoring system.

Allows registering additional scorer functions beyond the three built-in
dimensions (smoothness, consistency, completeness). When new metrics are
registered, the existing weights are rescaled proportionally so all weights
continue to sum to 1.0.

Weight rescaling example::

    Default: {smoothness: 0.40, consistency: 0.35, completeness: 0.25}
    Register "grip" with weight=0.20:
        scale_factor = 1.0 - 0.20 = 0.80
        New weights: {smoothness: 0.32, consistency: 0.28, completeness: 0.20, grip: 0.20}

Thread Safety:
    The registry is NOT thread-safe in R1. Do not register or reset metrics
    from multiple threads concurrently. Thread safety will be addressed in R2.

Usage::

    import torq.quality as q

    def grip_quality(episode):
        return float(np.mean(episode.actions[:, -1]))

    q.register("grip", grip_quality, weight=0.20)
    q.score(episodes)  # scores using rescaled weights including grip
    q.reset()          # restore DEFAULT_QUALITY_WEIGHTS
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable

from torq._config import DEFAULT_QUALITY_WEIGHTS, _quality_reset_hooks
from torq.errors import TorqQualityError

__all__ = ["register", "get_metrics", "reset"]

logger = logging.getLogger(__name__)


class _Registry:
    """Internal singleton registry for custom quality scoring metrics.

    Maintains built-in weights (which may be rescaled when custom metrics are
    added) and the set of custom scorer functions.

    Not intended for direct use — call module-level ``register``,
    ``get_metrics``, and ``reset`` instead.
    """

    def __init__(self) -> None:
        # Built-in weights — rescaled proportionally when custom metrics added.
        self._built_in_weights: dict[str, float] = dict(DEFAULT_QUALITY_WEIGHTS)
        # Custom metrics: name → (callable, weight)
        self._custom: dict[str, tuple[Callable, float]] = {}

    def register(self, name: str, fn: Callable, weight: float) -> None:
        """Register a custom quality scoring function.

        Args:
            name: Metric name. Must not collide with built-in names
                (smoothness, consistency, completeness).
            fn: Callable ``(episode: Episode) -> float`` returning [0.0, 1.0].
                Return value is validated at scoring time, not at registration.
            weight: Weight for this metric in (0.0, 1.0). All existing weights
                (built-in + previously registered custom) are rescaled
                proportionally by ``(1.0 - weight)`` to keep the total at 1.0.

        Raises:
            TorqQualityError: If ``name`` collides with a built-in metric name,
                or if ``weight`` is not strictly in (0.0, 1.0).
        """
        if not callable(fn):
            raise TorqQualityError(
                f"Custom metric scorer must be callable, got {type(fn).__name__!r}. "
                f"Pass a function with signature (episode: Episode) -> float."
            )
        if name in DEFAULT_QUALITY_WEIGHTS:
            raise TorqQualityError(
                f"Cannot register custom metric with built-in name '{name}'. "
                f"Built-in metrics are: {sorted(DEFAULT_QUALITY_WEIGHTS)}. "
                f"Choose a different name for your custom metric."
            )
        if not (0.0 < weight < 1.0):
            raise TorqQualityError(
                f"Custom metric weight must be in (0.0, 1.0), got {weight!r}. "
                f"Adjust the weight to a value strictly between 0 and 1."
            )

        if name in self._custom:
            warnings.warn(
                f"Re-registering existing custom metric '{name}'. "
                f"Previous scorer will be replaced.",
                UserWarning,
                stacklevel=3,
            )
            # Reverse the old weight contribution before applying the new one.
            # Without this, re-registration double-scales existing weights.
            old_weight = self._custom[name][1]
            reverse_factor = 1.0 / (1.0 - old_weight)
            self._built_in_weights = {
                k: round(v * reverse_factor, 8) for k, v in self._built_in_weights.items()
            }
            self._custom = {
                k: (fn_existing, round(w_existing * reverse_factor, 8))
                for k, (fn_existing, w_existing) in self._custom.items()
                if k != name  # remove the old entry; will re-add below
            }

        scale_factor = 1.0 - weight

        # Rescale built-in weights proportionally.
        self._built_in_weights = {
            k: round(v * scale_factor, 8) for k, v in self._built_in_weights.items()
        }
        # Rescale previously registered custom weights proportionally.
        self._custom = {
            k: (fn_existing, round(w_existing * scale_factor, 8))
            for k, (fn_existing, w_existing) in self._custom.items()
        }
        self._custom[name] = (fn, weight)

        logger.debug("Registered custom metric '%s' (weight=%.4f)", name, weight)

    def get_metrics(self) -> dict[str, float]:
        """Return a snapshot of all metrics and their current weights.

        Returns:
            Dict mapping metric name → weight (built-in + custom).
            Sum of all values equals 1.0 within floating-point precision.
        """
        result = dict(self._built_in_weights)
        result.update({name: w for name, (_, w) in self._custom.items()})
        return result

    def get_built_in_weights(self) -> dict[str, float]:
        """Return the current built-in dimension weights (may be rescaled).

        Returns:
            Dict with keys smoothness, consistency, completeness.
        """
        return dict(self._built_in_weights)

    def get_custom_scorers(self) -> dict[str, tuple[Callable, float]]:
        """Return registered custom scorers and their weights.

        Returns:
            Dict mapping name → (callable, weight).
        """
        return dict(self._custom)

    def has_custom_metrics(self) -> bool:
        """Return True if any custom metrics are registered."""
        return bool(self._custom)

    def reset(self) -> None:
        """Clear all custom metrics and restore DEFAULT_QUALITY_WEIGHTS."""
        self._custom.clear()
        self._built_in_weights = dict(DEFAULT_QUALITY_WEIGHTS)
        logger.debug("Registry reset to DEFAULT_QUALITY_WEIGHTS")


# Module-level singleton — the single source of registry state.
_registry = _Registry()

# Register the registry reset as a hook on _config so that
# tq.config.reset_quality_weights() also clears custom metrics.
# This avoids a circular import: _config ← registry ← _config.
_quality_reset_hooks.append(_registry.reset)


def register(name: str, fn: Callable, weight: float) -> None:
    """Register a custom quality scoring function.

    When a new metric is registered, the existing built-in (and previously
    registered custom) weights are rescaled proportionally so all weights
    continue to sum to 1.0.

    Args:
        name: Unique metric identifier. Must not be a built-in name
            (smoothness, consistency, completeness).
        fn: Callable ``(episode: Episode) -> float`` returning [0.0, 1.0].
            Validated at scoring time — a value outside [0.0, 1.0] raises
            ``TorqQualityError`` during ``tq.quality.score()``.
        weight: Fraction of the composite score assigned to this metric.
            Must be strictly in (0.0, 1.0).

    Raises:
        TorqQualityError: If name collides with a built-in, or weight is
            not in (0.0, 1.0).

    Example::

        def grip_quality(episode):
            return float(np.mean(episode.actions[:, -1]))

        tq.quality.register("grip", grip_quality, weight=0.20)
        # get_metrics() → {smoothness: 0.32, consistency: 0.28,
        #                   completeness: 0.20, grip: 0.20}
    """
    _registry.register(name, fn, weight)


def get_metrics() -> dict[str, float]:
    """Return all registered metrics and their current weights.

    Returns:
        Dict mapping metric name → weight (built-in + any custom metrics).
        Sum of all values equals 1.0 within floating-point precision.

    Example::

        tq.quality.register("grip", fn, weight=0.2)
        tq.quality.get_metrics()
        # {'smoothness': 0.32, 'consistency': 0.28,
        #  'completeness': 0.20, 'grip': 0.20}
    """
    return _registry.get_metrics()


def reset() -> None:
    """Clear all custom metrics and restore DEFAULT_QUALITY_WEIGHTS.

    Safe to call in test teardown after testing custom metric registration.

    Example::

        tq.quality.reset()
        tq.quality.get_metrics()
        # {'smoothness': 0.40, 'consistency': 0.35, 'completeness': 0.25}
    """
    _registry.reset()
