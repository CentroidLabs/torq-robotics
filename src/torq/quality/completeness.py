"""Completeness quality scorer for robot episodes.

Completeness answers the question: "Did this episode fully accomplish its
intended task?"  In R1, the primary signal is the ``success`` flag in episode
metadata.  When that flag is absent, a duration-based heuristic is used.
"""

from __future__ import annotations

import logging

import numpy as np

from torq.episode import Episode
from torq.errors import TorqQualityError
from torq.quality._validation import validate_episode
from torq.types import QualityScore

__all__ = ["score"]

logger = logging.getLogger(__name__)

# Duration (seconds) at which the heuristic score plateaus at 1.0.
EXPECTED_DURATION: float = 30.0


def score(episode: Episode) -> QualityScore:
    """Score the completeness of an episode.

    Primary signal — metadata ``success`` flag::

        success = True  → 1.0
        success = False → 0.0

    Fallback (no ``success`` key) — duration heuristic::

        duration_score = min(1.0, duration_seconds / EXPECTED_DURATION)

    Args:
        episode: The Episode to score.  Uses ``episode.metadata`` and
            ``episode.timestamps`` (shape [T], dtype np.int64 nanoseconds).
            Timestamps must be at least as long as ``episode.actions``.

    Returns:
        A float in [0.0, 1.0], or ``None`` when the episode has fewer than
        10 timesteps, has misaligned timestamps, or contains NaN values in
        the actions array.

    Raises:
        TorqQualityError: If an unexpected computation error occurs after
            validation passes.
    """
    if not validate_episode(episode, logger):
        return None

    try:
        success = episode.metadata.get("success")

        if success is True:
            return 1.0
        if success is False:
            return 0.0

        # No success flag → duration-based heuristic.
        duration_ns = float(episode.timestamps[-1] - episode.timestamps[0])
        duration_seconds = duration_ns / 1e9
        duration_score = min(1.0, duration_seconds / EXPECTED_DURATION)
        return float(np.clip(duration_score, 0.0, 1.0))
    except Exception as exc:
        raise TorqQualityError(
            f"completeness.score() failed for episode '{episode.episode_id}': {exc}. "
            "Check that episode.metadata and episode.timestamps are valid."
        ) from exc
