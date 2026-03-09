"""Shared validation helpers for quality scoring modules.

Centralises the minimum-timestep, timestamps-alignment, and NaN checks that
all three scorers (smoothness, consistency, completeness) must apply before
computing scores.
"""

from __future__ import annotations

import logging

import numpy as np

from torq.episode import Episode

__all__: list[str] = []  # private module — no public exports

MIN_TIMESTEPS = 10


def validate_episode(episode: Episode, logger: logging.Logger) -> bool:
    """Check that an episode is valid for quality scoring.

    Validates:

    1. ``len(episode.actions) >= MIN_TIMESTEPS`` — too-short episodes cannot
       be meaningfully scored.
    2. ``len(episode.timestamps) >= len(episode.actions)`` — misaligned arrays
       would cause ``IndexError`` in duration-based computations.
    3. ``np.any(np.isnan(episode.actions)) == False`` — NaN propagation is
       silently dangerous; reject early.

    Args:
        episode: The Episode to validate.
        logger: Logger from the calling scorer module.  Its ``name`` attribute
            identifies which scorer triggered the warning.

    Returns:
        ``True`` if the episode is valid for scoring, ``False`` otherwise
        (a ``logger.warning()`` is emitted for each failure reason).
    """
    scorer_name = logger.name

    if len(episode.actions) < MIN_TIMESTEPS:
        logger.warning(
            "%s: Episode '%s' has %d timestep(s) (minimum %d required). Returning None.",
            scorer_name,
            episode.episode_id,
            len(episode.actions),
            MIN_TIMESTEPS,
        )
        return False

    if len(episode.timestamps) < len(episode.actions):
        logger.warning(
            "%s: Episode '%s' has misaligned arrays: %d timestamps vs %d action rows. "
            "Returning None.",
            scorer_name,
            episode.episode_id,
            len(episode.timestamps),
            len(episode.actions),
        )
        return False

    if np.any(np.isnan(episode.actions)):
        logger.warning(
            "%s: Episode '%s' contains NaN values in actions array. Returning None.",
            scorer_name,
            episode.episode_id,
        )
        return False

    return True
