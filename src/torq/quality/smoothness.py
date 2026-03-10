"""Smoothness quality scorer for robot episodes.

Smoothness is measured via jerk (3rd derivative of joint positions).
Low jerk → fluid, human-like motion → high score.
High jerk → erratic, jerky motion → low score.

R2 Note (non-uniform timestamps):
    The current implementation computes discrete differences assuming uniform
    time spacing (i.e., jerk = diff³(actions)).  If an episode has
    non-uniform timestamp intervals, jerk will be overestimated for
    short-interval steps and underestimated for long ones.  A future R2
    improvement should divide each derivative by the corresponding Δt before
    computing the next order (velocity = Δactions/Δt, acceleration = Δvelocity/Δt,
    jerk = Δacceleration/Δt).
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

# Tuning constant: RMS jerk at which the score is 0.5.
# Start at 1.0; calibrate against labelled datasets later.
REFERENCE_JERK: float = 1.0


def score(episode: Episode) -> QualityScore:
    """Score the smoothness of an episode's actions using jerk analysis.

    Computes the RMS jerk (3rd discrete derivative of the action trajectory)
    and maps it to [0.0, 1.0] via a sigmoid-style normalisation::

        smoothness = 1.0 / (1.0 + rms_jerk / REFERENCE_JERK)

    A perfectly smooth (constant-velocity) trajectory has zero jerk and
    returns 1.0.  Random noise returns a value close to 0.0.

    Args:
        episode: The Episode to score.  Uses ``episode.actions`` (shape [T, D]).

    Returns:
        A float in [0.0, 1.0], or ``None`` when the episode has fewer than
        10 timesteps or contains NaN values in the actions array.

    Raises:
        TorqQualityError: If an unexpected computation error occurs after
            validation passes.  Check that ``episode.actions`` is a valid
            numeric ndarray.
    """
    if not validate_episode(episode, logger):
        return None

    try:
        actions = episode.actions.astype(np.float64)

        velocity = np.diff(actions, axis=0)  # [T-1, D]
        acceleration = np.diff(velocity, axis=0)  # [T-2, D]
        jerk = np.diff(acceleration, axis=0)  # [T-3, D]

        rms_jerk: float = float(np.sqrt(np.mean(jerk**2)))
        smoothness = 1.0 / (1.0 + rms_jerk / REFERENCE_JERK)

        return float(np.clip(smoothness, 0.0, 1.0))
    except Exception as exc:
        raise TorqQualityError(
            f"smoothness.score() failed for episode '{episode.episode_id}': {exc}. "
            "Check that episode.actions is a valid numeric ndarray."
        ) from exc
