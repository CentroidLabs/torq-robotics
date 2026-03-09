"""Consistency quality scorer for robot episodes.

Consistency measures how predictable and directionally stable the action
trajectory is.  Oscillating or hesitating trajectories score low; smooth,
unidirectional ones score high.
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


def score(episode: Episode) -> QualityScore:
    """Score the consistency of an episode's action trajectory.

    Algorithm::

        1. deltas = diff(actions)               [T-1, D]
        2. magnitudes = L2 norm of each delta   [T-1]
        3. autocorr = Pearson correlation of magnitudes[:-1] vs magnitudes[1:]
        4. reversal_ratio = sign-change fraction across all dimensions
        5. consistency = max(0, autocorr) * (1.0 - reversal_ratio)

    High positive autocorrelation with few direction reversals → high score.
    Oscillating or hesitating trajectories → low score.

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
        deltas = np.diff(actions, axis=0)  # [T-1, D]

        magnitudes = np.linalg.norm(deltas, axis=1)  # [T-1]

        # Autocorrelation at lag-1 using Pearson correlation.
        if len(magnitudes) < 2:
            return float(np.clip(1.0, 0.0, 1.0))

        # Guard against floating-point noise in constant/near-constant trajectories.
        # When variance is negligible, magnitude is effectively constant → perfectly
        # consistent (autocorr = 1.0).
        std_lead = float(np.std(magnitudes[:-1]))
        std_lag = float(np.std(magnitudes[1:]))
        if std_lead < 1e-9 or std_lag < 1e-9:
            autocorr = 1.0
        else:
            corr = float(np.corrcoef(magnitudes[:-1], magnitudes[1:])[0, 1])
            autocorr = 0.0 if np.isnan(corr) else corr

        # Direction reversal ratio: fraction of consecutive delta pairs that
        # change sign in at least one dimension.
        if deltas.shape[0] >= 2:
            sign_changes = np.any(np.diff(np.sign(deltas), axis=0) != 0, axis=1)
            reversal_ratio = float(np.mean(sign_changes))
        else:
            reversal_ratio = 0.0

        consistency = max(0.0, autocorr) * (1.0 - reversal_ratio)
        return float(np.clip(consistency, 0.0, 1.0))
    except Exception as exc:
        raise TorqQualityError(
            f"consistency.score() failed for episode '{episode.episode_id}': {exc}. "
            "Check that episode.actions is a valid numeric ndarray."
        ) from exc
