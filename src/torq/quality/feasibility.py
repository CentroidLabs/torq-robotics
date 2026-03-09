"""Kinematic feasibility scorer — R1 stub.

In R1, this always returns 1.0.  Full URDF-based joint limit and collision
checking is deferred to R2 (QM-06).

Usage::

    from torq.quality import feasibility
    score = feasibility.score(episode)  # always 1.0 in R1
"""

from __future__ import annotations

from torq.episode import Episode

__all__ = ["score"]


def score(episode: Episode) -> float:
    """Return the kinematic feasibility score for an episode.

    R1 stub — always returns 1.0.  Full URDF-based validation (joint limits,
    collision detection) is planned for R2.

    Args:
        episode: The Episode to score (not used in R1).

    Returns:
        1.0 always.
    """
    return 1.0
