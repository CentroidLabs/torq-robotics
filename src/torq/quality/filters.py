"""torq.quality.filters — quality gate filtering for robot episodes.

Provides a pure filtering function that accepts a list of pre-scored episodes
and returns a new list containing only those that meet the quality threshold.
"""

from __future__ import annotations

import logging

from torq.episode import Episode
from torq.errors import TorqQualityError

__all__ = ["filter"]

logger = logging.getLogger(__name__)


def filter(
    episodes: list[Episode],
    *,
    min_score: float,
) -> list[Episode]:
    """Return a new list of episodes that meet the quality threshold.

    Excludes episodes that are unscored (``episode.quality is None``), have a
    ``None`` overall score (e.g. too short to score), or whose
    ``episode.quality.overall`` is strictly below ``min_score``.

    This function is *read-only*: it never modifies the input list or the
    episodes themselves.  It also never calls ``tq.quality.score()`` — episodes
    must already be scored before calling this function.

    Args:
        episodes: List of Episodes.  Each episode should have been scored via
            ``tq.quality.score()`` beforehand; unscored episodes are excluded.
        min_score: Minimum acceptable overall quality score, inclusive.  Must
            be in ``[0.0, 1.0]``.  An episode scoring exactly ``min_score``
            passes the gate.

    Returns:
        A new ``list[Episode]`` containing only episodes that passed the gate,
        preserving the original ordering.  Returns an empty list when no
        episodes pass (no exception is raised).

    Raises:
        TorqQualityError: If ``episodes`` is not a list, or if ``min_score``
            is outside the range ``[0.0, 1.0]``.

    Examples::

        import torq as tq

        episodes = tq.ingest("/data")
        tq.quality.score(episodes)
        good = tq.quality.filter(episodes, min_score=0.75)
    """
    # ── Input validation ──────────────────────────────────────────────────────
    if not isinstance(episodes, list):
        raise TorqQualityError(
            f"episodes must be a list[Episode], got {type(episodes).__name__!r}. "
            "Pass a list of Episode objects."
        )

    if not isinstance(min_score, (int, float)) or isinstance(min_score, bool):
        raise TorqQualityError(
            f"min_score must be a float in [0.0, 1.0], got {min_score!r}. "
            "Provide a numeric threshold, e.g. min_score=0.75."
        )

    min_score = float(min_score)

    if not (0.0 <= min_score <= 1.0):
        raise TorqQualityError(
            f"min_score must be in [0.0, 1.0], got {min_score}. "
            "Valid range is 0.0 (accept all scored) to 1.0 (only perfect scores)."
        )

    # ── Fast path: empty list ─────────────────────────────────────────────────
    if not episodes:
        return []

    total = len(episodes)
    passed: list[Episode] = []

    for ep in episodes:
        # Exclude unscored episodes
        if ep.quality is None:
            logger.warning(
                "Episode '%s' was never scored — excluded from filtered results. "
                "Call tq.quality.score() before filtering.",
                ep.episode_id,
            )
            continue

        # Exclude episodes where overall is None (too short to score, <10 timesteps)
        if ep.quality.overall is None:
            logger.warning(
                "Episode '%s' has no overall score (episode may be too short to score) "
                "— excluded from filtered results.",
                ep.episode_id,
            )
            continue

        # Apply threshold (inclusive)
        if ep.quality.overall >= min_score:
            passed.append(ep)

    n_passed = len(passed)

    logger.info(
        "Quality gate: %d/%d episodes passed (min_score=%.2f)",
        n_passed,
        total,
        min_score,
    )

    if n_passed == 0:
        logger.warning(
            "0/%d episodes passed quality gate (min_score=%.2f). Consider lowering the threshold.",
            total,
            min_score,
        )

    return passed
