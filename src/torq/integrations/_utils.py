"""Shared utilities for torq.integrations modules.

No optional-dependency imports here — this module must load without wandb,
mlflow, or any other optional tracker installed.
"""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torq.compose.dataset import Dataset

__all__ = ["_quality_stats"]


def _quality_stats(dataset: Dataset) -> dict:
    """Compute quality statistics from scored episodes.

    Args:
        dataset: Torq Dataset to inspect.

    Returns:
        Dict with keys ``torq_quality_mean``, ``torq_quality_std``,
        ``torq_quality_min``, ``torq_quality_max``, ``torq_quality_n_scored``.
        Numeric fields are ``None`` when no episodes have been scored.
    """
    scores = [
        ep.quality.overall
        for ep in dataset.episodes
        if ep.quality is not None and ep.quality.overall is not None
    ]
    if not scores:
        return {
            "torq_quality_mean": None,
            "torq_quality_std": None,
            "torq_quality_min": None,
            "torq_quality_max": None,
            "torq_quality_n_scored": 0,
        }
    return {
        "torq_quality_mean": round(sum(scores) / len(scores), 4),
        "torq_quality_std": round(statistics.stdev(scores), 4) if len(scores) > 1 else 0.0,
        "torq_quality_min": round(min(scores), 4),
        "torq_quality_max": round(max(scores), 4),
        "torq_quality_n_scored": len(scores),
    }
