"""QualityReport frozen dataclass for per-episode quality assessment.

Stores per-dimension quality scores (smoothness, consistency, completeness)
and computes a weighted composite ``overall`` score automatically in
``__post_init__``.

None-propagation rule: if ANY component score is ``None``, ``overall`` is
``None``. Partial composites are never returned.
"""

from __future__ import annotations

import logging
from dataclasses import InitVar, dataclass, field

from torq._config import DEFAULT_QUALITY_WEIGHTS
from torq.types import QualityScore

__all__ = ["QualityReport"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QualityReport:
    """Per-episode quality assessment with component and composite scores.

    All fields are immutable after creation (``frozen=True``). The ``overall``
    field is computed automatically from the three component scores and the
    provided weights in ``__post_init__``.

    None-propagation rule: if ANY of smoothness, consistency, or completeness
    is ``None``, ``overall`` is ``None``. No partial composite is ever computed.

    Args:
        smoothness: Smoothness score in [0.0, 1.0], or ``None`` when the
            episode is too short (<10 timesteps) or contains NaN values.
        consistency: Consistency score in [0.0, 1.0], or ``None`` for same
            reasons as above.
        completeness: Completeness score in [0.0, 1.0], or ``None`` for same
            reasons.
        weights: Optional weight dict with keys ``"smoothness"``,
            ``"consistency"``, ``"completeness"`` (and optionally additional
            custom metric keys). When ``None``, defaults to
            ``DEFAULT_QUALITY_WEIGHTS``. Used only for computing ``overall``
            — not stored as a field.
        overall: Weighted composite score. Auto-computed in ``__post_init__``;
            do not pass this — it will be overwritten.

    Example::

        report = QualityReport(smoothness=0.9, consistency=0.8, completeness=1.0)
        assert report.overall == 0.9 * 0.40 + 0.8 * 0.35 + 1.0 * 0.25

        # None propagation
        report_none = QualityReport(smoothness=None, consistency=0.8, completeness=1.0)
        assert report_none.overall is None
    """

    smoothness: QualityScore
    consistency: QualityScore
    completeness: QualityScore

    # InitVar: passed to __post_init__ but NOT stored as a dataclass field.
    weights: InitVar[dict[str, float] | None] = None

    # Computed in __post_init__; frozen=True requires object.__setattr__.
    overall: QualityScore = field(default=None, init=False)

    def __post_init__(self, weights: dict[str, float] | None) -> None:
        # frozen=True means self.overall = ... raises FrozenInstanceError.
        # object.__setattr__ bypasses the frozen guard — required pattern here.
        if any(v is None for v in (self.smoothness, self.consistency, self.completeness)):
            object.__setattr__(self, "overall", None)
        else:
            w = weights if weights is not None else DEFAULT_QUALITY_WEIGHTS
            composite = (
                self.smoothness * w["smoothness"]  # type: ignore[operator]
                + self.consistency * w["consistency"]  # type: ignore[operator]
                + self.completeness * w["completeness"]  # type: ignore[operator]
            )
            object.__setattr__(self, "overall", round(composite, 6))

    def __repr__(self) -> str:
        return (
            f"QualityReport("
            f"smoothness={self.smoothness}, "
            f"consistency={self.consistency}, "
            f"completeness={self.completeness}, "
            f"overall={self.overall})"
        )
