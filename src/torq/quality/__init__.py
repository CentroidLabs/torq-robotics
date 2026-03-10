"""torq.quality — automated quality scoring for robot episodes.

Entry point for scoring robot episodes across three built-in dimensions:
smoothness, consistency, and completeness.  The composite ``overall`` score
is the weighted average: ``smoothness×0.40 + consistency×0.35 + completeness×0.25``
(or with custom weights via per-call override or ``tq.config.quality_weights``).

Usage::

    import torq as tq

    episodes = tq.ingest("/path/to/data")
    tq.quality.score(episodes)

    for ep in episodes:
        print(ep.quality.overall)   # weighted composite
        print(ep.quality.smoothness)

Custom metric example::

    def grip_quality(episode):
        return float(np.mean(episode.actions[:, -1]))

    tq.quality.register("grip", grip_quality, weight=0.20)
    tq.quality.score(episodes)   # includes grip contribution
    tq.quality.reset()           # restore defaults
"""

from __future__ import annotations

import logging

import numpy as np

from torq._config import config
from torq._gravity_well import _gravity_well
from torq.episode import Episode
from torq.errors import TorqQualityError
from torq.quality import feasibility
from torq.quality.completeness import score as _completeness_score
from torq.quality.consistency import score as _consistency_score
from torq.quality.filters import filter as filter_episodes
from torq.quality.registry import _registry, get_metrics, register, reset
from torq.quality.report import QualityReport

# Built-in scorers — pure numpy, safe to import eagerly.
from torq.quality.smoothness import score as _smoothness_score

# Re-export filter under the name 'filter' for the tq.quality.filter(...) API.
# This intentionally shadows the Python builtin 'filter' in this module's namespace,
# which is acceptable since tq.quality.filter is always accessed via module attribute.
filter = filter_episodes  # noqa: A001

__all__ = ["score", "register", "get_metrics", "reset", "filter", "report", "feasibility"]

logger = logging.getLogger(__name__)

_REQUIRED_WEIGHT_KEYS: frozenset[str] = frozenset({"smoothness", "consistency", "completeness"})

try:
    from tqdm import tqdm  # type: ignore[import-untyped]
except ImportError:

    def tqdm(iterable, *, disable: bool = False, desc: str = "", **_kw):  # type: ignore[no-redef]
        """Passthrough when tqdm is not installed — accepts and ignores all kwargs."""
        return iterable


def score(
    episodes: Episode | list[Episode],
    weights: dict[str, float] | None = None,
) -> Episode | list[Episode]:
    """Score all quality dimensions and attach a QualityReport to each episode.

    Scores each episode in-place across three built-in dimensions (smoothness,
    consistency, completeness) and any registered custom metrics.  Attaches a
    ``QualityReport`` to ``episode.quality`` and returns the same object(s).

    Args:
        episodes: A single Episode or a list of Episodes to score.
        weights: Optional per-call weight override.  Must be a dict with exactly
            the keys ``"smoothness"``, ``"consistency"``, ``"completeness"``
            whose values sum to 1.0 (±0.001).  Uses ``tq.config.quality_weights``
            when ``None``.  Per-call weights do NOT modify global config and
            do NOT interact with the custom metric registry.

    Returns:
        The same Episode or list passed in (mutated in-place, same identity).

    Raises:
        TorqQualityError: If ``episodes`` is not an Episode or list[Episode],
            if per-call ``weights`` are missing required keys or do not sum to
            1.0 ±0.001, or if any scorer (built-in or custom) raises an error.

    Examples::

        # Basic list scoring
        tq.quality.score(episodes)

        # Single episode
        tq.quality.score(episode)

        # Per-call weight override (global config unchanged)
        tq.quality.score(
            episodes,
            weights={"smoothness": 0.5, "consistency": 0.3, "completeness": 0.2},
        )
    """
    # ── Input type validation ─────────────────────────────────────────────────
    if not isinstance(episodes, (Episode, list)):
        raise TorqQualityError(
            f"episodes must be an Episode or list[Episode], got {type(episodes).__name__!r}. "
            f"Pass a single Episode or a list of Episode objects."
        )

    # ── Empty list fast path ──────────────────────────────────────────────────
    if isinstance(episodes, list) and len(episodes) == 0:
        return episodes

    # ── Per-call weight validation ────────────────────────────────────────────
    if weights is not None:
        # Key presence check — must contain exactly the three required keys.
        provided_keys = set(weights.keys()) if hasattr(weights, "keys") else set()
        missing_keys = _REQUIRED_WEIGHT_KEYS - provided_keys
        if missing_keys:
            raise TorqQualityError(
                f"Quality weights dict is missing required keys: {sorted(missing_keys)}. "
                f"Provide all three keys: 'smoothness', 'consistency', 'completeness'. "
                f"Got keys: {sorted(provided_keys)}."
            )
        try:
            total = sum(weights.values())
        except (TypeError, AttributeError) as exc:
            raise TorqQualityError(
                f"Quality weights must be a dict of numeric values, got {weights!r}. "
                f"Provide weights as a dict like "
                f"{{'smoothness': 0.5, 'consistency': 0.3, 'completeness': 0.2}}."
            ) from exc
        if abs(total - 1.0) > 0.001:
            raise TorqQualityError(
                f"Quality weights must sum to 1.0 ±0.001. Got: {total:.4f}. "
                f"Adjust your weights so they sum to 1.0. "
                f"Current weights: {weights}"
            )
        effective_weights = weights
    else:
        # Use registry's rescaled built-in weights (accounts for custom metrics)
        # if custom metrics are registered, else fall back to global config.
        if _registry.has_custom_metrics():
            effective_weights = _registry.get_built_in_weights()
        else:
            effective_weights = config.quality_weights

    # ── Normalize input ───────────────────────────────────────────────────────
    single = isinstance(episodes, Episode)
    episode_list: list[Episode] = [episodes] if single else episodes  # type: ignore[list-item]

    # ── Custom scorer state ───────────────────────────────────────────────────
    # Only interact with the registry when weights is None (per-call override
    # bypasses registry custom metric logic by design — AC#4).
    use_custom = _registry.has_custom_metrics() and weights is None

    # ── Progress bar (skip for ≤1 episodes — overhead not worth it) ──────────
    disable_progress = config.quiet or len(episode_list) <= 1

    # ── Score each episode ────────────────────────────────────────────────────
    scored_overalls: list[float] = []

    for ep in tqdm(episode_list, disable=disable_progress, desc="Scoring episodes"):
        # Built-in scorers ─────────────────────────────────────────────────────
        try:
            s_score = _smoothness_score(ep)
        except TorqQualityError:
            raise
        except Exception as exc:
            raise TorqQualityError(
                f"smoothness scorer failed for episode '{ep.episode_id}': {exc}. "
                f"Ensure episode.actions is a valid numeric ndarray."
            ) from exc

        try:
            c_score = _consistency_score(ep)
        except TorqQualityError:
            raise
        except Exception as exc:
            raise TorqQualityError(
                f"consistency scorer failed for episode '{ep.episode_id}': {exc}. "
                f"Ensure episode.actions is a valid numeric ndarray."
            ) from exc

        try:
            co_score = _completeness_score(ep)
        except TorqQualityError:
            raise
        except Exception as exc:
            raise TorqQualityError(
                f"completeness scorer failed for episode '{ep.episode_id}': {exc}. "
                f"Ensure episode.actions is a valid numeric ndarray and "
                f"episode.metadata is a dict."
            ) from exc

        # Build QualityReport (overall auto-computed from built-in scores) ────
        report = QualityReport(
            smoothness=s_score,
            consistency=c_score,
            completeness=co_score,
            weights=effective_weights,
        )

        # Custom scorers: augment overall if any registered ───────────────────
        if use_custom and report.overall is not None:
            custom_scorers = _registry.get_custom_scorers()
            custom_contribution: float = 0.0
            custom_any_none = False

            for metric_name, (fn, metric_weight) in custom_scorers.items():
                try:
                    custom_val = fn(ep)
                except TorqQualityError:
                    raise
                except Exception as exc:
                    raise TorqQualityError(
                        f"Custom metric '{metric_name}' failed for episode "
                        f"'{ep.episode_id}': {exc}. "
                        f"Ensure your custom scorer returns a float in [0.0, 1.0]."
                    ) from exc

                if custom_val is None:
                    custom_any_none = True
                    break

                if not (0.0 <= float(custom_val) <= 1.0):
                    raise TorqQualityError(
                        f"Custom metric '{metric_name}' returned {custom_val!r} for "
                        f"episode '{ep.episode_id}'. "
                        f"Return value must be a float in [0.0, 1.0]. "
                        f"Fix your scorer function."
                    )
                custom_contribution += float(custom_val) * metric_weight

            if custom_any_none:
                # None from any custom scorer propagates — clear overall.
                object.__setattr__(report, "overall", None)
            else:
                new_overall = round(report.overall + custom_contribution, 6)
                # Frozen dataclass: must use object.__setattr__ to mutate.
                object.__setattr__(report, "overall", new_overall)

        ep.quality = report

        if report.overall is not None:
            scored_overalls.append(report.overall)

    # ── Gravity well (non-empty result only) ──────────────────────────────────
    if scored_overalls:
        avg_score = sum(scored_overalls) / len(scored_overalls)
        _gravity_well(f"Average quality: {avg_score:.2f}", "GW-SDK-01")

    return episodes


def report(episodes: list[Episode]) -> None:
    """Print a quality distribution summary for a list of scored episodes.

    Computes and prints descriptive statistics (min, max, mean, median, std)
    of the overall quality scores across all scored episodes.  Episodes where
    ``episode.quality`` or ``episode.quality.overall`` is ``None`` are excluded
    from statistics but are counted in the total.

    Outliers are defined as episodes whose overall score deviates more than 2
    standard deviations from the mean.  When all episodes have identical scores,
    std=0 and no outliers are reported.

    Args:
        episodes: List of episodes, typically already scored via
            ``tq.quality.score(episodes)``.  Episodes without scores are
            silently excluded from statistics.

    Returns:
        None — this function prints to stdout and returns nothing.

    Raises:
        TorqQualityError: If ``episodes`` is not a list.

    Examples::

        tq.quality.score(episodes)
        tq.quality.report(episodes)
    """
    if not isinstance(episodes, list):
        raise TorqQualityError(
            f"episodes must be a list[Episode], got {type(episodes).__name__!r}. "
            "Pass a list of Episode objects."
        )

    total = len(episodes)

    # Collect scored episodes (those with a non-None overall)
    scored: list[tuple[str, float]] = []
    for ep in episodes:
        if ep.quality is not None and ep.quality.overall is not None:
            scored.append((ep.episode_id, ep.quality.overall))

    n_scored = len(scored)

    print("Quality Distribution Report")
    print("===========================")
    print(f"Episodes scored : {n_scored} / {total}")

    if n_scored == 0:
        print("No scored episodes — run tq.quality.score(episodes) first.")
        return

    if n_scored < 3:
        logger.warning(
            "Distribution analysis is unreliable with fewer than 3 samples (%d scored). "
            "Score more episodes for meaningful statistics.",
            n_scored,
        )

    scores_arr = np.array([s for _, s in scored], dtype=float)
    s_min = float(np.min(scores_arr))
    s_max = float(np.max(scores_arr))
    s_mean = float(np.mean(scores_arr))
    s_median = float(np.median(scores_arr))
    s_std = float(np.std(scores_arr, ddof=0))

    print(f"Min             : {s_min:.3f}")
    print(f"Max             : {s_max:.3f}")
    print(f"Mean            : {s_mean:.3f}")
    print(f"Median          : {s_median:.3f}")
    print(f"Std Dev         : {s_std:.3f}")
    print()

    if s_std == 0.0:
        print("Outliers: none (zero variance — all episodes scored identically)")
    else:
        outliers = [
            (eid, score, (score - s_mean) / s_std)
            for eid, score in scored
            if abs(score - s_mean) > 2 * s_std
        ]
        if outliers:
            print(f"Outliers (> 2\u03c3 from mean): {len(outliers)} episodes")
            for eid, score, sigma in outliers:
                sign = "+" if sigma >= 0 else "\u2212"
                print(f"  - {eid}  score={score:.3f}  ({sign}{abs(sigma):.2f}\u03c3)")
        else:
            print("Outliers (> 2\u03c3 from mean): none")
