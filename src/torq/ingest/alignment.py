"""Multi-rate temporal alignment for robot sensor streams.

Aligns sensor streams recorded at different frequencies to a single common
timeline using linear interpolation (continuous signals) or nearest-frame
selection (image streams).

All timestamps are np.int64 nanoseconds throughout — float seconds are
never used in internal computation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

from torq.errors import TorqIngestError

__all__ = ["Stream", "align"]

logger = logging.getLogger(__name__)

_DEFAULT_GAP_THRESHOLD_NS: int = 1_000_000_000  # 1 second


@dataclass
class Stream:
    """A single sensor stream with timestamps and data.

    Args:
        timestamps: Nanosecond timestamps, dtype np.int64, shape [T].
        data: Stream data. Shape [T, D] for continuous; shape [T, H, W, C] or
            [T] (index array) for image streams.
        kind: ``"continuous"`` uses linear interpolation per dimension.
            ``"image"`` uses nearest-frame selection.
    """

    timestamps: np.ndarray  # np.int64 nanoseconds, shape [T]
    data: np.ndarray
    kind: Literal["continuous", "image"]


def align(
    streams: dict[str, Stream],
    target_hz: float,
    *,
    gap_threshold_ns: int = _DEFAULT_GAP_THRESHOLD_NS,
) -> dict[str, Stream]:
    """Align multiple sensor streams to a common target timeline.

    Builds a common timeline from the overlapping range of all stream windows
    (``t_start = max(first timestamps)``, ``t_end = min(last timestamps)``),
    then resamples each stream to that timeline.

    Args:
        streams: Dict mapping stream name → Stream. At least one stream required.
        target_hz: Target frequency in Hz (e.g. ``50.0`` for 50 Hz).
        gap_threshold_ns: Consecutive timestamp gap above which a warning is
            emitted (nanoseconds). Default: 1 second. Gaps never raise an
            exception — per AC #2.

    Returns:
        Dict mapping stream name → resampled Stream, all sharing the same
        target timeline.

    Raises:
        TorqIngestError: If ``streams`` is empty, or if any stream has fewer
            than 2 timesteps — too short to interpolate.

    Note:
        **R1 limitation — step_ns integer truncation:** The target timeline step
        is computed as ``step_ns = int(1e9 / target_hz)``.  For frequencies
        that do not evenly divide 1 GHz (e.g. 30 Hz → 33,333,333 ns instead of
        the exact 33,333,333.̄3 ns), a small drift accumulates over long
        episodes (~20 ns per second at 30 Hz).  Only exact-divisor frequencies
        such as 50, 100, and 200 Hz are perfectly drift-free.  Use
        ``np.linspace``-based alignment (future R2 feature) for drift-critical
        applications.
    """
    # Guard: invalid target_hz produces bare ZeroDivisionError or silent empty output
    if target_hz <= 0:
        raise TorqIngestError(
            f"target_hz must be positive, got {target_hz}. "
            f"Pass a frequency in Hz (e.g. target_hz=50.0)."
        )

    # Guard: empty dict crashes max()/min() with unhelpful ValueError
    if not streams:
        raise TorqIngestError(
            "align() requires at least one stream, but received an empty dict. "
            "Pass a dict with at least one Stream entry."
        )

    # Validate all streams have enough data to interpolate
    for name, stream in streams.items():
        if len(stream.timestamps) < 2:
            raise TorqIngestError(
                f"Stream '{name}' has {len(stream.timestamps)} timestep(s) and is "
                f"too short to interpolate. "
                f"At least 2 timesteps are required. "
                f"Check your source data or filter out streams shorter than 2 timesteps."
            )

    # Warn about timing gaps (advisory only — never raise)
    for name, stream in streams.items():
        gaps = np.diff(stream.timestamps)
        large_gaps = gaps[gaps > gap_threshold_ns]
        if len(large_gaps) > 0:
            max_gap_s = int(large_gaps.max()) / 1e9
            logger.warning(
                "Stream '%s' has %d gap(s) exceeding %.1fs (max gap: %.2fs). "
                "Alignment may produce interpolated values across the gap.",
                name,
                len(large_gaps),
                gap_threshold_ns / 1e9,
                max_gap_s,
            )

    # Build common target timeline using integer nanosecond arithmetic only.
    # Overlap window: [max(first_ts), min(last_ts)] ensures all streams have data.
    t_start = max(int(s.timestamps[0]) for s in streams.values())
    t_end = min(int(s.timestamps[-1]) for s in streams.values())
    step_ns: int = int(1e9 / target_hz)  # Hz → ns step (single float→int conversion)
    target_timestamps = np.arange(t_start, t_end + 1, step_ns, dtype=np.int64)

    aligned: dict[str, Stream] = {}
    for name, stream in streams.items():
        if stream.kind == "continuous":
            resampled_data = _interpolate_continuous(
                stream.timestamps, stream.data, target_timestamps
            )
        else:
            resampled_data = _nearest_frame(stream.timestamps, stream.data, target_timestamps)
        aligned[name] = Stream(
            timestamps=target_timestamps,
            data=resampled_data,
            kind=stream.kind,
        )

    return aligned


def _interpolate_continuous(
    src_ts: np.ndarray,
    src_data: np.ndarray,
    target_ts: np.ndarray,
) -> np.ndarray:
    """Linear interpolation per dimension for continuous signals.

    Args:
        src_ts: Source timestamps, np.int64 nanoseconds, shape [T_src].
        src_data: Source data, shape [T_src] or [T_src, D].
        target_ts: Target timestamps, np.int64 nanoseconds, shape [T_tgt].

    Returns:
        Interpolated array of shape [T_tgt] or [T_tgt, D].
    """
    if src_data.ndim == 1:
        return np.interp(target_ts, src_ts, src_data).astype(src_data.dtype)
    # Multi-dimensional: interpolate each column independently
    out = np.empty((len(target_ts), src_data.shape[1]), dtype=src_data.dtype)
    for dim in range(src_data.shape[1]):
        out[:, dim] = np.interp(target_ts, src_ts, src_data[:, dim])
    return out


def _nearest_frame(
    src_ts: np.ndarray,
    src_data: np.ndarray,
    target_ts: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbour frame selection for image streams.

    Args:
        src_ts: Source timestamps, np.int64 nanoseconds, shape [T_src].
        src_data: Source frames, shape [T_src, ...] (any trailing dims).
        target_ts: Target timestamps, np.int64 nanoseconds, shape [T_tgt].

    Returns:
        Frames selected by nearest timestamp, shape [T_tgt, ...].
    """
    # searchsorted gives insertion point; clip to valid range
    indices = np.searchsorted(src_ts, target_ts, side="left")
    indices = np.clip(indices, 0, len(src_ts) - 1)
    # For each target, compare distance to left and right candidate.
    # Cast to float64 before subtraction: int64 wraparound for large Unix
    # nanosecond timestamps (~1.7e18) produces wrong distances after np.abs().
    left = np.clip(indices - 1, 0, len(src_ts) - 1)
    right = indices
    left_dist = np.abs(target_ts.astype(np.float64) - src_ts[left].astype(np.float64))
    right_dist = np.abs(target_ts.astype(np.float64) - src_ts[right].astype(np.float64))
    nearest = np.where(left_dist <= right_dist, left, right)
    return src_data[nearest]
