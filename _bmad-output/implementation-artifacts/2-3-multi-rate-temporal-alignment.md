# Story 2.3: Multi-Rate Temporal Alignment

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer,
I want sensor streams at different frequencies to be aligned to a common timeline,
so that every Episode timestep has synchronised observations across all modalities.

## Acceptance Criteria

1. **Given** joint state data at 50Hz and camera data at 30Hz,
   **When** `alignment.align(streams, target_hz=50)` is called,
   **Then** all streams are resampled to 50Hz using linear interpolation for continuous signals and nearest-frame for image streams,
   **And** all timestamps are `np.int64` nanoseconds throughout (no float seconds in internal computation).

2. **Given** streams with a timing gap exceeding a configurable threshold,
   **When** alignment is attempted,
   **Then** a `logger.warning()` is emitted naming the gap and the streams affected (not a raised exception).

3. **Given** a stream with fewer than 2 timesteps,
   **When** alignment is attempted,
   **Then** `TorqIngestError` is raised with a message explaining the stream is too short to interpolate.

## Tasks / Subtasks

- [x] Task 1: Create `src/torq/ingest/alignment.py` — multi-rate temporal alignment (AC: #1, #2, #3)
  - [x] Define `Stream` dataclass with `timestamps`, `data`, `kind: Literal["continuous","image"]`
  - [x] Implement `align()` with validation, overlap-window timeline, per-stream resampling
  - [x] `_interpolate_continuous()` — `np.interp()` per dimension
  - [x] `_nearest_frame()` — `np.searchsorted()` + left/right distance comparison
  - [x] Gap detection → `logger.warning()` (never exception)
  - [x] `step_ns = int(1e9 / target_hz)` — single float→int conversion
  - [x] Google-style docstrings, `__all__`, module-level logger

- [x] Task 2: Update `src/torq/ingest/__init__.py` — expose `align` at `torq.ingest.align` (AC: #1)

- [x] Task 3: Write unit tests in `tests/unit/test_alignment.py` — 8 tests (AC: #1, #2, #3)
  - [x] `test_align_continuous_stream_resamples_to_target_hz`
  - [x] `test_align_multi_rate_returns_synchronised_timestamps`
  - [x] `test_align_timestamps_are_int64_nanoseconds`
  - [x] `test_align_no_float_seconds_internally`
  - [x] `test_align_image_stream_uses_nearest_frame`
  - [x] `test_align_timing_gap_emits_warning`
  - [x] `test_align_short_stream_raises_torq_ingest_error`
  - [x] `test_align_ingest_error_message_explains_issue`

- [x] Task 4: Run full test suite and verify no regressions (AC: all)
  - [x] All previous 110 tests still pass (118 total, 0 regressions)
  - [x] All 8 new tests pass in 0.08s
  - [x] `ruff check src/ && ruff format --check src/` clean

## Dev Notes

### Exact Implementation — `src/torq/ingest/alignment.py`

```python
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
        kind: "continuous" uses linear interpolation. "image" uses nearest-frame.
    """

    timestamps: np.ndarray   # np.int64 nanoseconds, shape [T]
    data: np.ndarray
    kind: Literal["continuous", "image"]


def align(
    streams: dict[str, Stream],
    target_hz: float,
    *,
    gap_threshold_ns: int = _DEFAULT_GAP_THRESHOLD_NS,
) -> dict[str, Stream]:
    """Align multiple sensor streams to a common target timeline.

    Args:
        streams: Dict mapping stream name → Stream. At least one stream required.
        target_hz: Target frequency in Hz (e.g. 50.0 for 50Hz).
        gap_threshold_ns: Gap between consecutive source timestamps (in nanoseconds)
            above which a warning is emitted. Default: 1 second.

    Returns:
        Dict mapping stream name → resampled Stream, all at target_hz.

    Raises:
        TorqIngestError: If any stream has fewer than 2 timesteps.
    """
    # Validate all streams
    for name, stream in streams.items():
        if len(stream.timestamps) < 2:
            raise TorqIngestError(
                f"Stream '{name}' has {len(stream.timestamps)} timestep(s) — "
                f"at least 2 are required for interpolation. "
                f"Check your source data or filter out streams shorter than 2 timesteps."
            )

    # Detect timing gaps per stream
    for name, stream in streams.items():
        gaps = np.diff(stream.timestamps)
        large_gaps = gaps[gaps > gap_threshold_ns]
        if len(large_gaps) > 0:
            max_gap_s = int(large_gaps.max()) / 1e9
            logger.warning(
                f"Stream '{name}' has {len(large_gaps)} gap(s) exceeding "
                f"{gap_threshold_ns / 1e9:.1f}s (max gap: {max_gap_s:.2f}s). "
                f"Alignment may produce interpolated values across the gap."
            )

    # Build common target timeline using integer nanosecond arithmetic only
    t_start = max(s.timestamps[0] for s in streams.values())
    t_end = min(s.timestamps[-1] for s in streams.values())
    step_ns: int = int(1e9 / target_hz)   # Hz → ns step (integer, no float seconds internally)
    target_timestamps = np.arange(t_start, t_end + 1, step_ns, dtype=np.int64)

    aligned: dict[str, Stream] = {}
    for name, stream in streams.items():
        if stream.kind == "continuous":
            resampled_data = _interpolate_continuous(
                stream.timestamps, stream.data, target_timestamps
            )
        else:
            resampled_data = _nearest_frame(
                stream.timestamps, stream.data, target_timestamps
            )
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
    """Linear interpolation per dimension for continuous signals."""
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
    """Nearest-neighbour frame selection for image streams."""
    # searchsorted gives insertion points; clip to valid index range
    indices = np.searchsorted(src_ts, target_ts, side="left")
    indices = np.clip(indices, 0, len(src_ts) - 1)
    # For each target, pick nearest between left and right candidate
    left = np.clip(indices - 1, 0, len(src_ts) - 1)
    right = indices
    left_dist = np.abs(target_ts - src_ts[left])
    right_dist = np.abs(target_ts - src_ts[right])
    nearest = np.where(left_dist <= right_dist, left, right)
    return src_data[nearest]
```

**CRITICAL implementation notes:**

- `step_ns = int(1e9 / target_hz)` — the `int()` cast is mandatory. This is the single point where Hz converts to nanoseconds. All subsequent arithmetic uses `np.int64` integers. Never pass float seconds to `np.arange` or `np.interp`.
- `np.interp(target_ts, src_ts, ...)` works with `np.int64` arrays — numpy promotes internally. No float conversion needed.
- For image streams, `src_data` may be a frame index array `[T]` or a full `[T, H, W, C]` array. `_nearest_frame` handles both via numpy indexing `src_data[nearest]`.
- `gap_threshold_ns` warning is advisory only — never raise an exception for gaps. Per AC #2: "a `logger.warning()` is emitted... (not a raised exception)".
- `t_start = max(...)`, `t_end = min(...)` gives the overlapping window where all streams have data. This is the safe alignment range.

### `src/torq/ingest/__init__.py` Update

Check the existing file first. Currently it's a stub. After this story it should export `align`:

```python
"""Torq ingest sub-package — format parsers and temporal alignment."""

from torq.ingest.alignment import align

__all__ = ["align"]
```

Do NOT add `ingest()` here yet — that's Story 2.7. Only `align` is exposed in this story.

### Testing Conventions

**caplog pattern for warning assertions (pytest built-in):**
```python
import logging

def test_align_timing_gap_emits_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="torq.ingest.alignment"):
        align(streams_with_gap, target_hz=50)
    assert any("gap" in r.message.lower() for r in caplog.records)
```

**Fixture for 50Hz joint stream + 30Hz image stream:**
```python
import numpy as np
from torq.ingest.alignment import Stream

def make_streams(n_joint=100, n_cam=60, t_start=0):
    """50Hz joint stream (100 frames) + 30Hz camera stream (60 frames), 2s duration."""
    joint_ts = np.arange(n_joint, dtype=np.int64) * 20_000_000 + t_start   # 20ms = 50Hz
    cam_ts   = np.arange(n_cam,   dtype=np.int64) * 33_333_333 + t_start   # 33.3ms ≈ 30Hz
    return {
        "joint_pos": Stream(
            timestamps=joint_ts,
            data=np.random.rand(n_joint, 6).astype(np.float32),
            kind="continuous",
        ),
        "wrist_cam": Stream(
            timestamps=cam_ts,
            data=np.zeros((n_cam, 48, 64, 3), dtype=np.uint8),
            kind="image",
        ),
    }
```

**Short-stream test (AC #3):**
```python
def test_align_short_stream_raises_torq_ingest_error():
    streams = {
        "joint": Stream(
            timestamps=np.array([0], dtype=np.int64),
            data=np.zeros((1, 6), dtype=np.float32),
            kind="continuous",
        )
    }
    with pytest.raises(TorqIngestError, match="too short"):
        align(streams, target_hz=50)
```

### Project Structure Notes

#### Files to create/modify in this story

```
src/torq/ingest/
├── __init__.py          ← MODIFY (add align export; currently stub)
└── alignment.py         ← CREATE

tests/unit/
└── test_alignment.py    ← CREATE (8 tests)
```

#### Files NOT touched in this story

```
src/torq/episode.py           ← No changes
src/torq/storage/             ← No changes (completed in Story 2.2)
src/torq/errors.py            ← TorqIngestError already exists; no additions needed
src/torq/__init__.py          ← No changes (alignment is torq.ingest.align, not top-level tq.align)
```

`alignment.py` is an internal utility used by the MCAP/HDF5/LeRobot ingesters in Stories 2.4–2.6. It is NOT exposed at the `tq.*` top level — only at `torq.ingest.align`.

### Architecture Compliance

| Rule | Requirement | This Story |
|---|---|---|
| `ingest/` import direction | imports episode, errors, storage, media, types | ✓ alignment.py only imports `errors` (no storage/media needed) |
| No circular imports | alignment ← errors (leaf) | ✓ |
| Timestamps are `np.int64` nanoseconds | single source of truth | ✓ `step_ns = int(1e9 / target_hz)`, all ops in int64 |
| No float seconds in internal computation | AC #1 explicit requirement | ✓ float only at boundary: `int(1e9 / target_hz)` |
| `TorqIngestError` for stream failures | not bare ValueError | ✓ stream < 2 timesteps |
| `logger.warning()` for gaps | not exception | ✓ per AC #2 |
| Google-style docstrings | all public classes and functions | ✓ |
| `logging.getLogger(__name__)` | module-level logger | ✓ `torq.ingest.alignment` |
| `ruff format` line length 100 | formatter standard | ✓ |

### Previous Story Intelligence (from Stories 2.1, 2.2)

- **95 + 15 = 110 tests passing** as of Story 2.2. Zero regressions is a hard requirement.
- **`ruff check src/` and `ruff format --check src/`** must both be clean before marking done.
- **Test speed**: all 8 unit tests must be < 1s each. Use small synthetic arrays (100 timesteps, 6 dims), never real video files in alignment tests.
- **`caplog` fixture** is pytest built-in — no extra import needed. Use `caplog.at_level(logging.WARNING, logger="torq.ingest.alignment")` for targeted capture.
- **`np.interp` works with int64 inputs** — confirmed safe, no float conversion needed by numpy.
- **`object.__setattr__` not needed here** — alignment creates new Stream objects, doesn't touch Episode fields.
- **Conftest `sample_episode` fixture** exists but not needed for alignment tests — alignment tests operate on raw `Stream` dicts, not Episode objects.

### References

- Story 2.3 AC: [Source: planning-artifacts/epics.md — Epic 2, Story 2.3]
- Alignment module specification: [Source: planning-artifacts/architecture.md — FR-to-File-Mapping: `ingest/alignment.py`]
- Timestamp format rule (np.int64 ns only): [Source: planning-artifacts/architecture.md — Timestamp Format]
- Exception hierarchy (TorqIngestError): [Source: src/torq/errors.py]
- Dependency direction (ingest/ allowed imports): [Source: planning-artifacts/architecture.md — Dependency Rules]
- Test count targets (8 tests): [Source: planning-artifacts/architecture.md — test_alignment.py: F09 — 8 tests]
- Multi-rate sensor use case (50Hz + 30Hz): [Source: planning-artifacts/epics.md — Story 2.3 AC1]
- `logger.warning` not exception for gaps: [Source: planning-artifacts/epics.md — Story 2.3 AC2]

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6

### Debug Log References

None — implementation matched spec exactly.

### Completion Notes List

- Implemented `Stream` dataclass + `align()` + `_interpolate_continuous()` + `_nearest_frame()` in `alignment.py`
- All 3 ACs satisfied: linear interp for continuous (AC1), warning-only for gaps (AC2), TorqIngestError for short streams (AC3)
- `step_ns = int(1e9 / target_hz)` — float→int at boundary only; all arithmetic in np.int64
- Overlap window strategy: `t_start = max(first_ts)`, `t_end = min(last_ts)` across all streams
- `torq.ingest.align` exported; `torq.__init__.py` NOT changed (alignment is ingest-internal)
- 118/118 tests, 0 regressions, ruff clean
- ✅ Resolved review finding [HIGH]: Documented step_ns integer truncation as R1 limitation in align() docstring
- ✅ Resolved review finding [MEDIUM]: Added empty-streams guard raising TorqIngestError with helpful message
- ✅ Resolved review finding [MEDIUM]: Fixed int64 overflow in _nearest_frame — distances now computed in float64
- ✅ Resolved review finding [LOW]: Fixed non-deterministic make_streams fixture — uses np.random.default_rng(42)
- ✅ Resolved review finding [LOW]: Added TestSingleStreamEdgeCase and TestNearestFrameOverflow and TestEmptyStreams test classes
- 121/121 tests passing, 0 regressions, ruff clean
- ✅ Resolved review-2 finding [HIGH]: Architecture deviation documented — `align()` with `Stream` dataclass is intentional simplification; architecture doc needs update
- ✅ Resolved review-2 finding [MEDIUM]: Fixed docstring `round()` → `int()` to match code
- ✅ Resolved review-2 finding [MEDIUM]: Added `target_hz > 0` guard with TorqIngestError
- ✅ Resolved review-2 finding [MEDIUM]: Fixed remaining `np.random.rand` → `_RNG.random()` in test
- 123/123 tests passing, 0 regressions, ruff clean

### File List

- `src/torq/ingest/alignment.py` (CREATED; MODIFIED — empty-streams guard, float64 cast in _nearest_frame, R1 limitation docstring, target_hz validation, docstring fix)
- `src/torq/ingest/__init__.py` (MODIFIED — added `align` export)
- `tests/unit/test_alignment.py` (CREATED — 8 tests; MODIFIED — fixed RNG seed, added 5 new test classes, 13 tests total)

## Review Follow-ups (AI)

- [x] [AI-Review][HIGH] `step_ns = int(1e9 / target_hz)` truncates for non-integer-divisible frequencies (e.g. 30Hz → 33333333ns, drifts ~2μs over 2s). `np.arange` with `t_end + 1` can produce ±1 timestep error. Only exact-divisor Hz (50, 100, 200) are drift-free. Document as R1 limitation or use `np.linspace`. Test `test_align_no_float_seconds_internally` only checks 50Hz (exact). [src/torq/ingest/alignment.py:101-102]
- [x] [AI-Review][MEDIUM] `align({}, target_hz=50)` crashes with `ValueError: max() iterable argument is empty` instead of a helpful `TorqIngestError`. Add empty-streams validation. [src/torq/ingest/alignment.py:99]
- [x] [AI-Review][MEDIUM] `_nearest_frame` int64 subtraction `target_ts - src_ts[left]` can overflow for large Unix nanosecond timestamps (~1.7e18). `np.abs()` on wrapped int64 produces wrong distances. Cast to float64 before abs, or use relative timestamps. [src/torq/ingest/alignment.py:166-167]
- [x] [AI-Review][LOW] `make_streams` test helper uses `np.random.rand` without fixed seed — non-deterministic fixtures violate TESTING.md convention. Use `np.random.default_rng(42)`. [tests/unit/test_alignment.py:24,33]
- [x] [AI-Review][LOW] No test for `align()` with a single image stream — edge case for overlap window calculation with one stream. [tests/unit/test_alignment.py]

### Review 2 Follow-ups (AI)

- [x] [AI-Review-2][HIGH] Architecture spec defines `align_to_frequency()` with separate data/timestamps/continuous_keys/discrete_keys params and tuple return. Implementation uses `align()` with `Stream` dataclass — simpler API but architecture deviation. Architecture doc should be updated to match implementation. [src/torq/ingest/alignment.py:45, architecture.md:1060]
- [x] [AI-Review-2][MEDIUM] Docstring says `round(1e9 / target_hz)` but code uses `int(1e9 / target_hz)`. Fixed: docstring now says `int()`. [src/torq/ingest/alignment.py:74]
- [x] [AI-Review-2][MEDIUM] `target_hz=0` raises bare `ZeroDivisionError`, `target_hz<0` silently returns empty streams. Fixed: added `target_hz > 0` guard with `TorqIngestError`. [src/torq/ingest/alignment.py:82-86]
- [x] [AI-Review-2][MEDIUM] `test_align_continuous_stream_resamples_to_target_hz` uses `np.random.rand` (unfixed seed) despite module-level `_RNG`. Fixed: replaced with `_RNG.random()`. [tests/unit/test_alignment.py:59]

## Change Log

- 2026-03-06: Implemented multi-rate temporal alignment — alignment.py, updated ingest/__init__.py. Added 8 tests. 118/118 passing, ruff clean.
- 2026-03-06: Code review completed — 1 HIGH, 2 MEDIUM, 2 LOW issues found. 5 action items created.
  Status moved to in-progress. All ACs implemented but step_ns truncation drift for non-exact Hz,
  missing empty-streams guard, and potential int64 overflow in nearest-frame distance calc.
- 2026-03-06: Addressed code review findings — 5 items resolved (Date: 2026-03-06).
  [HIGH] Documented step_ns R1 limitation in align() docstring.
  [MEDIUM] Added empty-streams TorqIngestError guard.
  [MEDIUM] Fixed int64 overflow in _nearest_frame (cast to float64 before abs).
  [LOW] Fixed non-deterministic fixture (np.random.default_rng(42)).
  [LOW] Added 3 new test classes: TestEmptyStreams, TestNearestFrameOverflow, TestSingleStreamEdgeCase.
  121/121 tests passing, 0 regressions, ruff clean. Status moved to review.
- 2026-03-06: Code review 2 completed — 1 HIGH, 3 MEDIUM, 2 LOW issues found. All HIGH+MEDIUM fixed:
  [HIGH] Architecture deviation documented (align vs align_to_frequency — intentional, arch doc needs update).
  [MEDIUM] Fixed docstring round()→int() mismatch.
  [MEDIUM] Added target_hz>0 validation guard.
  [MEDIUM] Fixed remaining np.random.rand non-determinism in test.
  123/123 tests passing, 0 regressions, ruff clean. Status moved to done.
