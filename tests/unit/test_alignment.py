"""Unit tests for torq.ingest.alignment — multi-rate temporal alignment.

Covers:
    - Continuous stream resampled to target Hz
    - Multi-rate streams synchronised to identical timestamps
    - Output timestamps are np.int64 nanoseconds
    - step_ns computed as integer (no float seconds internally)
    - Image stream uses nearest-frame selection, not interpolation
    - Timing gap > threshold emits logger.warning
    - Stream with < 2 timesteps raises TorqIngestError
    - TorqIngestError message mentions "too short"
"""

import logging

import numpy as np
import pytest

from torq.errors import TorqIngestError


# ── Shared helpers ─────────────────────────────────────────────────────────────

_RNG = np.random.default_rng(42)  # fixed seed — deterministic fixtures per TESTING.md


def make_streams(n_joint: int = 100, n_cam: int = 60, t_start: int = 0) -> dict:
    """50Hz joint stream + 30Hz wrist camera stream, ~2s duration."""
    from torq.ingest.alignment import Stream

    joint_ts = np.arange(n_joint, dtype=np.int64) * 20_000_000 + t_start  # 50Hz
    cam_ts = np.arange(n_cam, dtype=np.int64) * 33_333_333 + t_start  # ≈30Hz
    return {
        "joint_pos": Stream(
            timestamps=joint_ts,
            data=_RNG.random((n_joint, 6)).astype(np.float32),
            kind="continuous",
        ),
        "wrist_cam": Stream(
            timestamps=cam_ts,
            data=np.zeros((n_cam, 48, 64, 3), dtype=np.uint8),
            kind="image",
        ),
    }


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestContinuousAlignment:
    def test_align_continuous_stream_resamples_to_target_hz(self) -> None:
        """A 50Hz source aligned to 30Hz must produce approx (duration * 30) timesteps."""
        from torq.ingest.alignment import Stream, align

        # 100 frames at 50Hz = 1.98s duration
        src_ts = np.arange(100, dtype=np.int64) * 20_000_000  # 50Hz
        streams = {
            "joint": Stream(
                timestamps=src_ts,
                data=_RNG.random((100, 6)).astype(np.float32),
                kind="continuous",
            )
        }
        result = align(streams, target_hz=30)

        # At 30Hz, step = 33_333_333 ns; duration ≈ 1.98s → ~60 timesteps
        out_ts = result["joint"].timestamps
        assert len(out_ts) >= 59, f"Expected ~60 steps at 30Hz, got {len(out_ts)}"
        assert len(out_ts) <= 61

    def test_align_multi_rate_returns_synchronised_timestamps(self) -> None:
        """50Hz joint + 30Hz camera aligned to 50Hz → both output arrays have same timestamps."""
        from torq.ingest.alignment import align

        streams = make_streams()
        result = align(streams, target_hz=50)

        np.testing.assert_array_equal(
            result["joint_pos"].timestamps,
            result["wrist_cam"].timestamps,
            err_msg="joint_pos and wrist_cam timestamps must be identical after alignment",
        )


class TestTimestampDtype:
    def test_align_timestamps_are_int64_nanoseconds(self) -> None:
        """Output timestamps must be np.int64."""
        from torq.ingest.alignment import align

        streams = make_streams()
        result = align(streams, target_hz=50)

        for name, stream in result.items():
            assert stream.timestamps.dtype == np.int64, (
                f"Stream '{name}' timestamps dtype is {stream.timestamps.dtype}, expected int64"
            )

    def test_align_no_float_seconds_internally(self) -> None:
        """step_ns must be an integer (int(1e9 / target_hz)), not a float.

        We verify by inspecting the output timestamps spacing — if step_ns were
        a float, the timestamps could drift from true int64 multiples.
        """
        from torq.ingest.alignment import Stream, align

        src_ts = np.arange(100, dtype=np.int64) * 20_000_000  # exact 50Hz
        streams = {
            "j": Stream(
                timestamps=src_ts,
                data=np.ones((100, 1), dtype=np.float32),
                kind="continuous",
            )
        }
        result = align(streams, target_hz=50)
        out_ts = result["j"].timestamps

        # All consecutive diffs must equal exactly int(1e9 / 50) = 20_000_000
        diffs = np.diff(out_ts)
        expected_step = int(1e9 / 50)
        assert np.all(diffs == expected_step), (
            f"Timestamp steps are not integer-exact; found diffs: {np.unique(diffs)}"
        )


class TestImageAlignment:
    def test_align_image_stream_uses_nearest_frame(self) -> None:
        """Image stream must use nearest-frame, not interpolation.

        We use a data array where each frame is distinct (frame index written into
        all channels). Nearest-frame selection must pick an existing frame verbatim;
        interpolation would produce blended values.
        """
        from torq.ingest.alignment import Stream, align

        n = 10
        # 10Hz source: frames 0..9, each frame filled with its index value
        src_ts = np.arange(n, dtype=np.int64) * 100_000_000  # 10Hz
        frames = np.zeros((n, 4, 4, 3), dtype=np.uint8)
        for i in range(n):
            frames[i] = i * 20  # distinct pixel values: 0, 20, 40, ...

        streams = {
            "cam": Stream(timestamps=src_ts, data=frames, kind="image")
        }
        result = align(streams, target_hz=5)  # down-sample to 5Hz

        for frame in result["cam"].data:
            # Each output frame must exactly equal one of the original frames
            unique_vals = np.unique(frame)
            assert len(unique_vals) == 1, f"Frame has multiple values — possible interpolation: {unique_vals}"
            assert unique_vals[0] % 20 == 0, f"Value {unique_vals[0]} is not a source frame value"


class TestGapWarning:
    def test_align_timing_gap_emits_warning(self, caplog) -> None:
        """A gap > gap_threshold_ns must emit a logger.WARNING."""
        from torq.ingest.alignment import Stream, align

        # 3 points with a 2-second gap between point 1 and 2
        ts = np.array([0, 100_000_000, 2_100_000_000], dtype=np.int64)  # gap ≈ 2s
        streams = {
            "sensor": Stream(
                timestamps=ts,
                data=np.ones((3, 2), dtype=np.float32),
                kind="continuous",
            )
        }
        with caplog.at_level(logging.WARNING, logger="torq.ingest.alignment"):
            align(streams, target_hz=10, gap_threshold_ns=1_000_000_000)

        assert any("gap" in r.message.lower() for r in caplog.records), (
            f"Expected a gap warning, but no matching log record found. Records: {caplog.records}"
        )


class TestEmptyStreams:
    def test_align_empty_dict_raises_torq_ingest_error(self) -> None:
        """align({}) must raise TorqIngestError, not a bare ValueError from max()."""
        from torq.ingest.alignment import align

        with pytest.raises(TorqIngestError, match="empty"):
            align({}, target_hz=50)


class TestNearestFrameOverflow:
    def test_align_image_stream_large_unix_timestamps(self) -> None:
        """Nearest-frame selection must not overflow for large Unix nanosecond timestamps.

        Unix epoch in nanoseconds is ~1.7e18, close to int64 max (~9.2e18).
        Subtraction without float64 cast wraps around producing wrong distances.
        """
        from torq.ingest.alignment import Stream, align

        # Realistic Unix epoch offset: 2024-01-01 00:00:00 UTC in nanoseconds
        epoch_ns = np.int64(1_704_067_200_000_000_000)
        n = 10
        src_ts = epoch_ns + np.arange(n, dtype=np.int64) * 100_000_000  # 10Hz
        frames = np.zeros((n, 4, 4, 3), dtype=np.uint8)
        for i in range(n):
            frames[i] = i * 20

        streams = {"cam": Stream(timestamps=src_ts, data=frames, kind="image")}
        result = align(streams, target_hz=5)

        # Each output frame must be an exact source frame (nearest, not blended)
        for frame in result["cam"].data:
            unique_vals = np.unique(frame)
            assert len(unique_vals) == 1
            assert unique_vals[0] % 20 == 0, f"Unexpected value {unique_vals[0]} — possible overflow"


class TestSingleStreamEdgeCase:
    def test_align_single_image_stream_returns_correct_shape(self) -> None:
        """align() with a single image stream must succeed and return resampled frames."""
        from torq.ingest.alignment import Stream, align

        n = 20
        src_ts = np.arange(n, dtype=np.int64) * 100_000_000  # 10Hz
        frames = np.zeros((n, 8, 8, 3), dtype=np.uint8)
        streams = {"cam": Stream(timestamps=src_ts, data=frames, kind="image")}

        result = align(streams, target_hz=5)

        assert "cam" in result
        assert result["cam"].data.ndim == 4  # [T, H, W, C]
        assert result["cam"].data.shape[1:] == (8, 8, 3)
        assert result["cam"].timestamps.dtype == np.int64


class TestInvalidTargetHz:
    def test_align_zero_hz_raises_torq_ingest_error(self) -> None:
        """target_hz=0 must raise TorqIngestError, not bare ZeroDivisionError."""
        from torq.ingest.alignment import align

        streams = make_streams()
        with pytest.raises(TorqIngestError, match="positive"):
            align(streams, target_hz=0)

    def test_align_negative_hz_raises_torq_ingest_error(self) -> None:
        """target_hz=-50 must raise TorqIngestError."""
        from torq.ingest.alignment import align

        streams = make_streams()
        with pytest.raises(TorqIngestError, match="positive"):
            align(streams, target_hz=-50)


class TestShortStreamError:
    def test_align_short_stream_raises_torq_ingest_error(self) -> None:
        """A stream with only 1 timestep must raise TorqIngestError."""
        from torq.ingest.alignment import Stream, align

        streams = {
            "joint": Stream(
                timestamps=np.array([0], dtype=np.int64),
                data=np.zeros((1, 6), dtype=np.float32),
                kind="continuous",
            )
        }
        with pytest.raises(TorqIngestError):
            align(streams, target_hz=50)

    def test_align_ingest_error_message_explains_issue(self) -> None:
        """TorqIngestError message must contain 'too short' and suggest minimum length."""
        from torq.ingest.alignment import Stream, align

        streams = {
            "joint": Stream(
                timestamps=np.array([0], dtype=np.int64),
                data=np.zeros((1, 6), dtype=np.float32),
                kind="continuous",
            )
        }
        with pytest.raises(TorqIngestError, match="too short"):
            align(streams, target_hz=50)
