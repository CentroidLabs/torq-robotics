"""Unit tests for torq.ingest.mcap — MCAP / ROS 2 ingestion.

Covers:
    - Returns list[Episode] with observations and actions (AC #1)
    - Timestamps are np.int64 nanoseconds (AC #1)
    - source_path set to MCAP file path (AC #1)
    - Unknown schema topics emit logger.warning (AC #4)
    - Corrupt CDR messages emit warning and are skipped (AC #3)
    - Empty MCAP returns [] (AC #3 edge case)
    - Gripper data takes priority over velocity in auto-detection (AC #2)
    - boundary_strategy='none' returns single episode
    - boundary_strategy='manual' splits at specified timestamps
    - boundary_detection.mcap with auto-detection returns 5 episodes (gripper)
    - Velocity-only boundary detection produces multiple episodes (priority 2)
    - Invalid boundary_strategy raises TorqIngestError
    - Corrupt message warning includes decode-related keywords
"""

import logging
from pathlib import Path

import numpy as np
import pytest

# ── Fixture paths ─────────────────────────────────────────────────────────────

FIXTURES_DATA = Path(__file__).parent.parent / "fixtures" / "data"


@pytest.fixture(scope="session")
def sample_mcap() -> Path:
    """Path to the 2-topic, 100-message sample MCAP fixture."""
    p = FIXTURES_DATA / "sample.mcap"
    if not p.exists():
        pytest.skip("sample.mcap not found — run: python tests/fixtures/generate_fixtures.py")
    return p


@pytest.fixture(scope="session")
def boundary_mcap() -> Path:
    """Path to the 3-episode gripper-boundary MCAP fixture."""
    p = FIXTURES_DATA / "boundary_detection.mcap"
    if not p.exists():
        pytest.skip(
            "boundary_detection.mcap not found — run: python tests/fixtures/generate_fixtures.py"
        )
    return p


@pytest.fixture(scope="session")
def velocity_only_mcap() -> Path:
    """Path to the 3-episode velocity-only (no gripper) MCAP fixture."""
    p = FIXTURES_DATA / "velocity_only.mcap"
    if not p.exists():
        pytest.skip(
            "velocity_only.mcap not found — run: python tests/fixtures/generate_fixtures.py"
        )
    return p


@pytest.fixture(scope="session")
def corrupt_mcap() -> Path:
    """Path to the partially-corrupt MCAP fixture."""
    p = FIXTURES_DATA / "corrupt.mcap"
    if not p.exists():
        pytest.skip("corrupt.mcap not found — run: python tests/fixtures/generate_fixtures.py")
    return p


@pytest.fixture(scope="session")
def empty_mcap() -> Path:
    """Path to the zero-message MCAP fixture."""
    p = FIXTURES_DATA / "empty.mcap"
    if not p.exists():
        pytest.skip("empty.mcap not found — run: python tests/fixtures/generate_fixtures.py")
    return p


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestBasicIngest:
    def test_ingest_returns_list_of_episodes(self, sample_mcap: Path) -> None:
        """ingest() must return a non-empty list of Episode objects."""
        from torq.episode import Episode
        from torq.ingest.mcap import ingest

        episodes = ingest(sample_mcap)

        assert isinstance(episodes, list), "ingest() must return a list"
        assert len(episodes) > 0, "Expected at least one episode from sample.mcap"
        for ep in episodes:
            assert isinstance(ep, Episode), f"Expected Episode, got {type(ep)}"

    def test_ingest_episodes_have_observations_and_actions(self, sample_mcap: Path) -> None:
        """Returned episodes must have non-empty observations and actions arrays."""
        from torq.ingest.mcap import ingest

        episodes = ingest(sample_mcap)

        ep = episodes[0]
        assert ep.observations, "Episode observations dict must not be empty"
        assert ep.actions is not None and ep.actions.size > 0, (
            "Episode actions must be a non-empty array"
        )
        # Observations must include at least joint data
        assert "joint_pos" in ep.observations, "Expected 'joint_pos' in observations"

    def test_ingest_timestamps_are_int64_nanoseconds(self, sample_mcap: Path) -> None:
        """Episode timestamps must be np.int64 (nanoseconds, per architecture spec)."""
        from torq.ingest.mcap import ingest

        episodes = ingest(sample_mcap)

        for ep in episodes:
            assert ep.timestamps.dtype == np.int64, (
                f"timestamps dtype is {ep.timestamps.dtype}, expected int64"
            )
            # Timestamps should be in nanosecond range (>= 1e9 since we start at t=1s)
            assert ep.timestamps[0] >= 1_000_000_000, (
                f"Unexpected timestamp value {ep.timestamps[0]} — expected nanoseconds"
            )

    def test_ingest_episode_source_path_is_mcap_file(self, sample_mcap: Path) -> None:
        """episode.source_path must equal the input MCAP file path."""
        from torq.ingest.mcap import ingest

        episodes = ingest(sample_mcap)

        for ep in episodes:
            assert ep.source_path == sample_mcap, (
                f"Expected source_path={sample_mcap}, got {ep.source_path}"
            )


class TestUnknownTopicWarning:
    def test_ingest_unknown_topic_emits_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """A topic with an unrecognised schema must emit a logger.warning.

        Uses boundary_detection.mcap — all its schemas ARE known. We create
        a tiny in-memory fixture with a custom schema to trigger the warning.
        """
        import io
        from types import SimpleNamespace

        from mcap.writer import Writer
        from mcap_ros2._dynamic import serialize_dynamic

        from torq.ingest.mcap import ingest

        # Build a tiny MCAP with one known topic and one unknown schema topic
        KNOWN_SCHEMA = "float64[] data\n"
        UNKNOWN_SCHEMA = "uint32 mystery_field\n"

        enc_known = serialize_dynamic("std_msgs/msg/Float64MultiArray", KNOWN_SCHEMA)
        enc_unknown = serialize_dynamic("custom_msgs/msg/Mystery", UNKNOWN_SCHEMA)
        encode_known = enc_known["std_msgs/msg/Float64MultiArray"]
        encode_unknown = enc_unknown["custom_msgs/msg/Mystery"]

        buf = io.BytesIO()
        writer = Writer(buf)
        writer.start(profile="ros2", library="test")

        # Known topic: /action (Float64MultiArray)
        sid_known = writer.register_schema(
            name="std_msgs/msg/Float64MultiArray", encoding="ros2msg", data=KNOWN_SCHEMA.encode()
        )
        sid_unknown = writer.register_schema(
            name="custom_msgs/msg/Mystery", encoding="ros2msg", data=UNKNOWN_SCHEMA.encode()
        )
        ch_known = writer.register_channel(
            topic="/action", message_encoding="cdr", schema_id=sid_known
        )
        ch_unknown = writer.register_channel(
            topic="/mystery", message_encoding="cdr", schema_id=sid_unknown
        )

        t_ns = 1_000_000_000
        for i in range(5):
            ts = t_ns + i * 20_000_000
            writer.add_message(
                channel_id=ch_known,
                log_time=ts,
                publish_time=ts,
                sequence=i,
                data=encode_known(SimpleNamespace(data=[float(i)] * 6)),
            )
            writer.add_message(
                channel_id=ch_unknown,
                log_time=ts,
                publish_time=ts,
                sequence=i,
                data=encode_unknown(SimpleNamespace(mystery_field=i)),
            )

        writer.finish()
        buf.seek(0)

        # Write to a temp file (ingest() requires a path)
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".mcap", delete=False) as tmp:
            tmp.write(buf.getvalue())
            tmp_path = Path(tmp.name)

        try:
            with caplog.at_level(logging.WARNING, logger="torq.ingest.mcap"):
                ingest(tmp_path)
            assert any(
                "custom_msgs" in r.message or "Mystery" in r.message for r in caplog.records
            ), f"Expected unknown-schema warning, got: {[r.message for r in caplog.records]}"
        finally:
            tmp_path.unlink(missing_ok=True)


class TestCorruptHandling:
    def test_ingest_corrupt_mcap_skips_and_warns(
        self, corrupt_mcap: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """corrupt.mcap must emit a warning and return partial results — not raise."""
        from torq.ingest.mcap import ingest

        with caplog.at_level(logging.WARNING, logger="torq.ingest.mcap"):
            episodes = ingest(corrupt_mcap)

        # Must not raise; must return at least one episode from valid messages
        assert isinstance(episodes, list), "ingest() must return a list even for corrupt files"
        assert len(episodes) >= 1, "Expected at least one episode from the valid messages"

        # Must have emitted a warning specifically about decoding failure
        assert any(
            "decode" in r.message.lower() or "failed" in r.message.lower() for r in caplog.records
        ), (
            "Expected a warning about failed decoding for the corrupt message, "
            f"got: {[r.message for r in caplog.records]}"
        )

    def test_ingest_empty_mcap_returns_empty_list(
        self, empty_mcap: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """empty.mcap must return [] and emit an informational warning."""
        from torq.ingest.mcap import ingest

        with caplog.at_level(logging.WARNING, logger="torq.ingest.mcap"):
            episodes = ingest(empty_mcap)

        assert episodes == [], f"Expected [], got {episodes}"
        assert any(
            "empty" in r.message.lower() or "no channels" in r.message.lower()
            for r in caplog.records
        ), f"Expected 'empty'/'no channels' warning, got: {[r.message for r in caplog.records]}"


class TestBoundaryDetection:
    def test_ingest_boundary_detection_gripper_priority(self, boundary_mcap: Path) -> None:
        """When gripper AND velocity data are present, gripper strategy takes priority.

        boundary_detection.mcap has /joint_states (with velocity) AND /gripper.
        Gripper strategy detects open↔close transitions (4 in this fixture),
        producing 5 episodes. Velocity strategy would only produce 3.
        """
        from torq.ingest.mcap import ingest

        episodes = ingest(boundary_mcap, boundary_strategy="auto")

        # Gripper has 4 transitions (open→close at 49,109 and close→open at 59,119)
        # producing 5 segments; velocity alone would yield 3
        assert len(episodes) == 5, f"Expected 5 episodes (gripper priority), got {len(episodes)}"

    def test_ingest_boundary_strategy_none_returns_single_episode(
        self, boundary_mcap: Path
    ) -> None:
        """boundary_strategy='none' must return exactly one episode per file."""
        from torq.ingest.mcap import ingest

        episodes = ingest(boundary_mcap, boundary_strategy="none")

        assert len(episodes) == 1, (
            f"Expected 1 episode with boundary_strategy='none', got {len(episodes)}"
        )

    def test_ingest_velocity_only_boundary_detection(self, velocity_only_mcap: Path) -> None:
        """Velocity strategy (priority 2) used when no gripper topic present.

        velocity_only.mcap has /joint_states (with velocity) but NO /gripper topic.
        Two near-zero velocity gaps at idx 50-59 and 110-119 should produce 3 episodes.
        """
        from torq.ingest.mcap import ingest

        episodes = ingest(velocity_only_mcap, boundary_strategy="auto")

        assert len(episodes) == 3, f"Expected 3 episodes (velocity boundaries), got {len(episodes)}"

    def test_ingest_invalid_boundary_strategy_raises(self, sample_mcap: Path) -> None:
        """Passing an invalid boundary_strategy must raise TorqIngestError."""
        from torq.errors import TorqIngestError
        from torq.ingest.mcap import ingest

        with pytest.raises(TorqIngestError, match="Invalid boundary_strategy='foo'"):
            ingest(sample_mcap, boundary_strategy="foo")

    def test_ingest_boundary_strategy_manual(self, sample_mcap: Path) -> None:
        """boundary_strategy='manual' must split at the specified timestamp markers.

        sample.mcap: 100 msgs at 50Hz, t=[1e9, 1e9+20ms, ..., 1e9+99*20ms].
        Splitting at t=1e9+50*20ms=2e9 ns should produce 2 episodes.
        """
        from torq.ingest.mcap import ingest

        marker_ns = 1_000_000_000 + 50 * 20_000_000  # midpoint of the file

        episodes = ingest(sample_mcap, boundary_strategy="manual", markers=[marker_ns])

        assert len(episodes) == 2, (
            f"Expected 2 episodes with manual split at {marker_ns} ns, got {len(episodes)}"
        )
        # First episode ends before or at marker; second starts after
        assert episodes[0].timestamps[-1] <= marker_ns
        assert episodes[1].timestamps[0] >= marker_ns
