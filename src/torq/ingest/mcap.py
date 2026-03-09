"""MCAP / ROS 2 ingestion for the Torq SDK.

Reads MCAP recordings from ALOHA-2, ROS 2 Humble/Iron/Jazzy, and any
CDR-encoded ROS 2 bag file, returning a list of :class:`~torq.episode.Episode`
objects aligned to a common 50 Hz timeline.

All timestamps are ``np.int64`` nanoseconds throughout — MCAP ``log_time``
is already in nanosecond integer format, so no conversion is needed.

Optional dependency: ``mcap-ros2-support`` (core ``mcap`` is required).
Install the full ROS 2 stack with::

    pip install torq-robotics[ros2]
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from torq.episode import Episode
from torq.errors import TorqImportError, TorqIngestError
from torq.ingest.alignment import Stream, align

try:
    from mcap_ros2.decoder import DecoderFactory as _DecoderFactory

    _HAS_ROS2 = True
except ImportError:  # pragma: no cover
    _HAS_ROS2 = False

from mcap.reader import make_reader

__all__ = ["ingest"]

logger = logging.getLogger(__name__)

# Schema names whose message structure is understood and mapped to numpy arrays
_KNOWN_SCHEMAS: frozenset[str] = frozenset(
    {
        "sensor_msgs/msg/JointState",
        "sensor_msgs/JointState",
        "std_msgs/msg/Float64MultiArray",
        "std_msgs/Float64MultiArray",
        "geometry_msgs/msg/Twist",
        "geometry_msgs/Twist",
        "sensor_msgs/msg/Image",
        "sensor_msgs/Image",
        "sensor_msgs/msg/CompressedImage",
        "sensor_msgs/CompressedImage",
    }
)


# ── Public API ────────────────────────────────────────────────────────────────


def ingest(
    path: str | Path,
    *,
    boundary_strategy: str = "auto",
    markers: list[int] | None = None,
) -> list[Episode]:
    """Ingest an MCAP / ROS 2 recording into a list of :class:`~torq.episode.Episode` objects.

    Reads all CDR-encoded topics from the file, aligns them to 50 Hz using
    :func:`torq.ingest.alignment.align`, detects episode boundaries, and returns
    one :class:`~torq.episode.Episode` per detected segment.

    Args:
        path: Path to the MCAP file.
        boundary_strategy: Episode boundary detection strategy.

            - ``"auto"`` — composite: gripper state change (priority 1),
              near-zero velocity (priority 2), manual markers (priority 3).
              Falls back to a single episode if no boundary signal found.
            - ``"none"`` — single episode per file.
            - ``"manual"`` — split only at ``markers`` timestamps.

        markers: Nanosecond timestamps at which to split episodes. Only used
            when ``boundary_strategy`` is ``"auto"`` (fallback) or ``"manual"``.

    Returns:
        List of :class:`~torq.episode.Episode` objects.  Empty list if the file
        has no messages or all messages are undecodable.

    Raises:
        TorqImportError: If ``mcap-ros2-support`` is not installed.
        TorqIngestError: If the file does not exist or is not a regular file.

    Note:
        **Corrupt message tolerance**: corrupt or truncated messages emit a
        ``logger.warning()`` and are skipped — ingestion never aborts on a
        single bad message (AC #3).

        **Unknown schemas**: topics with unrecognised schema names emit a
        ``logger.warning()`` and are skipped (AC #4).

        **R1 alignment target**: always 50 Hz.  Streams with fewer than 2
        timesteps raise :class:`~torq.errors.TorqIngestError` inside
        :func:`~torq.ingest.alignment.align` and are surfaced as a warning.
    """
    _VALID_STRATEGIES = {"auto", "none", "manual"}
    if boundary_strategy not in _VALID_STRATEGIES:
        raise TorqIngestError(
            f"Invalid boundary_strategy='{boundary_strategy}'. "
            f"Must be one of: {', '.join(sorted(_VALID_STRATEGIES))}."
        )

    path = Path(path)
    if not path.exists():
        raise TorqIngestError(
            f"MCAP file not found: '{path}'. Check the path and ensure the file exists."
        )
    if not path.is_file():
        raise TorqIngestError(
            f"Expected a file, but '{path}' is a directory. Pass the full path to an .mcap file."
        )

    if not _HAS_ROS2:  # pragma: no cover
        raise TorqImportError(
            "mcap-ros2-support is required for MCAP / ROS 2 ingestion but is not installed. "
            "Install with: pip install torq-robotics[ros2]"
        )

    decoder_factory = _DecoderFactory()

    # topic → list of (log_time_ns, decoded_message_obj, schema_name)
    topic_messages: dict[str, list[tuple[int, Any, str]]] = defaultdict(list)
    warned_schemas: set[str] = set()

    with open(path, "rb") as f:
        reader = make_reader(f)

        try:
            summary = reader.get_summary()
        except Exception as exc:
            logger.warning(
                "Failed to read MCAP summary from '%s': %s — returning empty list.",
                path,
                exc,
            )
            return []

        if summary is None or not summary.channels:
            logger.warning("MCAP file '%s' has no channels (empty file).", path)
            return []

        # Pre-flight: warn about unknown schemas
        for channel in summary.channels.values():
            schema = summary.schemas.get(channel.schema_id)
            if schema is None:
                logger.warning(
                    "Topic '%s' has no schema in MCAP summary — will be skipped.",
                    channel.topic,
                )
                continue
            if schema.name not in _KNOWN_SCHEMAS and schema.name not in warned_schemas:
                logger.warning(
                    "Unknown schema '%s' on topic '%s' — will attempt decode; "
                    "unrecognised fields are ignored.",
                    schema.name,
                    channel.topic,
                )
                warned_schemas.add(schema.name)

        # Read messages — wrap entire loop to tolerate truncated files
        try:
            for schema, channel, message in reader.iter_messages():
                try:
                    decode_fn = decoder_factory.decoder_for(channel.message_encoding, schema)
                    if decode_fn is None:
                        if channel.message_encoding not in warned_schemas:
                            logger.warning(
                                "No CDR decoder for encoding '%s' on topic '%s' — skipping.",
                                channel.message_encoding,
                                channel.topic,
                            )
                            warned_schemas.add(channel.message_encoding)
                        continue
                    decoded = decode_fn(message.data)
                    topic_messages[channel.topic].append((message.log_time, decoded, schema.name))
                except Exception as exc:
                    logger.warning(
                        "Failed to decode message on topic '%s' in '%s': %s — skipping.",
                        channel.topic,
                        path,
                        exc,
                    )
        except Exception as exc:
            logger.warning(
                "MCAP reading stopped early in '%s': %s. Processing %d topic(s) collected so far.",
                path,
                exc,
                len(topic_messages),
            )

    if not topic_messages:
        return []

    streams = _build_streams(topic_messages, warned_schemas)

    if not streams:
        return []

    # Temporal alignment to 50 Hz
    try:
        aligned = align(streams, target_hz=50.0)
    except TorqIngestError as exc:
        logger.warning("Temporal alignment failed for '%s': %s", path, exc)
        return []

    # Episode segmentation
    timestamps = next(iter(aligned.values())).timestamps
    data_by_key = {k: v.data for k, v in aligned.items()}
    boundary_indices = _detect_boundaries(data_by_key, timestamps, boundary_strategy, markers)

    return _build_episodes(aligned, timestamps, boundary_indices, path)


# ── Stream construction ───────────────────────────────────────────────────────


def _build_streams(
    topic_messages: dict[str, list[tuple[int, Any, str]]],
    warned_schemas: set[str],
) -> dict[str, Stream]:
    """Convert decoded per-topic message lists into :class:`~torq.ingest.alignment.Stream` dicts."""
    streams: dict[str, Stream] = {}

    for topic, messages in topic_messages.items():
        if not messages:
            continue

        timestamps = np.array([m[0] for m in messages], dtype=np.int64)
        schema_name = messages[0][2]
        decoded_msgs = [m[1] for m in messages]

        if "JointState" in schema_name:
            try:
                pos = np.array([list(m.position) for m in decoded_msgs], dtype=np.float64)
                streams["joint_pos"] = Stream(timestamps=timestamps, data=pos, kind="continuous")
                vel_lists = [list(m.velocity) for m in decoded_msgs]
                if vel_lists and any(v for v in vel_lists):
                    vel = np.array(vel_lists, dtype=np.float64)
                    streams["joint_vel"] = Stream(
                        timestamps=timestamps, data=vel, kind="continuous"
                    )
            except (AttributeError, ValueError) as exc:
                logger.warning(
                    "Failed to extract JointState fields from topic '%s': %s",
                    topic,
                    exc,
                )

        elif "Float64MultiArray" in schema_name:
            try:
                data = np.array([list(m.data) for m in decoded_msgs], dtype=np.float64)
                key = _sanitize_topic(topic)
                streams[key] = Stream(timestamps=timestamps, data=data, kind="continuous")
            except (AttributeError, ValueError) as exc:
                logger.warning(
                    "Failed to extract Float64MultiArray from topic '%s': %s",
                    topic,
                    exc,
                )

        elif "Twist" in schema_name:
            try:
                data = np.array(
                    [
                        [
                            m.linear.x,
                            m.linear.y,
                            m.linear.z,
                            m.angular.x,
                            m.angular.y,
                            m.angular.z,
                        ]
                        for m in decoded_msgs
                    ],
                    dtype=np.float64,
                )
                streams[_sanitize_topic(topic)] = Stream(
                    timestamps=timestamps, data=data, kind="continuous"
                )
            except (AttributeError, ValueError) as exc:
                logger.warning("Failed to extract Twist fields from topic '%s': %s", topic, exc)

        else:
            if schema_name not in warned_schemas:
                logger.warning(
                    "Unknown schema '%s' on topic '%s' — stream skipped.",
                    schema_name,
                    topic,
                )
                warned_schemas.add(schema_name)

    return streams


def _sanitize_topic(topic: str) -> str:
    """Convert a ROS 2 topic name to a valid stream key.

    Examples:
        ``/action``         → ``"action"``
        ``/gripper``        → ``"gripper"``
        ``/my/nested/topic``→ ``"my_nested_topic"``
    """
    return topic.lstrip("/").replace("/", "_")


# ── Episode boundary detection ────────────────────────────────────────────────


def _detect_boundaries(
    streams: dict[str, np.ndarray],
    timestamps: np.ndarray,
    strategy: str,
    markers: list[int] | None,
) -> list[int]:
    """Return sorted boundary indices into the aligned timestamp array.

    Args:
        streams: Dict of stream key → data array (already aligned).
        timestamps: Common np.int64 timestamp array.
        strategy: ``"auto"``, ``"none"``, or ``"manual"``.
        markers: Nanosecond marker timestamps (for ``"manual"`` or auto fallback).

    Returns:
        Sorted list of boundary indices.  Each index ``b`` causes a split
        such that episode i ends at ``b`` (inclusive) and episode i+1 starts
        at ``b + 1``.
    """
    if strategy == "none":
        return []
    if strategy == "manual":
        return _boundaries_from_markers(timestamps, markers or [])

    # Auto: gripper (priority 1) → velocity (priority 2) → manual markers (priority 3)
    has_gripper = any("gripper" in k for k in streams)
    has_velocity = "joint_vel" in streams

    if has_gripper:
        return _gripper_boundaries(streams, timestamps)
    if has_velocity:
        return _velocity_boundaries(streams, timestamps)
    if markers:
        return _boundaries_from_markers(timestamps, markers)
    return []


def _gripper_boundaries(
    streams: dict[str, np.ndarray],
    timestamps: np.ndarray,  # noqa: ARG001 — reserved for future time-gating
) -> list[int]:
    """Detect gripper open ↔ closed transitions as episode boundaries.

    Returns indices of ALL open↔close and close↔open transitions so that each
    continuous gripper-state period becomes its own candidate segment.
    Short segments (< 2 timesteps) are filtered at episode construction time.

    Args:
        streams: Dict containing at least one key with ``"gripper"`` in its name.
            Data shape ``[T]`` or ``[T, 1]``; first column used if 2-D.
        timestamps: Aligned timestamp array (unused in R1; reserved).

    Returns:
        Sorted list of transition indices.
    """
    gripper_key = next(k for k in streams if "gripper" in k)
    gripper = streams[gripper_key]
    if gripper.ndim > 1:
        gripper = gripper[:, 0]
    is_open = gripper > 0.02  # threshold: <0.02 m = closed (ALOHA-2 range 0–0.044 m)
    transitions = np.where(np.diff(is_open.astype(np.int8)) != 0)[0]
    return transitions.tolist()


def _velocity_boundaries(
    streams: dict[str, np.ndarray],
    timestamps: np.ndarray,
    vel_threshold: float = 0.01,
    min_duration_ns: int = 100_000_000,  # 100 ms
) -> list[int]:
    """Detect sustained near-zero velocity periods as episode boundaries.

    Args:
        streams: Dict containing ``"joint_vel"`` key, shape ``[T, D]``.
        timestamps: Aligned np.int64 timestamp array.
        vel_threshold: Max joint velocity (rad/s) considered "at rest".
        min_duration_ns: Minimum pause duration (ns) to count as a boundary.

    Returns:
        Sorted list of midpoint indices within each qualifying pause period.
    """
    vel = streams["joint_vel"]
    max_vel = np.abs(vel).max(axis=1)
    near_zero = max_vel < vel_threshold

    boundaries: list[int] = []
    in_pause = False
    pause_start = 0

    for i, is_zero in enumerate(near_zero):
        if is_zero and not in_pause:
            in_pause = True
            pause_start = i
        elif not is_zero and in_pause:
            in_pause = False
            duration = int(timestamps[i]) - int(timestamps[pause_start])
            if duration >= min_duration_ns:
                midpoint = pause_start + (i - pause_start) // 2
                boundaries.append(midpoint)

    # Handle case where recording ends during a pause
    if in_pause:
        end_idx = len(near_zero) - 1
        duration = int(timestamps[end_idx]) - int(timestamps[pause_start])
        if duration >= min_duration_ns:
            midpoint = pause_start + (end_idx - pause_start) // 2
            boundaries.append(midpoint)

    return boundaries


def _boundaries_from_markers(
    timestamps: np.ndarray,
    markers: list[int],
) -> list[int]:
    """Convert nanosecond marker timestamps to nearest aligned timestamp indices.

    Args:
        timestamps: Aligned np.int64 timestamp array.
        markers: Nanosecond timestamps marking episode boundaries.

    Returns:
        Sorted, deduplicated list of nearest indices.
    """
    indices: list[int] = []
    for t in sorted(markers):
        idx = int(np.searchsorted(timestamps, t))
        idx = max(0, min(idx, len(timestamps) - 1))
        indices.append(idx)
    return sorted(set(indices))


# ── Episode construction ──────────────────────────────────────────────────────


def _build_episodes(
    aligned: dict[str, Stream],
    timestamps: np.ndarray,
    boundary_indices: list[int],
    source_path: Path,
) -> list[Episode]:
    """Slice aligned streams into Episodes at each boundary index.

    Segments with fewer than 2 timesteps are silently discarded (they cannot
    form a valid :class:`~torq.episode.Episode`).

    Args:
        aligned: Dict of stream key → aligned :class:`~torq.ingest.alignment.Stream`.
        timestamps: Common np.int64 aligned timestamp array.
        boundary_indices: Indices after which episode splits occur.
        source_path: Provenance path set on each returned episode.

    Returns:
        List of :class:`~torq.episode.Episode` objects, one per retained segment.
    """
    # Build monotone split-point list: 0, b1+1, b2+1, ..., T
    raw = [0] + [b + 1 for b in sorted(set(boundary_indices))] + [len(timestamps)]
    split_points = sorted(set(raw))

    # Identify action stream — exact match "action" or prefix "action_*"
    action_key: str | None = next(
        (k for k in aligned if k == "action" or k.startswith("action_")),
        None,
    )

    episodes: list[Episode] = []
    for start, end in zip(split_points[:-1], split_points[1:]):
        if end - start < 2:
            continue  # too short to be a valid episode

        seg_ts = timestamps[start:end]

        obs: dict[str, np.ndarray] = {
            k: v.data[start:end] for k, v in aligned.items() if k != action_key
        }

        if action_key is not None:
            actions = aligned[action_key].data[start:end]
        else:
            actions = np.empty((end - start, 0), dtype=np.float64)

        episodes.append(
            Episode(
                episode_id="",
                observations=obs,
                actions=actions,
                timestamps=seg_ts,
                source_path=source_path,
                metadata={"task": "", "embodiment": ""},
            )
        )

    return episodes
