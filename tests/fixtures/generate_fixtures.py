"""Generate deterministic test fixtures for Torq SDK MCAP ingestion tests.

Run once to create all fixture files:
    python tests/fixtures/generate_fixtures.py

All fixtures are deterministic (fixed seed), small (<1MB each), and
checked into the repository. They do NOT require a live ROS 2 system.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from mcap.writer import Writer
from mcap_ros2._dynamic import serialize_dynamic

FIXTURES_DIR = Path(__file__).parent / "data"

# ── ROS 2 message schemas (simplified: no Header field for CDR compactness) ──

JOINT_STATE_SCHEMA = "string[] name\nfloat64[] position\nfloat64[] velocity\nfloat64[] effort\n"

FLOAT64_ARRAY_SCHEMA = "float64[] data\n"

JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

# Pre-compile CDR encoders once at module load
_js_encoders = serialize_dynamic("sensor_msgs/msg/JointState", JOINT_STATE_SCHEMA)
_fa_encoders = serialize_dynamic("std_msgs/msg/Float64MultiArray", FLOAT64_ARRAY_SCHEMA)
_encode_joint = _js_encoders["sensor_msgs/msg/JointState"]
_encode_float64 = _fa_encoders["std_msgs/msg/Float64MultiArray"]


def _register_joint_schema(writer: Writer) -> int:
    return writer.register_schema(
        name="sensor_msgs/msg/JointState",
        encoding="ros2msg",
        data=JOINT_STATE_SCHEMA.encode(),
    )


def _register_float64_schema(writer: Writer) -> int:
    return writer.register_schema(
        name="std_msgs/msg/Float64MultiArray",
        encoding="ros2msg",
        data=FLOAT64_ARRAY_SCHEMA.encode(),
    )


def _joint_msg(pos: np.ndarray, vel: np.ndarray, eff: np.ndarray | None = None) -> bytes:
    if eff is None:
        eff = np.zeros_like(pos)
    return _encode_joint(
        SimpleNamespace(
            name=JOINT_NAMES,
            position=pos.tolist(),
            velocity=vel.tolist(),
            effort=eff.tolist(),
        )
    )


def _action_msg(data: np.ndarray) -> bytes:
    return _encode_float64(SimpleNamespace(data=data.tolist()))


# ── Fixture 1: sample.mcap ────────────────────────────────────────────────────


def generate_sample_mcap() -> Path:
    """Minimal 2-topic MCAP: /joint_states (50Hz) + /action (50Hz), 2s, 100 steps.

    Fixture properties:
    - 100 messages per topic at 50Hz
    - t_start = 1_000_000_000 ns (t=1s since epoch)
    - 6-DOF joint states and 6-DOF actions
    - Deterministic: seed=42
    """
    rng = np.random.default_rng(42)
    path = FIXTURES_DIR / "sample.mcap"
    path.parent.mkdir(parents=True, exist_ok=True)

    t_start_ns = 1_000_000_000  # 1s epoch
    step_ns = 20_000_000  # 50Hz = 20ms
    n_steps = 100

    with open(path, "wb") as f:
        writer = Writer(f)
        writer.start(profile="ros2", library="torq-fixtures/1.0")

        schema_joint = _register_joint_schema(writer)
        schema_action = _register_float64_schema(writer)
        ch_joint = writer.register_channel(
            topic="/joint_states", message_encoding="cdr", schema_id=schema_joint
        )
        ch_action = writer.register_channel(
            topic="/action", message_encoding="cdr", schema_id=schema_action
        )

        for i in range(n_steps):
            t_ns = t_start_ns + i * step_ns
            pos = rng.uniform(-0.5, 0.5, 6).astype(np.float64)
            vel = rng.uniform(-0.1, 0.1, 6).astype(np.float64)
            action = rng.uniform(-0.2, 0.2, 6).astype(np.float64)

            writer.add_message(
                channel_id=ch_joint,
                log_time=t_ns,
                publish_time=t_ns,
                sequence=i,
                data=_joint_msg(pos, vel),
            )
            writer.add_message(
                channel_id=ch_action,
                log_time=t_ns,
                publish_time=t_ns,
                sequence=i,
                data=_action_msg(action),
            )

        writer.finish()

    print(f"Generated {path} ({path.stat().st_size:,} bytes)")
    return path


# ── Fixture 2: boundary_detection.mcap ───────────────────────────────────────


def generate_boundary_detection_mcap() -> Path:
    """3-episode MCAP with gripper-state and velocity boundaries.

    Episode structure (50Hz, 170 total steps):
        Ep1  (idx   0– 49): active, gripper open (0.04m)
        Gap1 (idx  50– 59): gripper closed (0.005m), near-zero vel
        Ep2  (idx  60–109): active, gripper open
        Gap2 (idx 110–119): gripper closed, near-zero vel
        Ep3  (idx 120–169): active, gripper open

    Ground truth gripper close transitions: idx 49 and idx 109
    (i.e. is_open changes from True→False at those indices).

    Metadata key "ground_truth_boundaries":
        "boundary_transition_indices": "49,109"
    """
    rng = np.random.default_rng(7)
    path = FIXTURES_DIR / "boundary_detection.mcap"

    t_start_ns = 1_000_000_000
    step_ns = 20_000_000  # 50Hz
    n_total = 170

    # Episode/gap mask
    ACTIVE_VEL = 0.15  # clearly non-zero
    NEAR_ZERO_VEL = 0.002  # below 0.01 threshold
    GRIPPER_OPEN = 0.040
    GRIPPER_CLOSED = 0.005

    with open(path, "wb") as f:
        writer = Writer(f)
        writer.start(profile="ros2", library="torq-fixtures/1.0")

        schema_joint = _register_joint_schema(writer)
        schema_action = _register_float64_schema(writer)

        # Gripper uses same Float64MultiArray schema (single-element data array)
        schema_gripper = writer.register_schema(
            name="std_msgs/msg/Float64MultiArray",
            encoding="ros2msg",
            data=FLOAT64_ARRAY_SCHEMA.encode(),
        )

        ch_joint = writer.register_channel(
            topic="/joint_states", message_encoding="cdr", schema_id=schema_joint
        )
        ch_action = writer.register_channel(
            topic="/action", message_encoding="cdr", schema_id=schema_action
        )
        ch_gripper = writer.register_channel(
            topic="/gripper", message_encoding="cdr", schema_id=schema_gripper
        )

        # Gap periods
        _gap1 = set(range(50, 60))
        _gap2 = set(range(110, 120))

        for i in range(n_total):
            t_ns = t_start_ns + i * step_ns
            is_gap = i in _gap1 or i in _gap2

            vel_mag = NEAR_ZERO_VEL if is_gap else ACTIVE_VEL
            pos = rng.uniform(-0.5, 0.5, 6).astype(np.float64)
            vel = rng.uniform(-vel_mag, vel_mag, 6).astype(np.float64)
            action = rng.uniform(-0.2, 0.2, 6).astype(np.float64)
            gripper_val = GRIPPER_CLOSED if is_gap else GRIPPER_OPEN

            writer.add_message(
                channel_id=ch_joint,
                log_time=t_ns,
                publish_time=t_ns,
                sequence=i,
                data=_joint_msg(pos, vel),
            )
            writer.add_message(
                channel_id=ch_action,
                log_time=t_ns,
                publish_time=t_ns,
                sequence=i,
                data=_action_msg(action),
            )
            writer.add_message(
                channel_id=ch_gripper,
                log_time=t_ns,
                publish_time=t_ns,
                sequence=i,
                data=_action_msg(np.array([gripper_val])),
            )

        # Ground truth metadata: close transition happens at is_open[49]→is_open[50]
        # np.diff(is_open)[49] = False - True < 0 → transition at index 49
        writer.add_metadata(
            name="ground_truth_boundaries",
            data={"boundary_transition_indices": "49,109"},
        )

        writer.finish()

    print(f"Generated {path} ({path.stat().st_size:,} bytes)")
    return path


# ── Fixture 2b: velocity_only.mcap ────────────────────────────────────────────


def generate_velocity_only_mcap() -> Path:
    """3-episode MCAP using velocity boundaries only (no gripper topic).

    Episode structure (50Hz, 170 total steps):
        Ep1  (idx   0– 49): active velocity (~0.15 rad/s)
        Gap1 (idx  50– 59): near-zero velocity (<0.01 rad/s)
        Ep2  (idx  60–109): active velocity
        Gap2 (idx 110–119): near-zero velocity
        Ep3  (idx 120–169): active velocity

    No /gripper topic — forces velocity boundary detection (priority 2).
    """
    rng = np.random.default_rng(13)
    path = FIXTURES_DIR / "velocity_only.mcap"

    t_start_ns = 1_000_000_000
    step_ns = 20_000_000  # 50Hz
    n_total = 170

    ACTIVE_VEL = 0.15
    NEAR_ZERO_VEL = 0.002

    with open(path, "wb") as f:
        writer = Writer(f)
        writer.start(profile="ros2", library="torq-fixtures/1.0")

        schema_joint = _register_joint_schema(writer)
        schema_action = _register_float64_schema(writer)

        ch_joint = writer.register_channel(
            topic="/joint_states", message_encoding="cdr", schema_id=schema_joint
        )
        ch_action = writer.register_channel(
            topic="/action", message_encoding="cdr", schema_id=schema_action
        )

        _gap1 = set(range(50, 60))
        _gap2 = set(range(110, 120))

        for i in range(n_total):
            t_ns = t_start_ns + i * step_ns
            is_gap = i in _gap1 or i in _gap2

            vel_mag = NEAR_ZERO_VEL if is_gap else ACTIVE_VEL
            pos = rng.uniform(-0.5, 0.5, 6).astype(np.float64)
            vel = rng.uniform(-vel_mag, vel_mag, 6).astype(np.float64)
            action = rng.uniform(-0.2, 0.2, 6).astype(np.float64)

            writer.add_message(
                channel_id=ch_joint,
                log_time=t_ns,
                publish_time=t_ns,
                sequence=i,
                data=_joint_msg(pos, vel),
            )
            writer.add_message(
                channel_id=ch_action,
                log_time=t_ns,
                publish_time=t_ns,
                sequence=i,
                data=_action_msg(action),
            )

        writer.finish()

    print(f"Generated {path} ({path.stat().st_size:,} bytes)")
    return path


# ── Fixture 3: corrupt.mcap ───────────────────────────────────────────────────


def generate_corrupt_mcap() -> Path:
    """MCAP with 5 valid messages, 1 corrupt (bad CDR data), 5 more valid messages.

    The corrupt message has a valid MCAP message record wrapper but contains
    random bytes that fail CDR decoding. The ingestion layer should:
    - Emit a logger.warning for the corrupt message
    - Continue and return episodes from the 10 valid messages
    """
    rng = np.random.default_rng(99)
    path = FIXTURES_DIR / "corrupt.mcap"

    t_start_ns = 1_000_000_000
    step_ns = 20_000_000

    with open(path, "wb") as f:
        writer = Writer(f)
        writer.start(profile="ros2", library="torq-fixtures/1.0")

        schema_joint = _register_joint_schema(writer)
        ch_joint = writer.register_channel(
            topic="/joint_states", message_encoding="cdr", schema_id=schema_joint
        )

        # 5 valid messages
        for i in range(5):
            t_ns = t_start_ns + i * step_ns
            pos = rng.uniform(-0.5, 0.5, 6).astype(np.float64)
            vel = rng.uniform(-0.1, 0.1, 6).astype(np.float64)
            writer.add_message(
                channel_id=ch_joint,
                log_time=t_ns,
                publish_time=t_ns,
                sequence=i,
                data=_joint_msg(pos, vel),
            )

        # 1 corrupt message: valid MCAP record wrapper, invalid CDR payload
        corrupt_payload = bytes(range(32))  # 32 bytes of 0x00..0x1f — invalid CDR
        writer.add_message(
            channel_id=ch_joint,
            log_time=t_start_ns + 5 * step_ns,
            publish_time=t_start_ns + 5 * step_ns,
            sequence=5,
            data=corrupt_payload,
        )

        # 5 more valid messages
        for i in range(6, 11):
            t_ns = t_start_ns + i * step_ns
            pos = rng.uniform(-0.5, 0.5, 6).astype(np.float64)
            vel = rng.uniform(-0.1, 0.1, 6).astype(np.float64)
            writer.add_message(
                channel_id=ch_joint,
                log_time=t_ns,
                publish_time=t_ns,
                sequence=i,
                data=_joint_msg(pos, vel),
            )

        writer.finish()

    print(f"Generated {path} ({path.stat().st_size:,} bytes)")
    return path


# ── Fixture 4: empty.mcap ─────────────────────────────────────────────────────


def generate_empty_mcap() -> Path:
    """Valid MCAP header with zero messages and zero channels."""
    path = FIXTURES_DIR / "empty.mcap"

    with open(path, "wb") as f:
        writer = Writer(f)
        writer.start(profile="ros2", library="torq-fixtures/1.0")
        writer.finish()

    print(f"Generated {path} ({path.stat().st_size:,} bytes)")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# HDF5 (robomimic format) fixtures
# ══════════════════════════════════════════════════════════════════════════════


def generate_robomimic_hdf5() -> None:
    """Generate robomimic-format HDF5 fixtures: simple, images, corrupt."""
    import h5py

    rng = np.random.default_rng(99)

    # ── simple: 2 demos, joint_pos + joint_vel + actions, no images ──
    path = FIXTURES_DIR / "robomimic_simple.hdf5"
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        for demo_idx, t_len in [(0, 30), (1, 20)]:
            demo = data.create_group(f"demo_{demo_idx}")
            demo.attrs["num_samples"] = t_len
            demo.create_dataset("actions", data=rng.random((t_len, 6)).astype(np.float32))
            obs = demo.create_group("obs")
            obs.create_dataset("joint_pos", data=rng.random((t_len, 6)).astype(np.float32))
            obs.create_dataset("joint_vel", data=rng.random((t_len, 6)).astype(np.float32))
    print(f"Generated {path} ({path.stat().st_size:,} bytes)")

    # ── images: 1 demo with agentview_image ──
    path = FIXTURES_DIR / "robomimic_images.hdf5"
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        demo = data.create_group("demo_0")
        t_len = 10
        demo.attrs["num_samples"] = t_len
        demo.create_dataset("actions", data=rng.random((t_len, 6)).astype(np.float32))
        obs = demo.create_group("obs")
        obs.create_dataset("joint_pos", data=rng.random((t_len, 6)).astype(np.float32))
        obs.create_dataset(
            "agentview_image",
            data=rng.integers(0, 255, (t_len, 48, 64, 3), dtype=np.uint8),
        )
    print(f"Generated {path} ({path.stat().st_size:,} bytes)")

    # ── corrupt: valid HDF5 then truncated ──
    path = FIXTURES_DIR / "corrupt.hdf5"
    tmp = path.with_suffix(".tmp.hdf5")
    with h5py.File(tmp, "w") as f:
        data = f.create_group("data")
        demo = data.create_group("demo_0")
        demo.create_dataset("actions", data=rng.random((10, 6)).astype(np.float32))
    raw = tmp.read_bytes()
    path.write_bytes(raw[:-100])
    tmp.unlink()
    print(f"Generated {path} ({path.stat().st_size:,} bytes)")


# ══════════════════════════════════════════════════════════════════════════════
# LeRobot v3.0 format fixtures
# ══════════════════════════════════════════════════════════════════════════════


def _write_stub_mp4(path: Path, n_frames: int, height: int, width: int, rng) -> None:
    """Write a minimal MP4 using av (PyAV). Falls back to stub bytes if av unavailable."""
    try:
        import av

        with av.open(str(path), mode="w") as container:
            stream = container.add_stream("libx264", rate=50)
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            for _ in range(n_frames):
                frame_np = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
                av_frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
                for packet in stream.encode(av_frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    except ImportError:
        # Stub file so path exists — ImageSequence is lazy, won't decode at ingest
        path.write_bytes(b"\x00" * 64)


def generate_lerobot_fixture() -> None:
    """Generate a minimal LeRobot v3.0 dataset fixture (2 episodes, Parquet + MP4)."""
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(77)
    root = FIXTURES_DIR / "lerobot"

    # Create directory structure
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "videos" / "chunk-000").mkdir(parents=True, exist_ok=True)

    # ── meta/info.json ──
    info = {
        "fps": 50,
        "robot_type": "aloha",
        "total_episodes": 2,
        "total_frames": 60,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [14]},
            "action": {"dtype": "float32", "shape": [14]},
            "observation.images.top": {
                "dtype": "video",
                "shape": [3, 48, 64],
                "video_info": {"fps": 50},
            },
            "timestamp": {"dtype": "float32", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    print(f"Generated {root / 'meta' / 'info.json'}")

    # ── Parquet episode files ──
    t_per_ep = 30
    for ep_idx in range(2):
        rows = {
            "episode_index": pa.array([ep_idx] * t_per_ep, type=pa.int64()),
            "frame_index": pa.array(list(range(t_per_ep)), type=pa.int64()),
            "index": pa.array(
                list(range(ep_idx * t_per_ep, (ep_idx + 1) * t_per_ep)), type=pa.int64()
            ),
            "timestamp": pa.array([i / 50.0 for i in range(t_per_ep)], type=pa.float32()),
        }
        for i in range(14):
            rows[f"observation.state_{i}"] = pa.array(
                rng.standard_normal(t_per_ep).astype(np.float32).tolist(), type=pa.float32()
            )
            rows[f"action_{i}"] = pa.array(
                rng.standard_normal(t_per_ep).astype(np.float32).tolist(), type=pa.float32()
            )
        table = pa.table(rows)
        parquet_path = root / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
        pq.write_table(table, str(parquet_path))
        print(f"Generated {parquet_path} ({parquet_path.stat().st_size:,} bytes)")

    # ── MP4 video files (minimal 30-frame 48×64) ──
    for ep_idx in range(2):
        mp4_path = (
            root / "videos" / "chunk-000" / f"observation.images.top_episode_{ep_idx:06d}.mp4"
        )
        _write_stub_mp4(mp4_path, t_per_ep, 48, 64, rng)
        print(f"Generated {mp4_path}")

    # ── lerobot_no_info: same structure but without info.json (for AC #3) ──
    no_info_root = FIXTURES_DIR / "lerobot_no_info"
    (no_info_root / "meta").mkdir(parents=True, exist_ok=True)
    (no_info_root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    print(f"Generated {no_info_root} (no info.json)")


# ── Entry point ───────────────────────────────────────────────────────────────


if __name__ == "__main__":
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing fixtures to {FIXTURES_DIR}\n")

    generate_sample_mcap()
    generate_boundary_detection_mcap()
    generate_velocity_only_mcap()
    generate_corrupt_mcap()
    generate_empty_mcap()
    generate_robomimic_hdf5()
    generate_lerobot_fixture()

    print("\nAll fixtures generated successfully.")
