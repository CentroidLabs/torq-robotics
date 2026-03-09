# Story 2.6: LeRobot v3.0 Ingestion

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a robotics researcher,
I want to load LeRobot v3.0 datasets (Parquet + MP4 format),
so that I can work with datasets prepared in the LeRobot format without a custom loader.

## Acceptance Criteria

1. **Given** a LeRobot v3.0 directory with `meta/info.json`, `data/chunk-*/` Parquet files, and `videos/chunk-*/` MP4 files,
   **When** `lerobot.ingest(path)` is called,
   **Then** one Episode is returned per episode in the dataset with features mapped from `info.json`.

2. **Given** a LeRobot dataset with camera observations,
   **When** ingestion runs,
   **Then** video data is returned as `ImageSequence` objects with lazy loading (no frames decoded at ingest time).

3. **Given** a LeRobot dataset where `meta/info.json` is missing or malformed,
   **When** ingestion is called,
   **Then** `TorqIngestError` is raised with the path and a message explaining that `meta/info.json` is required.

## Tasks / Subtasks

- [x] Task 1: Extend `tests/fixtures/generate_fixtures.py` — LeRobot fixture (prerequisite for Task 3)
  - [x] Add function `generate_lerobot_fixture()` to the existing `generate_fixtures.py`
  - [x] Fixture: `tests/fixtures/data/lerobot/` — minimal LeRobot v3.0 directory structure:
    - `meta/info.json` — episode count, features schema, fps, robot_type
    - `data/chunk-000/episode_000000.parquet` — episode 0 data
    - `data/chunk-000/episode_000001.parquet` — episode 1 data
    - `videos/chunk-000/observation.images.top_episode_000000.mp4` — camera for episode 0
    - `videos/chunk-000/observation.images.top_episode_000001.mp4` — camera for episode 1
  - [x] `meta/info.json` content:
    ```json
    {
      "fps": 50,
      "robot_type": "aloha",
      "total_episodes": 2,
      "total_frames": 60,
      "features": {
        "observation.state": {"dtype": "float32", "shape": [14], "names": {"joints": [...]}},
        "action": {"dtype": "float32", "shape": [14]},
        "observation.images.top": {"dtype": "video", "shape": [3, 480, 640], "video_info": {"fps": 50}},
        "timestamp": {"dtype": "float32", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]}
      }
    }
    ```
  - [x] Each Parquet file columns: `observation.state` (14 float32 cols), `action` (14 float32 cols), `timestamp` (float32, seconds), `episode_index` (int64), `frame_index` (int64), `index` (int64)
  - [x] Episode 0: 30 rows, timestamps 0.0–0.58s at 50Hz; Episode 1: 30 rows
  - [x] MP4 files: minimal 30-frame 48×64 pixel videos (use imageio if available, else create 1-frame MP4)
  - [x] Fixture: `tests/fixtures/data/lerobot_no_info/` — same structure but without `meta/info.json` (for AC #3 test)
  - [x] All fixtures: `rng = np.random.default_rng(77)`, deterministic
  - [x] Update `tests/conftest.py` with path helpers: `lerobot_dataset`, `lerobot_no_info` (NOT tests/fixtures/conftest.py — see Story 2.5 R2)

- [x] Task 2: Create `src/torq/ingest/lerobot.py` — LeRobot v3.0 ingestion (AC: #1, #2, #3)
  - [x] Implement `ingest(path: str | Path) -> list[Episode]`
  - [x] **Step 1 — Load info.json**:
    - Path: `{path}/meta/info.json`
    - Raise `TorqIngestError` if missing or JSON-malformed (AC #3)
    - Extract: `fps`, `total_episodes`, `features` dict
  - [x] **Step 2 — Discover Parquet chunks**:
    - Glob `{path}/data/chunk-*/*.parquet`, sort by episode_index
    - Load all Parquet files with `pyarrow.parquet.read_table()`, concatenate via `pyarrow.concat_tables()`
    - Group rows by `episode_index` column
  - [x] **Step 3 — Discover video files**:
    - Glob `{path}/videos/chunk-*/*.mp4` — group by episode index from filename
    - Filename pattern: `{feature_key}_episode_{NNNNNN}.mp4` where feature_key is the camera feature name (e.g., `observation.images.top`)
  - [x] **Step 4 — Feature mapping** (per episode):
    - `observation.state` columns → `observations["state"]` numpy array `[T, D]` float32
    - `action` columns → `actions` numpy array `[T, A]` float32
    - `timestamp` column → convert float seconds to `np.int64` nanoseconds: `(timestamps_s * 1e9).astype(np.int64)`
    - Camera features (key contains `"images"` or `"video"` dtype in features): video path → `ImageSequence(path)` lazy (AC #2)
  - [x] **Step 5 — Episode construction** (per episode_index):
    - `episode_id = ""` (placeholder)
    - `observations = {"state": arr, "camera_name": ImageSequence(mp4_path), ...}`
    - `actions = arr`
    - `timestamps = np.int64 nanoseconds`
    - `source_path = Path(path)`
    - `metadata = {"task": "", "embodiment": info.get("robot_type", "")}`
  - [x] **Column name discovery**: do NOT hardcode column names. Use prefix-based detection:
    - `observation.state` → columns starting with `"observation.state"` (or exact match if single vector column)
    - `action` → columns starting with `"action"`
    - Use `info["features"]` to know expected shape and dtype
  - [x] If `total_episodes` in info.json is 0 → return `[]` + `logger.warning()`
  - [x] If a Parquet chunk file is missing → `logger.warning()` + skip (continue with available data)
  - [x] Add Google-style docstrings, `__all__ = ["ingest"]`, module-level logger

- [x] Task 3: Update `src/torq/ingest/__init__.py` — expose `ingest_lerobot`
  - [x] Add `from torq.ingest.lerobot import ingest as ingest_lerobot`
  - [x] Add `"ingest_lerobot"` to `__all__`

- [x] Task 4: Write unit tests in `tests/unit/test_ingest_lerobot.py` — 5 tests (AC: #1, #2, #3)
  - [x] `test_ingest_lerobot_returns_one_episode_per_episode_index` — fixture → 2 episodes
  - [x] `test_ingest_lerobot_state_and_action_arrays_populated` — obs["state"] and actions are non-empty float32 arrays
  - [x] `test_ingest_lerobot_timestamps_are_int64_nanoseconds` — timestamps dtype `np.int64`, values > 0
  - [x] `test_ingest_lerobot_camera_is_lazy_image_sequence` — `observations["top"]` is ImageSequence, `_cache is None` (not loaded)
  - [x] `test_ingest_lerobot_missing_info_json_raises_torq_ingest_error` — `lerobot_no_info` path → `TorqIngestError` with "info.json" in message

- [x] Task 5: Run full test suite and verify no regressions
  - [x] All previous 148 tests still pass (153 total with 5 new)
  - [x] All 5 new tests pass
  - [x] `ruff check src/ tests/ tests/fixtures/ && ruff format --check src/ tests/ tests/fixtures/` clean on changed files
  - [x] `python tests/fixtures/generate_fixtures.py` runs cleanly

## Dev Notes

### LeRobot v3.0 Directory Structure

```
{dataset_root}/
├── meta/
│   └── info.json          ← REQUIRED — episode count, fps, features schema
├── data/
│   ├── chunk-000/
│   │   ├── episode_000000.parquet   ← all rows for episode 0
│   │   └── episode_000001.parquet   ← all rows for episode 1
│   └── chunk-001/                   ← next chunk (if > N episodes per chunk)
│       └── episode_000100.parquet
└── videos/
    └── chunk-000/
        ├── observation.images.top_episode_000000.mp4
        └── observation.images.top_episode_000001.mp4
```

**Parquet column names** (LeRobot uses dot-notation feature keys):
```
observation.state_0, observation.state_1, ... observation.state_13  ← 14 cols for ALOHA
action_0, action_1, ... action_13                                    ← 14 cols
timestamp                                                            ← float32, seconds
episode_index                                                        ← int64
frame_index                                                          ← int64
index                                                                ← int64 (global frame idx)
```

**Note:** LeRobot Parquet files may have either:
- Individual columns per dimension: `observation.state_0`, `observation.state_1`, ...
- Or a single list-type column: `observation.state` containing `list<float32>`

Check `info["features"]["observation.state"]["shape"]` to determine expected dims. Use prefix match `[c for c in columns if c.startswith("observation.state")]` to collect all dimension columns.

### Exact Implementation — `src/torq/ingest/lerobot.py`

```python
"""LeRobot v3.0 dataset ingestion for the Torq SDK.

Parses LeRobot format datasets (Parquet + MP4) and returns canonical Episodes.
Requires meta/info.json to be present for feature schema discovery.

Timestamps in LeRobot are float seconds — converted to np.int64 nanoseconds
at ingest boundary (single conversion point).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from torq.episode import Episode
from torq.errors import TorqIngestError
from torq.media.image_sequence import ImageSequence

__all__ = ["ingest"]

logger = logging.getLogger(__name__)


def ingest(path: str | Path) -> list[Episode]:
    """Ingest a LeRobot v3.0 dataset directory and return one Episode per episode index.

    Args:
        path: Root directory of the LeRobot dataset (must contain ``meta/info.json``).

    Returns:
        List of Episode objects sorted by episode_index.

    Raises:
        TorqIngestError: If ``meta/info.json`` is missing or malformed, or if
            required data files cannot be read.
    """
    root = Path(path)
    info = _load_info(root)
    fps: float = info.get("fps", 50)
    robot_type: str = info.get("robot_type", "")
    features: dict = info.get("features", {})

    # Identify camera feature keys (those with dtype "video" or "image")
    camera_keys = {
        k for k, v in features.items()
        if v.get("dtype") in ("video", "image") or "images" in k
    }

    # Load all Parquet data, grouped by episode_index
    all_tables = _load_parquet_chunks(root)
    if all_tables is None:
        return []

    # Group rows by episode_index
    import pyarrow as pa
    import pyarrow.compute as pc

    episode_indices = all_tables.column("episode_index").to_pylist()
    unique_episodes = sorted(set(episode_indices))

    if not unique_episodes:
        logger.warning("LeRobot dataset at '%s' has no episodes.", root)
        return []

    # Discover video files: {(camera_key_suffix, episode_idx) → Path}
    video_map = _discover_videos(root)

    episodes: list[Episode] = []
    for ep_idx in unique_episodes:
        mask = pc.equal(all_tables.column("episode_index"), ep_idx)
        ep_table = all_tables.filter(mask)
        episode = _build_episode(ep_table, ep_idx, root, camera_keys, video_map, robot_type, fps)
        episodes.append(episode)

    return episodes


def _load_info(root: Path) -> dict:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise TorqIngestError(
            f"LeRobot dataset at '{root}' is missing 'meta/info.json'. "
            f"This file is required for feature schema discovery. "
            f"Ensure the dataset follows LeRobot v3.0 format with a 'meta/' directory."
        )
    try:
        return json.loads(info_path.read_text())
    except json.JSONDecodeError as exc:
        raise TorqIngestError(
            f"'meta/info.json' at '{root}' is malformed JSON: {exc}. "
            f"Validate it with: python -c \"import json; json.load(open('{info_path}'))\""
        ) from exc


def _load_parquet_chunks(root: Path):
    """Load and concatenate all Parquet chunk files, sorted by episode_index."""
    import pyarrow as pa

    chunk_files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not chunk_files:
        logger.warning("LeRobot dataset at '%s' has no Parquet data files.", root)
        return None

    tables = []
    for chunk_path in chunk_files:
        try:
            tables.append(pq.read_table(str(chunk_path)))
        except Exception as exc:
            logger.warning("Skipping unreadable Parquet chunk '%s': %s", chunk_path, exc)

    if not tables:
        return None
    return pa.concat_tables(tables)


def _discover_videos(root: Path) -> dict[tuple[str, int], Path]:
    """Map (camera_key_suffix, episode_idx) → MP4 path."""
    video_map: dict[tuple[str, int], Path] = {}
    for mp4 in (root / "videos").glob("chunk-*/*.mp4"):
        stem = mp4.stem  # e.g. "observation.images.top_episode_000000"
        if "_episode_" not in stem:
            continue
        key_part, ep_part = stem.rsplit("_episode_", 1)
        try:
            ep_idx = int(ep_part)
        except ValueError:
            continue
        # Normalise camera key to short name: "observation.images.top" → "top"
        short_key = key_part.split(".")[-1]
        video_map[(short_key, ep_idx)] = mp4
    return video_map


def _build_episode(
    table,
    ep_idx: int,
    root: Path,
    camera_keys: set[str],
    video_map: dict,
    robot_type: str,
    fps: float,
) -> Episode:
    """Build one Episode from a filtered PyArrow table (rows for one episode)."""
    columns = set(table.column_names)

    # ── Timestamps: float seconds → np.int64 nanoseconds ──
    if "timestamp" in columns:
        ts_s = table.column("timestamp").to_pylist()
        timestamps = (np.array(ts_s, dtype=np.float64) * 1e9).astype(np.int64)
    else:
        # Synthesise at fps if timestamp column absent
        T = len(table)
        step_ns = int(1e9 / fps)
        timestamps = np.arange(T, dtype=np.int64) * step_ns

    # ── Actions ──
    action_cols = sorted([c for c in columns if c == "action" or c.startswith("action_")])
    if action_cols:
        actions = np.stack(
            [table.column(c).to_pylist() for c in action_cols], axis=1
        ).astype(np.float32)
    else:
        actions = np.empty((len(timestamps), 0), dtype=np.float32)

    # ── Observations (continuous) ──
    observations: dict[str, object] = {}
    state_cols = sorted([c for c in columns if c == "observation.state" or c.startswith("observation.state_")])
    if state_cols:
        observations["state"] = np.stack(
            [table.column(c).to_pylist() for c in state_cols], axis=1
        ).astype(np.float32)

    # ── Camera observations (lazy ImageSequence) ──
    for short_key, path in {
        k: v for (k, eidx), v in video_map.items() if eidx == ep_idx
    }.items():
        observations[short_key] = ImageSequence(path)

    return Episode(
        episode_id="",
        observations=observations,
        actions=actions,
        timestamps=timestamps,
        source_path=root,
        metadata={"task": "", "embodiment": robot_type},
    )
```

### Fixture Generation — `generate_lerobot_fixture()`

The fixture generator needs `pyarrow` (core dep) and optionally `imageio` for MP4 files. Use a fallback for MP4 if imageio is not available:

```python
def generate_lerobot_fixture():
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(77)
    root = FIXTURES_DIR / "lerobot"
    root.mkdir(parents=True, exist_ok=True)
    (root / "meta").mkdir(exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "videos" / "chunk-000").mkdir(parents=True, exist_ok=True)

    # info.json
    info = {
        "fps": 50,
        "robot_type": "aloha",
        "total_episodes": 2,
        "total_frames": 60,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [14]},
            "action": {"dtype": "float32", "shape": [14]},
            "observation.images.top": {"dtype": "video", "shape": [3, 48, 64]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
        }
    }
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=2))

    # Parquet episodes
    for ep_idx, T in [(0, 30), (1, 30)]:
        rows = {
            "episode_index": pa.array([ep_idx] * T, type=pa.int64()),
            "frame_index": pa.array(list(range(T)), type=pa.int64()),
            "index": pa.array(list(range(ep_idx * T, (ep_idx + 1) * T)), type=pa.int64()),
            "timestamp": pa.array(
                [i / 50.0 for i in range(T)], type=pa.float32()
            ),
        }
        for i in range(14):
            rows[f"observation.state_{i}"] = pa.array(
                rng.standard_normal(T).astype(np.float32).tolist(), type=pa.float32()
            )
            rows[f"action_{i}"] = pa.array(
                rng.standard_normal(T).astype(np.float32).tolist(), type=pa.float32()
            )
        table = pa.table(rows)
        pq.write_table(
            table,
            str(root / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet")
        )

    # MP4 files (minimal — 1x1 pixel if imageio unavailable)
    for ep_idx in range(2):
        mp4_path = root / "videos" / "chunk-000" / f"observation.images.top_episode_{ep_idx:06d}.mp4"
        try:
            import imageio
            frames = rng.integers(0, 255, (T, 48, 64, 3), dtype=np.uint8)
            imageio.v3.imwrite(str(mp4_path), frames, fps=50)
        except ImportError:
            # Write a stub file so path exists (ImageSequence will fail on access, but lazy)
            mp4_path.write_bytes(b"\x00" * 64)
        print(f"Generated {mp4_path}")

    # lerobot_no_info — same structure without info.json (for AC #3 test)
    no_info_root = FIXTURES_DIR / "lerobot_no_info"
    no_info_root.mkdir(parents=True, exist_ok=True)
    (no_info_root / "meta").mkdir(exist_ok=True)
    (no_info_root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    # Intentionally no info.json written
    print(f"Generated {no_info_root} (no info.json)")
```

### Timestamp Conversion — Critical Detail

LeRobot stores timestamps as **float seconds** in the Parquet `timestamp` column. This is the single point where float seconds enter the Torq system:

```python
# CORRECT — single conversion at ingest boundary
timestamps = (np.array(ts_s, dtype=np.float64) * 1e9).astype(np.int64)

# WRONG — loses precision
timestamps = np.array(ts_s, dtype=np.float32) * 1e9  # float32 only has ~7 decimal digits
```

Use `float64` for the intermediate multiplication, then cast to `int64`. Never use `float32` for timestamp arithmetic.

### `src/torq/ingest/__init__.py` After This Story

```python
"""Torq ingest sub-package — format parsers and temporal alignment."""

from torq.ingest.alignment import align
from torq.ingest.hdf5 import ingest as ingest_hdf5
from torq.ingest.lerobot import ingest as ingest_lerobot
from torq.ingest.mcap import ingest as ingest_mcap

__all__ = ["align", "ingest_hdf5", "ingest_lerobot", "ingest_mcap"]
```

### Project Structure Notes

#### Files to create/modify

```
tests/fixtures/
├── generate_fixtures.py              ← MODIFY (add generate_lerobot_fixture function)
├── conftest.py                       ← MODIFY (add lerobot_dataset, lerobot_no_info fixtures)
└── data/
    ├── lerobot/                      ← GENERATE
    │   ├── meta/info.json
    │   ├── data/chunk-000/episode_000000.parquet
    │   ├── data/chunk-000/episode_000001.parquet
    │   └── videos/chunk-000/observation.images.top_episode_000000.mp4
    │   └── videos/chunk-000/observation.images.top_episode_000001.mp4
    └── lerobot_no_info/              ← GENERATE (no info.json)
        └── meta/

src/torq/ingest/
└── lerobot.py                        ← CREATE

tests/unit/
└── test_ingest_lerobot.py            ← CREATE (5 tests)
```

#### Files to modify

```
src/torq/ingest/__init__.py           ← MODIFY (add ingest_lerobot)
```

#### Files NOT touched

```
src/torq/ingest/mcap.py              ← No changes
src/torq/ingest/hdf5.py              ← No changes
src/torq/ingest/alignment.py         ← No changes
src/torq/storage/                    ← No changes (hasattr duck-typing already in place from Story 2.5)
src/torq/__init__.py                 ← No changes (tq.ingest() is Story 2.7)
```

### Architecture Compliance

| Rule | Requirement | This Story |
|---|---|---|
| `ingest/` import direction | imports episode, errors, media, types | ✓ lerobot.py → episode, errors, media.image_sequence |
| No circular imports | lerobot → episode, errors, ImageSequence | ✓ |
| Timestamps are `np.int64` nanoseconds | float seconds → int64 ns at ingest boundary | ✓ `(ts_float64 * 1e9).astype(np.int64)` |
| Float seconds only at boundary | single conversion point in `_build_episode` | ✓ |
| `ImageSequence` for camera obs | lazy loading, no frames at ingest | ✓ `ImageSequence(mp4_path)` |
| `episode_id = ""` placeholder | storage layer assigns real ID | ✓ |
| `TorqIngestError` for missing info.json | with path + reason + resolution | ✓ per AC #3 |
| `logger.warning()` for skipped chunks | not exception | ✓ |
| `pathlib.Path` everywhere | never os.path | ✓ |
| Google-style docstrings | all public classes and functions | ✓ |
| `logging.getLogger(__name__)` | → `torq.ingest.lerobot` | ✓ |
| `ruff format` line length 100 | formatter standard | ✓ |
| `ruff check` on tests and fixtures too | Story 2.4 lesson | ✓ run on src/ tests/ tests/fixtures/ |

### Previous Story Intelligence (from Stories 2.1–2.5)

- **147 tests passing** as of Story 2.5 (after two code review rounds). Zero regressions is a hard requirement.
- **`hasattr(obs, 'frames')` duck-typing** is now the pattern in `storage/_impl.py` and `storage/parquet.py` (Story 2.5 fix). `ImageSequence` has `.frames`, so it will correctly route to MP4 save and skip Parquet serialisation.
- **`FrameProvider` Protocol** exists in `src/torq/types.py` (added in Story 2.5 R2 review). `save_video()` in `storage/video.py` now accepts `FrameProvider` type. `ImageSequence` satisfies this protocol natively.
- **`Episode.observations`** is typed `dict[str, np.ndarray | object]` (widened in Story 2.5 R2 review) — the `object` union handles `ImageSequence` and `_InMemoryFrames` without a circular import.
- **Fixture helpers are in `tests/conftest.py`** (not `tests/fixtures/conftest.py`) — Story 2.5 R2 review moved shared fixtures there for visibility across all test subdirectories. Add LeRobot path helpers to `tests/conftest.py`.
- **`tests/fixtures/conftest.py`** exists as a placeholder doc only — do NOT put fixture helpers there.
- **`episode_id = ""` placeholder** — established across MCAP (2.4) and HDF5 (2.5). Same here.
- **`ruff check` must include test and fixture files** — Story 2.4 debug log: ruff violations were found in tests/fixtures after claiming "ruff clean". Run `ruff check src/ tests/ tests/fixtures/`.
- **Fixture generator `generate_fixtures.py`** — already has MCAP and HDF5 generators. ADD to it, never replace. Call `generate_lerobot_fixture()` in the `if __name__ == "__main__":` block.
- **`pyarrow` is a core dep** — no import guard needed. Use directly.
- **`pyarrow.compute`** (`import pyarrow.compute as pc`) is part of pyarrow — no additional install needed. Use `pc.equal()` for filtering by episode_index.
- **LeRobot camera key normalisation** — the video filename uses the full dotted key (`observation.images.top`) but the observation dict key should be the short suffix (`top`). This simplifies user access: `episode.observations["top"]` not `episode.observations["observation.images.top"]`.
- **Test speed**: all 5 unit tests must be < 1s. LeRobot tests read from small fixture files (Parquet + stub MP4). The `test_ingest_lerobot_camera_is_lazy_image_sequence` test checks `_cache is None` on the ImageSequence — this is fast since no decoding happens.

### References

- Story 2.6 AC: [Source: planning-artifacts/epics.md — Epic 2, Story 2.6]
- LeRobot ingestion module: [Source: planning-artifacts/architecture.md — FR-to-File-Mapping: `ingest/lerobot.py`]
- LeRobot v3.0 format: Parquet + MP4, meta/info.json required [Source: planning-artifacts/epics.md — Story 2.6 AC1]
- `ImageSequence` for lazy camera loading: [Source: planning-artifacts/epics.md — Story 2.6 AC2]
- Timestamp format (float seconds → int64 ns at boundary): [Source: planning-artifacts/architecture.md — Timestamp Format]
- Episode `episode_id=""` placeholder: [Source: planning-artifacts/architecture.md — Episode ID Format]
- Test count targets (5 tests): [Source: planning-artifacts/architecture.md — test_ingest_lerobot.py: F11 — 5 tests]
- `hasattr(obs, 'frames')` duck-typing now in storage layer: [Source: 2-5-hdf5-robomimic-ingestion.md — File List]
- `FrameProvider` Protocol in `src/torq/types.py`: [Source: 2-5-hdf5-robomimic-ingestion.md — Story 2.5 R2 Review]
- Shared fixture helpers are in `tests/conftest.py`: [Source: 2-5-hdf5-robomimic-ingestion.md — Story 2.5 R2 Review]
- Fixture generator already exists: [Source: tests/fixtures/generate_fixtures.py — Stories 2.4, 2.5]
- `ingest/__init__.py` current state: [Source: src/torq/ingest/__init__.py]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

No blocking issues encountered during implementation.

### Completion Notes List

- Implemented `src/torq/ingest/lerobot.py` — full LeRobot v3.0 ingestion with Parquet + MP4 support
- Feature mapping: `observation.state_*` columns → `observations["state"]` float32 array, `action_*` → `actions` float32 array
- Camera observations loaded as lazy `ImageSequence` objects (no frame decoding at ingest time) — AC #2
- Timestamps converted from float seconds → `np.int64` nanoseconds at ingest boundary using float64 intermediate
- `TorqIngestError` raised for missing/malformed `meta/info.json` with actionable message — AC #3
- Added `generate_lerobot_fixture()` to existing `tests/fixtures/generate_fixtures.py` (deterministic, seed=77)
- Created `lerobot/` and `lerobot_no_info/` fixture directories
- Exposed `ingest_lerobot` in `src/torq/ingest/__init__.py`
- Added `lerobot_dataset` and `lerobot_no_info` path helpers to `tests/conftest.py`
- All 5 new tests pass, all 148 previous tests pass (153 total), zero regressions
- `ruff check src/ tests/ tests/fixtures/` and `ruff format --check` clean
- ✅ Resolved review finding [MEDIUM]: `camera_keys` now used for validation — warns if camera feature in info.json has no matching MP4
- ✅ Resolved review finding [MEDIUM]: Bare `KeyError` on missing `episode_index` column now wrapped in `TorqIngestError` with available columns
- ✅ Resolved review finding [MEDIUM]: Generic observation discovery via `_group_observation_columns()` — extracts ALL `observation.*` non-camera columns, not just `observation.state`
- ✅ Resolved review finding [MEDIUM]: Added test for malformed JSON info.json (AC3 full coverage)
- ✅ Resolved review finding [LOW]: Added 3 edge case tests — empty dataset, no videos dir, unreadable Parquet chunk
- ✅ Resolved review finding [LOW]: Added test for `str` path acceptance
- ✅ Resolved review finding [LOW]: Added `root.is_dir()` validation with clear `TorqIngestError`
- ✅ Resolved review finding [LOW]: Added `isinstance(features, dict)` validation for null/non-dict features
- After review fixes: 12 tests pass (5 original + 7 new), 160 total suite, zero regressions

### File List

- `src/torq/ingest/lerobot.py` — **Created**, **Modified** (review fixes: is_dir check, features validation, episode_index error wrapping, generic obs discovery, camera_keys validation)
- `src/torq/ingest/__init__.py` — **Modified** (added `ingest_lerobot` export)
- `tests/unit/test_ingest_lerobot.py` — **Created**, **Modified** (12 tests: 5 original + 7 review follow-up tests)
- `tests/fixtures/generate_fixtures.py` — **Modified** (added `generate_lerobot_fixture()`)
- `tests/conftest.py` — **Modified** (added `lerobot_dataset`, `lerobot_no_info` fixtures)
- `tests/fixtures/data/lerobot/` — **Generated** (fixture directory with meta/info.json, Parquet, MP4)
- `tests/fixtures/data/lerobot_no_info/` — **Generated** (fixture directory without info.json)

## Review Follow-ups (AI)

- [x] [AI-Review][MEDIUM] `camera_keys` is dead code — computed from info.json (lerobot.py:51-53), passed to `_build_episode` (line 75, param line 176), but never referenced in the function body. Camera obs loop (lines 227-229) uses `video_map` filtered by `ep_idx` only. Either remove the parameter or use it for validation (e.g., warn if a camera feature in info.json has no matching MP4). [src/torq/ingest/lerobot.py:51-53, 176]
- [x] [AI-Review][MEDIUM] Bare `KeyError` escapes when Parquet lacks `episode_index` column. `combined.column("episode_index")` at lerobot.py:61 raises `KeyError: 'Field "episode_index" does not exist in schema'`. Violates SDK error hierarchy ("No module raises bare Python exceptions"). Wrap in try/except and re-raise as `TorqIngestError` with message about expected LeRobot Parquet schema. [src/torq/ingest/lerobot.py:61]
- [x] [AI-Review][MEDIUM] Only `observation.state` extracted from Parquet — all other observation feature columns (`observation.velocity`, `observation.effort`, `observation.ee_pos`, etc.) are silently dropped with no warning. At minimum, discover and extract ALL `observation.*` prefixed non-camera columns, or log a warning listing skipped observation features. [src/torq/ingest/lerobot.py:218-224]
- [x] [AI-Review][MEDIUM] AC3 specifies "missing **or malformed**" info.json but only missing is tested. The `json.JSONDecodeError` handler at lerobot.py:102-106 is untested. Add a test with corrupt JSON content (e.g., `{bad json`) to cover the full AC. [tests/unit/test_ingest_lerobot.py]
- [x] [AI-Review][LOW] No edge case tests for: empty dataset (0 episodes in Parquet), dataset without `videos/` dir, dataset with missing/unreadable Parquet chunks. Code paths exist (lerobot.py:64-66, 119-121, 132-133) but are untested. [tests/unit/test_ingest_lerobot.py]
- [x] [AI-Review][LOW] No test for `ingest()` accepting `str` path — signature is `str | Path` but all tests pass `Path` objects from fixtures. [tests/unit/test_ingest_lerobot.py]
- [x] [AI-Review][LOW] No validation that `path` is a directory — if a file path is passed, the error will be "missing meta/info.json" which is misleading. Add `root.is_dir()` check with a clear `TorqIngestError`. [src/torq/ingest/lerobot.py:44]
- [x] [AI-Review][LOW] `features` dict from info.json not type-validated — `info.get("features", {})` at line 48 assumes features is a dict. If info.json has `"features": null`, line 52 crashes with `AttributeError` instead of `TorqIngestError`. [src/torq/ingest/lerobot.py:48]

## Change Log

- 2026-03-07: Code review completed — 0 HIGH, 4 MEDIUM, 4 LOW issues found. 8 action items created.
  Key findings: `camera_keys` dead code parameter, bare KeyError on missing episode_index column,
  only observation.state extracted (other obs features silently dropped), malformed JSON AC3 path
  untested. Story status set to in-progress for follow-up fixes.
- 2026-03-07: Addressed all 8 code review findings — 4 MEDIUM, 4 LOW resolved.
  Source changes: added `is_dir()` path validation, `features` dict type validation, `episode_index`
  KeyError wrapping in TorqIngestError, generic observation column discovery via
  `_group_observation_columns()`, and `camera_keys` validation (warns on missing MP4).
  Test changes: added 7 new tests (malformed JSON, str path, file path, null features, empty dataset,
  no videos dir, unreadable Parquet chunk). 160 total tests pass, zero regressions.
