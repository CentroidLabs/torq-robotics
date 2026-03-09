# Story 2.1: Episode Dataclass and ImageSequence Loader

Status: done

## Story

As a developer,
I want a canonical Episode object that holds aligned robot data,
So that I have a single consistent structure regardless of source format.

## Acceptance Criteria

1. **Given** `src/torq/episode.py` with the Episode dataclass,
   **When** an Episode is created with `episode_id`, `observations`, `actions`, and `timestamps`,
   **Then** all fields are accessible and `repr(episode)` shows duration, timestep count, and modality list without method calls.

2. **Given** an Episode with immutable fields (`episode_id`, `observations`, `actions`, `timestamps`),
   **When** code attempts to set `episode.episode_id = "new_id"` after creation,
   **Then** `EpisodeImmutableFieldError` is raised with a message explaining the field is locked and to create a new Episode instead.

3. **Given** an Episode with mutable fields (`quality`, `metadata`, `tags`),
   **When** `episode.quality = report` or `episode.tags = ["pick"]` is set,
   **Then** the assignment succeeds without error.

4. **Given** `src/torq/media/image_sequence.py` with `ImageSequence`,
   **When** an `ImageSequence` is constructed from a file path,
   **Then** no frames are loaded from disk until `.frames` is accessed (lazy loading),
   **And** accessing `.frames` returns a numpy array of shape `[T, H, W, C]`.

## Tasks / Subtasks

- [x] Task 1: Create `src/torq/episode.py` — Episode dataclass (AC: #1, #2, #3)
  - [x] Define `Episode` dataclass with all fields (see Dev Notes for complete spec)
  - [x] Implement `__setattr__` immutability guard for `_IMMUTABLE_FIELDS`
  - [x] Implement `__repr__` showing duration, timestep count, and modality list
  - [x] Add `observation_keys` and `action_keys` auto-populated from dict keys
  - [x] Add `duration_ns` as a computed field from `timestamps`
  - [x] Write Google-style docstrings on the class and all public methods
  - [x] Add `__all__ = ["Episode"]` (do NOT export `_IMMUTABLE_FIELDS`)

- [x] Task 2: Create `src/torq/media/__init__.py` and `src/torq/media/image_sequence.py` (AC: #4)
  - [x] Implement `ImageSequence` with lazy loading via `@property frames`
  - [x] Constructor accepts `pathlib.Path` (preferred) or `str` path to an MP4 file
  - [x] Cache decoded frames after first access — subsequent `.frames` calls return cached array
  - [x] `imageio-ffmpeg` import must be **inside** the `frames` property body, never at module level
  - [x] If `imageio` is not installed, raise `TorqImportError` with install instructions: `pip install torq-robotics[vision]`
  - [x] `frames` property returns `np.ndarray` of shape `[T, H, W, C]`, dtype `np.uint8`
  - [x] Write Google-style docstrings on class and `frames` property
  - [x] Add `__all__ = ["ImageSequence"]`
  - [x] `media/__init__.py` exports `ImageSequence`

- [x] Task 3: Update `src/torq/__init__.py` to export `Episode` and `ImageSequence` (AC: #1, #4)
  - [x] Add `from torq.episode import Episode`
  - [x] Add `from torq.media import ImageSequence`
  - [x] Update `__all__` (keep alphabetical order)

- [x] Task 4: Write unit tests in `tests/unit/test_episode.py` — 18 tests (AC: #1, #2, #3)
  - [x] `test_episode_creation_and_field_access` — all fields accessible
  - [x] `test_repr_shows_duration` — `repr()` contains duration string
  - [x] `test_repr_shows_timestep_count` — `repr()` contains timestep count
  - [x] `test_repr_shows_modality_list` — `repr()` contains observation key names
  - [x] `test_immutable_episode_id` — set post-init → `EpisodeImmutableFieldError`
  - [x] `test_immutable_observations` — set post-init → `EpisodeImmutableFieldError`
  - [x] `test_immutable_actions` — set post-init → `EpisodeImmutableFieldError`
  - [x] `test_immutable_timestamps` — set post-init → `EpisodeImmutableFieldError`
  - [x] `test_immutable_error_message_quality` — error message names field, says "create a new episode"
  - [x] `test_mutable_quality` — `episode.quality = mock_report` succeeds
  - [x] `test_mutable_metadata` — `episode.metadata["key"] = "val"` succeeds
  - [x] `test_mutable_tags` — `episode.tags = ["pick"]` succeeds
  - [x] `test_observation_keys_populated` — `observation_keys` matches `observations.keys()`
  - [x] `test_action_keys_populated` — `action_keys` is non-empty list
  - [x] `test_duration_ns_calculated` — `duration_ns == timestamps[-1] - timestamps[0]`
  - [x] `test_source_path_tracking` — `source_path` stored as `pathlib.Path`
  - [x] `test_quality_initializes_to_none` — freshly created episode has `quality is None`
  - [x] `test_episode_imports_nothing_from_torq` — source scan: no `from torq.` / `import torq.` lines

- [x] Task 5: Write unit tests in `tests/unit/test_image_sequence.py` — 12 tests (AC: #4)
  - [x] `test_construction_from_path` — constructs without error
  - [x] `test_lazy_no_frames_at_construction` — `_cache` is None immediately after construction
  - [x] `test_frames_access_triggers_load` — `img_seq.frames` returns array
  - [x] `test_frames_shape_is_t_h_w_c` — shape is 4D: `(T, H, W, C)`
  - [x] `test_frames_dtype_uint8` — dtype is `np.uint8`
  - [x] `test_frames_cached_on_second_access` — second access returns same array object (identity check)
  - [x] `test_missing_file_raises_torq_error` — `TorqError` or subclass for missing file
  - [x] `test_imageio_not_installed_raises_torq_import_error` — mock missing import → `TorqImportError`
  - [x] `test_torq_import_error_mentions_install_command` — error message contains `pip install`
  - [x] `test_accepts_pathlib_path` — constructor works with `pathlib.Path`
  - [x] `test_accepts_str_path` — constructor works with `str`
  - [x] `test_image_sequence_imports_nothing_from_torq_except_errors` — source scan

- [x] Task 6: Run full test suite and verify no regressions (AC: all)
  - [x] All Story 1.1 + 1.2 + 1.3 + 1.4 tests still pass
  - [x] All 30 new tests pass (18 episode + 12 image_sequence)
  - [x] `ruff check src/ && ruff format --check src/` clean

## Dev Notes

### Exact Implementation — `src/torq/episode.py`

```python
"""Canonical Episode representation for robot learning data.

Episode is the central data structure of the Torq SDK. All ingestion formats
(MCAP, HDF5, LeRobot) produce Episodes. All quality scoring, composition,
and serving operates on Episodes.

Episode is NOT designed for subclassing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from torq.quality.report import QualityReport  # type-check only — no runtime import

__all__ = ["Episode"]

logger = logging.getLogger(__name__)

_IMMUTABLE_FIELDS: frozenset[str] = frozenset({"episode_id", "observations", "actions", "timestamps"})


@dataclass
class Episode:
    """A single robot demonstration episode with aligned observations and actions.

    Fields are split into two categories:
    - **Immutable** (locked after init): episode_id, observations, actions, timestamps
    - **Mutable** (can be set after init): quality, metadata, tags

    Args:
        episode_id: Unique identifier in ``ep_{n:04d}`` format.
        observations: Dict mapping modality name → array of shape [T, D].
            Values are np.ndarray. Images should use ImageSequence, not raw arrays.
        actions: Action array of shape [T, action_dim] (np.float32 or np.float64).
        timestamps: Nanosecond timestamps array of shape [T], dtype np.int64.
        source_path: Provenance — path to the original source file.
        metadata: Free-form dict for user tags, success flags, etc. Mutable.
        quality: QualityReport attached by tq.quality.score(). None until scored.
        tags: User labels (e.g. ["pick", "success"]). Mutable.
    """

    episode_id: str
    observations: dict[str, np.ndarray]
    actions: np.ndarray
    timestamps: np.ndarray  # np.int64 nanoseconds, shape [T]
    source_path: Path

    metadata: dict = field(default_factory=dict)
    quality: "QualityReport | None" = field(default=None)
    tags: list[str] = field(default_factory=list)

    # Derived fields — populated in __post_init__
    observation_keys: list[str] = field(init=False)
    action_keys: list[str] = field(init=False)
    duration_ns: int = field(init=False)

    def __post_init__(self) -> None:
        self.observation_keys = list(self.observations.keys())
        self.action_keys = ["actions"]  # standard key; multi-action support in R2
        if len(self.timestamps) >= 2:
            object.__setattr__(self, "duration_ns", int(self.timestamps[-1] - self.timestamps[0]))
        else:
            object.__setattr__(self, "duration_ns", 0)
        # Convert source_path to Path if given as str
        if not isinstance(self.source_path, Path):
            object.__setattr__(self, "source_path", Path(self.source_path))

    def __setattr__(self, name: str, value: object) -> None:
        if name in _IMMUTABLE_FIELDS and hasattr(self, name):
            raise EpisodeImmutableFieldError(  # noqa: F821 — defined below
                f"'{name}' cannot be changed after episode creation. "
                f"Create a new Episode with the updated value instead."
            )
        super().__setattr__(name, value)

    def __repr__(self) -> str:
        duration_s = self.duration_ns / 1e9
        n_steps = len(self.timestamps)
        modalities = ", ".join(self.observation_keys) or "(none)"
        return (
            f"Episode(id={self.episode_id!r}, "
            f"steps={n_steps}, "
            f"duration={duration_s:.2f}s, "
            f"modalities=[{modalities}])"
        )
```

**CRITICAL implementation notes:**
- `EpisodeImmutableFieldError` is imported from `torq.errors` — add `from torq.errors import EpisodeImmutableFieldError` at the top.
- The `if TYPE_CHECKING:` guard prevents a circular import: `QualityReport` is defined in `torq.quality.report` which will later import `Episode`. Using `TYPE_CHECKING` keeps it type-check-only.
- `duration_ns` and `observation_keys` are derived fields using `field(init=False)` — they are set in `__post_init__` via `object.__setattr__` to bypass the immutability guard.
- `source_path` is NOT in `_IMMUTABLE_FIELDS` — it is mutable (storage layer may update it on save).
- Do NOT add `torq.quality.report` to normal imports — that module doesn't exist yet. Use string annotation `"QualityReport | None"` plus `TYPE_CHECKING` guard.

**Full import block for episode.py:**

```python
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from torq.errors import EpisodeImmutableFieldError

if TYPE_CHECKING:
    pass  # QualityReport forward reference handled by string annotation
```

### Exact Implementation — `src/torq/media/image_sequence.py`

```python
"""Lazy-loading image sequence backed by an MP4 file.

Frames are decoded from disk only on first access to the ``.frames`` property.
Decoded frames are cached in memory; subsequent accesses return the same array.

Requires the ``[vision]`` extra: ``pip install torq-robotics[vision]``
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from torq.errors import TorqImportError

__all__ = ["ImageSequence"]

logger = logging.getLogger(__name__)


class ImageSequence:
    """A lazy-loading sequence of RGB frames backed by an MP4 file.

    Args:
        path: Path to the MP4 file (str or pathlib.Path).

    Example:
        >>> seq = ImageSequence("videos/ep_0001.mp4")
        >>> frames = seq.frames   # decoded on first access
        >>> frames.shape          # (T, H, W, C)
        (120, 480, 640, 3)
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._cache: np.ndarray | None = None

    @property
    def frames(self) -> np.ndarray:
        """Decoded frames as an array of shape [T, H, W, C], dtype uint8.

        Frames are decoded from disk on first access and cached in memory.

        Returns:
            np.ndarray: RGB frames, shape [T, H, W, C], dtype uint8.

        Raises:
            TorqImportError: If imageio is not installed.
            TorqStorageError: If the file cannot be read.
        """
        if self._cache is not None:
            return self._cache
        self._cache = self._load_frames()
        return self._cache

    def _load_frames(self) -> np.ndarray:
        """Load and decode all frames from the MP4 file."""
        try:
            import imageio  # noqa: PLC0415 — intentional lazy import
        except ImportError:
            raise TorqImportError(
                f"imageio is required to decode video frames. "
                f"Install it with: pip install torq-robotics[vision]"
            ) from None

        from torq.errors import TorqStorageError  # local import to avoid circular deps

        if not self._path.exists():
            raise TorqStorageError(
                f"Video file not found: '{self._path}'. "
                f"Ensure the path is correct and the file has not been moved."
            )

        reader = imageio.get_reader(str(self._path), format="ffmpeg")
        frames = [frame for frame in reader]
        reader.close()
        return np.stack(frames, axis=0)  # [T, H, W, C]

    def __repr__(self) -> str:
        loaded = "loaded" if self._cache is not None else "lazy"
        return f"ImageSequence(path={self._path.name!r}, status={loaded!r})"
```

**CRITICAL implementation notes:**
- `import imageio` is **inside** `_load_frames()`, never at module level. This is mandatory. Non-vision users must be able to `import torq` without imageio installed.
- `imageio-ffmpeg` is a peer dependency of `imageio` — both are in the `[vision]` extra in `pyproject.toml`. You do not need to import `imageio_ffmpeg` directly; imageio handles it.
- `TorqStorageError` is imported inside `_load_frames()` (also lazy) to avoid any future circular import risk.
- The `_cache` attribute uses a leading underscore — it is private, not exposed as a property.

### `src/torq/media/__init__.py`

```python
"""Torq media sub-package — lazy-loading video and image utilities."""

from torq.media.image_sequence import ImageSequence

__all__ = ["ImageSequence"]
```

### `src/torq/__init__.py` Update

After this story, `__init__.py` should be:

```python
"""Torq — Robot Learning Data Infrastructure SDK."""

from torq._config import config
from torq._version import __version__
from torq.cloud import cloud
from torq.episode import Episode
from torq.errors import TorqError
from torq.media import ImageSequence

__all__ = ["Episode", "ImageSequence", "__version__", "cloud", "config", "TorqError"]
```

### Dependency Rules — Import Graph

`episode.py` is the **root** of the dependency graph. It must import from `torq.errors` only (plus stdlib and numpy). No other `torq.*` imports allowed.

```
errors.py          ← imports NOTHING from torq (leaf)
types.py           ← imports NOTHING from torq (leaf)
_version.py        ← imports NOTHING from torq (leaf)
_config.py         ← imports errors only
_gravity_well.py   ← imports _config only
cloud.py           ← imports _gravity_well only
episode.py         ← imports errors only              ← THIS STORY
media/
  image_sequence.py ← imports errors only             ← THIS STORY
__init__.py        ← imports _version, _config, cloud, errors, episode, media
```

**CI test in `tests/test_imports.py`** must verify:
- `episode.py` has zero `from torq.` imports (except `from torq.errors`)
- `image_sequence.py` has zero `from torq.` imports at module level (except `from torq.errors`)
- `import torq` still works without imageio installed

### Project Structure Notes

#### Files to create in this story

```
src/torq/
├── episode.py                     ← CREATE
└── media/
    ├── __init__.py                ← CREATE (already exists as stub — check contents first)
    └── image_sequence.py          ← CREATE

tests/unit/
├── test_episode.py                ← CREATE (18 tests)
└── test_image_sequence.py         ← CREATE (12 tests)
```

**IMPORTANT:** `src/torq/media/__init__.py` already exists as a stub (it was created in Story 1.1 scaffolding). Read it first before writing. If it's empty, replace with the content above.

#### Files NOT touched in this story

```
src/torq/errors.py         ← Already has EpisodeImmutableFieldError from Story 1.2
src/torq/types.py          ← Already has EpisodeID, Timestamp aliases from Story 1.2
src/torq/_config.py        ← Already complete from Story 1.3
src/torq/_gravity_well.py  ← Already complete from Story 1.4
src/torq/cloud.py          ← Already complete from Story 1.4
pyproject.toml             ← No new core deps (imageio is [vision] extra, already declared)
```

#### pyproject.toml check

Before adding any new deps, verify `imageio` and `imageio-ffmpeg` are already listed under `[vision]` optional deps from Story 1.1. If not, add them there — do NOT add them to core deps.

### Testing Conventions (from previous stories)

1. **Source file import scanning** — proven pattern from Stories 1.2, 1.3, 1.4:
   ```python
   def test_episode_imports_nothing_from_torq():
       src = Path("src/torq/episode.py").read_text()
       # Only torq.errors is allowed
       torq_imports = [
           line for line in src.splitlines()
           if ("from torq." in line or "import torq." in line)
           and "torq.errors" not in line
       ]
       assert torq_imports == []
   ```

2. **capsys pattern** — not needed in this story (no print output)

3. **Fixture helper** — create a shared `make_episode()` fixture in `conftest.py` if one doesn't already exist:
   ```python
   @pytest.fixture
   def sample_episode(tmp_path):
       return Episode(
           episode_id="ep_0001",
           observations={"joint_pos": np.zeros((10, 6), dtype=np.float32)},
           actions=np.zeros((10, 6), dtype=np.float32),
           timestamps=np.arange(10, dtype=np.int64) * 20_000_000,  # 20ms spacing
           source_path=tmp_path / "sample.mcap",
       )
   ```

4. **Lazy loading test pattern** — check the private `_cache` attribute directly:
   ```python
   def test_lazy_no_frames_at_construction(tmp_path):
       seq = ImageSequence(tmp_path / "fake.mp4")
       assert seq._cache is None
   ```

5. **Test speed** — all 30 tests must be < 1s each. No `@pytest.mark.slow` needed (no file I/O in Episode tests; ImageSequence tests use mocks or minimal test videos).

### Architecture Compliance

| Rule | Requirement | This Story |
|---|---|---|
| episode.py is dependency root | imports nothing from torq (except errors) | ✓ only `from torq.errors import EpisodeImmutableFieldError` |
| No circular imports | episode ← errors only | ✓ `TYPE_CHECKING` guard for QualityReport forward ref |
| No torch/jax at module level | optional imports only | ✓ imageio import inside property body |
| Timestamps are `np.int64` nanoseconds | single source of truth | ✓ enforced by type hint and test |
| `None` not `NaN` for uncomputable scores | quality field defaults to None | ✓ `quality: QualityReport | None = None` |
| Helpful error messages | [what] + [why] + [what to do] | ✓ all errors follow this format |
| Google-style docstrings | all public classes and functions | ✓ Episode and ImageSequence both documented |
| `pathlib.Path` for file ops | never `os.path` or string concat | ✓ both files use `Path` throughout |
| `logging.getLogger(__name__)` | module-level logger | ✓ both files define `logger` |
| `ruff format` line length 100 | formatter standard | ✓ apply before committing |

### Previous Story Intelligence (from Stories 1.1–1.4)

- **`__all__` must not include `_`-prefixed names** (Story 1.2 finding) — `_IMMUTABLE_FIELDS` must NOT be in `__all__`.
- **Test scan pattern for import graph compliance** works well — reuse it (Stories 1.2, 1.4).
- **`object.__setattr__`** is required to set derived fields in `__post_init__` without triggering the custom `__setattr__` guard.
- **All 62 tests pass** as of Story 1.4. Any regression in the existing test suite is a blocker — fix before submitting for review.
- **`ruff check` and `ruff format --check`** must both be clean. Run them before marking the story done.

### References

- Episode interface contract: [Source: planning-artifacts/architecture.md — Episode Interface Contract section]
- Immutability guard pattern: [Source: planning-artifacts/architecture.md — Episode Interface Contract section]
- ImageSequence lazy loading: [Source: planning-artifacts/architecture.md — ImageSequence specification]
- Dependency rules / import graph: [Source: planning-artifacts/architecture.md — Project Structure]
- Timestamp format: [Source: planning-artifacts/architecture.md — Timestamp Format]
- Error hierarchy: [Source: planning-artifacts/architecture.md — Exception Base Classes]
- `None` vs `NaN` rule: [Source: planning-artifacts/architecture.md — Edge Case Handling]
- Gravity well trigger map (reference only): [Source: planning-artifacts/architecture.md — FR-to-File-Mapping]
- `EpisodeImmutableFieldError` already declared: [Source: src/torq/errors.py]
- Test count targets (18 + 12): [Source: CLAUDE.md — TESTING.md section, F01/F02]
- Story 2.1 acceptance criteria: [Source: planning-artifacts/epics.md — Epic 2, Story 2.1]

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6

### Debug Log References

- CI gate `test_episode_py_has_no_torq_imports` was too strict — it forbade all `torq.*` imports
  including `torq.errors`. Updated to allow `from torq.errors` since errors.py is a true leaf (imports
  nothing from torq), so no circular import risk exists. Architecture doc confirms: `episode.py ← errors only`.

### Completion Notes List

- Implemented `Episode` dataclass with immutable fields (`episode_id`, `observations`, `actions`,
  `timestamps`) enforced via `__setattr__` guard, and mutable fields (`quality`, `metadata`, `tags`).
- Derived fields (`observation_keys`, `action_keys`, `duration_ns`) set via `object.__setattr__` in
  `__post_init__` to bypass the immutability guard.
- `QualityReport` forward reference uses string annotation + `# noqa: F821` to keep type-check-only
  without runtime import (future module not yet created).
- Implemented `ImageSequence` with lazy-loading `frames` property; `imageio` imported inside
  `_load_frames()` only, never at module level. Frames cached on first access.
- Added `vision = ["imageio>=2.31", "imageio-ffmpeg>=0.4"]` to `pyproject.toml` optional deps.
- All 30 new tests pass; 64 existing tests have zero regressions. Total: 94/94 passing.
- Linting: `ruff check src/` and `ruff format --check src/` both clean.

### File List

- `src/torq/episode.py` — CREATED
- `src/torq/media/image_sequence.py` — CREATED
- `src/torq/media/__init__.py` — MODIFIED (replaced stub with full export)
- `src/torq/__init__.py` — MODIFIED (added Episode, ImageSequence exports)
- `pyproject.toml` — MODIFIED (added vision optional dependency group)
- `tests/unit/test_episode.py` — CREATED (18 tests)
- `tests/unit/test_image_sequence.py` — CREATED (12 tests)
- `tests/test_imports.py` — MODIFIED (relaxed episode.py CI gate to allow torq.errors import)
- `tests/unit/test_episode.py` — MODIFIED (added 0-length timestamp edge case test; 19 tests total)
- `tests/unit/test_image_sequence.py` — MODIFIED (updated patches to imageio.v3 API, cleaned up dead code; 12 tests)

## Review Follow-ups (AI)

- [x] [AI-Review][HIGH] `ImageSequence._load_frames()` uses deprecated `imageio.get_reader()` API — replace with `imageio.mimread()` or v3 API `imageio.v3.imread(path, index=None)` [src/torq/media/image_sequence.py:76]
- [x] [AI-Review][HIGH] `ImageSequence._load_frames()` loads ALL frames into memory with no guard — add warning/limit for large videos (e.g., >10k frames) [src/torq/media/image_sequence.py:77]
- [x] [AI-Review][MEDIUM] `test_missing_file_raises_torq_error` has dead code (`mock_import`, `_imageio_available`) and is fragile — remove dead code, mock imageio to succeed, assert `TorqStorageError` specifically [tests/unit/test_image_sequence.py:124-142]
- [x] [AI-Review][MEDIUM] `ImageSequence` reader not closed on exception during frame iteration — wrap in try/finally or context manager [src/torq/media/image_sequence.py:76-79]
- [x] [AI-Review][MEDIUM] `Episode.__setattr__` immutability bypass via `object.__setattr__` is undocumented — add comment noting guard is for accidental mutation, not deliberate circumvention [src/torq/episode.py:80-86]
- [x] [AI-Review][LOW] `ImageSequence` constructor doesn't validate file extension — consider warning on non-video extensions [src/torq/media/image_sequence.py:36-38]
- [x] [AI-Review][LOW] No test for 0-length timestamp Episode edge case [src/torq/episode.py:72-75]
- [x] [AI-Review][LOW] Redundant string quoting on `quality` annotation — `from __future__ import annotations` already defers evaluation [src/torq/episode.py:61]

## Change Log

- 2026-03-06: Implemented Episode dataclass, ImageSequence lazy loader, updated public API exports,
  added vision deps, wrote 30 unit tests. All ACs satisfied. Story ready for review.
- 2026-03-06: Code review completed — 2 HIGH, 3 MEDIUM, 3 LOW issues found. 8 action items created.
  Status moved to in-progress. All ACs are implemented but code quality issues remain.
- 2026-03-06: All 8 review items resolved. Switched to imageio v3 API (eliminates deprecated
  get_reader and the reader-not-closed issue simultaneously), added large-video warning (>10k frames),
  added extension warning in constructor, cleaned up fragile test, added object.__setattr__ comment,
  added 0-length timestamp test, removed redundant string quoting. 95/95 tests passing. Status: review.
- 2026-03-06: Second code review pass — fixed remaining fragility in test_missing_file
  (added pytest.importorskip guard). All issues resolved. 95/95 tests passing. Status: done.
