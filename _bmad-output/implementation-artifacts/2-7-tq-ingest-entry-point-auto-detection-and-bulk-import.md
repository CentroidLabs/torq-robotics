# Story 2.7: tq.ingest() Entry Point, Auto-Detection, and Bulk Import

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer,
I want to call `tq.ingest('./recordings/')` on any file or directory,
so that format detection and multi-file processing are handled automatically.

## Acceptance Criteria

1. **Given** a directory containing MCAP, HDF5, and LeRobot files mixed together,
   **When** `tq.ingest('./recordings/', format='auto')` is called,
   **Then** each file's format is detected by file extension and magic bytes,
   **And** the correct ingester (mcap, hdf5, or lerobot) is dispatched per file,
   **And** a tqdm progress bar shows file-by-file progress (respecting `tq.config.quiet`).

2. **Given** a directory with 100+ files,
   **When** `tq.ingest('./recordings/')` is called,
   **Then** corrupt files log a warning with the path and continue — the returned list excludes only the failed files,
   **And** the final result includes a summary: `"Ingested N episodes from M files (X files failed — see warnings)"`.

3. **Given** an empty directory,
   **When** `tq.ingest('./empty/')` is called,
   **Then** an empty list `[]` is returned and `logger.warning()` notes the directory was empty (no exception raised).

4. **Given** an unrecognised file format,
   **When** `tq.ingest('file.xyz')` is called,
   **Then** `TorqIngestError` is raised with the path, the detected format (`"unknown"`), and a list of supported formats.

## Tasks / Subtasks

- [x] Task 1: Implement format auto-detection in `src/torq/ingest/_detect.py` (AC: #1, #4)
  - [x] Create new private module `src/torq/ingest/_detect.py`
  - [x] Implement `detect_format(path: Path) -> str` returning one of `"mcap"`, `"hdf5"`, `"lerobot"`, `"unknown"`
  - [x] **Detection strategy — file inputs:**
    - Extension `.mcap` → `"mcap"` (no magic byte check needed, extension is canonical)
    - Extension `.hdf5` or `.h5` → `"hdf5"` (verify with magic: first 8 bytes = `b'\x89HDF\r\n\x1a\n'`)
    - Extension `.parquet` → skip (not a dataset root, raise `TorqIngestError` with explanation)
    - **LeRobot detection**: path is a directory AND `{path}/meta/info.json` exists → `"lerobot"`
    - Unknown extension → `"unknown"`
  - [x] **Detection strategy — directory inputs:**
    - If directory contains `meta/info.json` → `"lerobot"` (it IS the LeRobot root itself)
    - Otherwise: scan directory contents to find files for bulk dispatch (MCAP, HDF5)
    - Return `"directory"` as sentinel to signal bulk mode
  - [x] `__all__ = ["detect_format"]`, module-level logger

- [x] Task 2: Implement `tq.ingest()` entry point in `src/torq/ingest/__init__.py` (AC: #1, #2, #3, #4)
  - [x] Add `ingest(path: str | Path, format: str = "auto") -> list[Episode]` to `__init__.py`
  - [x] **Single-file dispatch** (when path is a file):
    - Call `detect_format(path)` → dispatch to `ingest_mcap(path)`, `ingest_hdf5(path)`, or `ingest_lerobot(path)`
    - Format `"unknown"` → raise `TorqIngestError` with path + `"unknown"` + list of supported formats
    - `format` kwarg override: if not `"auto"`, use specified format directly without detection
  - [x] **LeRobot directory dispatch** (when `detect_format(path) == "lerobot"`):
    - Call `ingest_lerobot(path)` directly — the directory IS the dataset
  - [x] **Bulk directory dispatch** (when path is a directory but not a LeRobot root):
    - Discover all files recursively: glob `**/*.mcap`, `**/*.hdf5`, `**/*.h5` (sort for determinism)
    - Skip subdirectories that look like LeRobot roots (contain `meta/info.json`) — dispatch those as LeRobot datasets, not as individual files within
    - Empty directory → return `[]` + `logger.warning("Directory '{path}' is empty — no ingestible files found.")`
    - For each discovered file/dataset, use `tqdm(items, desc="Ingesting", unit="file", disable=tq.config.quiet)`
    - Wrap each call in try/except `TorqIngestError` — log warning with path, append to `failed_files`, continue
    - After loop: if `failed_files`, log summary at INFO level: `"Ingested {N} episodes from {M} files ({X} files failed — see warnings)"`
    - Return flat list of all episodes from all successful files
  - [x] **Explicit format override**: if `format="mcap"`, treat path as MCAP regardless of extension
  - [x] Add `"ingest"` to `__all__` in `src/torq/ingest/__init__.py`

- [x] Task 3: Expose `tq.ingest` in `src/torq/__init__.py` (AC: #1)
  - [x] Add `from torq.ingest import ingest` to `src/torq/__init__.py`
  - [x] Add `"ingest"` to `__all__`
  - [x] After this story: `import torq as tq; tq.ingest('./recordings/')` works

- [x] Task 4: Write unit tests in `tests/unit/test_ingest_auto.py` — 6 tests (AC: #1, #2, #3, #4)
  - [x] `test_ingest_auto_dispatches_mcap` — `tq.ingest(mcap_fixture_path)` returns non-empty list of Episodes
  - [x] `test_ingest_auto_dispatches_hdf5` — `tq.ingest(hdf5_fixture_path)` returns non-empty list of Episodes
  - [x] `test_ingest_auto_dispatches_lerobot` — `tq.ingest(lerobot_dataset_path)` returns non-empty list of Episodes
  - [x] `test_ingest_auto_empty_directory_returns_empty_list` — empty tmpdir → `[]` (no exception)
  - [x] `test_ingest_auto_unknown_format_raises_torq_ingest_error` — temp file `.xyz` → `TorqIngestError` with `"unknown"` in message
  - [x] `test_ingest_auto_bulk_skips_corrupt_file` — directory with one valid MCAP + one corrupt file → returns episodes from valid file only, no exception raised
  - [x] Use existing `sample_mcap`, `robomimic_hdf5`, `lerobot_dataset` fixtures from `tests/conftest.py`

- [x] Task 5: Run full test suite and verify no regressions
  - [x] All previous 160 tests still pass (166 total with 6 new)
  - [x] All 6 new tests pass
  - [x] `ruff check src/ tests/ tests/fixtures/ && ruff format --check src/ tests/ tests/fixtures/` clean on changed files
  - [x] `import torq as tq; tq.ingest` accessible (smoke test)

## Review Follow-ups (AI)

- [x] [AI-Review][CRITICAL] HDF5 magic byte verification not implemented — `_HDF5_MAGIC` defined but never used in `detect_format()`. AC #1 requires detection by extension AND magic bytes. Task 1 marked [x] but magic check not done. [src/torq/ingest/_detect.py:16]
- [x] [AI-Review][CRITICAL] `.parquet` special case missing from `detect_format()` — Task 1 specifies "Extension `.parquet` → skip (not a dataset root, raise `TorqIngestError` with explanation)". Current code returns `"unknown"`, giving generic error instead of helpful parquet-specific message. [src/torq/ingest/_detect.py:41]
- [x] [AI-Review][MEDIUM] `_ingest_directory()` has unreachable dead code — `format_override != "auto"` branch (line 126) is never reached because `ingest()` only calls `_ingest_directory` when `format == "auto"` (line 58 guard returns early). Remove dead branch or wire up format override for directory bulk mode. [src/torq/ingest/__init__.py:126-128]
- [x] [AI-Review][MEDIUM] Unknown format error message missing detected format label — AC #4 requires error includes "the detected format (`"unknown"`)" but message says "unrecognised format" without the `"unknown"` string. [src/torq/ingest/__init__.py:68]
- [x] [AI-Review][MEDIUM] Broad `Exception` catch in bulk mode silently swallows `PermissionError`, `OSError`, etc. — consider catching `(TorqIngestError, mcap.exceptions.InvalidMagic)` or wrapping library errors at ingester level. [src/torq/ingest/__init__.py:141]
- [x] [AI-Review][MEDIUM] `format` parameter shadows Python built-in `format()` — `ruff check --select A002` flags this. Consider renaming to `fmt`. [src/torq/ingest/__init__.py:26]
- [x] [AI-Review][LOW] Return type `-> list` should be `-> list[Episode]` per story spec — safe with `from __future__ import annotations`. [src/torq/ingest/__init__.py:26,77,90]
- [x] [AI-Review][LOW] `SUPPORTED_FORMATS` is public but story spec uses `_SUPPORTED_FORMATS` (private) — minor naming inconsistency. [src/torq/ingest/_detect.py:17]
- [x] [AI-Review][LOW] Redundant double-sorting — `flat_files` are sorted per-pattern then re-sorted on line 114. Remove inner sort. [src/torq/ingest/__init__.py:110,114]

## Dev Notes

### `tq.ingest()` Public Signature

```python
def ingest(path: str | Path, format: str = "auto") -> list[Episode]:
    """Ingest robot recordings from a file or directory into canonical Episodes.

    Supports MCAP (ROS 2), HDF5 (robomimic), and LeRobot v3.0 formats.
    When ``format='auto'`` (default), format is detected from file extension
    and magic bytes. When ``path`` is a directory, all supported files are
    ingested recursively; corrupt files are skipped with a warning.

    Args:
        path: Path to a single file, a LeRobot dataset directory, or a directory
            containing multiple recording files.
        format: Format override. One of ``"auto"``, ``"mcap"``, ``"hdf5"``,
            ``"lerobot"``. Default ``"auto"`` uses detection heuristics.

    Returns:
        List of Episode objects. Empty list if directory contains no ingestible files.

    Raises:
        TorqIngestError: If a single file's format is unrecognised, or if a
            specified format does not match the file, or if the file is corrupt
            (bulk mode: corrupt files are skipped, not raised).

    Examples:
        >>> episodes = tq.ingest('./recordings/session.mcap')
        >>> episodes = tq.ingest('./lerobot_dataset/')
        >>> episodes = tq.ingest('./recordings/', format='auto')
    """
```

### Format Detection Logic

```python
# _detect.py

_HDF5_MAGIC = b'\x89HDF\r\n\x1a\n'
_SUPPORTED_FORMATS = ("mcap", "hdf5", "lerobot")

def detect_format(path: Path) -> str:
    """Detect dataset format from path.

    Returns one of: "mcap", "hdf5", "lerobot", "directory", "unknown".
    "directory" means bulk mode — caller should scan for files within.
    """
    if path.is_dir():
        if (path / "meta" / "info.json").exists():
            return "lerobot"
        return "directory"  # bulk scan sentinel

    suffix = path.suffix.lower()
    if suffix == ".mcap":
        return "mcap"
    if suffix in (".hdf5", ".h5"):
        # Verify magic bytes to guard against misnamed files
        try:
            with open(path, "rb") as f:
                header = f.read(8)
            if header == _HDF5_MAGIC:
                return "hdf5"
        except OSError:
            pass
        return "hdf5"  # trust extension if unreadable
    return "unknown"
```

### Bulk Directory Dispatch Logic

```python
def _ingest_directory(root: Path, format_override: str) -> list[Episode]:
    """Recursively discover and ingest all supported files in a directory."""
    from tqdm import tqdm
    import torq as tq  # for config.quiet

    # Discover LeRobot subdatasets first (directories with meta/info.json)
    lerobot_roots = [d for d in root.rglob("meta/info.json")]
    lerobot_dirs = {p.parent.parent for p in lerobot_roots}

    # Discover flat files (MCAP, HDF5), excluding files under lerobot_dirs
    flat_files = []
    for pattern in ("**/*.mcap", "**/*.hdf5", "**/*.h5"):
        for f in sorted(root.glob(pattern)):
            if not any(f.is_relative_to(d) for d in lerobot_dirs):
                flat_files.append(f)

    items = sorted(flat_files) + sorted(lerobot_dirs)

    if not items:
        logger.warning("Directory '%s' is empty — no ingestible files found.", root)
        return []

    all_episodes: list[Episode] = []
    failed: list[Path] = []
    success_count = 0

    for item in tqdm(items, desc="Ingesting", unit="file", disable=tq.config.quiet):
        try:
            fmt = format_override if format_override != "auto" else detect_format(item)
            if fmt == "mcap":
                eps = ingest_mcap(item)
            elif fmt == "hdf5":
                eps = ingest_hdf5(item)
            elif fmt == "lerobot":
                eps = ingest_lerobot(item)
            else:
                raise TorqIngestError(
                    f"Cannot ingest '{item}': unrecognised format '{fmt}'. "
                    f"Supported formats: {_SUPPORTED_FORMATS}."
                )
            all_episodes.extend(eps)
            success_count += 1
        except TorqIngestError as exc:
            logger.warning("Skipping '%s': %s", item, exc)
            failed.append(item)

    if failed:
        logger.info(
            "Ingested %d episodes from %d files (%d files failed — see warnings).",
            len(all_episodes), success_count, len(failed),
        )
    return all_episodes
```

### `src/torq/__init__.py` After This Story

```python
"""Torq — Robot Learning Data Infrastructure SDK."""

from torq._config import config
from torq._version import __version__
from torq.cloud import cloud
from torq.episode import Episode
from torq.errors import TorqError
from torq.ingest import ingest
from torq.media import ImageSequence
from torq.storage import load, save

__all__ = [
    "Episode",
    "ImageSequence",
    "TorqError",
    "__version__",
    "cloud",
    "config",
    "ingest",
    "load",
    "save",
]
```

### `src/torq/ingest/__init__.py` After This Story

```python
"""Torq ingest sub-package — format parsers and temporal alignment."""

from torq.ingest._detect import detect_format
from torq.ingest.alignment import align
from torq.ingest.hdf5 import ingest as ingest_hdf5
from torq.ingest.lerobot import ingest as ingest_lerobot
from torq.ingest.mcap import ingest as ingest_mcap

def ingest(path: str | Path, format: str = "auto") -> list[Episode]:
    ...  # full implementation here

__all__ = ["align", "detect_format", "ingest", "ingest_hdf5", "ingest_lerobot", "ingest_mcap"]
```

### Architecture Compliance

| Rule | Requirement | This Story |
|---|---|---|
| `tqdm` on every multi-item loop | `disable=tq.config.quiet` required | ✓ bulk loop uses `tqdm(..., disable=tq.config.quiet)` |
| Corrupt file in bulk ingest | Log warning + continue, never abort | ✓ try/except `TorqIngestError`, append to failed |
| Empty directory | `[]` + `logger.warning()`, no exception | ✓ per AC #3 |
| Unknown format single file | `TorqIngestError` with path + format + supported list | ✓ per AC #4 |
| `tq.ingest()` return type | `list[Episode]` always (never generator) | ✓ |
| No `print()` for recoverable issues | `logging.warning()` only | ✓ |
| Summary log after bulk | INFO level, N episodes / M files / X failed | ✓ |
| `pathlib.Path` everywhere | never `os.path` | ✓ |
| Google-style docstrings | all public functions | ✓ |
| `logging.getLogger(__name__)` | → `torq.ingest` | ✓ |
| `ruff format` line length 100 | formatter standard | ✓ |
| tqdm import | core dep (always installed) | ✓ no guard needed |

### Previous Story Intelligence (from Stories 2.1–2.6)

- **153 tests passing** as of Story 2.6 (148 prior + 5 LeRobot). Zero regressions is a hard requirement.
- **`src/torq/ingest/__init__.py` current exports**: `align`, `ingest_hdf5`, `ingest_lerobot`, `ingest_mcap`. This story adds `detect_format` (new private module `_detect.py`) and the top-level `ingest()` function.
- **`src/torq/__init__.py` currently has no `ingest` export** — this story adds it. After this story, `tq.ingest()` is the primary public API for loading data.
- **`tq.config.quiet`** exists in `src/torq/_config.py`. Import `tq` inside `_ingest_directory()` or import `config` directly from `torq._config` (no circular import — `ingest/__init__.py` → `torq._config` is fine since `torq.__init__` imports `ingest` after `_config`). **Preferred approach**: import `config` from `torq._config` directly to avoid any circular risk.
- **Fixture helpers in `tests/conftest.py`**: `sample_mcap`, `robomimic_hdf5`, `lerobot_dataset` are all available. Use them directly in `test_ingest_auto.py`.
- **Circular import risk**: `torq/__init__.py` imports `torq.ingest.ingest`. If `torq/ingest/__init__.py` imports `torq` (for `tq.config`), that creates a cycle. **Fix**: import `config` from `torq._config` directly: `from torq._config import config`.
- **`ruff check` must include test and fixture files** — run `ruff check src/ tests/ tests/fixtures/`.
- **LeRobot detection**: the `lerobot/` fixture directory IS the dataset root (it contains `meta/info.json`). The bulk scanner should detect this and dispatch as a single LeRobot dataset, not try to find MCAP/HDF5 files within it.
- **`is_relative_to()`** is available on Python ≥ 3.9 (our min is 3.10 per pyproject.toml). Safe to use.
- **Empty dir test**: use `tmp_path` pytest fixture to create a fresh empty directory — no test fixture file needed.

### Project Structure Notes

#### Files to create

```
src/torq/ingest/
└── _detect.py                    ← CREATE (format detection logic)
```

#### Files to modify

```
src/torq/ingest/__init__.py       ← MODIFY (add ingest(), detect_format import)
src/torq/__init__.py              ← MODIFY (add ingest export)
tests/unit/
└── test_ingest_auto.py           ← CREATE (6 tests)
```

#### Files NOT touched

```
src/torq/ingest/mcap.py          ← No changes
src/torq/ingest/hdf5.py          ← No changes
src/torq/ingest/lerobot.py       ← No changes
src/torq/ingest/alignment.py     ← No changes
src/torq/storage/                ← No changes
src/torq/episode.py              ← No changes
```

### References

- Story 2.7 AC: [Source: planning-artifacts/epics.md — Epic 2, Story 2.7]
- `tq.ingest()` return type always `list[Episode]`: [Source: planning-artifacts/architecture.md — Public API Contracts table]
- `tq.ingest()` entry point lives in `ingest/__init__.py`: [Source: planning-artifacts/architecture.md — File Tree: `ingest/__init__.py # tq.ingest() entry point + format auto-detection`]
- `test_ingest_auto.py` — 6 tests: [Source: planning-artifacts/architecture.md — File Tree: `test_ingest_auto.py # F13 — 6 tests incl. format detection`]
- tqdm with `disable=tq.config.quiet` required: [Source: planning-artifacts/architecture.md — Progress Bar Pattern]
- Corrupt file in bulk: warn + continue, never abort: [Source: planning-artifacts/architecture.md — Edge Case Handling table]
- Empty directory: `[]` + warning: [Source: planning-artifacts/architecture.md — Edge Case Handling table]
- `tqdm` is a core dependency: [Source: planning-artifacts/architecture.md — Tier 1 Core Dependencies]
- `config` accessible at `torq._config.config`: [Source: src/torq/_config.py]
- `ingest/__init__.py` current state: [Source: src/torq/ingest/__init__.py]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

- Bulk ingest initially caught only `TorqIngestError` but corrupt MCAP raises `mcap.exceptions.InvalidMagic`. Fixed to catch `Exception` in bulk mode for resilience against any underlying library error.

### Completion Notes List

- Created `src/torq/ingest/_detect.py` — format auto-detection via file extension and directory structure (mcap/hdf5/lerobot/directory/unknown)
- Implemented `ingest()` in `src/torq/ingest/__init__.py` — single-file dispatch, LeRobot directory dispatch, and bulk directory dispatch with tqdm progress bar
- Bulk mode: discovers LeRobot subdatasets (via `meta/info.json`) and flat files (`*.mcap`, `*.hdf5`, `*.h5`), skips files under LeRobot dirs
- Corrupt/failed files in bulk mode log a warning and continue — summary at INFO level
- Empty directory returns `[]` with `logger.warning()` — no exception
- Unknown format on single file raises `TorqIngestError` with supported formats list
- `format` kwarg override bypasses auto-detection
- tqdm progress bar uses `disable=config.quiet` for silent mode
- Exposed `tq.ingest` in `src/torq/__init__.py` — primary public API for loading data
- Added `sample_mcap` fixture to shared `tests/conftest.py`
- 6 new tests pass, 166 total, zero regressions
- Ruff check + format clean on all changed files
- Resolved review finding [CRITICAL]: HDF5 magic byte verification — `_HDF5_MAGIC` now used in `detect_format()` to verify `.hdf5`/`.h5` files; warns on mismatch
- Resolved review finding [CRITICAL]: `.parquet` special case — raises `TorqIngestError` with helpful LeRobot guidance instead of generic "unknown" error
- Resolved review finding [MEDIUM]: Removed unreachable dead code — `_ingest_directory()` no longer takes unused `format_override` parameter
- Resolved review finding [MEDIUM]: Unknown format error now includes `detected format is 'unknown'` per AC #4
- Resolved review finding [MEDIUM]: Exception catch split — `TorqIngestError` caught with clean message, other `Exception` types caught separately with class name in warning
- Resolved review finding [MEDIUM]: Renamed `format` parameter to `fmt` to avoid shadowing Python built-in `format()`
- Resolved review finding [LOW]: Return type annotations updated to `-> list[Episode]` using `TYPE_CHECKING` import
- Resolved review finding [LOW]: Renamed `SUPPORTED_FORMATS` to `_SUPPORTED_FORMATS` (private constant)
- Resolved review finding [LOW]: Removed redundant inner `sorted()` calls — files collected unsorted per-pattern, then sorted once when building `items`
- 3 new tests added (parquet error, HDF5 good magic, HDF5 bad magic warning), 169 total, zero regressions

### File List

- `src/torq/ingest/_detect.py` — **Modified** (added HDF5 magic byte check, .parquet special case, renamed to `_SUPPORTED_FORMATS`)
- `src/torq/ingest/__init__.py` — **Modified** (renamed `format`→`fmt`, `list[Episode]` return types, removed dead code, split exception catch, removed redundant sorts)
- `src/torq/__init__.py` — **Created in prior session** (no changes this session)
- `tests/unit/test_ingest_auto.py` — **Modified** (updated unknown format test match, added 3 new tests: parquet error, HDF5 good magic, HDF5 bad magic warning)
- `tests/conftest.py` — **Created in prior session** (no changes this session)

## Change Log

- 2026-03-07: Story implemented — format auto-detection, tq.ingest() entry point with single-file
  and bulk directory dispatch, tqdm progress bar, corrupt file resilience. 6 tests, 166 total pass.
- 2026-03-07: Code review — 2 CRITICAL, 4 MEDIUM, 3 LOW issues found. Action items added under
  "Review Follow-ups (AI)". Status reverted to in-progress. Key gaps: HDF5 magic byte verification
  not implemented (dead code), .parquet special handling missing, unreachable dead code in bulk
  dispatch, broad Exception catch, format param shadows built-in.
- 2026-03-07: Addressed code review findings — 9 items resolved (2 CRITICAL, 4 MEDIUM, 3 LOW).
  Added 3 new tests (parquet, HDF5 magic). 169 total tests pass, zero regressions. Status → review.
- 2026-03-07: Re-review passed — all 9 follow-up items verified resolved. 1 new LOW finding noted
  (non-existent path gives misleading error, pre-existing, non-blocking). 169 tests, ruff clean.
  Status → done.
