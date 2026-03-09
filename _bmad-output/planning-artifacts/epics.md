---
stepsCompleted: [1, 2, 3, 4]
status: 'complete'
completedAt: '2026-03-02'
inputDocuments:
  - /Users/ayushdas/Documents/CentroidLabs/torque/bmad-dev/prd.md
  - /Users/ayushdas/Documents/CentroidLabs/torque/bmad-dev/_bmad-output/planning-artifacts/architecture.md
workflowType: 'create-epics-and-stories'
project_name: 'Torque SDK'
user_name: 'Ayush'
date: '2026-03-02'
---

# Torque SDK - Epic Breakdown

## Overview

This document provides the complete epic and story breakdown for Torque SDK, decomposing the requirements from the PRD and Architecture into implementable stories.

## Requirements Inventory

### Functional Requirements

**R1 Alpha — P0 (Must Ship)**

FR1 (DI-01): Ingest MCAP/ROS2 bag files with automatic topic discovery and schema extraction. Completes in <10s for files under 1GB.
FR2 (DI-02): Ingest HDF5 (robomimic) and Parquet+MP4 (LeRobot v3.0) formats without TensorFlow dependency.
FR3 (DI-03): Automatic episode boundary detection from continuous teleoperation streams using composite strategy: gripper state change (priority 1) → velocity threshold (priority 2) → manual markers (priority 3). >90% accuracy against annotated fixtures. Segments 1-hour stream in <30s.
FR4 (DI-04): Generate canonical Episode representation with aligned action-observation pairs, nanosecond timestamps, provenance tracking, and extensible metadata.
FR5 (QM-01): Automated quality scoring across three dimensions — smoothness (jerk analysis), consistency (action autocorrelation), completeness (metadata.success + duration heuristic). Scores 0.0–1.0. Episodes <10 timesteps return None (not NaN). Processes 100 episodes in <60s.
FR6 (QM-02): Per-episode QualityReport with per-dimension scores, overall weighted composite score, and configurable weights.
FR7 (QM-03): Quality gates — configurable thresholds filter low-quality episodes from composed datasets. Override requires explicit flag.
FR8 (QM-04): Quality distribution reporting across a dataset with outlier detection (>2σ or IQR-based). (R1: text/CLI-based; interactive viz deferred to R2.)
FR9 (DC-01): Structured query builder `tq.query(task=..., quality_min=..., embodiment=...)` returning lazy iterators with <1s latency for 100K episodes via sharded JSON index.
FR10 (DC-02): Stratified and quality-weighted sampling strategies for dataset composition. Deterministic given same seed.
FR11 (DC-03): Dataset versioning — every `tq.compose()` call records full provenance (query, filters, sampling config, source episode IDs) as a named recipe.
FR12 (ML-01): Native PyTorch DataLoader (`tq.DataLoader`) with multi-worker support, pin_memory, and DistributedDataParallel compatibility. First batch within 5s on local storage.
FR13 (ML-03): W&B, MLflow, and TensorBoard integration — auto-logs dataset ID, version, quality statistics at training start. One-line setup: `tq.integrations.wandb.init()`.
FR14 (QE-02): Structured query API `tq.query()` — index-accelerated on task, embodiment, quality, date fields. Results are lazy-evaluated iterators.
FR15 (PD-01): SDK installable as `pip install torq-robotics`. Import as `import torq as tq` completes in <2s.
FR16 (PD-02): All documentation, README examples, and code use `import torq as tq`. Zero `import torque` references.
FR17 (GW-SDK-01): Gravity well prompt fires after `tq.quality.score()` completes successfully. Suppressible via `tq.config.quiet=True`.
FR18 (GW-SDK-02): Gravity well prompt fires after `tq.compose()` returns a non-empty dataset. Suppressible via `tq.config.quiet=True`.
FR19 (GW-SDK-04): `tq.cloud()` prints waitlist prompt and datatorq.ai URL.
FR20 (GW-SDK-06): All gravity wells use unified `_gravity_well()` helper. Consistent format: `💡 {message}\n   → {url}\n`. No network calls.

**R1 Alpha — P1 (Should Ship)**

FR21 (DI-06): Bulk import `tq ingest ./dir/` with progress bar and per-file error summary. Auto-detects format. Never aborts on corrupt file — logs warning + continues.
FR22 (QM-06): Kinematic feasibility scoring stub — returns 1.0 in R1 (full URDF validation deferred to R2 QM-06).
FR23 (QM-07): Custom quality metric plugins via `tq.quality.register(name, fn, weight)`. Existing weights rescaled proportionally. Idempotent (re-registration overwrites with UserWarning).
FR24 (GW-SDK-03): Gravity well when `tq.DataLoader()` is initialized on dataset >50GB. Suppressible.
FR25 (GW-SDK-05): Gravity well when `tq.query()` or `tq.compose()` returns <5 episodes. Suppressible.
FR26 (CLI): CLI commands — `tq ingest`, `tq list`, `tq info`, `tq export` — thin wrappers over SDK. All support `--json` flag.

### NonFunctional Requirements

NFR1 (NFR-P01): The SDK shall ingest a 1GB MCAP file in under 10 seconds (wall-clock, 16GB RAM / 4-core CPU / SSD).
NFR2 (NFR-P02): Metadata query shall return results in under 1 second for datasets with 100,000+ episodes. Requires sharded JSON index (not flat file).
NFR3 (NFR-P03): PyTorch DataLoader shall sustain throughput of 1,000+ episodes/second during training iteration (GPU-equipped machine).
NFR4 (NFR-P05): `import torq` shall complete in under 2 seconds. No heavy dependencies (torch, jax, opencv) loaded at import time.
NFR5 (NFR-R04): Zero silent data corruption. Failed files log warning with path + reason and continue; never drop data silently. Validated by corrupt MCAP and truncated HDF5 adversarial fixtures.
NFR6 (NFR-U01): A new user shall complete the full 5-line workflow (ingest → score → compose → DataLoader → training loop) in under 20 minutes from `pip install` with real data.
NFR7 (NFR-U02): Every SDK exception shall include a human-readable message stating what went wrong and what the user should do to resolve it. No bare exceptions.
NFR8 (NFR-U03): All CLI commands shall support `--json` flag for machine-readable output and return non-zero exit codes on any failure.
NFR9 (NFR-C01): Python 3.10+ on Linux (primary), macOS (development), Windows via WSL2. CI matrix validates all three.
NFR10 (NFR-C02): PyTorch 2.0+ including DistributedDataParallel support.
NFR11 (NFR-C04): ROS 2 Humble, Iron, and Jazzy support for MCAP ingestion.
NFR12 (NFR-C05): `import torq` shall function without torch, jax, or any ML framework installed. Framework imports occur only inside `serve/` and only when explicitly called.

### Additional Requirements

**Build & Package Setup (from Architecture):**
- Use Hatch (hatchling 1.29.0) build backend — `pip install hatch && hatch new torq-robotics`
- Src-layout: `src/torq/` (prevents import-from-source bugs in CI)
- pyproject.toml only (no setup.py, no setup.cfg)
- Python 3.10+ minimum (pytest 9.x requires this)
- Optional dependency extras: `[torch]`, `[jax]` (R2), `[dev]`
- Core dependencies: numpy, pyarrow, mcap, h5py, tqdm
- imageio-ffmpeg is an optional extra (not core — bundles 60MB FFmpeg binary)
- Entry point: `[project.scripts] tq = "torq.cli.main:app"` via Typer

**Architectural Constraints (from Architecture):**
- No circular imports — episode.py is the dependency root; enforced by CI `test_imports.py`
- episode.py must import NOTHING from torq.* — CI gate verifies this
- Optional deps (torch, jax, imageio, opencv, wandb, mlflow) imported ONLY inside functions that need them, never at module level
- All timestamps: np.int64 nanoseconds internally; float seconds only in user-facing repr
- All file operations: pathlib.Path only (no os.path, no string concatenation)
- All index writes: atomic via os.replace() write-then-rename (no partial writes)
- All tqdm calls include `disable=tq.config.quiet`
- All logging via `logging.getLogger(__name__)` — print() reserved for gravity wells + tqdm only
- Quality scores that cannot be computed return None (never NaN, never 0.0)
- Episode ID format: `ep_{n:04d}` — generated only in `src/torq/storage/index.py`

**Sharded JSON Index (from Architecture — required for NFR-P02):**
- Directory structure: `.torq/index/` with `by_task.json`, `by_embodiment.json`, `quality.json`, `manifest.json`
- Query execution: set intersection of inverted index results → binary search on quality
- String normalization: all categorical fields lowercased and stripped at ingest time

**Exception Hierarchy (from Architecture):**
- All exceptions defined in `src/torq/errors.py` — no module raises bare Python exceptions
- Hierarchy: TorqError → TorqIngestError, TorqStorageError, TorqQualityError, TorqConfigError, TorqImportError, EpisodeImmutableFieldError
- Every error message: [what failed] + [why] + [what user should try next]

**Episode Immutability Contract (from Architecture):**
- Fields locked post-init: `episode_id`, `observations`, `actions`, `timestamps`
- Mutable fields: `quality`, `metadata`, `tags`
- Mutation of locked fields raises EpisodeImmutableFieldError

**Quality Architecture (from Architecture):**
- Default weights: smoothness=0.40, consistency=0.35, completeness=0.25
- Per-call weight override supported; global via `tq.config.quality_weights`; reset via `tq.config.reset_quality_weights()`
- Weights must sum to 1.0 ± 0.001; raises QualityConfigError otherwise
- `tq.quality.score()` mutates episode.quality in-place and returns the same episode

**Gravity Well Pattern (from Architecture):**
- All 5 gravity wells call `_gravity_well()` from `src/torq/_gravity_well.py`
- Output format: `💡 {message}\n   → https://www.datatorq.ai\n`
- No network calls in R1; fire only after successful completion (never on error)
- Config: `tq.config.quiet = True` or `TORQ_QUIET=1` env var suppresses all wells

**CI/CD Requirements (from Architecture):**
- GitHub Actions matrix: Python 3.10/3.11/3.12 × ubuntu-latest + macos-latest (blocking) + windows-latest (informational)
- Fast gate (every push, <2 min): ruff check + ruff format --check + `python -c "import torq"` with no extras + unit tests + import graph tests
- Full gate (PR only): complete test suite + NFR-P02 benchmark on ubuntu-latest only

**Test Fixtures Required (Architectural Prerequisites):**
- Quality ground truth: 50 synthetic hand-labeled episodes (smooth, hesitant, jerky, incomplete) — required before QM-01 algorithm can be validated
- 100K episode benchmark dataset — required for NFR-P02 benchmark test
- Corrupt MCAP + truncated HDF5 adversarial fixtures — required for NFR-R04 tests
- Episode boundary detection fixture with 3 ground truth boundaries — required for DI-03 >90% accuracy test

### FR Coverage Map

| FR | Epic | Description |
|---|---|---|
| FR1 (DI-01) | Epic 2 | MCAP/ROS2 ingestion |
| FR2 (DI-02) | Epic 2 | HDF5 + LeRobot ingestion |
| FR3 (DI-03) | Epic 2 | Episode boundary detection |
| FR4 (DI-04) | Epic 2 | Canonical Episode representation |
| FR5 (QM-01) | Epic 3 | Automated quality scoring |
| FR6 (QM-02) | Epic 3 | QualityReport per-episode |
| FR7 (QM-03) | Epic 3 | Quality gates |
| FR8 (QM-04) | Epic 3 | Quality distribution reporting |
| FR9 (DC-01) | Epic 4 | Structured query builder |
| FR10 (DC-02) | Epic 4 | Stratified + quality-weighted sampling |
| FR11 (DC-03) | Epic 4 | Dataset versioning + recipe provenance |
| FR12 (ML-01) | Epic 5 | PyTorch DataLoader (first story in Epic 5) |
| FR13 (ML-03) | Epic 5 | W&B/MLflow integration (follow-up story in Epic 5) |
| FR14 (QE-02) | Epic 4 | Index-accelerated structured query API |
| FR15 (PD-01) | Epic 1 | pip install + import torq as tq |
| FR16 (PD-02) | Epic 1 | Zero import torque references |
| FR17 (GW-SDK-01) | Epic 3 | Gravity well after quality.score() |
| FR18 (GW-SDK-02) | Epic 4 | Gravity well after compose() |
| FR19 (GW-SDK-04) | Epic 1 | tq.cloud() stub |
| FR20 (GW-SDK-06) | Epic 1 | Unified _gravity_well() infrastructure |
| FR21 (DI-06) | Epic 2 | Bulk import with progress + error summary |
| FR22 (QM-06) | Epic 3 | Kinematic feasibility stub (returns 1.0) |
| FR23 (QM-07) | Epic 3 | Custom metric plugin registration |
| FR24 (GW-SDK-03) | Epic 5 | Gravity well when dataset >50GB |
| FR25 (GW-SDK-05) | Epic 4 | Gravity well when <5 episodes returned |
| FR26 (CLI) | Epic 6 | All CLI commands |

**Coverage: 26/26 FRs mapped.**

## Epic List

### Epic 1: Foundation — Installable, Importable SDK
A developer can `pip install torq-robotics`, `import torq as tq`, and get a working package with sensible defaults, helpful error messages, and the gravity well infrastructure ready.
**FRs covered:** FR15 (PD-01), FR16 (PD-02), FR19 (GW-SDK-04), FR20 (GW-SDK-06)
**Story ordering note:** Package scaffolding and pyproject.toml first; errors/types/version/config next; gravity well infrastructure last.

### Epic 2: Data Ingestion & Storage — Developer Can Load Robot Recordings
A developer can call `tq.ingest('./recordings/')` on MCAP, HDF5, or LeRobot files and get back a list of canonical `Episode` objects they can inspect, save, and reload locally.
**FRs covered:** FR1 (DI-01), FR2 (DI-02), FR3 (DI-03), FR4 (DI-04), FR21 (DI-06)
**Story ordering note:** Storage layer and Episode dataclass first (shared foundation). Then MCAP, HDF5, and LeRobot ingest stories run in parallel. Bulk import (DI-06) last.

### Epic 3: Quality Scoring — Developer Can Score and Filter Episodes
A developer can call `tq.quality.score(episodes)` and get per-episode quality scores with dimension breakdown, enabling confident data filtering before training.
**FRs covered:** FR5 (QM-01), FR6 (QM-02), FR7 (QM-03), FR8 (QM-04), FR22 (QM-06), FR23 (QM-07), FR17 (GW-SDK-01)
**Story ordering note:** Core scoring dimensions (smoothness, consistency, completeness) and QualityReport first. Quality gates next. Custom plugin registry and feasibility stub after. Gravity well last.

### Epic 4: Dataset Composition — Developer Can Build Training-Ready Datasets
A developer can call `tq.compose()` or `tq.query()` to filter, sample, and version training datasets from their scored episode pool, with provenance automatically recorded.
**FRs covered:** FR9 (DC-01), FR10 (DC-02), FR11 (DC-03), FR14 (QE-02), FR18 (GW-SDK-02), FR25 (GW-SDK-05)
**Story ordering note:** Structured query + index acceleration first. Sampling strategies next. Dataset class + versioning + recipe provenance after. Gravity wells last.

### Epic 5: ML Training Integration — Developer Can Train Directly from Torq Datasets
A developer can create `tq.DataLoader(dataset, batch_size=32)` and iterate batches in a PyTorch training loop, with dataset lineage auto-logged to W&B or MLflow.
**FRs covered:** FR12 (ML-01), FR13 (ML-03), FR24 (GW-SDK-03)
**Story ordering note:** PyTorch DataLoader (ML-01) is the first and critical story. W&B/MLflow integration (ML-03) is the follow-up story. Gravity well (>50GB trigger) last.

### Epic 6: CLI & Automation — Developer Can Run Torq Headlessly
A developer can run `tq ingest`, `tq list`, `tq info`, `tq export` from the command line in CI/CD pipelines with `--json` output and reliable exit codes for scripting.
**FRs covered:** FR26 (CLI)
**Story ordering note:** CLI comes last by design — it wraps the completed SDK. Ingest command first (highest user value), then list/info/export.

---

## Epic 1: Foundation — Installable, Importable SDK

A developer can `pip install torq-robotics`, `import torq as tq`, and get a working package with sensible defaults, helpful error messages, and the gravity well infrastructure ready.

### Story 1.1: Package Scaffolding and Build Configuration

As a developer,
I want a properly installable Python package,
So that I can `pip install torq-robotics` and `import torq as tq` without errors.

**Acceptance Criteria:**

**Given** the repository with a valid `pyproject.toml` using hatchling build backend and src-layout
**When** a developer runs `pip install -e ".[dev]"`
**Then** the installation completes without errors
**And** `import torq as tq` succeeds and `tq.__version__` returns a version string

**Given** a Python environment with only core dependencies installed (numpy, pyarrow, mcap, h5py, tqdm)
**When** `import torq` is executed
**Then** no ImportError is raised and torch/jax/opencv are NOT imported
**And** `python -c "import torq"` completes in under 2 seconds

**Given** the installed package
**When** `grep -r "import torque" src/` is run
**Then** zero matches are returned (all imports use `import torq`)

### Story 1.2: Core Types, Errors, and Version

As a developer,
I want typed, helpful error messages when the SDK fails,
So that I understand exactly what went wrong and how to fix it.

**Acceptance Criteria:**

**Given** `src/torq/errors.py` is implemented
**When** any Torq operation fails
**Then** a typed `TorqError` subclass is raised (never a bare `ValueError` or `Exception`)
**And** the error message contains: [what failed] + [why] + [what the user should try next]

**Given** the full exception hierarchy
**When** `from torq.errors import TorqError` is executed
**Then** all 7 classes are importable: `TorqError`, `TorqIngestError`, `TorqStorageError`, `TorqQualityError`, `TorqConfigError`, `TorqImportError`, `EpisodeImmutableFieldError`

**Given** `src/torq/types.py` and `src/torq/_version.py`
**When** `import torq` is executed
**Then** `tq.__version__` returns a semver string (e.g. `"0.1.0-alpha"`)
**And** `episode.py` imports nothing from `torq.*` (verified by CI import graph test)

### Story 1.3: Configuration Singleton

As a developer,
I want to configure SDK-wide behaviour (quiet mode, quality weights),
So that I can suppress output in CI and customise scoring without changing my code.

**Acceptance Criteria:**

**Given** `src/torq/_config.py` implementing the Config singleton
**When** `tq.config.quiet = True` is set
**Then** all subsequent gravity well prompts and tqdm bars are suppressed

**Given** the `TORQ_QUIET=1` environment variable is set before import
**When** `import torq as tq` is executed
**Then** `tq.config.quiet` is `True` automatically

**Given** `tq.config.quality_weights` is set to a custom dict
**When** the weights do NOT sum to 1.0 ± 0.001
**Then** `TorqConfigError` is raised with a message stating the actual sum and the correction needed

**Given** `tq.config.reset_quality_weights()` is called
**When** `tq.config.quality_weights` is read
**Then** it returns `DEFAULT_QUALITY_WEIGHTS` = `{smoothness: 0.40, consistency: 0.35, completeness: 0.25}`

### Story 1.4: Gravity Well Infrastructure and Cloud Stub

As a developer,
I want `tq.cloud()` to direct me to the cloud platform,
So that I know how to access collaborative and cloud-scale features.

**Acceptance Criteria:**

**Given** `src/torq/_gravity_well.py` is implemented
**When** `_gravity_well(message="...", feature="GW-01")` is called
**Then** output matches format: `💡 {message}\n   → https://www.datatorq.ai\n`
**And** no network calls are made

**Given** `tq.config.quiet = True`
**When** `_gravity_well()` is called
**Then** nothing is printed

**Given** `src/torq/cloud.py` is implemented
**When** `tq.cloud()` is called
**Then** the datatorq.ai URL and waitlist message are printed via `_gravity_well()`
**And** no exception is raised

**Given** any cloud-only keyword argument is passed to a local SDK function
**When** the function is called
**Then** `_gravity_well()` fires and the function continues without raising an unhandled exception

---

## Epic 2: Data Ingestion & Storage — Developer Can Load Robot Recordings

A developer can call `tq.ingest('./recordings/')` on MCAP, HDF5, or LeRobot files and get back a list of canonical `Episode` objects they can inspect, save, and reload locally.

### Story 2.1: Episode Dataclass and ImageSequence Loader

As a developer,
I want a canonical Episode object that holds aligned robot data,
So that I have a single consistent structure regardless of source format.

**Acceptance Criteria:**

**Given** `src/torq/episode.py` with the Episode dataclass
**When** an Episode is created with episode_id, observations, actions, and timestamps
**Then** all fields are accessible and `repr(episode)` shows duration, timestep count, and modality list without method calls

**Given** an Episode with immutable fields (episode_id, observations, actions, timestamps)
**When** code attempts to set `episode.episode_id = "new_id"` after creation
**Then** `EpisodeImmutableFieldError` is raised with a message explaining the field is locked and to create a new Episode instead

**Given** an Episode with mutable fields (quality, metadata, tags)
**When** `episode.quality = report` or `episode.tags = ["pick"]` is set
**Then** the assignment succeeds without error

**Given** `src/torq/media/image_sequence.py` with ImageSequence
**When** an ImageSequence is constructed from a file path
**Then** no frames are loaded from disk until `.frames` is accessed (lazy loading)
**And** accessing `.frames` returns a numpy array of shape [T, H, W, C]

### Story 2.2: Storage Layer — Save and Load Episodes

As a developer,
I want to save episodes to disk and reload them later,
So that I can persist ingested data without re-processing source files.

**Acceptance Criteria:**

**Given** a valid Episode object
**When** `tq.save(episode, path='./dataset/')` is called
**Then** a Parquet file is written to `episodes/` using atomic `os.replace()` write pattern
**And** the sharded JSON index (`by_task.json`, `by_embodiment.json`, `quality.json`, `manifest.json`) is updated atomically
**And** Episode ID is generated in `ep_{n:04d}` format by `storage/index.py` only

**Given** an Episode with image data (ImageSequence)
**When** `tq.save(episode, path='./dataset/')` is called
**Then** an MP4 file is written to `videos/` using a conditional imageio-ffmpeg import (not at module level)

**Given** a saved episode
**When** `tq.load(episode_id='ep_0001', path='./dataset/')` is called
**Then** an Episode is returned with all fields matching the original
**And** observations, actions, and timestamps arrays are numerically identical (no corruption)

**Given** a partial write that is interrupted
**When** the index is read afterwards
**Then** the index reflects the last successfully committed state (atomic write guarantee)

**Given** categorical fields (task, embodiment) with mixed casing on save
**When** the index is queried
**Then** `"ALOHA-2"`, `"aloha2"`, and `"Aloha 2"` all resolve to the same index bucket

### Story 2.3: Multi-Rate Temporal Alignment

As a developer,
I want sensor streams at different frequencies to be aligned to a common timeline,
So that every Episode timestep has synchronised observations across all modalities.

**Acceptance Criteria:**

**Given** joint state data at 50Hz and camera data at 30Hz
**When** `alignment.align(streams, target_hz=50)` is called
**Then** all streams are resampled to 50Hz using linear interpolation for continuous signals and nearest-frame for image streams
**And** all timestamps are `np.int64` nanoseconds throughout (no float seconds in internal computation)

**Given** streams with a timing gap exceeding a configurable threshold
**When** alignment is attempted
**Then** a `logger.warning()` is emitted naming the gap and the streams affected (not a raised exception)

**Given** a stream with fewer than 2 timesteps
**When** alignment is attempted
**Then** `TorqIngestError` is raised with a message explaining the stream is too short to interpolate

### Story 2.4: MCAP / ROS 2 Ingestion

As a robotics researcher,
I want to load MCAP files from my ALOHA-2 or ROS 2 teleoperation sessions,
So that I can work with my existing recordings without writing conversion scripts.

**Acceptance Criteria:**

**Given** a valid MCAP file containing joint_states and action topics
**When** `mcap.ingest(path)` is called
**Then** a list of Episode objects is returned with observations, actions, and nanosecond timestamps populated
**And** a 1GB MCAP file ingests in under 10 seconds on a standard dev machine

**Given** a continuous MCAP stream with no explicit episode markers
**When** ingestion runs with `boundary_strategy='auto'`
**Then** episode boundaries are detected using the composite strategy: gripper state changes (priority 1), velocity thresholds (priority 2), manual markers (priority 3)
**And** boundary detection achieves >90% accuracy against the annotated `boundary_detection.mcap` fixture

**Given** a MCAP file with a corrupt or truncated message
**When** ingestion is called
**Then** a `logger.warning()` is emitted with the file path and error reason
**And** ingestion continues processing remaining messages (never aborts)
**And** the returned Episode list excludes only the corrupt segment

**Given** a MCAP file with an unknown topic type
**When** ingestion runs
**Then** a `logger.warning()` is emitted for the skipped topic and ingestion continues

**Given** ROS 2 Humble, Iron, and Jazzy message schemas
**When** an MCAP file from any of these distributions is ingested
**Then** joint states and actions are correctly parsed without schema errors

### Story 2.5: HDF5 (Robomimic) Ingestion

As a robotics researcher,
I want to load robomimic HDF5 files from my teleoperation collection,
So that I can work with datasets in the standard robomimic format.

**Acceptance Criteria:**

**Given** a robomimic HDF5 file with `/data/demo_*` groups
**When** `hdf5.ingest(path)` is called
**Then** one Episode is returned per demo group with joint_pos and actions arrays correctly mapped
**And** HDF5 float-second timestamps are converted to `np.int64` nanoseconds at ingest

**Given** an HDF5 file with image data (`agentview_image`)
**When** ingestion runs
**Then** image data is returned as an ImageSequence attached to the episode's observations dict

**Given** a truncated or corrupt HDF5 file
**When** ingestion is called
**Then** `TorqIngestError` is raised with the file path, error reason, and a suggestion to validate the file with h5py directly

### Story 2.6: LeRobot v3.0 Ingestion

As a robotics researcher,
I want to load LeRobot v3.0 datasets (Parquet + MP4 format),
So that I can work with datasets prepared in the LeRobot format without a custom loader.

**Acceptance Criteria:**

**Given** a LeRobot v3.0 directory with `meta/info.json`, `data/chunk-*/` Parquet files, and `videos/chunk-*/` MP4 files
**When** `lerobot.ingest(path)` is called
**Then** one Episode is returned per episode in the dataset with features mapped from `info.json`

**Given** a LeRobot dataset with camera observations
**When** ingestion runs
**Then** video data is returned as ImageSequence objects with lazy loading (no frames decoded at ingest time)

**Given** a LeRobot dataset where `meta/info.json` is missing or malformed
**When** ingestion is called
**Then** `TorqIngestError` is raised with the path and a message explaining that `meta/info.json` is required

### Story 2.7: tq.ingest() Entry Point, Auto-Detection, and Bulk Import

As a developer,
I want to call `tq.ingest('./recordings/')` on any file or directory,
So that format detection and multi-file processing are handled automatically.

**Acceptance Criteria:**

**Given** a directory containing MCAP, HDF5, and LeRobot files mixed together
**When** `tq.ingest('./recordings/', format='auto')` is called
**Then** each file's format is detected by file extension and magic bytes
**And** the correct ingester (mcap, hdf5, or lerobot) is dispatched per file
**And** a tqdm progress bar shows file-by-file progress (respecting `tq.config.quiet`)

**Given** a directory with 100+ files
**When** `tq.ingest('./recordings/')` is called
**Then** corrupt files log a warning with the path and continue — the returned list excludes only the failed files
**And** the final result includes a summary: `"Ingested N episodes from M files (X files failed — see warnings)"`

**Given** an empty directory
**When** `tq.ingest('./empty/')` is called
**Then** an empty list `[]` is returned and `logger.warning()` notes the directory was empty (no exception raised)

**Given** an unrecognised file format
**When** `tq.ingest('file.xyz')` is called
**Then** `TorqIngestError` is raised with the path, the detected format (`"unknown"`), and a list of supported formats

---

## Epic 3: Quality Scoring — Developer Can Score and Filter Episodes

A developer can call `tq.quality.score(episodes)` and get per-episode quality scores with dimension breakdown, enabling confident data filtering before training.

### Story 3.1: Core Quality Scoring Dimensions

As a robotics researcher,
I want each episode automatically scored on smoothness, consistency, and completeness,
So that I have objective per-dimension quality metrics without writing custom analysis code.

**Acceptance Criteria:**

**Given** an Episode with an actions array of 10 or more timesteps
**When** `smoothness.score(episode)` is called
**Then** a float in [0.0, 1.0] is returned based on jerk analysis (3rd derivative of joint positions, normalised)

**Given** an Episode with an actions array of 10 or more timesteps
**When** `consistency.score(episode)` is called
**Then** a float in [0.0, 1.0] is returned based on action autocorrelation (penalises oscillation and hesitation)

**Given** an Episode with `metadata.success = True`
**When** `completeness.score(episode)` is called
**Then** a score close to 1.0 is returned using the metadata flag as primary signal

**Given** an Episode with fewer than 10 timesteps
**When** any scoring function (smoothness, consistency, or completeness) is called
**Then** `None` is returned (never `NaN`, never `0.0`)
**And** `logger.warning()` names the episode ID and timestep count

**Given** an Episode with NaN values in the actions array
**When** any scoring function is called
**Then** `None` is returned and a warning is logged (no exception raised, no NaN propagation)

### Story 3.2: QualityReport and tq.quality.score() Entry Point

As a robotics researcher,
I want a single API call that scores all dimensions and attaches a QualityReport to each episode,
So that quality results are accessible via `episode.quality` after a single function call.

**Acceptance Criteria:**

**Given** a list of scored Episodes
**When** `tq.quality.score(episodes)` is called
**Then** each episode's `.quality` field is populated with a `QualityReport` in-place
**And** the same list is returned (in-place mutation, same object identity)
**And** a tqdm progress bar shows scoring progress (respecting `tq.config.quiet`)

**Given** a single Episode (not a list)
**When** `tq.quality.score(episode)` is called
**Then** the episode's `.quality` is populated and the same episode is returned

**Given** a QualityReport
**When** `episode.quality.overall` is read
**Then** it returns the weighted composite: `smoothness×0.40 + consistency×0.35 + completeness×0.25` (using current `tq.config.quality_weights`)

**Given** per-call weight override: `tq.quality.score(episodes, weights={'smoothness': 0.5, 'consistency': 0.3, 'completeness': 0.2})`
**When** the call is made
**Then** the provided weights are used for this call only (global config unchanged)
**And** if weights do not sum to 1.0 ± 0.001, `TorqQualityError` is raised before any scoring begins

**Given** 100 episodes
**When** `tq.quality.score(episodes)` is called
**Then** all 100 episodes are scored in under 60 seconds

### Story 3.3: Quality Gates

As a robotics researcher,
I want to define quality thresholds that automatically reject low-quality episodes,
So that composed datasets are protected from contamination by bad demonstrations.

**Acceptance Criteria:**

**Given** a list of scored episodes and a quality threshold
**When** `tq.quality.filter(episodes, min_score=0.75)` is called
**Then** only episodes with `quality.overall >= 0.75` are returned
**And** a log message reports how many episodes were filtered and the threshold used

**Given** `min_score=0.75` that filters out all episodes
**When** `tq.quality.filter(episodes, min_score=0.75)` is called
**Then** an empty list is returned (no exception)
**And** `logger.warning()` states that 0 episodes passed the threshold and suggests lowering it

**Given** an episode where `episode.quality` is `None` (unscored or <10 timesteps)
**When** quality filtering is applied
**Then** the episode is excluded from the filtered result
**And** a warning is logged naming the episode ID

### Story 3.4: Quality Distribution Reporting

As a robotics researcher,
I want to see the quality distribution across my dataset with outlier detection,
So that I can understand data quality at a glance before composing a training dataset.

**Acceptance Criteria:**

**Given** a list of scored episodes
**When** `tq.quality.report(episodes)` is called
**Then** a text summary is printed showing: min, max, mean, median, and std of overall scores
**And** outliers (>2σ from mean) are listed by episode ID

**Given** a dataset where all episodes score identically
**When** `tq.quality.report(episodes)` is called
**Then** the report notes zero variance without raising an exception

**Given** fewer than 3 scored episodes
**When** `tq.quality.report(episodes)` is called
**Then** available statistics are shown and a warning notes that distribution analysis is unreliable below 3 samples

### Story 3.5: Custom Metric Registry and Kinematic Feasibility Stub

As a robotics researcher,
I want to register my own quality metrics alongside the built-in ones,
So that I can extend scoring with domain-specific criteria without modifying SDK internals.

**Acceptance Criteria:**

**Given** a callable `fn(episode) -> float` returning a value in [0.0, 1.0]
**When** `tq.quality.register('grip_force', fn, weight=0.2)` is called
**Then** existing weights are rescaled proportionally so all weights sum to 1.0
**And** example: `{smoothness: 0.40, consistency: 0.35, completeness: 0.25}` becomes `{smoothness: 0.32, consistency: 0.28, completeness: 0.20, grip_force: 0.20}`

**Given** a registered custom metric whose callable returns a value outside [0.0, 1.0]
**When** `tq.quality.score(episode)` is called and the custom fn returns e.g. `4.7`
**Then** `TorqQualityError` is raised immediately naming the metric, the offending value, the episode ID, and a normalisation suggestion (e.g. `score / max_possible` or `1 / (1 + exp(-x))`)
**And** no partial quality result is attached to the episode

**Given** a metric name already registered
**When** `tq.quality.register('grip_force', new_fn, weight=0.15)` is called again
**Then** the existing metric is overwritten and a `UserWarning` is emitted (not an error)
**And** weights are recalculated using the new weight value

**Given** `tq.quality.register()` is called and then `tq.config.reset_quality_weights()` is called
**When** `tq.config.quality_weights` is read
**Then** it returns `DEFAULT_QUALITY_WEIGHTS` with all custom metrics removed

**Given** `src/torq/quality/completeness.py` feasibility stub
**When** `feasibility.score(episode)` is called
**Then** `1.0` is always returned (full URDF validation deferred to R2)
**And** the function docstring states this is a R1 stub

### Story 3.6: Gravity Well After Quality Scoring

As a robotics researcher,
I want a prompt directing me to the cloud platform after scoring completes,
So that I know where to compare my quality scores against the community benchmark.

**Acceptance Criteria:**

**Given** `tq.quality.score(episodes)` completes successfully
**When** the function returns
**Then** `_gravity_well()` fires with a message including the computed average score and the datatorq.ai URL

**Given** `tq.config.quiet = True`
**When** `tq.quality.score(episodes)` completes
**Then** no gravity well output is printed

**Given** `tq.quality.score(episodes)` fails with an exception
**When** the exception is raised
**Then** `_gravity_well()` does NOT fire (gravity wells fire only on success)

---

## Epic 4: Dataset Composition — Developer Can Build Training-Ready Datasets

A developer can call `tq.compose()` or `tq.query()` to filter, sample, and version training datasets from their scored episode pool, with provenance automatically recorded.

### Story 4.1: Structured Query API

As a developer,
I want to query my episode pool by task, quality, embodiment, and date,
So that I can find relevant episodes in under 1 second even at 100K+ scale.

**Acceptance Criteria:**

**Given** a dataset of 100,000+ saved episodes with a populated sharded JSON index
**When** `tq.query(task='pick', quality_min=0.8, embodiment='aloha2')` is called
**Then** a lazy iterator of matching Episodes is returned in under 1 second
**And** query execution uses set intersection on inverted index shards, not a full Parquet scan

**Given** compound filters with AND/OR logic
**When** `tq.query(task=['pick', 'place'], quality_min=0.7, quality_max=0.95)` is called
**Then** all matching episodes across both tasks within the quality range are returned

**Given** `tq.query()` with no filters
**When** the iterator is consumed
**Then** all episodes in the index are returned in episode ID order

**Given** a query that matches zero episodes
**When** the iterator is consumed
**Then** an empty iterator is returned (no exception)
**And** `logger.warning()` notes the query parameters that matched nothing

### Story 4.2: Dataset Class

As a developer,
I want a Dataset object that wraps a collection of episodes with a named recipe,
So that I can pass a versioned, inspectable dataset to downstream training tools.

**Acceptance Criteria:**

**Given** a list of Episodes and a name
**When** `Dataset(episodes=episodes, name='pick_v1', recipe={...})` is constructed
**Then** `len(dataset)` returns the episode count
**And** `iter(dataset)` yields Episodes one at a time
**And** `repr(dataset)` shows name, episode count, and average quality score (e.g. `Dataset('pick_v1', 31 episodes, quality_avg=0.81)`)

**Given** a Dataset
**When** `dataset.recipe` is read
**Then** it returns the full composition query dict that created this dataset (filters, sampling config, seed)

### Story 4.3: Sampling Strategies

As a developer,
I want configurable sampling strategies when composing datasets,
So that I can balance task distribution or oversample high-quality episodes deterministically.

**Acceptance Criteria:**

**Given** episodes across 3 tasks with unequal counts (pick: 50, place: 20, pour: 10)
**When** `sampling='stratified'` is applied with `limit=30`
**Then** the resulting dataset has 10 episodes per task (±1 for rounding)

**Given** episodes with varying quality scores
**When** `sampling='quality_weighted'` is applied
**Then** higher-quality episodes are sampled more frequently
**And** the sampling distribution is proportional to `episode.quality.overall`

**Given** `seed=42` is provided
**When** the same sampling call is made twice with identical inputs
**Then** both calls return episodes in identical order (deterministic)

**Given** `sampling='none'`
**When** composition runs
**Then** all filtered episodes are returned without resampling

### Story 4.4: tq.compose() Entry Point and Dataset Versioning

As a developer,
I want a single `tq.compose()` call to filter, sample, and version a training dataset,
So that every dataset I build has full provenance recorded automatically.

**Acceptance Criteria:**

**Given** a scored episode pool
**When** `tq.compose(task='pick', quality_min=0.75, sampling='stratified', limit=50, name='pick_v1')` is called
**Then** a `Dataset` is returned containing episodes matching the filters
**And** `dataset.recipe` stores the exact query parameters, sampling config, seed, and source episode IDs

**Given** `quality_min=0.9` that results in fewer than 5 episodes
**When** `tq.compose()` is called
**Then** the Dataset is returned (no exception)
**And** `logger.warning()` reports the low episode count and suggests lowering `quality_min`

**Given** `tq.compose()` is called with a `name` matching an existing saved dataset
**When** the call completes
**Then** a new versioned Dataset is created without overwriting the existing one

**Given** `tq.compose()` returns a Dataset with 0 episodes
**When** the call completes
**Then** an empty Dataset is returned
**And** `logger.warning()` states which filter(s) eliminated all episodes

### Story 4.5: Gravity Wells for Composition

As a developer,
I want prompts that surface cloud capabilities at the right moments during composition,
So that I discover collaborative and scale features when they become relevant.

**Acceptance Criteria:**

**Given** `tq.compose()` returns a Dataset with more than 0 episodes
**When** the function returns
**Then** `_gravity_well()` fires with a message including the episode count and the datatorq.ai URL (GW-SDK-02)

**Given** `tq.query()` or `tq.compose()` returns fewer than 5 episodes
**When** the function returns
**Then** `_gravity_well()` fires with a message naming the task/embodiment queried and suggesting community datasets at datatorq.ai (GW-SDK-05)
**And** if both GW-SDK-02 and GW-SDK-05 conditions are met, only GW-SDK-05 fires (more specific wins)

**Given** `tq.config.quiet = True`
**When** any composition gravity well would fire
**Then** nothing is printed

---

## Epic 5: ML Training Integration — Developer Can Train Directly from Torq Datasets

A developer can create `tq.DataLoader(dataset, batch_size=32)` and iterate batches in a PyTorch training loop, with dataset lineage auto-logged to W&B or MLflow.

### Story 5.1: PyTorch DataLoader

As an ML engineer,
I want a drop-in PyTorch DataLoader that streams batches from a Torq Dataset,
So that I can replace my custom data loading script with a single `tq.DataLoader()` call.

**Acceptance Criteria:**

**Given** a Torq `Dataset` object
**When** `from torq.serve import DataLoader; loader = DataLoader(dataset, batch_size=32, shuffle=True)` is called
**Then** `DataLoader` is NOT importable from top-level `import torq` (requires explicit `from torq.serve import DataLoader`)
**And** `torch` is imported inside the function only — never at module level in `serve/`
**And** if torch is not installed, `TorqImportError` is raised with the message: `"PyTorch is required for tq.DataLoader(). Install it with: pip install torq-robotics[torch]"`

**Given** an initialised DataLoader
**When** `for batch in loader:` iterates
**Then** each batch is a dict with keys `observations` and `actions` as tensors of shape `[batch_size, T, D]`
**And** the first batch is available within 5 seconds on local storage

**Given** `num_workers=4` is set
**When** the DataLoader iterates
**Then** multi-worker loading works without deadlock or tensor sharing errors

**Given** variable-length episodes in the dataset
**When** a batch is collated
**Then** episodes are padded or truncated to a consistent length within the batch (no collation error)
**And** if a shape mismatch occurs, `TorqIngestError` is raised naming the offending episode ID and modality

**Given** a DistributedDataParallel training setup
**When** the DataLoader is used across multiple GPUs
**Then** it functions correctly with `torch.utils.data.distributed.DistributedSampler`

**Given** 1,000 episodes iterated in batches
**When** wall-clock time is measured
**Then** throughput is at least 1,000 episodes per second on a GPU-equipped machine

### Story 5.2: W&B, MLflow, and TensorBoard Integration

As an ML engineer,
I want dataset lineage automatically logged to my experiment tracker at training start,
So that I can trace any trained model back to the exact dataset version and quality statistics that produced it.

**Acceptance Criteria:**

**Given** W&B is installed and `import wandb; wandb.init()` has been called
**When** `from torq.integrations.wandb import init; init()` is called before training
**Then** dataset ID, version, composition recipe, quality statistics (mean, std, min, max), and episode count are logged to the active W&B run
**And** `wandb` is imported inside the function only — never at module level

**Given** MLflow is installed and an active MLflow run exists
**When** `from torq.integrations.mlflow import init; init()` is called
**Then** the same dataset metadata is logged as MLflow params and tags

**Given** neither W&B nor MLflow is installed
**When** `torq.integrations` is imported
**Then** no ImportError is raised at import time (conditional imports only)
**And** calling `init()` raises `TorqImportError` with the appropriate install instruction

**Given** `tq.config.quiet = True`
**When** integration logging runs
**Then** no console output is produced by the integration layer (logging only via `logging` module)

### Story 5.3: Gravity Well for Large Dataset Streaming

As an ML engineer,
I want a prompt directing me to cloud streaming when my local dataset exceeds 50GB,
So that I discover the cloud streaming option before storage becomes a blocker.

**Acceptance Criteria:**

**Given** a Dataset whose total on-disk size exceeds 50GB
**When** `DataLoader(dataset, ...)` is initialised
**Then** `_gravity_well()` fires with a message referencing the dataset size and the datatorq.ai streaming URL (GW-SDK-03)

**Given** a Dataset under 50GB
**When** `DataLoader(dataset, ...)` is initialised
**Then** no gravity well fires

**Given** `tq.config.quiet = True`
**When** the 50GB threshold is exceeded
**Then** no gravity well output is printed

---

## Epic 6: CLI & Automation — Developer Can Run Torq Headlessly

A developer can run `tq ingest`, `tq list`, `tq info`, `tq export` from the command line in CI/CD pipelines with `--json` output and reliable exit codes for scripting.

### Story 6.1: tq ingest CLI Command

As an ML engineer,
I want to run `tq ingest ./recordings/` from the command line in a nightly cron job,
So that new demonstrations are automatically ingested without writing Python scripts.

**Acceptance Criteria:**

**Given** a directory of MCAP files
**When** `tq ingest ./recordings/` is run
**Then** files are ingested and saved to the local dataset, progress is shown, and exit code 0 is returned on success

**Given** one or more files fail to ingest
**When** `tq ingest ./recordings/` completes
**Then** exit code 1 is returned
**And** stderr contains a summary of which files failed and why

**Given** the `--json` flag is passed
**When** `tq ingest ./recordings/ --json` runs
**Then** stdout contains a valid JSON object with keys: `episodes_ingested`, `files_processed`, `files_failed`, `duration_seconds`
**And** no human-readable progress output is printed to stdout (stderr only)

**Given** `tq ingest` is run in a headless CI environment with no tty
**When** it runs
**Then** no interactive prompts are shown and gravity wells are suppressed automatically (treat headless as quiet mode)

### Story 6.2: tq list and tq info Commands

As an ML engineer,
I want to inspect my local episode index from the command line,
So that I can audit what data is available without writing Python code.

**Acceptance Criteria:**

**Given** a populated local dataset
**When** `tq list` is run
**Then** a table is printed showing episode IDs, task, embodiment, duration, and quality score for all indexed episodes

**Given** `tq list --json` is run
**Then** stdout contains a valid JSON array of episode objects with the same fields

**Given** a valid episode ID
**When** `tq info ep_0001` is run
**Then** full episode metadata is printed: ID, source path, timesteps, duration, modalities, quality breakdown, tags

**Given** an episode ID that does not exist in the index
**When** `tq info ep_9999` is run
**Then** exit code 1 is returned and stderr contains a message naming the missing ID and suggesting `tq list` to see available IDs

**Given** `tq info ep_0001 --json` is run
**Then** stdout contains a valid JSON object with all episode metadata fields

### Story 6.3: tq export Command

As an ML engineer,
I want to export a composed dataset to a portable format,
So that I can share episodes with collaborators or import them into other tools.

**Acceptance Criteria:**

**Given** a named dataset in the local index
**When** `tq export pick_v1 --output ./export/` is run
**Then** all episodes in the dataset are written to `./export/` in Parquet + MP4 format
**And** a `recipe.json` is written alongside the data recording the composition provenance
**And** exit code 0 is returned on success

**Given** the `--json` flag is passed
**When** `tq export pick_v1 --output ./export/ --json` runs
**Then** stdout contains a valid JSON object with keys: `episodes_exported`, `output_path`, `size_bytes`

**Given** the output directory does not exist
**When** `tq export` is run
**Then** the directory is created automatically (no error for missing output dir)

**Given** a dataset name that does not exist
**When** `tq export unknown_dataset --output ./export/` is run
**Then** exit code 1 is returned and stderr names the missing dataset and suggests `tq list` to see available datasets
