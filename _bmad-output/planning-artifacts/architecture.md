---
stepsCompleted: [1, 2, 3, 4, 5, 6, 7, 8]
inputDocuments:
  - prd.md
  - prd-validation-report.md
workflowType: 'architecture'
lastStep: 8
status: 'complete'
completedAt: '2026-03-02'
project_name: 'Torque SDK'
user_name: 'Ayush'
date: '2026-03-02'
---

# Architecture Decision Document

_This document builds collaboratively through step-by-step discovery. Sections are appended as we work through each architectural decision together._

## Project Context Analysis

### Requirements Overview

**Functional Requirements:**
42 requirements across 7 categories. R1 P0 requirements (must ship in Alpha):
- Data Ingestion: DI-01 (MCAP), DI-02 (HDF5/LeRobot), DI-03 (episode segmentation), DI-04 (canonical episode)
- Quality Management: QM-01 (automated scoring), QM-02 (quality report), QM-03 (quality gates), QM-04 (distribution viz)
- Dataset Composition: DC-01 (query builder), DC-02 (stratified sampling), DC-03 (dataset versioning)
- ML Integration: ML-01 (PyTorch DataLoader), ML-03 (W&B/MLflow integration)
- Query Engine: QE-02 (structured query API only — QE-01 NL search corrected to R2)
- Package: PD-01, PD-02
- Gravity Wells: GW-SDK-01 through GW-SDK-06

**Scope Correction (validated with product owner):**
QE-01 (Natural Language Search) is R2, not R1. The PRD release plan places NL search under R2 Beta.
The zero-network-calls R1 constraint makes local NL parsing impractical at the P0 accuracy threshold (>85%).
Removing QE-01 from R1 P0 scope — only QE-02 (structured query API) ships in Alpha.

**Non-Functional Requirements:**
15 NFRs across 5 dimensions. R1-critical subset:
- NFR-P01: 1GB MCAP ingest < 10s
- NFR-P02: Metadata query < 1s for 100K+ episodes (demands sharded indexed JSON — not a flat file)
- NFR-P03: PyTorch DataLoader ≥ 1,000 episodes/second
- NFR-P05: `import torq` < 2s (mandatory lazy loading of all heavy/optional deps)
- NFR-R04: Zero silent data corruption (atomic index writes required; adversarial fixtures required)
- NFR-U01: Full 5-line workflow < 20 min from pip install
- NFR-U02: Human-readable exceptions with resolution steps everywhere
- NFR-U03: CLI --json flag and non-zero exit codes
- NFR-C01: Python 3.9+, Linux/macOS/WSL2
- NFR-C02: PyTorch 2.0+ including DistributedDataParallel
- NFR-C04: ROS 2 Humble, Iron, Jazzy
- NFR-C05: Core import works WITHOUT any ML framework installed

**Scale & Complexity:**
- Primary domain: Python SDK / developer tooling
- Complexity level: medium-high (R1), high (R2+)
- Estimated R1 architectural components: 8 core modules

### Dependency Tiers

Two-tier dependency model:

**Tier 1 — Core (always installed with `pip install torq-robotics`):**
numpy, pyarrow, mcap, h5py, tqdm

**Tier 2 — Optional extras (installed on request only):**
- `torq-robotics[torch]` — PyTorch for tq.DataLoader()
- `torq-robotics[jax]` — JAX pipeline (R2)
- `torq-robotics[dev]` — pytest, ruff, etc.

torch, jax, opencv are NEVER imported at module level anywhere in the codebase.
All optional framework imports are deferred inside the functions that require them,
with a helpful ImportError message if the extra is not installed.

### Technical Constraints & Dependencies

- **Storage**: Parquet + MP4 + JSON index (closed decision). No database.
- **Network**: Zero network calls in R1. No HTTP, no telemetry, no API calls of any kind.
- **Framework imports**: torch, jax, opencv are optional extras. Module-level imports forbidden outside `serve/`.
- **Circular imports**: Forbidden. `episode.py` is the dependency root. CI-enforced import test required — not just a convention.
- **Determinism**: DataLoader and composition must be deterministic given the same seed and inputs.
- **Compatibility**: Python 3.9+, ROS 2 Humble/Iron/Jazzy, PyTorch 2.0+, Linux/macOS/WSL2.
- **Hardware scope**: ALOHA-2, Franka, UR5 as reference. Hardware-agnostic episode model.

### JSON Index Architecture (locked)

NFR-P02 requires sub-second queries on 100K+ episodes without a database.
A flat JSON file fails at this scale. Architecture mandates a sharded index directory:

```
.torq/index/
├── by_task.json          # inverted index: task → [ep_ids]
├── by_embodiment.json    # inverted index: embodiment → [ep_ids]
├── quality.json          # sorted list: [(score, ep_id)] — binary search for range queries
└── manifest.json         # episode count, schema version, last_updated
```

**Query execution:** Set intersection of inverted index results → quality range filter (binary search).
**Write safety:** All index shard updates use atomic `os.replace()` write-then-rename. Partial writes
are silent data corruption (NFR-R04 violation).
**Index role:** Routing only. Returns episode IDs → Parquet loader fetches full metadata.
**String normalization:** All categorical fields (task, embodiment) lowercased and stripped at ingest.
`"ALOHA-2"`, `"aloha2"`, `"Aloha 2"` must resolve to the same `by_embodiment` bucket.

### Fixture Prerequisites (Architectural, Not Implementation)

These fixtures must exist before their dependent modules are implemented:

| Fixture | Required By | Status |
|---|---|---|
| Quality ground truth (50 hand-labeled episodes) | QM-01 scoring algorithm design | not_created |
| 100K episode benchmark | NFR-P02 validation | not_created |
| Corrupt MCAP / truncated HDF5 | NFR-R04 adversarial tests | not_created |
| NL query benchmark (50 annotated pairs) | QE-01 — deferred to R2 | deferred |

### Cross-Cutting Concerns Identified

1. **Lazy loading** — enforced at package architecture level; CI test verifies `import torq` completes without torch/jax/opencv
2. **No circular imports** — `episode.py` is the dependency root; CI import graph test enforces this
3. **Error handling** — all public functions raise typed errors with human-readable message + resolution step
4. **Determinism** — same inputs + seed → identical outputs (critical for ML pipeline trust, UJ-02 Step 2)
5. **Progress reporting** — progress bars required for ingest, scoring, and bulk operations
6. **Gravity wells** — print-only in R1, triggered after successful completions only, suppressible via `tq.config.quiet`
7. **Format agnosticism** — internal Episode representation is fully decoupled from source format
8. **Index query performance** — sharded directory with binary search on quality, set intersection for compound queries
9. **Atomic index writes** — `os.replace()` pattern; no partial writes permitted

## Starter Template Evaluation

### Primary Technology Domain

Python SDK / developer library. No web framework, no UI. Pure Python package
distributed via PyPI.

### Selected Starter: Hatch (hatchling build backend)

**Rationale:** PyPA-recommended modern build backend. Native support for optional
dependency extras ([torch], [jax], [dev]). Compatible with ruff and pytest.
No lock-file opinions that conflict with conda environments used by robotics researchers.

**Initialization Command:**

```bash
pip install hatch
hatch new torq-robotics
```

**Architectural Decisions Established:**

**Language & Runtime:**
- Python 3.10+ (bumped from PRD 3.9 — Python 3.9 is EOL as of Oct 2025; pytest 9.x requires 3.10+)
- pyproject.toml only — no setup.py, no setup.cfg

**Build Backend:**
- hatchling 1.29.0 (current stable)

**Project Layout:**
- Src layout: `src/torq/` (PyPA-recommended; prevents import-from-source bugs in CI)
- Tests at: `tests/unit/`, `tests/integration/`, `tests/acceptance/`, `tests/fixtures/`
- `pytest.ini_options` sets `pythonpath = ["src"]` so tests resolve against installed package
- All module paths use `src/torq/` prefix (e.g., `src/torq/episode.py`)

**Rationale for src-layout over flat:**
Flat layout allows pytest to import directly from source without installation, masking
packaging bugs. Src-layout forces `pip install -e .` before any import resolves — tests
always run against the real installed package, identical to what users receive.
PyPA now recommends src-layout as the default for all distributed packages.

**Linting & Formatting:**
- ruff 0.15.4 (format + lint + isort, line-length=100)

**Testing Framework:**
- pytest 9.0.2
- Markers: `@pytest.mark.slow` for integration/acceptance tests (excluded from fast unit runs)

**Optional Extras (pyproject.toml):**

```toml
[project.optional-dependencies]
torch = ["torch>=2.0"]
jax = ["jax>=0.4"]          # R2
dev = ["pytest>=9.0", "ruff>=0.15", "pytest-cov"]
```

**Core Dependencies:**

```toml
[project.dependencies]
numpy, pyarrow, mcap, h5py, tqdm
```

**Development Workflow:**

```bash
pip install -e ".[dev]"          # core + dev tools
pip install -e ".[dev,torch]"    # with PyTorch DataLoader
ruff check src/torq/ && ruff format src/torq/
pytest tests/ -v
pytest tests/unit/ -v            # fast pass only
```

**Note:** Project scaffolding (`pyproject.toml` + directory structure) is the
first implementation story before any feature code is written.

## Core Architectural Decisions

### Decision Priority Analysis

**Critical Decisions (Block Implementation):**
- Episode data model: mutable with `__setattr__` field-level mutation guard
- Episode boundary detection: composite strategy (gripper → velocity → manual markers)
  — tie-break rule: gripper always wins when both gripper topic AND velocity data present
- Quality scoring: jerk + autocorrelation + completeness heuristic, configurable weights
- CLI framework: Typer
- Configuration: module singleton + TORQ_QUIET env var
- File paths: `pathlib.Path` everywhere (enables Windows support)

**Important Decisions (Shape Architecture):**
- Quality weight configurability: per-call override + global config + `reset_quality_weights()`
- Custom metric registration: `tq.quality.register()` — proportional rescale on add,
  idempotent (overwrite existing name with UserWarning, not error)
- Windows support: targeted from R1, CI informational gate (non-blocking until verified stable)
- ImageSequence: lazy loading on first access
- imageio-ffmpeg: optional extra only, not core (moves to `[vision]` or `[torch]` extra)

**Deferred Decisions (Post-R1):**
- JAX pipeline (R2)
- NL search / QE-01 (R2)
- Kinematic feasibility scoring — QM-06 stub returns 1.0 in R1
- Telemetry opt-in — GW-SDK-07 (R2)

### Data Architecture

**Episode Data Model:**
- Type: Python dataclass, mutable with field-level `__setattr__` guard
- Immutable fields (locked post-init): `episode_id`, `observations`, `actions`, `timestamps`
- Mutable fields: `quality` (QualityReport | None), `metadata` (dict), `tags` (list)
- Guard implementation:
  ```python
  _IMMUTABLE_FIELDS = frozenset({'episode_id', 'observations', 'actions', 'timestamps'})

  def __setattr__(self, name, value):
      if name in self._IMMUTABLE_FIELDS and hasattr(self, name):
          raise EpisodeImmutableFieldError(
              f"'{name}' cannot be changed after episode creation. "
              f"Create a new episode instead."
          )
      super().__setattr__(name, value)
  ```
- `Episode` is NOT designed for subclassing (documented in class docstring)
- Required CI test: assert that setting `episode_id` post-init raises `EpisodeImmutableFieldError`

**ImageSequence:**
- Loading: lazy — frames loaded from disk only on first access to `.frames` property
- Storage: MP4 via imageio-ffmpeg — imageio-ffmpeg is an OPTIONAL extra, not core
- Moves to `[vision]` or `[torch]` extra (bundles ~60MB FFmpeg binary)
- Conditional import inside `src/torq/storage/video.py` only — never at module level
- Rationale: joint-state-only users should not pay the 60MB install cost

**Episode Boundary Detection (DI-03):**
- Strategy: composite with fixed fallback order
- Priority 1: gripper state change (open↔close transitions)
- Priority 2: velocity threshold (near-zero velocity periods)
- Priority 3: manual markers (user-provided timestamps)
- Tie-break rule: if both gripper topic AND velocity data are present,
  gripper strategy always wins — explicitly specified, not a runtime heuristic
- Configuration: `tq.ingest(..., boundary_strategy='auto')` — inspects available
  topics before selecting strategy
- Accuracy target: >90% against 10 annotated continuous-stream fixtures (DI-03 AC)

**Storage:**
- Non-image: Parquet (pyarrow) — one file per episode in `episodes/`
- Images: MP4 (imageio-ffmpeg, optional extra) — one file per episode in `videos/`
- Index: Sharded JSON directory (`by_task.json`, `by_embodiment.json`, `quality.json`, `manifest.json`)
- All writes: atomic `os.replace()` write-then-rename pattern

### Quality Scoring Architecture

**Scoring Dimensions (R1):**

| Dimension | Algorithm | Input | Default Weight |
|---|---|---|---|
| smoothness | Jerk analysis — normalised 3rd derivative of joint positions | actions array | 0.40 |
| consistency | Autocorrelation of action deltas — penalises oscillation/hesitation | actions array | 0.35 |
| completeness | `metadata.success` if present; else duration heuristic | metadata + duration | 0.25 |
| feasibility | Stub — returns 1.0 (URDF validation deferred to QM-06, R2) | — | — |

**Default weights stored as named constants:**
```python
DEFAULT_QUALITY_WEIGHTS = {
    'smoothness': 0.40,
    'consistency': 0.35,
    'completeness': 0.25,
}
```

**Weight configurability:**
- Per-call: `tq.quality.score(episodes, weights={'smoothness': 0.5, ...})`
- Global: `tq.config.quality_weights = {...}`
- Reset: `tq.config.reset_quality_weights()` → restores `DEFAULT_QUALITY_WEIGHTS`
- Validation: weights must sum to 1.0 ± 0.001; raises `QualityConfigError` otherwise

**Custom metric plugins (R1 lightweight):**
- `tq.quality.register('my_metric', fn, weight=0.2)` — fn must return float in [0.0, 1.0]
- Rescaling contract: existing weights scaled proportionally so all weights sum to 1.0
  Example: existing {s:0.40, c:0.35, co:0.25} + new at w=0.20
  → scale factor = (1-0.20)/1.0 = 0.80 → {s:0.32, c:0.28, co:0.20, new:0.20}
- Idempotent: registering same name twice overwrites silently with `UserWarning`
- Supports notebook re-run patterns (no error on re-registration)

**Quality attachment:**
- `tq.quality.score(episode)` → mutates `episode.quality` in-place, returns episode
- `tq.quality.score(episodes)` → batch, mutates all in-place, returns list

### API & Communication Patterns

**CLI Framework: Typer (`typer[all]`, latest stable)**
- Commands: `tq ingest`, `tq list`, `tq info`, `tq export`
- All commands: `--json` flag for machine-readable output (NFR-U03)
- All commands: non-zero exit code on any failure (NFR-U03)
- Entry point: `[project.scripts] tq = "torq.cli.main:app"`

**Internal API conventions:**
- All public functions: type hints required, Google-style docstrings
- All exceptions: typed `TorqError` hierarchy, human-readable message + resolution step
- Progress bars: tqdm for any operation estimated >1s

### Configuration Architecture

- `tq.config.quiet = True` — suppress gravity wells programmatically
- `tq.config.quality_weights = {...}` — override quality weights globally
- `tq.config.reset_quality_weights()` — restore `DEFAULT_QUALITY_WEIGHTS`
- `TORQ_QUIET=1` — environment variable override for CI/cron/headless use
- Config file: `~/.torq/config.toml` — stub created in R1, written in R2 (telemetry preference)

### Infrastructure & Deployment

**CI/CD: GitHub Actions**

Matrix:
- Python: 3.10, 3.11, 3.12
- OS: ubuntu-latest (blocking), macos-latest (blocking),
  windows-latest (`continue-on-error: true` — informational until verified stable)

Gates — every push (fast, <2 min):
- `ruff check src/ && ruff format --check src/`
- `python -c "import torq"` with no extras installed (enforces NFR-C05 + lazy loading)
- `pytest tests/unit/ -v`
- `pytest tests/test_imports.py` (import graph — enforces no circular imports)

Gates — PR only (full suite):
- `pytest tests/ -v --tb=short` (all unit + integration + acceptance)
- Performance benchmark on ubuntu-latest only — 100K episode query < 1s (NFR-P02)
  NOTE: benchmark excluded from windows-latest (slower runners cause false positives)

**File path standard:**
- `pathlib.Path` everywhere — never `os.path` or string path concatenation
- Enforced in code review; required for Windows compatibility

**Windows support:**
- Scope change from PRD Section 0.2 (WSL2 only → native Windows targeted from R1)
- Confirmed by product owner
- Enabled by: `pathlib.Path` standard, `imageio-ffmpeg` as optional extra, mcap Windows wheels
- R1: CI informational gate; promoted to blocking once test suite is stable on Windows

## Implementation Patterns & Consistency Rules

> This section prevents AI agent implementation conflicts. Every rule addresses a specific
> way two modules could silently break each other if left unspecified.

### Naming Conventions

| Construct | Convention | Example |
|---|---|---|
| Public class | CamelCase | `Episode`, `QualityReport`, `Dataset` |
| Public function/method | snake_case | `tq.quality.score()`, `tq.ingest()` |
| Private helper | _snake_case | `_gravity_well()`, `_require_torch()` |
| Module constant | UPPER_SNAKE_CASE | `DEFAULT_QUALITY_WEIGHTS` |
| Module file | snake_case | `episode.py`, `image_sequence.py` |
| Exception class | CamelCase + Error suffix | `TorqIngestError`, `EpisodeImmutableFieldError` |
| Test file | `test_{module}.py` | `test_episode.py`, `test_ingest_mcap.py` |
| Logger | `torq.{module}` | `logging.getLogger('torq.ingest')` |

**Episode ID format:**
- Pattern: `ep_{n:04d}` — zero-padded 4-digit integer from manifest counter
- Examples: `ep_0001`, `ep_0042`, `ep_1000`
- Generated by: `src/torq/storage/index.py` only — no other module generates IDs

### Exception Hierarchy

All exceptions defined in `src/torq/errors.py`. No module raises bare Python exceptions.

```python
class TorqError(Exception): ...                    # base — users catch this
class TorqIngestError(TorqError): ...              # file parsing, boundary detection
class TorqStorageError(TorqError): ...             # read/write/index failures
class TorqQualityError(TorqError): ...             # scoring config/computation
class TorqConfigError(TorqError): ...              # invalid config values
class TorqImportError(TorqError): ...              # optional dep not installed
class EpisodeImmutableFieldError(TorqError): ...   # mutation guard trigger
```

**Error message format — mandatory:**
```python
raise TorqIngestError(
    f"Could not detect episode boundaries in '{path}': no gripper topic found "
    f"and velocity threshold found 0 episodes. "
    f"Try: tq.ingest('{path}', boundary_strategy='manual', markers=[...])"
)
```
Every message: [what failed] + [why] + [what the user should try next].

### Logging vs Printing

| Use case | Method | Example |
|---|---|---|
| Recoverable warning (corrupt file, skipped topic) | `logging.getLogger('torq.{module}').warning(...)` | `logger.warning(f"Skipping malformed MCAP: {path}")` |
| Gravity well prompt | `_gravity_well()` only | never `print()` |
| Progress feedback | `tqdm(...)` only | never `print(f"Processing {i}...")` |
| Debug info | `logging.getLogger('torq.{module}').debug(...)` | suppressed by default |

**Logger naming:** one logger per module, named after the module path:
```python
# at top of each module file
import logging
logger = logging.getLogger(__name__)  # e.g. 'torq.ingest.mcap'
```

**Rule:** `print()` is reserved exclusively for gravity wells (via `_gravity_well()`) and
tqdm progress bars. All other output uses the `logging` module. This ensures Jake's
`tq ingest --json` pipeline can suppress prints while warnings remain on stderr.

### Timestamp Format

**Single source of truth: `np.int64` nanoseconds since epoch, everywhere internally.**

| Context | Format | Parquet column name |
|---|---|---|
| Internal storage | `np.int64` nanoseconds | `timestamp_ns` |
| External repr / user-facing | float seconds | `episode.duration` in seconds |
| MCAP source | nanoseconds (native) | direct pass-through |
| HDF5 source | convert to ns on ingest | `int(t * 1e9)` |
| LeRobot source | convert to ns on ingest | per-format converter |

**Anti-pattern:** Never mix float seconds and nanoseconds in the same computation.
All functions that receive timestamps receive `np.int64` nanoseconds.

### Parquet Schema Conventions

Column names are **templates**, not fixed strings — actual column count varies by embodiment.
Templates defined as constants in `src/torq/storage/parquet.py` and imported everywhere.

| Data | Template | Access pattern |
|---|---|---|
| Timestamps | `timestamp_ns` | `df['timestamp_ns']` |
| Joint positions | `obs_joint_pos_{i}` | `[c for c in df.columns if c.startswith('obs_joint_pos_')]` |
| Joint velocities | `obs_joint_vel_{i}` | `[c for c in df.columns if c.startswith('obs_joint_vel_')]` |
| Actions | `action_{i}` | `[c for c in df.columns if c.startswith('action_')]` |
| Force-torque | `obs_ft_{i}` | `[c for c in df.columns if c.startswith('obs_ft_')]` |
| Success flag | `metadata_success` | `df['metadata_success']` |

**Rule:** Never hardcode column names outside `parquet.py`. Always use the constants
or prefix-based column discovery. Column count is determined at ingest time per embodiment.

### Episode Interface Contract

`Episode` must expose these fields — set at ingest, consumed by compose and serve layers:

```python
@dataclass
class Episode:
    episode_id: str                          # ep_0001 format
    observations: dict[str, np.ndarray]      # modality_name → array [T, D]
    actions: np.ndarray                      # [T, action_dim]
    timestamps: np.ndarray                   # np.int64 nanoseconds [T]
    observation_keys: list[str]              # e.g. ['joint_pos', 'wrist_cam']
    action_keys: list[str]                   # e.g. ['joint_vel']
    duration_ns: int                         # total duration in nanoseconds
    source_path: Path                        # provenance — source file
    metadata: dict                           # mutable — user tags, success flag, etc.
    quality: QualityReport | None            # mutable — attached by tq.quality.score()
    tags: list[str]                          # mutable — user labels
```

`observation_keys` and `action_keys` are populated at ingest and used by the DataLoader
to build tensors. These are the contract between ingest and serve layers.

### Dataset Interface Contract

`Dataset` must expose this interface — contract between compose and serve layers:

```python
class Dataset:
    episodes: list[Episode]      # the actual episodes
    recipe: dict                 # the composition query that created this dataset
    name: str                    # version string e.g. 'pick_place_v3'
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Episode]: ...
    def __repr__(self) -> str: ...  # "Dataset('pick_place_v3', 31 episodes, quality_avg=0.81)"
```

### Return Type Conventions

| Function | Return Type | Notes |
|---|---|---|
| `tq.ingest(path)` | `list[Episode]` | Always subscriptable, never a generator |
| `tq.quality.score(episode)` | `Episode` | In-place mutation, returns same object |
| `tq.quality.score(episodes)` | `list[Episode]` | In-place batch, returns same list |
| `tq.query(...)` | `Iterator[Episode]` | Lazy — may be 100K+ episodes |
| `tq.compose(...)` | `Dataset` | Named object with recipe attached |
| `tq.DataLoader(dataset)` | `TorqDataLoader` | Subclasses torch DataLoader |

### Edge Case Handling — Mandatory Rules

| Situation | Required Behaviour | Anti-Pattern |
|---|---|---|
| Empty directory ingest | `[]` + `logger.warning(...)` | Raise exception |
| Corrupt file in bulk ingest | Log warning + continue remaining files | Abort entire ingest |
| Episode < 10 timesteps | Quality score = `None` + `logger.warning(...)` | `NaN` or `0.0` |
| `tq.compose()` returns 0 episodes | Empty `Dataset` + `logger.warning(...)` | Raise exception |
| Optional dep not installed | `TorqImportError` with install instruction | Bare `ImportError` |
| Unknown MCAP topic type | `logger.warning(...)` + skip topic | Raise or silently drop |

**`None` vs `NaN` rule:** Quality scores that cannot be computed return `None`, never
`float('nan')`. NaN propagates silently through numpy. None fails loudly and visibly.

**Required test:** Every quality scoring function must have an explicit test case with
an episode of < 10 timesteps asserting `score is None` — not `NaN`, not `0.0`.

### Progress Bar Pattern

```python
# CORRECT
for episode in tqdm(episodes, desc="Scoring episodes", unit="ep",
                    disable=tq.config.quiet):
    ...

# WRONG — missing disable hook
for episode in tqdm(episodes):
    ...

# WRONG — manual print
print(f"Processing {i} of {n}...")
```

Every tqdm call must include `disable=tq.config.quiet`.

### Optional Import Pattern

```python
# CORRECT — deferred inside the function, in src/torq/errors.py or local
def _require_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise TorqImportError(
            "PyTorch is required for tq.DataLoader(). "
            "Install it with: pip install torq-robotics[torch]"
        ) from None

# WRONG — module level (breaks `import torq` for non-torch users)
import torch
```

Applies to: `torch`, `jax`, `imageio`, `opencv`, `wandb`, `mlflow`.

### Public API Surface (`src/torq/__init__.py`)

```python
from torq._version import __version__
from torq.ingest import ingest
from torq.quality import quality          # namespace: tq.quality.score()
from torq.compose import compose, query, Dataset
from torq.storage import save, load
from torq.config import config
from torq.cloud import cloud
from torq.errors import TorqError         # base only — users catch this
# NOTE: DataLoader is NOT exported here — requires explicit import:
# from torq.serve import DataLoader
# This signals it is framework-specific and prevents import failure for non-torch users.
```

Rule: Nothing not in this list is part of the public API.
`DataLoader` requires `from torq.serve import DataLoader` — never from top-level `torq`.

### Gravity Well Pattern

All gravity wells call `_gravity_well()` from `src/torq/_gravity_well.py`. Never print directly.

```python
# CORRECT
from torq._gravity_well import _gravity_well
_gravity_well(
    message=f"Your episodes scored {score:.2f}. See how you compare to the community?",
    feature="GW-01"
)

# WRONG
print(f"💡 Your episodes scored {score:.2f}...")
```

Output format (owned exclusively by `_gravity_well()`):
```
💡 {message}
   → https://www.datatorq.ai
```

### Enforcement Summary

**All AI agents MUST:**
- Import `TorqError` subclasses from `src/torq/errors.py` — never define local exceptions
- Use `pathlib.Path` for all file operations — never `os.path` or string concatenation
- Store and pass timestamps as `np.int64` nanoseconds — convert at ingest boundary only
- Use Parquet column name templates from `src/torq/storage/parquet.py` via prefix discovery
- Include `disable=tq.config.quiet` on every tqdm call
- Use `logging.getLogger(__name__)` for warnings — never `print()` for recoverable issues
- Return `None` (not `NaN`, not `0.0`) for uncomputable quality scores
- Call `_gravity_well()` — never print gravity prompts directly
- Never import torch, jax, imageio, or opencv at module level

**Anti-patterns caught by CI:**
- `raise ValueError(...)` in `src/torq/` → use typed TorqError subclass
- `import torch` at module level → import graph CI test catches this
- `os.path.join(...)` → pathlib.Path standard
- `float('nan')` as quality return value → tested explicitly per scoring function
- `print(f"💡...")` outside `_gravity_well.py` → code review
- Hardcoded Parquet column names outside `parquet.py` → code review

## Project Structure & Boundaries

### Complete Project Directory Structure

```
torq-robotics/
├── pyproject.toml                        # hatchling build backend, deps, extras
├── README.md                             # 5-line workflow promise + install guide
├── CLAUDE.md                             # AI agent operating manual
├── ARCHITECTURE.md                       # points to planning-artifacts/architecture.md
├── TESTING.md                            # 130+ test cases in Given/When/Then format
├── progress.yml                          # build state tracker (session continuity)
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml                        # GitHub Actions matrix (3 Python × 3 OS)
│
├── src/
│   └── torq/
│       ├── __init__.py                   # minimal public API surface (see Patterns section)
│       ├── _version.py                   # __version__ string only
│       ├── _gravity_well.py              # _gravity_well() — single owner of prompt format
│       ├── _config.py                    # Config singleton — tq.config
│       ├── errors.py                     # complete TorqError hierarchy
│       ├── types.py                      # EpisodeID, Timestamp, QualityScore, TaskName, EmbodimentName
│       ├── cloud.py                      # tq.cloud() — wraps GW-04 _gravity_well() call
│       ├── episode.py                    # Episode dataclass — DEPENDENCY ROOT
│       │
│       ├── media/
│       │   ├── __init__.py
│       │   └── image_sequence.py         # ImageSequence — lazy MP4 frame loader
│       │
│       ├── storage/
│       │   ├── __init__.py               # save_episode(), load_episode()
│       │   ├── parquet.py                # Parquet r/w + COLUMN NAME TEMPLATES (source of truth)
│       │   ├── video.py                  # MP4 r/w via imageio-ffmpeg (conditional import)
│       │   └── index.py                  # Sharded JSON index — episode ID generation here only
│       │
│       ├── ingest/
│       │   ├── __init__.py               # tq.ingest() entry point + format auto-detection
│       │   ├── alignment.py              # multi-rate temporal alignment (np.int64 ns)
│       │   ├── mcap.py                   # MCAP/ROS2 → Episode (Humble/Iron/Jazzy)
│       │   ├── lerobot.py                # LeRobot v3.0 (Parquet+MP4) → Episode
│       │   └── hdf5.py                   # robomimic HDF5 → Episode
│       │
│       ├── quality/
│       │   ├── __init__.py               # tq.quality namespace + tq.quality.score() re-export
│       │   ├── smoothness.py             # jerk-based scoring (pure numpy)
│       │   ├── consistency.py            # autocorrelation scoring (pure numpy)
│       │   ├── completeness.py           # heuristic completeness (metadata.success + duration)
│       │   ├── report.py                 # QualityReport dataclass + COMPOSITE SCORING LOGIC
│       │   └── registry.py               # custom metric plugin registry + weight rescaling
│       │
│       ├── compose/
│       │   ├── __init__.py               # tq.compose(), tq.query()
│       │   ├── filters.py                # filter predicates: task, quality, embodiment, date
│       │   ├── sampling.py               # stratified, quality_weighted, none strategies
│       │   └── dataset.py                # Dataset class (episodes, recipe, name, __len__, __iter__)
│       │
│       ├── integrations/                 # ML experiment tracking (ML-03)
│       │   ├── __init__.py               # _notify_integrations() generic hook
│       │   ├── wandb.py                  # W&B integration (conditional wandb import)
│       │   └── mlflow.py                 # MLflow integration (conditional mlflow import)
│       │
│       ├── serve/
│       │   ├── __init__.py               # NOT in torq.__init__ — explicit import only
│       │   └── torch_loader.py           # TorqDataLoader — calls _notify_integrations() at init
│       │
│       └── cli/
│           ├── __init__.py
│           └── main.py                   # Typer app: tq ingest, tq list, tq info, tq export
│
├── tests/
│   ├── conftest.py                       # shared pytest fixtures
│   ├── test_imports.py                   # import graph CI gate — 3 mandatory cases:
│   │                                     #   1. `import torq` succeeds with core deps only
│   │                                     #   2. `import torq.serve` does NOT import torch
│   │                                     #   3. episode.py has no imports from torq.*
│   ├── unit/
│   │   ├── test_episode.py               # F01 — 18 tests incl. immutability guard
│   │   ├── test_image_sequence.py        # F02 — 12 tests incl. lazy loading
│   │   ├── test_storage_parquet.py       # F05 — 6 tests
│   │   ├── test_storage_video.py         # F06 — 3 tests
│   │   ├── test_storage_index.py         # F07 — 4 tests incl. atomic write, string normalization
│   │   ├── test_alignment.py             # F09 — 8 tests incl. multi-rate interpolation
│   │   ├── test_ingest_mcap.py           # F10 — 10 tests incl. corrupt MCAP fixture
│   │   ├── test_ingest_lerobot.py        # F11 — 5 tests
│   │   ├── test_ingest_hdf5.py           # F12 — 3 tests
│   │   ├── test_ingest_auto.py           # F13 — 6 tests incl. format detection
│   │   ├── test_quality_smoothness.py    # F14 — 8 tests incl. <10 timesteps → None assertion
│   │   ├── test_quality_consistency.py   # F15 — 4 tests incl. <10 timesteps → None assertion
│   │   ├── test_quality_completeness.py  # F16 — 5 tests incl. <10 timesteps → None assertion
│   │   ├── test_quality_report.py        # F17 — 6 tests incl. weight rescaling
│   │   ├── test_quality_registry.py      # registry — idempotency, weight normalization
│   │   ├── test_compose_filters.py       # F18 — 7 tests
│   │   ├── test_compose_sampling.py      # F19 — 9 tests incl. determinism with seed
│   │   ├── test_compose_dataset.py       # F19 — Dataset interface contract
│   │   ├── test_gravity_well.py          # GW-SDK-06 — all 5 wells via single helper
│   │   ├── test_config.py                # config singleton, TORQ_QUIET env var
│   │   └── test_cli.py                   # F22 — 8 tests incl. --json flag, exit codes
│   │
│   ├── integration/
│   │   ├── test_ingest_storage.py        # ingest → save → load round-trip
│   │   ├── test_quality_pipeline.py      # ingest → score → quality attached
│   │   ├── test_compose_pipeline.py      # ingest → score → compose → Dataset
│   │   └── test_dataloader.py            # Dataset → TorqDataLoader → batch iteration
│   │
│   ├── acceptance/
│   │   └── test_five_line_workflow.py    # F23 — the README promise, SC-1.3
│   │
│   ├── benchmarks/
│   │   └── test_nfr_performance.py       # @pytest.mark.benchmark — NFR-P02 (100K query <1s)
│   │                                     # ubuntu-latest only; excluded from windows-latest
│   │
│   └── fixtures/
│       ├── generate_fixtures.py           # deterministic fixture generator (run once)
│       ├── conftest.py                    # fixture path helpers
│       └── data/
│           ├── sample.mcap                # minimal 2-topic MCAP (joint_states + actions)
│           ├── multi_camera.mcap          # MCAP with 2 camera topics + joint data
│           ├── boundary_detection.mcap    # synthetic ~30s continuous stream, 3 episodes,
│           │                             # no explicit markers — tests DI-03 segmentation
│           │                             # ground truth: 3 boundary timestamps in fixture metadata
│           ├── corrupt.mcap               # truncated MCAP (NFR-R04 adversarial)
│           ├── empty.mcap                 # valid header, zero messages
│           ├── robomimic_simple.hdf5      # 2 demos, joint_pos + actions
│           ├── corrupt.hdf5               # truncated HDF5 (NFR-R04 adversarial)
│           ├── lerobot/                   # minimal LeRobot v3.0 directory structure
│           │   ├── meta/info.json
│           │   ├── data/chunk-000/
│           │   └── videos/chunk-000/
│           ├── benchmark/                 # gitignored — generated by generate_fixtures.py
│           │   └── 100k_episodes/         # 100K synthetic episodes for NFR-P02 benchmark
│           └── quality_ground_truth/
│               └── labeled_episodes.pkl   # 50 synthetic episodes with hand-labeled quality scores
```

### Dependency Rules (enforced by `test_imports.py`)

```
episode.py        ← imports NOTHING from torq (dependency root)
errors.py         ← imports NOTHING from torq
types.py          ← imports NOTHING from torq
_version.py       ← imports NOTHING from torq
_config.py        ← imports errors only
_gravity_well.py  ← imports _config only
cloud.py          ← imports _gravity_well only

media/            ← imports episode, errors
storage/          ← imports episode, errors, media, types
ingest/           ← imports episode, errors, storage, media, types
quality/          ← imports episode, errors, types
compose/          ← imports episode, errors, storage, quality, types
integrations/     ← imports types only (all framework imports conditional)
serve/            ← imports episode, errors, compose, integrations, types
cli/              ← imports all modules (thin wrappers only)
```

**Forbidden import directions (caught by CI):**
- `quality/` importing from `ingest/`, `compose/`, or `serve/`
- `compose/` importing from `serve/` or `ingest/`
- `storage/` importing from `ingest/`, `quality/`, `compose/`, or `serve/`
- `episode.py` importing from any other `torq.*` module

### FR Category → File Mapping

| FR | Primary File(s) | Notes |
|---|---|---|
| DI-01 MCAP ingest | `ingest/mcap.py` | boundary detection + topic discovery |
| DI-02 HDF5/LeRobot | `ingest/hdf5.py`, `ingest/lerobot.py` | |
| DI-03 Boundary detection | `ingest/mcap.py` (segmentation) | composite strategy lives here |
| DI-04 Canonical episode | `episode.py`, `storage/parquet.py` | column templates in parquet.py |
| QM-01 Quality scoring | `quality/smoothness.py`, `consistency.py`, `completeness.py` | |
| QM-02 Quality report | `quality/report.py` | COMPOSITE SCORING LOGIC lives here |
| QM-03 Quality gates | `compose/filters.py` | quality_min predicate |
| QM-04 Distribution (R1) | `quality/__init__.py` | text summary only in R1 |
| DC-01 Query builder | `compose/filters.py`, `storage/index.py` | index lookup + predicate evaluation |
| DC-02 Stratified sampling | `compose/sampling.py` | deterministic with seed |
| DC-03 Dataset versioning | `compose/dataset.py`, `storage/index.py` | recipe stored with Dataset |
| ML-01 PyTorch DataLoader | `serve/torch_loader.py` | |
| ML-03 W&B/MLflow | `integrations/wandb.py`, `integrations/mlflow.py` | called via `_notify_integrations()` |
| QE-02 Structured query | `compose/__init__.py`, `storage/index.py` | tq.query() entry point |
| GW-SDK-01–06 | `_gravity_well.py` | all 5 wells route through here |
| GW-SDK-01 | `quality/__init__.py` | fires after score() |
| GW-SDK-02 | `compose/__init__.py` | fires after compose() |
| GW-SDK-03 | `serve/torch_loader.py` | fires at DataLoader init if >50GB |
| GW-SDK-04 | `cloud.py` | tq.cloud() explicit call |
| GW-SDK-05 | `compose/__init__.py` | fires when result < 5 episodes |
| PD-01/PD-02 | `pyproject.toml`, `__init__.py` | |
| CLI (NFR-U03) | `cli/main.py` | |

### `src/torq/types.py` Contents

```python
from pathlib import Path
import numpy as np

EpisodeID = str          # format: "ep_0001"
Timestamp = np.int64     # nanoseconds since epoch
QualityScore = float | None  # None when episode too short to score
TaskName = str           # normalised: lowercase, stripped
EmbodimentName = str     # normalised: lowercase, stripped
```

### Data Flow

```
Raw files on disk (MCAP / HDF5 / LeRobot)
        │
        ▼  src/torq/ingest/
           format detection → parser → temporal alignment → episode segmentation
        │
        ▼  list[Episode]
           observation_keys, action_keys populated at ingest
           all timestamps as np.int64 nanoseconds
        │
        ▼  src/torq/storage/
           save: Parquet (obs/actions) + MP4 (images, optional) + JSON index update (atomic)
           load: Parquet → numpy arrays → Episode reconstruction
        │
        ▼  src/torq/quality/
           score: smoothness + consistency + completeness → QualityReport (report.py)
           episode.quality mutated in-place
        │
        ▼  src/torq/compose/
           query: index shard lookup (set intersection) → episode IDs
           filter: quality gate + task/embodiment predicates (filters.py)
           sample: stratified / quality_weighted (sampling.py)
           → Dataset(episodes, recipe, name)
        │
        ▼  src/torq/serve/
           TorqDataLoader → _notify_integrations() → collate → batch tensors
           obs tensor: [batch, T, obs_dim] built from episode.observation_keys
           action tensor: [batch, T, action_dim] built from episode.action_keys
        │
        ▼  Training loop
```

### Storage Layout on Disk

```
~/.torq/
└── config.toml               # R1: stub created; R2: telemetry preference written

{dataset_root}/               # user-chosen storage root passed to tq.ingest()
├── episodes/
│   ├── ep_0001.parquet
│   ├── ep_0002.parquet
│   └── ...
├── videos/                   # only created if episodes contain image modalities
│   ├── ep_0001.mp4
│   └── ...
└── index/
    ├── manifest.json         # episode count, schema version, last_updated
    ├── by_task.json          # inverted index: task → [ep_ids]
    ├── by_embodiment.json    # inverted index: embodiment → [ep_ids]
    └── quality.json          # sorted [(score, ep_id)] for binary search range queries
```

## Architecture Validation Results

### Coherence Validation ✅

**Decision Compatibility:**
All technology choices are compatible and complementary. numpy 2.0 + pyarrow 12 + mcap 1.1 have no known compatibility conflicts. hatchling 1.29 handles the src-layout correctly when `packages = ["src/torq"]` is set. Typer's `[all]` extra bundles rich (pretty CLI output) and shellingham (shell auto-completion detection) — both appropriate for UX goals. Python 3.10+ is required by pytest 9.x, aligns with all chosen library versions, and supports structural pattern matching if needed in R2. No version conflicts identified.

**Pattern Consistency:**
Implementation patterns fully support architectural decisions. The `__setattr__` mutation guard aligns with the mutable-but-protected Episode model. The optional import `_require_torch()` pattern is consistent with lazy loading and the two-tier dependency model. The sharded JSON index design aligns with the NFR-P02 query performance requirement and eliminates the need for any database. Parquet column templates (not fixed strings) align with the hardware-agnostic Episode model. All patterns use `pathlib.Path` as required for Windows compatibility.

**Structure Alignment:**
The src-layout (`src/torq/`) enables the dependency direction rules to be enforced by CI — `test_imports.py` can verify that `episode.py` imports nothing from `torq.*` without ambiguity. The `integrations/` directory cleanly isolates ML framework coupling (ML-03) from the core pipeline. All gravity wells route through `_gravity_well.py` — the single owner enforced by structure. The `cli/` module is the only module allowed to import from all others — the structure makes this boundary explicit.

---

### Requirements Coverage Validation ✅

**Functional Requirements Coverage:**

| FR Category | Coverage Status | Primary Files |
|---|---|---|
| DI-01 MCAP ingest | ✅ Full | `ingest/mcap.py` + `ingest/alignment.py` |
| DI-02 HDF5/LeRobot | ✅ Full | `ingest/hdf5.py`, `ingest/lerobot.py` |
| DI-03 Boundary detection | ✅ Full | `ingest/mcap.py` (composite strategy, tie-break rule specified) |
| DI-04 Canonical episode | ✅ Full | `episode.py` (interface contract), `storage/parquet.py` (column templates) |
| QM-01 Quality scoring | ✅ Full | `quality/smoothness.py`, `quality/consistency.py`, `quality/completeness.py` |
| QM-02 Quality report | ✅ Full | `quality/report.py` (composite scoring, weight rescaling) |
| QM-03 Quality gates | ✅ Full | `compose/filters.py` (quality_min predicate) |
| QM-04 Distribution | ✅ R1 text-summary | `quality/__init__.py` |
| DC-01 Query builder | ✅ Full | `compose/filters.py` + `storage/index.py` |
| DC-02 Stratified sampling | ✅ Full | `compose/sampling.py` (deterministic with seed) |
| DC-03 Dataset versioning | ✅ Full | `compose/dataset.py` + `storage/index.py` (recipe stored) |
| ML-01 PyTorch DataLoader | ✅ Full | `serve/torch_loader.py` |
| ML-03 W&B/MLflow | ✅ Full | `integrations/` via `_notify_integrations()` |
| QE-01 NL search | ✅ Deferred R2 | Scope correction confirmed by product owner |
| QE-02 Structured query | ✅ Full | `compose/__init__.py` + `storage/index.py` |
| GW-SDK-01–06 | ✅ Full | `_gravity_well.py` (single owner) |
| PD-01/PD-02 | ✅ Full | `pyproject.toml`, `__init__.py` |

**Non-Functional Requirements Coverage:**

| NFR | Coverage | Architectural Mechanism |
|---|---|---|
| NFR-P01 (1GB ingest < 10s) | ✅ | Streaming MCAP parser; no in-memory buffering of full file |
| NFR-P02 (query < 1s, 100K+) | ✅ | Sharded JSON index + set intersection + binary search on quality |
| NFR-P03 (DataLoader ≥ 1,000 ep/s) | ✅ | TorqDataLoader subclasses torch.DataLoader; pin_memory, num_workers |
| NFR-P05 (import torq < 2s) | ✅ | Lazy loading of all optional deps; CI gate enforces no-extras import |
| NFR-R04 (zero silent corruption) | ✅ | Atomic `os.replace()` on all index writes; adversarial corrupt fixtures |
| NFR-U01 (5-line workflow < 20min) | ✅ | F23 acceptance test; README promise |
| NFR-U02 (human-readable exceptions) | ✅ | TorqError hierarchy with [what] + [why] + [what to try] mandate |
| NFR-U03 (--json + non-zero exit) | ✅ | Typer CLI with --json flag and exit code discipline |
| NFR-C01 (Python 3.10+) | ✅ | `requires-python = ">=3.10"` in pyproject.toml |
| NFR-C02 (PyTorch 2.0+ incl. DDP) | ✅ | TorqDataLoader subclasses torch.DataLoader (DDP compatible by inheritance) |
| NFR-C04 (ROS 2 Humble/Iron/Jazzy) | ✅ | MCAP ingester handles all three MCAP schema versions |
| NFR-C05 (core import without ML) | ✅ | CI gate: `python -c "import torq"` with core deps only |
| Windows support | ✅ | `pathlib.Path` everywhere; CI informational gate |

---

### Implementation Readiness Validation ✅

**Decision Completeness:**
All critical decisions are documented with specific library versions. The pyproject.toml is fully specified (including hatchling wheel target, numpy 2.0+, mcap 1.1+, strict markers, coverage config). No agent implementing F04 will need to invent any dependency version or project configuration detail.

**Structure Completeness:**
Every file in the project tree is named with its responsibility noted. The FR-to-file mapping table leaves no ambiguity about which file owns which requirement. Dependency direction rules are fully specified and CI-enforceable via `test_imports.py`. Episode ID generation ownership is pinned to `storage/index.py` exclusively.

**Pattern Completeness:**
All nine cross-cutting consistency rules are documented with correct and incorrect examples. Parquet column naming templates prevent hardcoded column strings across modules. Timestamp format (`np.int64` nanoseconds) is the single source of truth with conversion rules at ingest boundaries. The `None` vs `NaN` rule is explicit with mandatory test coverage. Progress bar pattern includes the `disable=tq.config.quiet` hook on every call.

---

### Gap Analysis Results

**Critical Gaps — CLOSED via Party Mode:**

**Gap 1 — pyproject.toml exact content (CLOSED ✅)**

The complete `pyproject.toml` is now specified. No agent will need to invent dependency versions or configuration structure. See "Resolved Specification: pyproject.toml" below.

Key additions from Party Mode review:
- `numpy>=2.0` (not 1.24 — numpy 2.x is current for 2026 deployments)
- `mcap>=1.1` (not 1.0 — 1.1 guarantees stable MCAP v2 schema reader API)
- `[tool.hatch.build.targets.wheel] packages = ["src/torq"]` — critical for src-layout builds
- `addopts = "--strict-markers"` in pytest config — catches undefined marker typos
- `integration` marker added to the list
- `[tool.coverage.run]` section pointing to `src/torq` — prevents test files in coverage reports
- `hatchling` added to `[dev]` extra so contributors can build from source

**Gap 2 — Temporal alignment algorithm (CLOSED ✅)**

The complete temporal alignment algorithm is now specified including function signature, interpolation rules per modality type, edge case behaviour, and mandatory test cases. See "Resolved Specification: Temporal Alignment Algorithm" below.

Key decisions from Party Mode review:
- Target timestamps generated via `np.linspace()` — never copied from any source sensor (eliminates jitter drift)
- Nearest-neighbor for images is a **hard architectural rule** — never linear blending of frames
- Pure numpy only in `alignment.py` — no scipy dependency
- Caller (`mcap.py`, `hdf5.py`, `lerobot.py`) classifies `continuous_keys` vs `discrete_keys` — alignment module does not classify
- Explicit function signature specified (see below)

**Nice-to-Have — INCLUDED:**

`_notify_integrations()` signature formalised: `_notify_integrations(dataset: Dataset, config: dict) -> None`. Called by `TorqDataLoader.__init__()`. Silently skips if wandb/mlflow not installed (conditional import inside wandb.py and mlflow.py). No return value. No exception raised on missing optional deps.

**No remaining gaps.**

---

### Resolved Specification: pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "torq-robotics"
version = "0.1.0-alpha"
requires-python = ">=3.10"
dependencies = [
    "numpy>=2.0",
    "pyarrow>=12",
    "mcap>=1.1",
    "h5py>=3.8",
    "tqdm>=4.65",
    "typer[all]>=0.9",
]

[project.optional-dependencies]
torch  = ["torch>=2.0"]
vision = ["imageio-ffmpeg>=0.4"]
jax    = ["jax>=0.4"]
dev    = ["pytest>=9.0", "pytest-cov", "ruff>=0.15", "hatchling"]

[project.scripts]
tq = "torq.cli.main:app"

[tool.hatch.build.targets.wheel]
packages = ["src/torq"]

[tool.ruff]
line-length = 100
[tool.ruff.lint]
extend-select = ["I"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
addopts = "--strict-markers"
markers = [
    "slow: marks tests as slow (excluded from fast unit gate)",
    "benchmark: marks performance tests (ubuntu-latest only)",
    "integration: marks integration tests (require fixture files on disk)",
]

[tool.coverage.run]
source = ["src/torq"]
omit   = ["tests/*"]

[tool.coverage.report]
exclude_lines = ["if TYPE_CHECKING:"]
```

---

### Resolved Specification: Temporal Alignment Algorithm

**Module:** `src/torq/ingest/alignment.py`
**Feature:** F09

**Function Signature:**
```python
def align_to_frequency(
    data: dict[str, np.ndarray],        # modality_name → [T, D] array (D may be 1)
    timestamps: dict[str, np.ndarray],  # modality_name → [T] timestamps, np.int64 nanoseconds
    target_hz: float,                   # output frequency in Hz
    continuous_keys: list[str],         # these modalities use linear interpolation
    discrete_keys: list[str],           # these modalities use nearest-neighbor (no blending)
    start_ns: int | None = None,        # output window start; None → min of all input timestamps
    end_ns: int | None = None,          # output window end; None → max of all input timestamps
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Align multi-rate sensor data to a uniform target frequency.

    Returns:
        (aligned_timestamps_ns, aligned_data)
        aligned_timestamps_ns: np.int64 array [T'], exactly uniform at target_hz
        aligned_data: dict[str, ndarray [T', D]] for each key in data
    """
```

**Algorithm rules (hard rules — not heuristics):**

1. **Target timestamp generation:** `np.linspace(start_ns, end_ns, n_steps, dtype=np.int64)` where `n_steps = int((end_ns - start_ns) / (1e9 / target_hz)) + 1`. Never copy timestamps from any source sensor — source sensors have jitter.

2. **Target frequency selection (called by ingest parsers before calling this function):** highest Hz among all input sensors. If two sensors tie, either is valid — target is still that Hz.

3. **Continuous signals (joint_pos, joint_vel, force-torque, actions) → linear interpolation:**
   ```python
   # per-column (D may be > 1)
   aligned_col = np.interp(target_ts_ns, source_ts_ns, source_col)
   ```
   Interpolate each column independently. `np.interp` with no `left`/`right` kwargs → clamps at boundary values automatically. **No extrapolation.**

4. **Discrete signals (images, camera frames) → nearest-neighbor:**
   ```python
   indices = np.clip(np.searchsorted(source_ts_ns, target_ts_ns), 0, len(source_ts_ns) - 1)
   aligned_frames = source_frames[indices]
   ```
   **Hard rule:** output frames must be exact copies of input frames — never blended. Verified by test: `assert np.array_equal(aligned[cam_key][i], source[cam_key][nearest_idx])`.

5. **Edge clamping:** Values at or before first source timestamp → first known value. Values at or after last source timestamp → last known value. `np.interp` handles this automatically for continuous; the `np.clip` in searchsorted handles it for discrete.

6. **No scipy dependency.** Pure numpy only. `np.interp` and `np.searchsorted` are sufficient.

7. **Caller contract:** `mcap.py`, `hdf5.py`, and `lerobot.py` each classify topics into `continuous_keys` and `discrete_keys` based on topic type before calling `align_to_frequency()`. The alignment module trusts this classification — it does not inspect data shape or dtype to guess.

**Mandatory test cases for `tests/unit/test_alignment.py`:**

| Test | Description |
|---|---|
| T-ALN-01 | Multi-rate nominal: joint_states @ 200Hz + camera @ 30Hz → output @ 200Hz; camera frames nearest-neighbor only |
| T-ALN-02 | Jitter elimination: two sensors nominally @ 50Hz with ±2ms noise → output timestamps exactly uniform at 50Hz |
| T-ALN-03 | Empty input: `T=0` for all sensors → returns `(np.array([], dtype=np.int64), {})` without raising |
| T-ALN-04 | Single timestep: `T=1` → identity (output equals input value) |
| T-ALN-05 | Out-of-bounds clamping: `start_ns` before first source timestamp → first known value, no NaN |
| T-ALN-06 | Nearest-neighbor integrity: `aligned_data[cam_key][i]` equals `source_cam[nearest_idx]` exactly (no blending) |

---

### Resolved Specification: Integration Hook

**`_notify_integrations()` signature (in `src/torq/integrations/__init__.py`):**

```python
def _notify_integrations(dataset: Dataset, config: dict) -> None:
    """
    Notify all installed ML experiment tracking integrations.

    Called by TorqDataLoader.__init__() after dataset is prepared.
    Silently skips any integration whose optional dep is not installed.
    Never raises — failures are logged as warnings, not propagated.

    Args:
        dataset: the Dataset being loaded
        config: dict of DataLoader configuration (batch_size, chunk_size, etc.)
    """
```

Called with: `_notify_integrations(dataset, {'batch_size': batch_size, 'chunk_size': chunk_size, ...})`

Each integration module (`wandb.py`, `mlflow.py`) wraps its optional dep import in try/except and silently returns if the library is not installed.

---

### Architecture Completeness Checklist

**✅ Requirements Analysis**
- [x] Project context thoroughly analyzed (42 FRs, 15 NFRs, 8-layer architecture)
- [x] Scale and complexity assessed (100K+ episodes, multi-format ingestion)
- [x] Technical constraints identified (no network, no DB, no torch at module level)
- [x] Cross-cutting concerns mapped (9 cross-cutting rules documented)
- [x] Scope corrections validated with product owner (QE-01 → R2, Windows → R1)

**✅ Architectural Decisions**
- [x] Critical decisions documented with specific library versions
- [x] Technology stack fully specified (pyproject.toml content locked)
- [x] Integration patterns defined (`_notify_integrations` hook, conditional imports)
- [x] Performance considerations addressed (sharded index, atomic writes, lazy loading)
- [x] Episode data model specified (mutable + `__setattr__` guard, interface contract)

**✅ Implementation Patterns**
- [x] Naming conventions established (all constructs covered)
- [x] Exception hierarchy specified (`TorqError` tree, mandatory message format)
- [x] Timestamp format established (`np.int64` nanoseconds, single source of truth)
- [x] Parquet schema conventions (column templates, prefix-based discovery)
- [x] Logging vs printing rules (logging module everywhere except gravity wells + tqdm)
- [x] Optional import pattern (deferred `_require_*()` helpers)
- [x] Progress bar pattern (`disable=tq.config.quiet` on every tqdm call)
- [x] Gravity well pattern (`_gravity_well()` single owner)
- [x] Edge case handling table (6 mandatory behaviours, `None` vs `NaN` rule)

**✅ Project Structure**
- [x] Complete directory structure defined (every file named with responsibility)
- [x] Component boundaries established (dependency direction rules)
- [x] CI/CD pipeline specified (GitHub Actions matrix, fast gate + PR gate)
- [x] FR-to-file mapping table (every R1 FR mapped to primary file)
- [x] Fixture prerequisites identified (boundary_detection.mcap, quality ground truth, etc.)

**✅ Validation Gaps Closed**
- [x] pyproject.toml exact content specified (Gap 1 — closed)
- [x] Temporal alignment algorithm specified (Gap 2 — closed)
- [x] `_notify_integrations()` signature formalised (nice-to-have — included)

---

### Architecture Readiness Assessment

**Overall Status: READY FOR IMPLEMENTATION**

**Confidence Level: HIGH**

Every F01–F23 feature can be implemented by an AI agent without needing to invent architectural decisions. All dependencies, patterns, boundaries, and edge case behaviours are specified to the level of function signatures and test case descriptions.

**Key Strengths:**
- Complete pyproject.toml eliminates F04 ambiguity — the most common source of early divergence
- Temporal alignment algorithm spec prevents three ingest parsers from independently inventing incompatible approaches
- Dependency direction rules + `test_imports.py` make circular imports a CI failure, not a code review finding
- `None` vs `NaN` ruling with explicit test mandate prevents silent quality score propagation bugs
- Windows support enabled architecturally from day one (`pathlib.Path` standard, CI gate)
- Atomic index writes with `os.replace()` ensure NFR-R04 compliance is structural, not optional

**Areas for Future Enhancement (R2+):**
- QE-01 Natural Language Search (deferred — requires network + model hosting)
- JAX pipeline (deferred — duplicate of torch pipeline, R2 priority)
- `~/.torq/config.toml` telemetry preference (stub created in R1, written in R2)
- QM-06 Kinematic feasibility scoring via URDF validation (stub in R1 returns 1.0)
- Benchmark tests (`@pytest.mark.benchmark`) may need `pytest-benchmark` plugin in R2

---

### Implementation Handoff

**AI Agent Guidelines:**

- Follow all architectural decisions exactly as documented — no local exceptions
- Use implementation patterns consistently across all components — the enforcement table is your contract
- Respect module dependency directions — `episode.py` imports nothing; `cli/` imports everything
- All timestamps: `np.int64` nanoseconds internally, convert only at ingest boundary
- All file operations: `pathlib.Path` — zero `os.path` or string concatenation
- All quality scores: `None` when uncomputable — never `NaN`, never `0.0`
- Refer to this document for all architectural questions before asking the user

**First Implementation Priority:**

```bash
# Story 1 — Package Scaffolding (F04)
# Create pyproject.toml (exact content above), src/torq/__init__.py,
# src/torq/_version.py, src/torq/errors.py, src/torq/types.py
# Verify: pip install -e ".[dev]" && python -c "import torq; print(torq.__version__)"

# Story 2 — Episode Dataclass (F01)
# Implement src/torq/episode.py with __setattr__ guard
# Write all 18 tests in tests/unit/test_episode.py
# Verify: pytest tests/unit/test_episode.py -v
```

Build order: F01 → F02 → F03 → F04 → F05 → F06 → F07 → F08 → F09 → F10 → F11 → F12 → F13 → F14 → F15 → F16 → F17 → F18 → F19 → F20 → F21 → F22 → F23
