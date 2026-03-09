# Story 3.1: Core Quality Scoring Dimensions

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a robotics researcher,
I want each episode automatically scored on smoothness, consistency, and completeness,
so that I have objective per-dimension quality metrics without writing custom analysis code.

## Acceptance Criteria

1. **Given** an Episode with an actions array of 10 or more timesteps,
   **When** `smoothness.score(episode)` is called,
   **Then** a float in [0.0, 1.0] is returned based on jerk analysis (3rd derivative of joint positions, normalised).

2. **Given** an Episode with an actions array of 10 or more timesteps,
   **When** `consistency.score(episode)` is called,
   **Then** a float in [0.0, 1.0] is returned based on action autocorrelation (penalises oscillation and hesitation).

3. **Given** an Episode with `metadata.success = True`,
   **When** `completeness.score(episode)` is called,
   **Then** a score close to 1.0 is returned using the metadata flag as primary signal.

4. **Given** an Episode with fewer than 10 timesteps,
   **When** any scoring function (smoothness, consistency, or completeness) is called,
   **Then** `None` is returned (never `NaN`, never `0.0`),
   **And** `logger.warning()` names the episode ID and timestep count.

5. **Given** an Episode with NaN values in the actions array,
   **When** any scoring function is called,
   **Then** `None` is returned and a warning is logged (no exception raised, no NaN propagation).

## Tasks / Subtasks

- [x] Task 1: Create `src/torq/quality/smoothness.py` (AC: #1, #4, #5)
  - [x] Implement `score(episode: Episode) -> float | None`
  - [x] Compute jerk (3rd derivative) from `episode.actions` array
  - [x] Normalise RMS jerk to [0.0, 1.0] via sigmoid: `1.0 / (1.0 + rms_jerk / reference_jerk)`
  - [x] Return `None` + `logger.warning()` for episodes < 10 timesteps
  - [x] Return `None` + `logger.warning()` for episodes with NaN in actions
  - [x] Module logger: `logging.getLogger(__name__)`
  - [x] Pure numpy — no torch, no scipy required

- [x] Task 2: Create `src/torq/quality/consistency.py` (AC: #2, #4, #5)
  - [x] Implement `score(episode: Episode) -> float | None`
  - [x] Compute action deltas between consecutive timesteps
  - [x] Calculate autocorrelation of action deltas
  - [x] Penalise oscillation (frequent direction reversals) and hesitation (near-zero action periods)
  - [x] Normalise to [0.0, 1.0]
  - [x] Return `None` + `logger.warning()` for episodes < 10 timesteps
  - [x] Return `None` + `logger.warning()` for episodes with NaN in actions
  - [x] Pure numpy — no torch, no scipy required

- [x] Task 3: Create `src/torq/quality/completeness.py` (AC: #3, #4, #5)
  - [x] Implement `score(episode: Episode) -> float | None`
  - [x] Primary signal: `episode.metadata.get('success')` — if `True` -> ~1.0, if `False` -> ~0.0
  - [x] Fallback heuristic when no success flag: duration-based scoring (longer episodes score higher up to a plateau)
  - [x] Return `None` + `logger.warning()` for episodes < 10 timesteps
  - [x] Return `None` + `logger.warning()` for episodes with NaN in actions
  - [x] Pure numpy — no torch, no scipy required

- [x] Task 4: Write unit tests `tests/unit/test_quality_smoothness.py` (8 tests)
  - [x] Test smooth trajectory scores high (close to 1.0)
  - [x] Test jerky trajectory scores low (close to 0.0)
  - [x] Test constant-velocity trajectory scores ~1.0 (zero jerk)
  - [x] Test single-jerk spike in otherwise smooth trajectory
  - [x] Test episode < 10 timesteps returns `None` (not NaN, not 0.0)
  - [x] Test episode with NaN values returns `None`
  - [x] Test score is always in [0.0, 1.0] range for valid episodes
  - [x] Test single-dimension vs multi-dimension action arrays

- [x] Task 5: Write unit tests `tests/unit/test_quality_consistency.py` (4 tests)
  - [x] Test consistent unidirectional trajectory scores high
  - [x] Test oscillating trajectory scores low
  - [x] Test episode < 10 timesteps returns `None`
  - [x] Test episode with NaN values returns `None`

- [x] Task 6: Write unit tests `tests/unit/test_quality_completeness.py` (5 tests)
  - [x] Test `metadata.success = True` returns score close to 1.0
  - [x] Test `metadata.success = False` returns score close to 0.0
  - [x] Test no success flag uses duration heuristic
  - [x] Test episode < 10 timesteps returns `None`
  - [x] Test episode with NaN values returns `None`

### Review Follow-ups (AI)

- [x] [AI-Review][HIGH] No TorqQualityError usage — scorers raise raw numpy/Python exceptions instead of wrapping in TorqQualityError as required by architecture. Add try/except wrapping in each scorer's `score()` function. [smoothness.py, consistency.py, completeness.py]
- [x] [AI-Review][HIGH] `_validation.py` doesn't validate timestamps length alignment with actions — `completeness.py:57` (`episode.timestamps[-1]`) will crash with IndexError if timestamps array is empty or shorter than actions. Add timestamps length check to `validate_episode()`. [src/torq/quality/_validation.py]
- [x] [AI-Review][MEDIUM] `make_episode` helper duplicated across all 3 test files — extract to shared fixture in `tests/conftest.py` or a test helper module. [tests/unit/test_quality_smoothness.py, test_quality_consistency.py, test_quality_completeness.py]
- [x] [AI-Review][MEDIUM] No score range validation tests for consistency and completeness — smoothness has `test_score_always_in_range` but the other two scorers lack equivalent fuzz-style [0.0, 1.0] range tests. [tests/unit/test_quality_consistency.py, test_quality_completeness.py]
- [x] [AI-Review][MEDIUM] Consistency test threshold too weak — `test_consistent_unidirectional_scores_high` asserts `> 0.5` for a perfect monotonic ramp; tighten to `> 0.8`. [tests/unit/test_quality_consistency.py:38]
- [x] [AI-Review][LOW] Smoothness scorer ignores non-uniform timestamp spacing — jerk computation assumes uniform dt. Known R1 limitation, document for R2. [src/torq/quality/smoothness.py]
- [x] [AI-Review][LOW] `_validation.py` creates logger via `getLogger(scorer_name)` per call — consider accepting caller's logger object directly. [src/torq/quality/_validation.py:30]

## Dev Notes

### Architecture Patterns and Constraints

- **Pure numpy only.** No scipy, no torch, no external ML libraries for scoring. All three scorers use `numpy` exclusively.
- **No circular imports.** `quality/` imports from `episode.py`, `errors.py`, `types.py` only. Never imports from `ingest/`, `compose/`, `serve/`, or `storage/`.
- **`None` vs `NaN` rule is CRITICAL.** Quality scores that cannot be computed return `None`, never `float('nan')`. NaN propagates silently through numpy; None fails loudly. Every scorer must have an explicit test for `score is None` on short episodes.
- **Minimum timestep threshold: 10.** Episodes with `len(episode.actions) < 10` must return `None` from all three scorers. This is a hard architectural rule (not configurable in R1).
- **Logger per module:** Each scorer file gets `logger = logging.getLogger(__name__)` at module top. Warnings use `logger.warning()`, never `print()`.
- **Error hierarchy:** Use `TorqQualityError` from `src/torq/errors.py` for any scoring computation errors. Never raise bare `ValueError` or `RuntimeError`.
- **Type alias:** `QualityScore = float | None` is defined in `src/torq/types.py` — use it as return type annotation.
- **Score range:** All scores MUST be in [0.0, 1.0]. Clamp if necessary. Never return negative values or values > 1.0.
- **Episode field access:** Read `episode.actions` (np.ndarray, shape [T, action_dim]), `episode.timestamps` (np.int64 nanoseconds), `episode.metadata` (dict), `episode.episode_id` (str).
- **`episode.quality` field:** This story does NOT populate `episode.quality` — that is Story 3.2's responsibility (QualityReport + entry point). This story only creates the individual scoring functions.
- **Function signature:** Each scorer exposes a single public function `score(episode: Episode) -> QualityScore` — no class wrappers, no factory pattern. Keep it simple.
- **`__all__` export:** Each module must define `__all__ = ["score"]`.
- **Docstrings:** Google style on all public functions. Include Args, Returns, and Raises sections.

### Scoring Algorithm Details

**Smoothness (jerk-based):**
```python
# 1. Compute velocity: np.diff(actions, axis=0) along time axis -> shape [T-1, D]
# 2. Compute acceleration: np.diff(velocity, axis=0) -> shape [T-2, D]
# 3. Compute jerk: np.diff(acceleration, axis=0) -> shape [T-3, D]
# 4. RMS jerk = np.sqrt(np.mean(jerk**2))
# 5. Normalize: smoothness = 1.0 / (1.0 + rms_jerk / REFERENCE_JERK)
# REFERENCE_JERK is a tuning constant — start with 1.0, calibrate later
# Result: smooth motion -> low jerk -> high score; jerky motion -> high jerk -> low score
```

**Consistency (autocorrelation):**
```python
# 1. Compute action deltas: np.diff(actions, axis=0) -> shape [T-1, D]
# 2. Flatten deltas to 1D magnitude: np.linalg.norm(deltas, axis=1) -> shape [T-1]
# 3. Compute autocorrelation at lag=1:
#    autocorr = np.corrcoef(magnitudes[:-1], magnitudes[1:])[0, 1]
# 4. High positive autocorrelation -> consistent motion -> high score
# 5. Detect direction reversals: sign changes in deltas per dimension
#    reversal_ratio = num_sign_changes / (T - 2)
# 6. Combine: consistency = max(0, autocorr) * (1.0 - reversal_ratio)
# 7. Clamp to [0.0, 1.0]
```

**Completeness (heuristic):**
```python
# 1. If metadata.get('success') is True -> return 1.0
# 2. If metadata.get('success') is False -> return 0.0
# 3. If no success flag present:
#    - Compute duration_seconds from timestamps: (timestamps[-1] - timestamps[0]) / 1e9
#    - duration_score = min(1.0, duration_seconds / EXPECTED_DURATION)
#    - EXPECTED_DURATION = 30.0 seconds (tuning constant)
#    - Return duration_score
```

### NaN Detection Pattern

```python
import numpy as np

MIN_TIMESTEPS = 10

def _validate_episode(episode: Episode, scorer_name: str) -> bool:
    """Returns True if episode is valid for scoring, False otherwise (with warning logged)."""
    if len(episode.actions) < MIN_TIMESTEPS:
        logger.warning(
            f"{scorer_name}: Episode '{episode.episode_id}' has {len(episode.actions)} "
            f"timesteps (minimum {MIN_TIMESTEPS} required). Returning None."
        )
        return False
    if np.any(np.isnan(episode.actions)):
        logger.warning(
            f"{scorer_name}: Episode '{episode.episode_id}' contains NaN values "
            f"in actions array. Returning None."
        )
        return False
    return True
```

Consider extracting this shared validation to a private helper `src/torq/quality/_validation.py` to avoid duplicating the pattern across all three scorers.

### Episode Construction for Tests

Use the existing `Episode` dataclass directly. Example test fixture pattern:

```python
from pathlib import Path
import numpy as np
from torq.episode import Episode

def make_episode(actions: np.ndarray, metadata: dict | None = None) -> Episode:
    """Helper to create test episodes with minimal boilerplate."""
    T = len(actions)
    return Episode(
        episode_id="ep_test",
        observations={"joint_pos": np.zeros((T, 7))},
        actions=actions,
        timestamps=np.arange(T, dtype=np.int64) * int(1e9 / 30),  # 30 Hz
        source_path=Path("/test/fixture.mcap"),
        metadata=metadata or {},
    )

# Smooth trajectory: sine wave (low jerk)
smooth_actions = np.sin(np.linspace(0, 2 * np.pi, 50)).reshape(-1, 1)

# Jerky trajectory: random noise
jerky_actions = np.random.RandomState(42).randn(50, 1) * 10

# Oscillating trajectory: alternating sign
oscillating_actions = np.array([(-1)**i for i in range(50)]).reshape(-1, 1).astype(float)

# Short episode (< 10 timesteps)
short_actions = np.zeros((5, 1))

# NaN-containing episode
nan_actions = np.ones((20, 1))
nan_actions[10] = float('nan')
```

### Previous Story Intelligence

Epic 2 (Data Ingestion & Storage) is fully complete with all 7 stories done. Key patterns established:
- All modules follow `logger = logging.getLogger(__name__)` at module top
- Each module has `__all__` exports defined
- Tests use the `Episode` dataclass directly with synthetic data
- `conftest.py` in `tests/` provides shared fixtures
- File format: Google-style docstrings on all public functions
- Ruff formatting with 100 char line length

### Default Quality Weights (for reference — used in Story 3.2, not this story)

```python
DEFAULT_QUALITY_WEIGHTS = {
    'smoothness': 0.40,
    'consistency': 0.35,
    'completeness': 0.25,
}
```

### Project Structure Notes

- **Files to CREATE:**
  - `src/torq/quality/smoothness.py`
  - `src/torq/quality/consistency.py`
  - `src/torq/quality/completeness.py`
  - `src/torq/quality/_validation.py` (optional shared helper)
  - `tests/unit/test_quality_smoothness.py`
  - `tests/unit/test_quality_consistency.py`
  - `tests/unit/test_quality_completeness.py`
- **Files that EXIST (do NOT modify):**
  - `src/torq/quality/__init__.py` — placeholder, expanded in Story 3.2
  - `src/torq/__init__.py` — quality namespace export is Story 3.2
  - `src/torq/episode.py` — Episode dataclass (read-only dependency)
  - `src/torq/errors.py` — TorqQualityError (import from here)
  - `src/torq/types.py` — QualityScore type alias (import from here)
- Alignment with project structure in architecture.md: paths match exactly

### References

- [Source: architecture.md#Quality Scoring Architecture] — scoring dimensions, algorithms, default weights
- [Source: architecture.md#Edge Case Handling] — None vs NaN rule, < 10 timesteps rule
- [Source: architecture.md#Implementation Patterns] — logging, error hierarchy, naming conventions
- [Source: architecture.md#Episode Interface Contract] — Episode field definitions
- [Source: architecture.md#Project Structure] — file paths: `src/torq/quality/smoothness.py`, `consistency.py`, `completeness.py`
- [Source: epics.md#Story 3.1] — acceptance criteria in BDD format
- [Source: epics.md#Epic 3 overview] — epic objectives and cross-story context

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6

### Debug Log References

- Fixed consistency scorer: `np.corrcoef` returns spurious negative values (~-0.22) for near-constant magnitude arrays due to floating-point noise. Fixed by checking `np.std < 1e-9` before computing corrcoef and treating near-constant variance as perfectly consistent (autocorr = 1.0).

### Completion Notes List

- Implemented shared validation helper `_validation.py` to avoid duplicating MIN_TIMESTEPS and NaN checks across all three scorers (pattern suggested in Dev Notes).
- `smoothness.py`: jerk-based scorer using 3rd discrete derivative; REFERENCE_JERK = 1.0; sigmoid normalisation; pure numpy.
- `consistency.py`: autocorrelation + direction-reversal scorer; guarded against floating-point noise in constant-magnitude trajectories.
- `completeness.py`: metadata `success` flag primary signal; duration heuristic fallback with EXPECTED_DURATION = 30s.
- All 19 new tests pass (8 smoothness, 5 consistency, 6 completeness). Full regression suite: 188/188 passed.
- ✅ Resolved review finding [HIGH]: Added `try/except TorqQualityError` wrapping to all three `score()` functions.
- ✅ Resolved review finding [HIGH]: Added `len(episode.timestamps) < len(episode.actions)` guard to `validate_episode()` in `_validation.py`.
- ✅ Resolved review finding [MEDIUM]: Extracted `make_quality_episode` factory fixture to `tests/conftest.py`; removed duplicate helpers from all 3 test files.
- ✅ Resolved review finding [MEDIUM]: Added `test_score_always_in_range` to consistency and completeness test files.
- ✅ Resolved review finding [MEDIUM]: Tightened consistency unidirectional threshold from `> 0.5` to `> 0.8`.
- ✅ Resolved review finding [LOW]: Documented non-uniform timestamp limitation in `smoothness.py` module docstring with R2 improvement plan.
- ✅ Resolved review finding [LOW]: `validate_episode()` now accepts `logging.Logger` object directly; each scorer passes its module-level `logger`.

### File List

- `src/torq/quality/_validation.py` (created; updated in review round)
- `src/torq/quality/smoothness.py` (created; updated in review round)
- `src/torq/quality/consistency.py` (created; updated in review round)
- `src/torq/quality/completeness.py` (created; updated in review round)
- `tests/conftest.py` (modified — added `make_quality_episode` fixture)
- `tests/unit/test_quality_smoothness.py` (created; updated in review round)
- `tests/unit/test_quality_consistency.py` (created; updated in review round)
- `tests/unit/test_quality_completeness.py` (created; updated in review round)
