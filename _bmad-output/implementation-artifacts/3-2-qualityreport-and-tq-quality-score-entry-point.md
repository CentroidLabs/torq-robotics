# Story 3.2: QualityReport and tq.quality.score() Entry Point

Status: done

## Story

As a robotics researcher,
I want a single API call that scores all dimensions and attaches a QualityReport to each episode,
So that quality results are accessible via `episode.quality` after a single function call.

## Acceptance Criteria

1. **Given** a list of scored Episodes,
   **When** `tq.quality.score(episodes)` is called,
   **Then** each episode's `.quality` field is populated with a `QualityReport` in-place,
   **And** the same list is returned (in-place mutation, same object identity),
   **And** a tqdm progress bar shows scoring progress (respecting `tq.config.quiet`).

2. **Given** a single Episode (not a list),
   **When** `tq.quality.score(episode)` is called,
   **Then** the episode's `.quality` is populated and the same episode is returned.

3. **Given** a QualityReport,
   **When** `episode.quality.overall` is read,
   **Then** it returns the weighted composite: `smoothness×0.40 + consistency×0.35 + completeness×0.25` (using current `tq.config.quality_weights`),
   **And** if any component score is `None`, `overall` is also `None`.

4. **Given** per-call weight override: `tq.quality.score(episodes, weights={'smoothness': 0.5, 'consistency': 0.3, 'completeness': 0.2})`,
   **When** the call is made,
   **Then** the provided weights are used for this call only (global config unchanged),
   **And** if weights do not sum to 1.0 ± 0.001, `TorqQualityError` is raised before any scoring begins.

5. **Given** 100 episodes,
   **When** `tq.quality.score(episodes)` is called,
   **Then** all 100 episodes are scored in under 60 seconds.

6. **Given** an empty list `[]`,
   **When** `tq.quality.score([])` is called,
   **Then** an empty list is returned immediately with no progress bar and no gravity well.

## Tasks / Subtasks

- [x] Task 1: Create `src/torq/quality/report.py` - QualityReport dataclass (AC: #1, #2, #3)
  - [x] Define `QualityReport` dataclass with fields: `smoothness: QualityScore`, `consistency: QualityScore`, `completeness: QualityScore`, `overall: QualityScore`
  - [x] Use `@dataclass(frozen=True)` — all fields immutable after creation
  - [x] Compute `overall` in `__post_init__` using `object.__setattr__` (required for frozen dataclasses)
  - [x] Add docstring documenting fields, None-propagation rule, and weight override behavior
  - [x] Import `QualityScore` from `torq.types`

- [x] Task 2: Create `src/torq/quality/__init__.py` - Entry point and registry management (AC: #1, #2, #3, #4, #6)
  - [x] Implement `score(episodes: Episode | list[Episode], weights: dict[str, float] | None = None) -> Episode | list[Episode]`
  - [x] Early return `[]` immediately when `episodes` is an empty list (no progress bar, no gravity well)
  - [x] Accept both single Episode and list[Episode] for ergonomic API
  - [x] Return same object (in-place mutation for efficiency)
  - [x] Validate per-call weight override: must sum to 1.0 ± 0.001, raise `TorqQualityError` if invalid BEFORE any scoring starts
  - [x] Import scorers: `from .smoothness import score as smoothness_score` etc.
  - [x] Wrap each scorer call in try/except, re-raise as `TorqQualityError` with episode ID and dimension name
  - [x] For each episode, call all three scorers and construct `QualityReport`
  - [x] Add tqdm progress bar (skip for 0 or 1 episodes): `tqdm(episodes, disable=tq.config.quiet, desc="Scoring episodes")`
  - [x] Fire gravity well on success (non-empty result only): `_gravity_well(f"Average quality: {avg_score:.2f}", "GW-SDK-01")`
  - [x] Export from `__all__`: `["score", "register", "get_metrics", "reset"]`
  - [x] Add module-level logger: `logger = logging.getLogger(__name__)`

- [x] Task 3: Update `src/torq/__init__.py` - Wire `tq.quality` namespace (AC: #1, #2)
  - [x] Add import: `from torq import quality`
  - [x] Add `"quality"` to `__all__`

- [x] Task 4: Create `src/torq/quality/registry.py` - Custom metric plugin support (AC: #4 prep)
  - [x] Implement `register(name: str, fn: Callable, weight: float) -> None`
  - [x] Validate fn returns float in [0.0, 1.0], else raise `TorqQualityError` at scoring time
  - [x] Implement weight rescaling: when new metric added, scale existing weights proportionally
  - [x] Allow idempotent re-registration (overwrite silently with `UserWarning`)
  - [x] Support `get_metrics() -> dict` to inspect current registered metrics + weights
  - [x] Support `reset()` to clear all custom metrics and restore `DEFAULT_QUALITY_WEIGHTS`
  - [x] Raise `TorqQualityError` if callable returns value outside [0.0, 1.0]

- [x] Task 5: Update `src/torq/types.py` - Verify QualityScore type alias
  - [x] Verified `QualityScore = float | None` exists — no changes needed

- [x] Task 6: Write unit tests `tests/unit/test_quality_report.py` (37 tests)
  - [x] Test QualityReport composite scoring: `overall = s×0.40 + c×0.35 + co×0.25`
  - [x] Test None propagation: if ANY component is None, `overall` is None (not partial composite)
  - [x] Test single Episode scoring: single Episode in, single Episode out (same object identity)
  - [x] Test list[Episode] scoring: list returned with same object identity
  - [x] Test weight override validation: non-summing-to-1.0 weights raise `TorqQualityError` BEFORE scoring
  - [x] Test weight override usage: `score(..., weights={...})` uses provided weights for that call only
  - [x] Test empty list: `tq.quality.score([]) == []` and gravity well not called
  - [x] Test tqdm progress bar disabled for single episode and when quiet=True
  - [x] Test gravity well fires on success with average score message
  - [x] Test gravity well does NOT fire on exception

- [x] Task 7: Write integration tests `tests/integration/test_quality_score.py` (7 tests)
  - [x] Test 100-episode scoring completes in <60s (mark `@pytest.mark.slow`)
  - [x] Test round-trip: score → save/load → re-score produces same quality (R1: QualityReport not persisted by storage layer)
  - [x] Test custom metric registration + scoring with rescaled weights

### Review Follow-ups (AI)

- [x] [AI-Review][MEDIUM] Per-call weight override validates sum but not key presence — added `_REQUIRED_WEIGHT_KEYS` check in `score()` before sum validation; missing keys raise `TorqQualityError` with clear message. [src/torq/quality/__init__.py]
- [x] [AI-Review][MEDIUM] `score()` doesn't validate `episodes` parameter type — added early `isinstance(episodes, (Episode, list))` check raising `TorqQualityError` before any other logic. [src/torq/quality/__init__.py]
- [x] [AI-Review][MEDIUM] Custom metric returning `None` doesn't propagate to `overall` — when `custom_any_none` is True, `object.__setattr__(report, "overall", None)` is now called explicitly. [src/torq/quality/__init__.py]
- [x] [AI-Review][MEDIUM] Re-registration rescaling is cumulative — registry now reverses the old weight contribution (divides by `1 - old_weight`) before applying the new rescaling when re-registering an existing metric. [src/torq/quality/registry.py]
- [x] [AI-Review][MEDIUM] `test_progress_bar_disabled_for_single_episode` conditionally asserts — changed to unconditional `assert mock_tqdm.called` followed by `assert kwargs.get("disable") is True`. [tests/unit/test_quality_report.py]
- [x] [AI-Review][LOW] `tqdm` fallback passthrough silently drops kwargs — fallback now explicitly accepts `disable` and `desc` kwargs (and absorbs the rest via `**_kw`). [src/torq/quality/__init__.py]
- [x] [AI-Review][LOW] Integration test directly mutates `ep.quality = None` — replaced with a fresh episode object (`ep2`) for the re-score comparison, avoiding any potential guard issues. [tests/integration/test_quality_score.py]

## Dev Notes

### Key Implementation Contract

**QualityReport Structure:**
```python
from __future__ import annotations
from dataclasses import dataclass
from torq.types import QualityScore

@dataclass(frozen=True)
class QualityReport:
    smoothness: QualityScore    # 0.0–1.0 or None (episode too short/NaN)
    consistency: QualityScore   # 0.0–1.0 or None
    completeness: QualityScore  # 0.0–1.0 or None
    overall: QualityScore       # weighted composite, or None if ANY component is None

    def __post_init__(self) -> None:
        # frozen=True means we MUST use object.__setattr__ here — self.overall = ... raises FrozenInstanceError
        if any(v is None for v in (self.smoothness, self.consistency, self.completeness)):
            object.__setattr__(self, 'overall', None)
        else:
            from torq._config import DEFAULT_QUALITY_WEIGHTS
            # weights can be injected via a separate constructor parameter if needed
            w = DEFAULT_QUALITY_WEIGHTS
            composite = self.smoothness * w['smoothness'] + self.consistency * w['consistency'] + self.completeness * w['completeness']
            object.__setattr__(self, 'overall', round(composite, 6))
```

**Entry point signature:**
```python
def score(
    episodes: Episode | list[Episode],
    weights: dict[str, float] | None = None,
) -> Episode | list[Episode]:
```

**Gravity well call — exact signature:**
```python
from torq._gravity_well import _gravity_well
# Takes TWO arguments: message (str) and feature identifier (str)
_gravity_well(f"Average quality: {avg_score:.2f}", "GW-SDK-01")
# Output format (owned by _gravity_well.py):
#   💡 Average quality: 0.87
#      → https://www.datatorq.ai
```

**`DEFAULT_QUALITY_WEIGHTS` location:**
```python
from torq._config import DEFAULT_QUALITY_WEIGHTS
# {'smoothness': 0.40, 'consistency': 0.35, 'completeness': 0.25}
# Do NOT redefine this constant anywhere — single source of truth is _config.py
```

**`src/torq/__init__.py` wiring:**
```python
# Add to existing imports in src/torq/__init__.py:
from torq import quality
# Add "quality" to __all__
# quality is pure numpy — eager import is safe (won't breach <2s NFR-P05)
```

### Architecture Patterns and Constraints

- **QualityReport dataclass:** `frozen=True`. Use `object.__setattr__` in `__post_init__` to set `overall`. DO NOT try `self.overall = ...` — this raises `FrozenInstanceError` at runtime on frozen dataclasses.
- **None propagation rule:** If ANY of smoothness, consistency, completeness is `None`, `overall` MUST be `None`. Never compute a partial composite.
- **Entry point mutation:** `tq.quality.score()` mutates episodes IN-PLACE by setting `.quality` field. Same object identity returned for chaining.
- **Error types:**
  - Per-call weight validation in `score()` → `TorqQualityError` ("Quality weights must sum to 1.0 ± 0.001. Got: {total}. Try: {w / sum(w.values()) for w in weights.items()}")
  - Global config `tq.config.quality_weights = {...}` → `TorqConfigError` (handled in `_config.py` — do NOT duplicate this logic)
- **Progress bar:** Skip for empty list and single Episode (overhead not worth it). For 2+ episodes: `tqdm(episodes, disable=tq.config.quiet, desc="Scoring episodes")`
- **Gravity well:** Only fires on successful completion of non-empty input. NOT called when `episodes=[]` or when an exception is raised. Exact call: `_gravity_well(f"Average quality: {avg_score:.2f}", "GW-SDK-01")`
- **Performance:** Must score 100 episodes in <60s. Individual scorers are O(T) per episode. Total = O(100×T). No quadratic loops.
- **No circular imports:** `quality/` imports from `episode.py`, `errors.py`, `types.py`, `_config.py`, `_gravity_well.py` only. Never imports from `ingest/`, `compose/`, `serve/`, or `storage/`.
- **Composite scoring formula:** `overall = smoothness×0.40 + consistency×0.35 + completeness×0.25` (with proportional rescaling for custom metrics)

### Flow Diagram

```
score(episodes, weights={})
  ↓
Early return [] if episodes is empty list (no gravity well, no tqdm)
  ↓
Validate per-call weights (sum to 1.0 ± 0.001) → raise TorqQualityError if invalid
  ↓
Normalise to list if single Episode
  ↓
For each episode in tqdm(episodes, disable=config.quiet):
  ↓
  Call smoothness_score(episode) → wrap exceptions in TorqQualityError
  Call consistency_score(episode) → wrap exceptions in TorqQualityError
  Call completeness_score(episode) → wrap exceptions in TorqQualityError
  ↓
  Create QualityReport(s=score_s, c=score_c, co=score_co)
  (overall computed automatically in __post_init__, propagates None if any component is None)
  ↓
  episode.quality = report  (mutate in-place — episode.quality is a mutable field)
  ↓
Compute avg_score from non-None overall values
  ↓
Fire gravity well: _gravity_well(f"Average quality: {avg_score:.2f}", "GW-SDK-01")
  ↓
Return same episodes object (same identity — single Episode or list)
```

### Previous Story Intelligence (Story 3.1)

**Established patterns (use these, do not reinvent):**

1. **None vs NaN rule is CRITICAL:** Scorers return `None` for short episodes (<10 timesteps) and NaN-containing episodes. QualityReport MUST handle None via any-None → overall=None rule.

2. **Shared validation helper:** `src/torq/quality/_validation.py` exists (created in Story 3.1). The `validate_episode(episode, logger)` function handles both the <10 timestep check and NaN check. Import and reuse, do not duplicate.

3. **Error wrapping (Story 3.1 HIGH review fix — already resolved):** All three scorers now wrap exceptions in `TorqQualityError`. Story 3.2's `score()` entry point must still wrap scorer calls defensively in case a custom registered scorer raises a raw exception.

4. **`make_quality_episode` fixture:** Extracted to `tests/conftest.py` in Story 3.1. Use it in Story 3.2 tests. Do NOT define another `make_episode` helper — use the shared fixture.

5. **Performance baseline:** Individual scorers run in microseconds. 100 episodes is trivially within the 60s budget. The bottleneck is tqdm overhead, not computation.

6. **Timestamp assumption:** Scorers assume uniform 30 Hz sampling (R1 limitation, documented in `smoothness.py`). Story 3.2 inherits this without changes.

7. **`_validation.py` logger pattern:** `validate_episode()` now accepts a `logging.Logger` object directly. Pass the module-level logger from each caller.

8. **Test isolation:** Use `tq.config.reset_quality_weights()` in test teardown/fixtures when testing weight overrides to prevent state leakage between tests.

### Architecture Requirements

**QM-02 Quality Report (FR6):**
- Per-episode QualityReport with per-dimension scores and overall weighted composite score
- Configurable weights (default: s=0.40, c=0.35, co=0.25)
- Entry point: `tq.quality.score(episodes)`

**GW-SDK-01 (FR17):** Gravity well fires after `tq.quality.score()` completes successfully on non-empty input
- Shows average score — call `_gravity_well(message, feature)` with feature="GW-SDK-01"
- Only fires on success, not on exception, not on empty input
- `_gravity_well` handles quiet suppression internally — no need to check `tq.config.quiet` before calling

**Custom metrics (FR23 — P1 Should Ship):** `tq.quality.register()` in `registry.py`

### File Structure Requirements

- `src/torq/quality/__init__.py` — `score()` entry point, re-exports `register`, `get_metrics`, `reset`
- `src/torq/quality/report.py` — `QualityReport` frozen dataclass
- `src/torq/quality/registry.py` — custom metric registration + weight rescaling
- `src/torq/__init__.py` — add `from torq import quality` and `"quality"` to `__all__`
- Existing from Story 3.1 (DO NOT MODIFY): `smoothness.py`, `consistency.py`, `completeness.py`, `_validation.py`

### Testing Standards

- **Unit tests:** `test_quality_report.py` — mock scorers, test composite logic, None propagation, weight validation, empty list
- **Integration tests:** `test_quality_score.py` — real scorers, full round-trip, performance
- **Test fixtures:** Use `make_quality_episode` from `tests/conftest.py` (fixed seed 42)
- **Performance test:** Assert 100 episodes score in <60s (`@pytest.mark.slow`)
- **State cleanup:** Call `tq.config.reset_quality_weights()` in teardown after any test that modifies global weights

### Code Style & Conventions

- **Type hints:** All public functions. Use `Episode | list[Episode]` union syntax (Python 3.10+).
- **Docstrings:** Google style on all public functions and classes (Args, Returns, Raises).
- **Logger name:** `logging.getLogger(__name__)` in each module → resolves to `torq.quality`, `torq.quality.report`, etc.
- **Naming:** CamelCase for classes (`QualityReport`), snake_case for functions (`score()`)
- **Line length:** 100 chars (ruff format standard)
- **Constants:** Import `DEFAULT_QUALITY_WEIGHTS` from `torq._config` — do NOT redefine

### References

- **Architecture:** `architecture.md` § Quality Scoring Architecture, § Implementation Patterns, § Data Architecture
- **Requirements:** `epics.md` § Story 3.2, FR6 (QM-02), FR17 (GW-SDK-01), FR23 (QM-07)
- **Previous story learnings:** `3-1-core-quality-scoring-dimensions.md`
- **Type definitions:** `src/torq/types.py` — `QualityScore = float | None`
- **Error hierarchy:** `src/torq/errors.py` — `TorqQualityError` (scoring/computation), `TorqConfigError` (config values)
- **Config singleton:** `src/torq/_config.py` — `DEFAULT_QUALITY_WEIGHTS`, `config.quiet`, `config.quality_weights`, `config.reset_quality_weights()`
- **Gravity well:** `src/torq/_gravity_well.py` — `_gravity_well(message: str, feature: str) -> None`
- **Public namespace:** `src/torq/__init__.py` — must add `quality` import here

### Known Risks / Watch Out For

- Weight validation should happen before any episodes are modified (fail-fast)
- Progress bar must respect `tq.config.quiet` to avoid breaking `tq ingest --json` pipelines
- `frozen=True` + computed `overall` in `__post_init__` requires `object.__setattr__` — forgetting this causes `FrozenInstanceError` at class instantiation
- None propagation: if ANY component is None, overall must be None — do not attempt partial composite
- Registry state must be thread-safe for concurrent scoring workflows (R2 scope — document this limitation in `registry.py` module docstring, do not implement locking in R1)
- `_gravity_well` takes TWO args `(message, feature)` — passing one arg raises `TypeError` at runtime
- `DEFAULT_QUALITY_WEIGHTS` lives in `_config.py`, not `types.py` — import from the right place

## Dev Agent Record

### Agent Model Used
Claude Haiku 4.5

### Completion Notes List

- [x] QualityReport dataclass created with immutable frozen fields (`frozen=True`) and None-propagating overall
- [x] `object.__setattr__` pattern used for `overall` in `__post_init__` (frozen dataclass requirement)
- [x] `weights` implemented as `InitVar[dict[str, float] | None]` — passed to `__post_init__` but not stored as a field; defaults to `DEFAULT_QUALITY_WEIGHTS`
- [x] Entry point `tq.quality.score()` handles single Episode and list[Episode] with same-object-identity return
- [x] Empty list returns immediately with no progress bar and no gravity well
- [x] Weight validation (sum to 1.0 ± 0.001) implemented before any scoring, raises `TorqQualityError`
- [x] tqdm imported at module level (enables patching in tests); `disable=True` for quiet mode and ≤1 episodes
- [x] Gravity well fires on success with average score — `_gravity_well(f"Average quality: {avg:.2f}", "GW-SDK-01")`
- [x] Custom metrics call `object.__setattr__` on the frozen QualityReport to add their contribution to `overall` after built-in scoring
- [x] Registry supports `register`, `get_metrics`, `reset`, proportional weight rescaling, UserWarning on re-registration
- [x] `src/torq/__init__.py` updated: `from torq import quality` added, `"quality"` in `__all__`
- [x] 37 unit tests created and passing (`tests/unit/test_quality_report.py`)
- [x] 7 integration tests created and passing (`tests/integration/test_quality_score.py`)
- [x] 100-episode scoring under 0.02s (well within 60s limit, AC#5)
- [x] All 228 tests pass — no regressions
- [x] R1 note: QualityReport not persisted by storage layer (R1 limitation); round-trip test validates re-scoring after load produces same result

### File List

- [x] `src/torq/quality/report.py` — QualityReport frozen dataclass (created)
- [x] `src/torq/quality/__init__.py` — score() entry point + re-exports (updated)
- [x] `src/torq/quality/registry.py` — custom metric registration (created)
- [x] `src/torq/__init__.py` — add quality namespace (updated)
- [x] `tests/unit/test_quality_report.py` — unit tests (created)
- [x] `tests/integration/test_quality_score.py` — integration tests (created)
- [x] `src/torq/types.py` — verified QualityScore = float | None exists (no changes)
