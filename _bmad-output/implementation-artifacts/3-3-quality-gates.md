# Story 3.3: Quality Gates

Status: review

## Story

As a robotics researcher,
I want to define quality thresholds that automatically reject low-quality episodes,
so that composed datasets are protected from contamination by bad demonstrations.

## Acceptance Criteria

1. **Given** a list of scored episodes and a quality threshold,
   **When** `tq.quality.filter(episodes, min_score=0.75)` is called,
   **Then** only episodes with `quality.overall >= 0.75` are returned,
   **And** a log message reports how many episodes were filtered and the threshold used.

2. **Given** `min_score=0.75` that filters out all episodes,
   **When** `tq.quality.filter(episodes, min_score=0.75)` is called,
   **Then** an empty list is returned (no exception),
   **And** `logger.warning()` states that 0 episodes passed the threshold and suggests lowering it.

3. **Given** an episode where `episode.quality` is `None` (unscored or <10 timesteps),
   **When** quality filtering is applied,
   **Then** the episode is excluded from the filtered result,
   **And** a warning is logged naming the episode ID.

## Tasks / Subtasks

- [x] Task 1: Create `src/torq/quality/filters.py` — `filter()` function (AC: #1, #2, #3)
  - [x] Implement `filter(episodes: list[Episode], *, min_score: float) -> list[Episode]`
  - [x] Validate `min_score` is in [0.0, 1.0], raise `TorqQualityError` if not
  - [x] Validate `episodes` is a list, raise `TorqQualityError` if not
  - [x] Iterate episodes: exclude those with `episode.quality is None` (log warning with episode ID)
  - [x] Exclude episodes with `episode.quality.overall is None` (log warning with episode ID)
  - [x] Exclude episodes with `episode.quality.overall < min_score`
  - [x] Log info message: "Quality gate: {passed}/{total} episodes passed (min_score={min_score})"
  - [x] If 0 episodes pass, log warning suggesting lowering threshold
  - [x] Return new list (do NOT mutate input list)

- [x] Task 2: Update `src/torq/quality/__init__.py` — export `filter` (AC: #1)
  - [x] Add import: `from torq.quality.filters import filter as filter_episodes`
  - [x] Re-export as `filter` in `__all__` — note: shadows builtin `filter`, which is intentional for API ergonomics (`tq.quality.filter(...)`)
  - [x] Add convenience alias in module: `filter = filter_episodes`

- [x] Task 3: Write unit tests `tests/unit/test_quality_filter.py` (AC: #1, #2, #3)
  - [x] Test basic filtering: 5 episodes with varying scores, min_score=0.5 returns only those >= 0.5
  - [x] Test all-filtered: min_score so high that no episodes pass → returns empty list, no exception
  - [x] Test warning logged when all episodes filtered (check logger.warning call)
  - [x] Test unscored episodes excluded: episode.quality is None → excluded with warning
  - [x] Test None overall excluded: episode.quality.overall is None → excluded with warning
  - [x] Test min_score=0.0 passes all scored episodes
  - [x] Test min_score=1.0 only passes perfect scores
  - [x] Test min_score validation: values < 0 or > 1 raise TorqQualityError
  - [x] Test empty input list returns empty list immediately
  - [x] Test return is new list (not same object identity as input)
  - [x] Test info log message format includes count and threshold

### Review Follow-ups (AI)

- [x] [AI-Review][HIGH] Warning message for `quality.overall is None` is identical to `quality is None` — differentiate so users can distinguish "never scored" from "too short to score" [`src/torq/quality/filters.py:96-101`]
- [x] [AI-Review][MEDIUM] No test for `min_score=True` or `min_score=False` — implementation rejects bools (line 65) but this path has zero test coverage [`tests/unit/test_quality_filter.py`]
- [x] [AI-Review][MEDIUM] No test for `min_score` as `int` (e.g. `min_score=1`) — implementation accepts and coerces ints but path is untested [`tests/unit/test_quality_filter.py`]
- [x] [AI-Review][MEDIUM] No mixed-type integration test combining scored-above, scored-below, unscored, and None-overall episodes in a single `filter()` call — verify count logging is correct for realistic usage [`tests/unit/test_quality_filter.py`]
- [x] [AI-Review][LOW] `test_min_score_below_zero_raises` and `test_min_score_above_one_raises` accept `make_quality_episode` fixture but never use it — remove unused parameter [`tests/unit/test_quality_filter.py:135-144`]

## Dev Notes

### Key Implementation Contract

**Filter function signature:**
```python
def filter(
    episodes: list[Episode],
    *,
    min_score: float,
) -> list[Episode]:
```

**Logging patterns (use module-level logger):**
```python
logger = logging.getLogger(__name__)

# Info — always logged on successful filter
logger.info("Quality gate: %d/%d episodes passed (min_score=%.2f)", passed, total, min_score)

# Warning — unscored episodes
logger.warning("Episode '%s' has no quality score — excluded from filtered results", ep.episode_id)

# Warning — all episodes filtered
logger.warning(
    "0/%d episodes passed quality gate (min_score=%.2f). "
    "Consider lowering the threshold.",
    total, min_score,
)
```

### Architecture Patterns and Constraints

- **Pure function:** `filter()` returns a NEW list. Does not mutate input. Does not modify episodes.
- **No scoring:** `filter()` does NOT call `tq.quality.score()`. Episodes must already be scored. If `episode.quality` is None, the episode is simply excluded with a warning.
- **Import guard:** `filters.py` imports only from `torq.episode`, `torq.errors`. No circular imports.
- **Naming collision:** The function name `filter` shadows Python's builtin. This is intentional — it's accessed as `tq.quality.filter(...)`, never as bare `filter(...)`. The import in `__init__.py` should use `from torq.quality.filters import filter as filter_episodes` then alias.
- **No compose dependency:** Despite architecture mapping QM-03 to `compose/filters.py`, this story implements the quality-level filtering API. The compose module (Epic 4) will call `tq.quality.filter()` internally when `quality_min` is passed to `tq.compose()`.
- **Threshold semantics:** `>=` comparison (inclusive). An episode scoring exactly `min_score` passes the gate.
- **Return order:** Preserve input order of episodes in the returned list.

### Previous Story Intelligence (Story 3.2)

**Established patterns (use these, do not reinvent):**

1. **`make_quality_episode` fixture:** Located in `tests/conftest.py`. Factory creates Episodes with configurable actions, timesteps, metadata, and duration. Use this to create test episodes, then call `tq.quality.score()` to attach QualityReport before testing filter.

2. **QualityReport access:** `episode.quality` is either `None` (unscored) or a `QualityReport` instance. Access `.overall` for composite score. `.overall` can also be `None` if episode was too short (<10 timesteps).

3. **Error type:** Use `TorqQualityError` for all quality-related validation failures. Import from `torq.errors`.

4. **Logger convention:** `logger = logging.getLogger(__name__)` at module level. Resolves to `torq.quality.filters`.

5. **Type hints:** All public functions. Use `list[Episode]` (Python 3.10+ syntax).

6. **Test isolation:** Use `tq.config.reset_quality_weights()` in teardown if tests modify global config (unlikely for filter tests, but noted for consistency).

7. **State cleanup for tests:** After calling `tq.quality.score()` in test setup, episodes have `.quality` attached. No cleanup needed since filter is read-only.

### File Structure Requirements

- `src/torq/quality/filters.py` — NEW file, contains `filter()` function
- `src/torq/quality/__init__.py` — UPDATE to export `filter`
- `tests/unit/test_quality_filter.py` — NEW file, unit tests

### Testing Standards

- **Unit tests only** — no integration tests needed for this simple filtering function
- **Use `make_quality_episode` fixture** from `tests/conftest.py` to create episodes
- **Pre-score episodes** in test setup using `tq.quality.score()` or by manually setting `ep.quality = QualityReport(...)`
- **Test logging** using `caplog` pytest fixture (preferred) or mock logger
- **No `@pytest.mark.slow`** — filter is O(n), no performance concerns

### Code Style & Conventions

- **Formatter:** `ruff format` (line length 100)
- **Linter:** `ruff check` (default rules + I = isort)
- **Docstrings:** Google style on all public functions (Args, Returns, Raises)
- **Line length:** 100 chars
- **Import order:** stdlib → third-party → torq (enforced by ruff isort)

### Project Structure Notes

- Quality module path: `src/torq/quality/`
- Test path: `tests/unit/`
- Config singleton: `src/torq/_config.py`
- Error hierarchy: `src/torq/errors.py`
- Episode dataclass: `src/torq/episode.py`

### References

- [Source: epics.md § Story 3.3 — Quality Gates acceptance criteria]
- [Source: architecture.md § QM-03 — quality gates requirement, maps to compose/filters.py quality_min predicate]
- [Source: architecture.md § User Journey UJ-02 — Jake trusts quality gates for automated pipelines]
- [Source: 3-2-qualityreport-and-tq-quality-score-entry-point.md — previous story patterns and QualityReport API]
- [Source: src/torq/quality/__init__.py — existing score() entry point and __all__ exports]
- [Source: src/torq/quality/report.py — QualityReport frozen dataclass]
- [Source: src/torq/errors.py — TorqQualityError]
- [Source: tests/conftest.py — make_quality_episode fixture]

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6

### Debug Log References

None — implementation went smoothly.

### Completion Notes List

- Implemented pure `filter()` function in `src/torq/quality/filters.py`
- Exports `filter` from `torq.quality.__init__` as `filter = filter_episodes` alias (intentionally shadows builtin, accessible as `tq.quality.filter(...)`)
- 17 unit tests written covering all ACs, boundary cases, logging, and validation
- All 17 new tests pass; full suite 256/256 passing with no regressions
- Info log format: "Quality gate: N/M episodes passed (min_score=X.XX)"
- Warning logged per-episode for unscored/None-overall, and once when 0 pass gate

### File List

- `src/torq/quality/filters.py` — NEW
- `src/torq/quality/__init__.py` — MODIFIED (added filter import and export)
- `tests/unit/test_quality_filter.py` — NEW

### Change Log

- 2026-03-09: Story 3.3 implemented — quality gate filter function with full unit test coverage
