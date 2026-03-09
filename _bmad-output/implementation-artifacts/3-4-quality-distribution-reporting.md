# Story 3.4: Quality Distribution Reporting

Status: done

## Story

As a robotics researcher,
I want to see the quality distribution across my dataset with outlier detection,
so that I can understand data quality at a glance before composing a training dataset.

## Acceptance Criteria

1. **Given** a list of scored episodes, **When** `tq.quality.report(episodes)` is called, **Then** a text summary is printed to stdout showing: min, max, mean, median, and std of overall scores, **And** outliers (>2σ from mean) are listed by episode ID.

2. **Given** a dataset where all episodes score identically, **When** `tq.quality.report(episodes)` is called, **Then** the report notes zero variance without raising an exception.

3. **Given** fewer than 3 scored episodes, **When** `tq.quality.report(episodes)` is called, **Then** available statistics are shown **And** a `logger.warning()` notes that distribution analysis is unreliable below 3 samples.

## Tasks / Subtasks

- [x] Task 1: Implement `report()` distribution function in `src/torq/quality/__init__.py` (AC: #1, #2, #3)
  - [x] Add `report(episodes: list[Episode]) -> None` function
  - [x] Validate input is a list; raise `TorqQualityError` with helpful message if not
  - [x] Filter to only scored episodes (those where `episode.quality is not None` and `episode.quality.overall is not None`)
  - [x] If 0 scored episodes: print a warning and return early (no exception)
  - [x] If < 3 scored episodes: compute available stats AND emit `logger.warning()` about unreliable distribution
  - [x] Compute: min, max, mean, median, std of overall scores using numpy
  - [x] Detect outliers: episodes where `|score - mean| > 2 * std`; handle zero-std case (no outliers)
  - [x] Print formatted text summary to stdout (use `print()`, not `logger`)
  - [x] Add `report` to `__all__` in `src/torq/quality/__init__.py`

- [x] Task 2: Write unit tests `tests/unit/test_quality_distribution.py` (AC: #1, #2, #3)
  - [x] Test basic distribution: list of 10 episodes with varying scores — verify printed output contains "min", "max", "mean", "median", "std"
  - [x] Test outlier detection: inject an episode with score 0.01 in a set of scores around 0.9 — verify its ID appears in output
  - [x] Test zero-variance: all episodes have identical scores — verify no exception and report notes zero variance / no outliers
  - [x] Test < 3 episodes (1 episode): shows stats and logs warning about unreliability
  - [x] Test < 3 episodes (2 episodes): shows stats and logs warning
  - [x] Test 0 scored episodes: prints warning and returns without exception
  - [x] Test unscored episodes excluded: episodes with `quality=None` or `quality.overall=None` are silently skipped for stats
  - [x] Test return value is `None` (function prints, doesn't return data)
  - [x] Test output printed to stdout (capture with `capsys` pytest fixture)

### Review Follow-ups (AI)

- [ ] [AI-Review][MEDIUM] `report()` adds ~90 lines to `__init__.py` (now ~350 lines) — consider extracting to `quality/distribution.py` in a future story to prevent monolith growth [`src/torq/quality/__init__.py:260-348`]
- [x] [AI-Review][MEDIUM] No test verifies outlier sigma value formatting (Unicode minus `\u2212` and sigma `\u03c3`) — only checks episode ID appears, not the deviation string [`tests/unit/test_quality_distribution.py:94-101`]
- [x] [AI-Review][MEDIUM] Test helpers `_make_scored`/`_make_unscored` duplicate Episode construction instead of reusing `make_quality_episode` conftest fixture — will break if `Episode.__init__` changes [`tests/unit/test_quality_distribution.py:32-63`]
- [x] [AI-Review][MEDIUM] `test_zero_variance_no_exception` uses `or` instead of `and` — passes if `0.000` appears anywhere even without the "zero variance" message [`tests/unit/test_quality_distribution.py:118`]
- [x] [AI-Review][LOW] `test_basic_distribution_output_contains_stats` checks labels but not computed values — with deterministic input, assert exact min/max/mean/median [`tests/unit/test_quality_distribution.py:67-81`]
- [x] [AI-Review][LOW] No validation test for tuple or single Episode input — only tests string; `tq.quality.report(episode)` is a common user mistake [`tests/unit/test_quality_distribution.py:204-207`]

## Dev Notes

### Function Signature

```python
def report(episodes: list[Episode]) -> None:
    """Print a quality distribution summary for a list of scored episodes.

    Computes and prints descriptive statistics (min, max, mean, median, std)
    of the overall quality scores across all scored episodes.  Episodes where
    ``episode.quality`` or ``episode.quality.overall`` is ``None`` are excluded
    from statistics but are not an error.

    Outliers are defined as episodes whose overall score deviates more than 2
    standard deviations from the mean.  When all episodes have identical scores,
    std=0 and no outliers are reported.

    Args:
        episodes: List of episodes, typically already scored via
            ``tq.quality.score(episodes)``.  Episodes without scores are
            silently excluded from statistics.

    Raises:
        TorqQualityError: If ``episodes`` is not a list.
    """
```

### Expected Output Format (AC #1)

```
Quality Distribution Report
===========================
Episodes scored : 42 / 50
Min             : 0.312
Max             : 0.971
Mean            : 0.748
Median          : 0.763
Std Dev         : 0.134

Outliers (> 2σ from mean): 2 episodes
  - episode_003  score=0.312  (−3.25σ)
  - episode_017  score=0.951  (+1.52σ)  ← note: only those exceeding 2σ listed
```

When zero variance (AC #2):
```
Quality Distribution Report
===========================
Episodes scored : 5 / 5
Min             : 0.800
Max             : 0.800
Mean            : 0.800
Median          : 0.800
Std Dev         : 0.000

Outliers: none (zero variance — all episodes scored identically)
```

### Implementation Notes

- **Use `numpy` for stats** — it's already a core dependency. Use `np.min`, `np.max`, `np.mean`, `np.median`, `np.std` (ddof=0 for population std).
- **Zero-std edge case** — when `std == 0.0`, the outlier threshold `mean ± 2*std` degenerates to `mean ± 0`, so no episodes are outliers. Handle explicitly without division.
- **Unscored episodes** — silently excluded from the scored pool but included in the total count display (`42 / 50`).
- **Output via `print()`** — this is a user-facing display function, NOT a logging call. Use `print()` for the table. Use `logger.warning()` only for the "< 3 samples" reliability warning.
- **Return None** — this function is purely for display; it returns nothing.
- **Do NOT call `tq.quality.score()` internally** — episodes must already be scored by the caller.
- **No gravity well** — `_gravity_well()` fires after `score()` only (GW-SDK-01), not after `report()`.

### Architecture Constraint — Naming Collision

`tq.quality.report` (distribution function) vs `QualityReport` (class in `report.py`):
- The CLASS `QualityReport` is imported from `torq.quality.report` and re-exported as `QualityReport`
- The FUNCTION `report()` is a new distribution summary function added directly in `__init__.py`
- These are distinct names (`report` vs `QualityReport`) — no collision
- Architecture doc (QM-04): "Distribution (R1) → `quality/__init__.py` — text summary only in R1"

### File Structure Requirements

- `src/torq/quality/__init__.py` — ADD `report()` function AND add `"report"` to `__all__`
- `tests/unit/test_quality_distribution.py` — NEW file

### Where NOT to Add Code

- Do NOT add a new `quality/distribution.py` file — architecture says `quality/__init__.py` for R1
- Do NOT modify `quality/report.py` — that file defines the `QualityReport` dataclass, leave it intact
- Do NOT modify `quality/filters.py` — separate concern

### Import Block (what's already in `__init__.py`)

```python
# Already present — do NOT duplicate:
import logging
import numpy as np  # ← numpy is available, use it for stats
from torq.quality.report import QualityReport
from torq.quality.filters import filter as filter_episodes
filter = filter_episodes
```

Note: `numpy` is not currently explicitly imported in `quality/__init__.py`. You must add `import numpy as np` at the top of the file.

### Test Pattern from Story 3.3 (follow same conventions)

Story 3.3 tests used:
- `make_quality_episode` fixture pattern (create episodes with `episode.quality` pre-populated)
- `pytest` with `capsys` for stdout capture
- `caplog` for log assertion

Create a helper `make_scored_episode(score: float, episode_id: str = "ep_XXX")` in the test file that constructs an `Episode` with `episode.quality = QualityReport(smoothness=score, consistency=score, completeness=score, weights=DEFAULT_QUALITY_WEIGHTS)`. Or reuse any existing conftest fixtures.

### References

- [Source: epics.md#Epic 3 > Story 3.4] — Story ACs and user story statement
- [Source: architecture.md#FR Category → File Mapping] — QM-04: `quality/__init__.py`, text summary only in R1
- [Source: architecture.md#Data Flow Quality Section] — Shows `report()` is called after `filter()` in user workflow
- [Source: implementation-artifacts/3-3-quality-gates.md] — Established patterns: pure functions, logging conventions, filter returns new list

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6

### Debug Log References

None.

### Completion Notes List

- Added `import numpy as np` to `quality/__init__.py` (was not previously imported)
- Implemented `report()` directly in `__init__.py` per architecture constraint (not a separate file)
- Zero-variance handled explicitly: when std==0.0, skip outlier threshold math and print "zero variance" note
- Outlier sigma values printed with sign prefix and Unicode σ symbol matching spec format
- 14 unit tests covering all ACs plus edge cases (empty list, None overall, stdout vs stderr, validation)
- All 274 tests passing, no regressions

### File List

- `src/torq/quality/__init__.py` — MODIFIED (added `import numpy as np`, `report()` function, `"report"` to `__all__`)
- `tests/unit/test_quality_distribution.py` — NEW

### Change Log

- 2026-03-09: Story 3.4 implemented — quality distribution reporting with outlier detection
