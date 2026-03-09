# Story 4.1: Structured Query API

Status: done

## Story

As a developer,
I want to query my episode pool by task, quality, embodiment, and date,
so that I can find relevant episodes in under 1 second even at 100K+ scale.

## Acceptance Criteria

1. **Given** a dataset of 100,000+ saved episodes with a populated sharded JSON index, **When** `tq.query(task='pick', quality_min=0.8, embodiment='aloha2')` is called, **Then** a lazy iterator of matching Episodes is returned in under 1 second **And** query execution uses set intersection on inverted index shards, not a full Parquet scan.

2. **Given** compound filters with AND/OR logic, **When** `tq.query(task=['pick', 'place'], quality_min=0.7, quality_max=0.95)` is called, **Then** all matching episodes across both tasks within the quality range are returned.

3. **Given** `tq.query()` with no filters, **When** the iterator is consumed, **Then** all episodes in the index are returned in episode ID order.

4. **Given** a query that matches zero episodes, **When** the iterator is consumed, **Then** an empty iterator is returned (no exception) **And** `logger.warning()` notes the query parameters that matched nothing.

## Tasks / Subtasks

- [x] Task 1: Fix bug in `src/torq/storage/index.py` — `composite` → `overall` (AC: #1 quality filter prerequisite)
  - [x] Line 163: change `getattr(episode.quality, "composite", None)` → `getattr(episode.quality, "overall", None)`
  - [x] Verify existing storage index tests still pass after fix

- [x] Task 2: Add `query_index()` to `src/torq/storage/index.py` — index-based episode ID resolution (AC: #1, #2, #3, #4)
  - [x] Implement `query_index(index_root, *, task=None, quality_min=None, quality_max=None, embodiment=None) -> list[str]`
  - [x] Load `by_task.json` and resolve task filter: if `task` is a str → single lookup; if list → union of lookups
  - [x] Load `by_embodiment.json` and resolve embodiment filter (same pattern)
  - [x] Load `quality.json` and resolve quality range: binary search on sorted score list
  - [x] Combine results via set intersection: task_ids ∩ embodiment_ids ∩ quality_ids
  - [x] If a filter dimension is not specified, treat as "all episode IDs" (no constraint from that shard)
  - [x] If no filters at all: return all episode IDs from `manifest.json` in episode ID order
  - [x] Return `list[str]` of matching episode IDs (sorted for determinism)
  - [x] Export `query_index` from `storage/index.py` `__all__`

- [x] Task 3: Create `src/torq/compose/filters.py` — filter predicate functions (AC: #1, #2)
  - [x] Implement `normalise(s: str) -> str` — lowercase + strip (mirrors `storage.index._normalise`)
  - [x] Implement `apply_task_filter(episode_ids: list[str], task: str | list[str] | None, by_task: dict) -> set[str]`
  - [x] Implement `apply_embodiment_filter(episode_ids, embodiment: str | list[str] | None, by_embodiment: dict) -> set[str]`
  - [x] Implement `apply_quality_filter(episode_ids, quality_min, quality_max, quality_list: list) -> set[str]`
  - [x] All filters return a `set[str]` of matching episode IDs

- [x] Task 4: Implement `tq.query()` in `src/torq/compose/__init__.py` (AC: #1, #2, #3, #4)
  - [x] Add `query(task=None, quality_min=None, quality_max=None, embodiment=None, *, store_path=None) -> Iterator[Episode]`
  - [x] Resolve `store_path`: raise `TorqComposeError` if not set (no global default in Config)
  - [x] Derive `index_root` from `store_path` (`Path(store_path) / "index"`)
  - [x] Call `query_index()` from `storage.index` to get matching episode IDs
  - [x] If 0 IDs matched: emit `logger.warning()` with the query parameters, yield nothing, return
  - [x] For each matching episode ID: lazy-load via `torq.storage.load(episode_id, store_path)` and yield
  - [x] Export `query` from `compose/__init__` `__all__`
  - [x] Re-export `query` from top-level `torq/__init__.py` as `tq.query`

- [x] Task 5: Write unit tests `tests/unit/test_compose_filters.py` (AC: #1, #2, #3, #4)
  - [x] Test task filter — single task string: returns only matching episode IDs
  - [x] Test task filter — list of tasks (OR logic): returns union across all specified tasks
  - [x] Test task filter — None (no filter): returns all episode IDs
  - [x] Test task filter — task not in index: returns empty set
  - [x] Test embodiment filter (same patterns as task)
  - [x] Test quality filter — min only: returns episodes with score >= min
  - [x] Test quality filter — max only: returns episodes with score <= max
  - [x] Test quality filter — range: returns episodes within [min, max]
  - [x] Test quality filter — episodes with None quality score: excluded when quality filter is active
  - [x] Test set intersection: task ∩ embodiment ∩ quality all applied simultaneously

- [x] Task 6: Write unit tests `tests/unit/test_compose_query.py` (AC: #1, #2, #3, #4)
  - [x] Test `query_index()` with populated index: task + quality_min → correct episode IDs returned
  - [x] Test `query_index()` with no filters: returns all episode IDs in manifest order
  - [x] Test `query_index()` with compound task list: union of tasks intersected with other filters
  - [x] Test `query_index()` with zero results: returns empty list (no exception)
  - [x] Test `tq.query()` returns lazy iterator: consuming it yields Episode objects
  - [x] Test `tq.query()` with no matches: empty iterator + warning logged
  - [x] Test `tq.query()` episode ID order: deterministic (sorted)

## Review Follow-ups (AI)

- [x] [AI-Review][HIGH] `compose/filters.py` is dead code — never imported by any production path. `query_index()` in `storage/index.py` duplicates all filtering inline. Either wire `filters.py` into the query path or delete it and its 27 tests. [src/torq/compose/filters.py]
  - Resolution: Refactored `_query.py` to load index shards directly and call `apply_task_filter`, `apply_embodiment_filter`, `apply_quality_filter` from `filters.py`. `filters.py` is now in the production code path. `query_index()` retains its own inline logic as a standalone storage-layer API.
- [x] [AI-Review][HIGH] `test_errors.py` partially updated — `TorqComposeError` missing from `test_all_exceptions_importable` (line 10-18), `test_all_subclasses_inherit_torq_error` (line 38-46), and `test_isinstance_check_for_each_subclass` (line 60-71). [tests/unit/test_errors.py]
  - Resolution: Added `TorqComposeError` to all three test methods.
- [x] [AI-Review][HIGH] Quality filter in `query_index()` uses O(n) linear scan instead of binary search (`bisect`) on the pre-sorted `quality.json`, contradicting dev notes and AC #1 performance intent. [src/torq/storage/index.py:276-286]
  - Resolution: Replaced linear scan with `bisect.bisect_left`/`bisect_right` in both `query_index()` and `apply_quality_filter()`.
- [x] [AI-Review][MEDIUM] Unused `import numpy as np` — will fail `ruff check` F401. [tests/unit/test_compose_query.py:20]
  - Resolution: Removed unused import.
- [x] [AI-Review][MEDIUM] No performance test validates AC #1's "under 1 second at 100K+" claim.
  - Resolution: Added `TestQueryPerformance::test_query_index_under_one_second_at_100k_episodes` — generates a 100K-episode synthetic index and asserts query completes in < 1s. Passes at ~0.2s.
- [x] [AI-Review][LOW] Normalisation logic duplicated between `filters.normalise()` and `index._normalise()` — moot if H1 resolved by removing `filters.py`. [src/torq/compose/filters.py:40, src/torq/storage/index.py:51]
  - Resolution: `filters.py` now imports `_normalise` from `storage.index` and re-exports it as `normalise`. Zero duplication.

## Dev Notes

### Pre-existing Bug to Fix First (Task 1)

**`src/torq/storage/index.py` line 163:**
```python
# CURRENT (BUG — always stores None for quality):
quality_score = getattr(episode.quality, "composite", None) if episode.quality else None

# CORRECT (QualityReport uses .overall, not .composite):
quality_score = getattr(episode.quality, "overall", None) if episode.quality else None
```

`QualityReport` (in `quality/report.py`) exposes `.overall` as the weighted composite score. The field `.composite` does not exist. The `getattr` with default `None` silently masks this — every saved episode has `None` stored as its quality score in the index, making quality-range queries return wrong results. Fix this BEFORE writing any query code.

### Index Structure (Already on Disk)

```
{store_path}/.torq/index/
├── manifest.json        # {episode_id: {task, embodiment, quality_score, timestamp, ...}}
├── by_task.json         # {"pick": ["ep_001", "ep_005", ...], "place": [...]}
├── by_embodiment.json   # {"aloha2": ["ep_001", ...], "franka": [...]}
└── quality.json         # [[0.62, "ep_003"], [0.71, "ep_001"], ..., [null, "ep_009"]]
                         #   sorted ascending; null scores at end
```

`quality.json` is a **sorted list** of `[score_or_null, episode_id]` pairs (ascending score, nulls last). Binary search for range queries is efficient — no full scan needed.

### `query_index()` Logic

```python
def query_index(
    index_root: Path,
    *,
    task: str | list[str] | None = None,
    quality_min: float | None = None,
    quality_max: float | None = None,
    embodiment: str | list[str] | None = None,
) -> list[str]:
```

Algorithm:
1. Start with "universe" = all episode IDs from `manifest.json`
2. If `task` filter: load `by_task.json`, compute union of task lookups → intersect with universe
3. If `embodiment` filter: load `by_embodiment.json`, compute union of lookups → intersect with current set
4. If `quality_min` or `quality_max`: load `quality.json`, filter by range → intersect with current set
   - Episodes with `None` quality score are excluded when any quality filter is specified
5. Return sorted list of remaining IDs

### Store Path Resolution for `tq.query()`

`tq.query()` needs to know where the index lives. Two options:
1. `store_path` kwarg passed directly by the caller
2. Fallback to `tq.config.store_path` (if it exists — check `_config.py` for this attribute)

Check `src/torq/_config.py` for whether `store_path` is already a config attribute. If not, the caller must always pass it, and the error message should say so.

### Lazy Iterator Contract

`tq.query()` returns an `Iterator[Episode]` — it should be a **generator** that loads each episode on demand via `storage.load()`. This avoids loading all 100K episodes into memory.

```python
def query(...) -> Iterator[Episode]:
    episode_ids = query_index(index_root, task=task, ...)
    if not episode_ids:
        logger.warning("tq.query() returned 0 episodes. Filters: task=%r, ...")
        return
    for ep_id in episode_ids:
        yield storage.load(ep_id, store_path)
```

### Architecture Constraints

- `compose/` may import from `storage/` and `episode` — NO imports from `ingest/`, `quality/`, or `serve/`
- `storage/index.py` must remain the only place that writes index shards
- `query_index()` is READ-ONLY — it never modifies shards
- String normalisation: use `_normalise()` in `index.py` for all task/embodiment lookups (lowercase + strip). `filters.py` must mirror this or import it

### Export Chain

```python
# torq/compose/__init__.py
__all__ = ["query"]

# torq/__init__.py  (check what's already exported and add):
from torq.compose import query
```

Verify `tq.query` works at the top level after wiring.

### String Normalisation Consistency

`storage/index.py` normalises task/embodiment keys when writing via `_normalise()` (lowercase + strip). The query path must apply the same normalisation when looking up keys, otherwise `task='Pick'` won't match index key `'pick'`. Either:
- Import `_normalise` from `storage.index` (leading underscore = internal, but same package)
- Duplicate the logic in `filters.py` (simple: `s.strip().lower()`)

Document whichever approach is chosen.

### File Structure Requirements

- `src/torq/storage/index.py` — MODIFY: fix `composite` bug, add `query_index()`
- `src/torq/compose/filters.py` — NEW
- `src/torq/compose/__init__.py` — ADD `query()` function + exports
- `src/torq/__init__.py` — ADD `tq.query` re-export
- `tests/unit/test_compose_filters.py` — NEW
- `tests/unit/test_compose_query.py` — NEW

### References

- [Source: architecture.md#JSON Index Architecture] — Sharded index structure, set intersection query execution
- [Source: architecture.md#FR Category > DC-01, QE-02] — compose/filters.py + storage/index.py for query
- [Source: architecture.md#Data Flow] — `tq.query()` returns `Iterator[Episode]`; `tq.compose()` returns `Dataset`
- [Source: architecture.md#Import graph] — compose may import storage/episode/errors/types only
- [Source: src/torq/storage/index.py:163] — Bug: `composite` should be `overall`
- [Source: src/torq/quality/report.py:68] — `QualityReport.overall` is the composite score field
- [Source: epics.md#Epic 4 > Story 4.1] — ACs and story statement

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6

### Debug Log References

None — implementation proceeded without errors after fixing `composite` → `overall` bug.

### Completion Notes List

- Fixed pre-existing bug: `storage/index.py` line 163 used `episode.quality.composite` (doesn't exist) instead of `episode.quality.overall`. This silently stored `None` for all quality scores, making quality-range queries broken.
- Implemented `query_index()` in `storage/index.py` using set intersection on shards. Universe derived from `quality.json` (all episodes are recorded there). Sorted list returned for determinism.
- Created `compose/filters.py` with standalone predicate functions for task, embodiment, and quality filtering. String normalisation matches `storage.index._normalise` (lowercase + strip + remove separators).
- Implemented `tq.query()` as a generator in `compose/_query.py`. `load` imported at module level for testability. `store_path` must be passed explicitly — no Config default exists.
- Added `TorqComposeError` to `errors.py` and updated `test_errors.py` expected `__all__` set accordingly.
- 362 tests pass (44 new: 27 filters + 17 query).

### File List

- `src/torq/storage/index.py` — MODIFIED: fixed `composite` → `overall` bug; added `query_index()`; updated `__all__`
- `src/torq/errors.py` — MODIFIED: added `TorqComposeError`; updated `__all__`
- `src/torq/compose/filters.py` — NEW
- `src/torq/compose/_query.py` — NEW
- `src/torq/compose/__init__.py` — MODIFIED: now exports `query` via `_query.py`
- `src/torq/__init__.py` — MODIFIED: added `query` import and `__all__` entry
- `tests/unit/test_compose_filters.py` — NEW (27 tests)
- `tests/unit/test_compose_query.py` — NEW (17 tests)
- `tests/unit/test_errors.py` — MODIFIED: updated expected `__all__` set to include `TorqComposeError`
