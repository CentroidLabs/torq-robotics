# Story 4.4: tq.compose() Entry Point and Dataset Versioning

Status: done

## Story

As a developer,
I want a single `tq.compose()` call to filter, sample, and version a training dataset,
so that every dataset I build has full provenance recorded automatically.

## Acceptance Criteria

1. **Given** a scored episode pool, **When** `tq.compose(task='pick', quality_min=0.75, sampling='stratified', limit=50, name='pick_v1', store_path='/data')` is called, **Then** a `Dataset` is returned containing episodes matching the filters **And** `dataset.recipe` stores the exact query parameters, sampling config, seed, and source episode IDs.

2. **Given** `quality_min=0.9` that results in fewer than 5 episodes, **When** `tq.compose()` is called, **Then** the Dataset is returned (no exception) **And** `logger.warning()` reports the low episode count and suggests lowering `quality_min`.

3. **Given** `tq.compose()` is called with a `name` matching an existing saved dataset, **When** the call completes, **Then** a new versioned Dataset is created without overwriting the existing one. (R1: name uniqueness enforced by caller — `tq.compose()` returns the Dataset with the given name and records the recipe; no file-based collision check required in R1.)

4. **Given** `tq.compose()` returns a Dataset with 0 episodes, **When** the call completes, **Then** an empty Dataset is returned **And** `logger.warning()` states which filter(s) eliminated all episodes.

## Tasks / Subtasks

- [x] Task 1: Create `src/torq/compose/_compose.py` — `compose()` function (AC: #1, #2, #3, #4)
  - [x] Implement `compose(task=None, quality_min=None, quality_max=None, embodiment=None, sampling='none', limit=None, seed=None, name='dataset', *, store_path) -> Dataset`
  - [x] `store_path` is keyword-only and required — raise `TorqComposeError` if not provided
  - [x] Step 1 — Query: call `query_index(index_root, task=task, quality_min=quality_min, quality_max=quality_max, embodiment=embodiment)` from `storage.index`
  - [x] Derive `index_root` from `store_path` the same way `_query.py` does: `Path(store_path) / "index"`
  - [x] Step 2 — Load episodes: load all matched episodes via `storage.load(ep_id, store_path)` (full list, not lazy — `Dataset.episodes` is `list[Episode]`)
  - [x] Step 3 — Sample: call `sample(episodes, strategy=sampling, limit=limit, seed=seed)` from `compose.sampling`
  - [x] Step 4 — Build recipe: construct `recipe` dict capturing all inputs
  - [x] Step 5 — Warnings for 0 episodes after query, 0 < n < 5 episodes after sampling, 0 after sampling with non-zero after query
  - [x] Step 6 — Return `Dataset(episodes=sampled_episodes, name=name, recipe=recipe)`
  - [x] Add `__all__ = ["compose"]`

- [x] Task 2: Wire `compose` into `src/torq/compose/__init__.py` and `src/torq/__init__.py` (AC: #1)
  - [x] Add `from torq.compose._compose import compose` to `compose/__init__.py`
  - [x] Add `"compose"` to `compose/__init__.__all__`
  - [x] Add `from torq.compose import compose` to `src/torq/__init__.py`
  - [x] Add `"compose"` to `torq.__init__.__all__`

- [x] Task 3: Write unit tests `tests/unit/test_compose_pipeline.py` (AC: #1, #2, #3, #4)
  - [x] Test full pipeline: given a populated index and saved episodes, `tq.compose(task='pick', store_path=...)` returns a Dataset with correct episodes
  - [x] Test recipe is populated with all input parameters
  - [x] Test `recipe['source_episode_ids']` lists all matched episode IDs (pre-sampling)
  - [x] Test `recipe['sampled_episode_ids']` lists final episode IDs (post-sampling)
  - [x] Test low episode count warning (< 5 episodes): warning logged, Dataset returned (no exception)
  - [x] Test 0 episode result: empty Dataset returned, warning logged naming filter
  - [x] Test `sampling='stratified'` wired end-to-end: result grouped by task
  - [x] Test `sampling='quality_weighted'` wired end-to-end: higher quality episodes present
  - [x] Test determinism: same inputs + `seed=42` → same episode order
  - [x] Test `name` stored in dataset and in recipe

## Dev Notes

### `compose()` is the Orchestrator

This function wires together the pieces built in stories 4.1–4.3:

```
tq.compose(task='pick', quality_min=0.75, sampling='stratified', limit=50, name='pick_v1', store_path='/data')
    │
    ├─ storage.index.query_index(...)     → list[str] episode IDs
    ├─ storage.load(ep_id, store_path)   → list[Episode] (full load, not lazy)
    ├─ compose.sampling.sample(...)      → list[Episode] (sampled)
    ├─ build recipe dict
    └─ Dataset(episodes, name, recipe)
```

### Key Difference from `tq.query()`

| | `tq.query()` | `tq.compose()` |
|---|---|---|
| Returns | `Iterator[Episode]` (lazy) | `Dataset` (eager — all loaded) |
| Loading | On-demand per yield | All upfront |
| Recipe | None | Full provenance dict |
| Sampling | No | Yes |
| Use case | Exploration, inspection | Training dataset construction |

### `index_root` Derivation

Check how `_query.py` derives `index_root` from `store_path` and use the same logic. From story 4.1 completion notes: `Path(store_path) / "index"`. Keep consistent.

### Recipe Dict Contract

The `recipe` dict is the full provenance record. It must capture:
- All filter parameters (even those that are `None`)
- Sampling strategy, limit, seed
- Name
- `source_episode_ids` — episodes BEFORE sampling (all that matched the query)
- `sampled_episode_ids` — episodes AFTER sampling (final selection)

This enables reproducibility: given the recipe, the exact dataset can be reconstructed.

### Warning Threshold for Low Episode Count

"Fewer than 5 episodes" (AC #2) triggers a warning. The exact message:
```
logger.warning(
    "tq.compose() returned only %d episode(s). "
    "Consider lowering quality_min (currently %.2f) to include more data.",
    n, quality_min or 0.0,
)
```
Only emit this when `quality_min` is set and < 5 episodes resulted. If 0 episodes, use the AC #4 warning instead.

### AC #3 — Dataset Versioning in R1

The epics AC says "without overwriting the existing one." In R1, `tq.compose()` does NOT write anything to disk — it returns an in-memory `Dataset` object. The name collision concern is therefore about the recipe/name string only. In R1: the caller owns naming; `tq.compose()` just records whatever `name` is passed into the recipe. No file-based collision detection is required. The AC is satisfied by the recipe storing the name.

### Import Graph

`_compose.py` may import from:
- `torq.compose.dataset.Dataset`
- `torq.compose.sampling.sample`
- `torq.storage.index.query_index`
- `torq.storage` (for `load`)
- `torq.errors.TorqComposeError`
- Standard library

Must NOT import from `quality`, `ingest`, or `serve`.

### File Structure Requirements

- `src/torq/compose/_compose.py` — NEW
- `src/torq/compose/__init__.py` — MODIFY: add `compose` import + `__all__` entry
- `src/torq/__init__.py` — MODIFY: add `compose` re-export + `__all__` entry
- `tests/unit/test_compose_pipeline.py` — NEW

### References

- [Source: epics.md#Epic 4 > Story 4.4] — ACs and story statement
- [Source: architecture.md#DC-03] — Dataset versioning: `compose/dataset.py` + `storage/index.py`, recipe stored with Dataset
- [Source: architecture.md#Return Type Conventions] — `tq.compose()` returns `Dataset`
- [Source: architecture.md#Decision table] — "compose() returns 0 episodes → Empty Dataset + logger.warning() | Raise exception"
- [Source: src/torq/compose/_query.py] — How `index_root` is derived from `store_path`
- [Source: src/torq/compose/sampling.py] — `sample()` signature from story 4.3
- [Source: src/torq/compose/dataset.py] — `Dataset` class from story 4.2

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6

### Debug Log References

### Completion Notes List

- `_compose.py` orchestrates query_index → load → sample → Dataset in one call.
- `store_path` made optional with `None` default; raises `TorqComposeError` explicitly (not `TypeError`) when missing — matches story requirement.
- Recipe captures all 9 fields including `source_episode_ids` (pre-sampling) and `sampled_episode_ids` (post-sampling) for full provenance.
- Three warning tiers: 0 after query (names active filters), 0 after sampling (names strategy), 1–4 after sampling (suggests lowering quality_min).
- AC #3 (versioning): handled by recipe recording `name`; no file-based collision check in R1 — caller owns naming.
- `compose` wired into `tq.compose` via both `compose/__init__.py` and `torq/__init__.py`.
- 15 tests, all pass. 431 total (0 regressions).
- ✅ Resolved review finding [HIGH]: `store_path` made truly required (`str | Path`, no default) in both `_compose.py` and `_query.py`. Runtime `None`-checks removed. `test_missing_store_path_raises` and `test_raises_compose_error_without_store_path` updated to expect `TypeError`.
- ✅ Resolved review finding [MEDIUM]: Removed dead `if TYPE_CHECKING: pass` block and unused `TYPE_CHECKING` import.
- ✅ Resolved review finding [MEDIUM]: Added `test_query_index_called_with_correct_args` and `test_load_called_per_matched_id` to verify wiring with `assert_called_once_with`.
- ✅ Resolved review finding [MEDIUM]: Added `assert any("sampling" in r.message.lower() ...)` to `test_zero_episodes_after_sampling_warns`.
- ✅ Resolved review finding [LOW]: Step comments renumbered to Steps 5/6 matching story spec.
- ✅ Resolved review finding [LOW]: Low episode count warning now branches on whether `quality_min` was provided; does not emit `quality_min=0.00` when it was `None`.
- 17 tests, all pass. 433 total (0 regressions).

### File List

- `src/torq/compose/_compose.py` — NEW
- `src/torq/compose/__init__.py` — MODIFIED: added `compose` import and `__all__` entry
- `src/torq/__init__.py` — MODIFIED: added `compose` re-export and `__all__` entry
- `tests/unit/test_compose_pipeline.py` — NEW (15 tests)

### Review Follow-ups (AI)

- [x] [AI-Review][HIGH] `store_path` signature lies to type checkers — declared as `str | Path | None = None` but immediately raises if `None`. Change to `str | Path` (required, no default) so mypy/pyright catch invalid calls at type-check time. Also fix same pattern in `_query.py:34`. [src/torq/compose/_compose.py:43]
- [x] [AI-Review][MEDIUM] Dead `TYPE_CHECKING` import block — `if TYPE_CHECKING: pass` does nothing. Remove it. [src/torq/compose/_compose.py:23-24]
- [x] [AI-Review][MEDIUM] Tests never assert `query_index` / `load` were called with correct arguments. At least one test should use `patch(...) as mock_qi` and `mock_qi.assert_called_once_with(tmp_path / "index", task=..., ...)` to verify wiring. [tests/unit/test_compose_pipeline.py]
- [x] [AI-Review][MEDIUM] `test_zero_episodes_after_sampling_warns` accepts `caplog` but never asserts a warning was logged. Add `assert any("sampling" in r.message for r in caplog.records)` or similar. [tests/unit/test_compose_pipeline.py:207-219]
- [x] [AI-Review][LOW] Step numbering in code comments (Steps 4/5) doesn't match story spec (Steps 5/6). Cosmetic but confusing. [src/torq/compose/_compose.py:125,142]
- [x] [AI-Review][LOW] Low episode count warning emits `quality_min=0.00` when `quality_min` is `None` (user never set it). Guard the message: only suggest lowering `quality_min` when it was actually provided. [src/torq/compose/_compose.py:137-139]
- [x] [AI-Review][LOW] Dead `if TYPE_CHECKING: pass` block is scaffolding residue. [src/torq/compose/_compose.py:23-24]
