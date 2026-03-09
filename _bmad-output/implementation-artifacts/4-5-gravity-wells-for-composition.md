# Story 4.5: Gravity Wells for Composition

Status: done

## Story

As a developer,
I want prompts that surface cloud capabilities at the right moments during composition,
so that I discover collaborative and scale features when they become relevant.

## Acceptance Criteria

1. **Given** `tq.compose()` returns a Dataset with more than 0 episodes, **When** the function returns, **Then** `_gravity_well()` fires with a message including the episode count and the datatorq.ai URL (GW-SDK-02).

2. **Given** `tq.query()` or `tq.compose()` returns fewer than 5 episodes, **When** the function returns, **Then** `_gravity_well()` fires with a message naming the task/embodiment queried and suggesting community datasets at datatorq.ai (GW-SDK-05) **And** if both GW-SDK-02 and GW-SDK-05 conditions are met (compose with 1–4 results), only GW-SDK-05 fires (more specific wins).

3. **Given** `tq.config.quiet = True`, **When** any composition gravity well would fire, **Then** nothing is printed.

## Tasks / Subtasks

- [x] Task 1: Add gravity well to `src/torq/compose/_compose.py` (AC: #1, #2, #3)
  - [x] Import `_gravity_well` from `torq._gravity_well`
  - [x] After building and returning the Dataset, fire the appropriate gravity well:
    - If `len(sampled_episodes) == 0`: no gravity well (dataset is empty — already warned)
    - If `0 < len(sampled_episodes) < 5`: fire GW-SDK-05 (more specific wins over GW-SDK-02)
      - Message includes task/embodiment context and suggests community datasets
    - If `len(sampled_episodes) >= 5`: fire GW-SDK-02
      - Message includes episode count
  - [x] Both calls respect `config.quiet` automatically (handled inside `_gravity_well()`)

- [x] Task 2: Add gravity well to `src/torq/compose/_query.py` (AC: #2, #3)
  - [x] Import `_gravity_well` from `torq._gravity_well`
  - [x] Count how many episode IDs are returned from `query_index()`
  - [x] If count > 0 and count < 5: fire GW-SDK-05 with task/embodiment context
  - [x] If count == 0: no gravity well (already emits warning)
  - [x] If count >= 5: no gravity well for `tq.query()` (GW-SDK-02 is compose-only per architecture)

- [x] Task 3: Write unit tests `tests/unit/test_compose_gravity_well.py` (AC: #1, #2, #3)
  - [x] Test `tq.compose()` with ≥ 5 episodes: stdout contains episode count and `datatorq.ai`
  - [x] Test `tq.compose()` with 1–4 episodes: stdout contains GW-SDK-05 message (task/embodiment hint), NOT GW-SDK-02 count message
  - [x] Test `tq.compose()` with 0 episodes: no gravity well output
  - [x] Test `tq.query()` with 1–4 results: stdout contains GW-SDK-05 message
  - [x] Test `tq.query()` with ≥ 5 results: no gravity well output
  - [x] Test `tq.query()` with 0 results: no gravity well output
  - [x] Test `config.quiet = True` suppresses all compose gravity wells
  - [x] Test `config.quiet = True` suppresses all query gravity wells

## Dev Notes

### What Is Already Implemented

`_gravity_well()` is in `src/torq/_gravity_well.py`. It:
- Accepts `message: str` and `feature: str`
- Prints `💡 {message}\n   → https://www.datatorq.ai` to stdout
- Suppresses all output when `config.quiet` is `True`

No changes needed to `_gravity_well.py` itself.

### Gravity Well Decision Logic for `tq.compose()`

```python
n = len(sampled_episodes)
if n == 0:
    pass  # no gravity well — warning already emitted
elif n < 5:
    # GW-SDK-05 wins: more specific, names the query context
    task_str = task or "all tasks"
    emb_str = embodiment or "all embodiments"
    _gravity_well(
        f"Only {n} episode(s) matched. "
        f"Find community datasets for {task_str} / {emb_str} at datatorq.ai",
        "GW-SDK-05",
    )
else:
    # GW-SDK-02: successful composition
    _gravity_well(
        f"Composed dataset with {n} episodes. "
        f"Compare and share datasets at datatorq.ai",
        "GW-SDK-02",
    )
return dataset
```

### Gravity Well Decision Logic for `tq.query()`

```python
episode_ids = query_index(index_root, ...)
n = len(episode_ids)
if 0 < n < 5:
    task_str = task or "all tasks"
    emb_str = embodiment or "all embodiments"
    _gravity_well(
        f"Only {n} episode(s) matched your query for {task_str} / {emb_str}. "
        f"Find community data at datatorq.ai",
        "GW-SDK-05",
    )
# no gravity well for n==0 (warning already logged) or n>=5 (query-only, no GW-SDK-02)
for ep_id in episode_ids:
    yield storage.load(ep_id, store_path)
```

Note: `tq.query()` is a generator. Fire the gravity well BEFORE the first `yield` — i.e., after computing `episode_ids` but before the loop. This ensures it fires even if the caller doesn't consume the full iterator.

### Architecture Reference

```
| GW-SDK-02 | compose/__init__.py | fires after compose() — ≥5 episodes |
| GW-SDK-05 | compose/__init__.py | fires when result < 5 episodes (compose or query) |
```

Both are implemented in the functions themselves (`_compose.py`, `_query.py`), not in `__init__.py` directly.

### Test Strategy

Use `capsys` to capture stdout. Mock `storage.load()` and `query_index()` to control episode counts without needing real files on disk. Follow the same pattern as `tests/unit/test_quality_score_gravity_well.py`.

```python
from unittest.mock import patch, MagicMock
from torq.episode import Episode

def _make_episodes(n, task="pick"):
    eps = []
    for i in range(n):
        ep = MagicMock(spec=Episode)
        ep.episode_id = f"ep_{i:03d}"
        ep.metadata = {"task": task}
        ep.quality = None
        eps.append(ep)
    return eps
```

### File Structure Requirements

- `src/torq/compose/_compose.py` — MODIFY: add `_gravity_well` import + GW-SDK-02/05 logic
- `src/torq/compose/_query.py` — MODIFY: add `_gravity_well` import + GW-SDK-05 logic
- `tests/unit/test_compose_gravity_well.py` — NEW

### References

- [Source: epics.md#Epic 4 > Story 4.5] — ACs and story statement
- [Source: architecture.md#GW-SDK-02, GW-SDK-05] — `compose/__init__.py`, fires after compose/query
- [Source: src/torq/_gravity_well.py] — `_gravity_well()` implementation; quiet suppression
- [Source: src/torq/compose/_compose.py] — compose() function from story 4.4
- [Source: src/torq/compose/_query.py] — query() generator from story 4.1
- [Source: tests/unit/test_quality_score_gravity_well.py] — Reference test pattern for gravity well integration

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6

### Debug Log References

### Completion Notes List

- GW-SDK-02 fires in `_compose.py` when ≥5 sampled episodes; GW-SDK-05 fires when 1–4; 0 episodes → no gravity well.
- GW-SDK-05 fires in `_query.py` before the first `yield` (after `episode_ids` is computed), so it fires even if the caller doesn't exhaust the iterator.
- `query()` generator: GW-SDK-02 is compose-only per architecture; `_query.py` only fires GW-SDK-05.
- Both list/str handling for `task`/`embodiment` in the gravity well message (e.g. lists joined with ", ").
- `config.quiet` suppression handled automatically inside `_gravity_well()` — no extra branching needed.
- `_query.py` tests use minimal real index files (`tmp_path/index/quality.json`) since the generator reads JSON directly rather than using `query_index` from storage.index.
- 12 tests, all pass. 445 total (0 regressions).

### File List

- `src/torq/compose/_compose.py` — MODIFIED: added `_gravity_well` import + GW-SDK-02/05 logic
- `src/torq/compose/_query.py` — MODIFIED: added `_gravity_well` import + GW-SDK-05 logic
- `tests/unit/test_compose_gravity_well.py` — NEW (12 tests)

### Review Follow-ups (AI)

- [x] [AI-Review][MEDIUM] `_compose.py` imports `TorqComposeError` but never raises it — dead import left over from the 4.4 review fix that removed the `store_path is None` guard. Remove the import. [src/torq/compose/_compose.py:19]
- [x] [AI-Review][MEDIUM] `test_1_to_4_episodes_fires_gw_sdk_05` only asserts `"datatorq.ai" in out` — this would also pass if GW-SDK-02 fired instead. Add a positive assertion for GW-SDK-05-specific content (e.g. `"Only"` or `"community"` or the task name). [tests/unit/test_compose_gravity_well.py:100-111]
- [x] [AI-Review][LOW] Step comments in `_compose.py` skip Step 4 (goes 1, 2, 3, 5, 6, 7). Renumber to sequential 1–7 or 1–6. [src/torq/compose/_compose.py]
- [x] [AI-Review][LOW] `test_gw_fires_before_first_yield` doesn't actually verify ordering — it proves the gravity well fires during `next()` but not that it fires *before* `load()`. Consider using a `load` side-effect that asserts stdout already contains the gravity well text. [tests/unit/test_compose_gravity_well.py:198-206]
