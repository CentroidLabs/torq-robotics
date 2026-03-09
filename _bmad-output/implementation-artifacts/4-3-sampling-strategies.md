# Story 4.3: Sampling Strategies

Status: done

## Story

As a developer,
I want configurable sampling strategies when composing datasets,
so that I can balance task distribution or oversample high-quality episodes deterministically.

## Acceptance Criteria

1. **Given** episodes across 3 tasks with unequal counts (pick: 50, place: 20, pour: 10), **When** `sampling='stratified'` is applied with `limit=30`, **Then** the resulting dataset has 10 episodes per task (±1 for rounding).

2. **Given** episodes with varying quality scores, **When** `sampling='quality_weighted'` is applied, **Then** higher-quality episodes are sampled more frequently **And** the sampling distribution is proportional to `episode.quality.overall`.

3. **Given** `seed=42` is provided, **When** the same sampling call is made twice with identical inputs, **Then** both calls return episodes in identical order (deterministic).

4. **Given** `sampling='none'`, **When** composition runs, **Then** all filtered episodes are returned without resampling.

## Tasks / Subtasks

- [x] Task 1: Create `src/torq/compose/sampling.py` — all three sampling strategies (AC: #1, #2, #3, #4)
  - [x] Implement `sample(episodes: list[Episode], strategy: str, limit: int | None = None, seed: int | None = None) -> list[Episode]`
  - [x] `strategy='none'`: return `episodes[:limit]` if limit set, else return all (preserve order)
  - [x] `strategy='stratified'`: group by `episode.metadata.get('task', '')`, allocate `limit // n_tasks` per group (±1 for remainder), sample within each group using `random.Random(seed)`
  - [x] `strategy='quality_weighted'`: weight each episode by `episode.quality.overall` (episodes with `None` quality get weight 0 — excluded), sample `limit` episodes without replacement using weighted shuffle; handle `seed` via `random.Random(seed)`
  - [x] Validate `strategy` is one of `{'none', 'stratified', 'quality_weighted'}` — raise `TorqComposeError` with helpful message listing valid options
  - [x] If `limit` is None: return all (for `none`); for stratified/quality_weighted treat as "no limit" — return all episodes after grouping/weighting
  - [x] If `limit > len(episodes)`: treat as `limit = len(episodes)` (never raise, just cap)
  - [x] Add `__all__ = ["sample"]`

- [x] Task 2: Write unit tests `tests/unit/test_compose_sampling.py` (AC: #1, #2, #3, #4)
  - [x] Test `strategy='none'` with limit: returns first N episodes, preserves order
  - [x] Test `strategy='none'` without limit: returns all episodes
  - [x] Test `strategy='stratified'` with 3 tasks, 30 limit: each task gets ~10 episodes (±1)
  - [x] Test `strategy='stratified'` sums to exactly `limit` (no rounding shortfall)
  - [x] Test `strategy='stratified'` without limit: all episodes returned, grouped
  - [x] Test `strategy='quality_weighted'` samples more high-quality episodes than low-quality (statistical, run with large N)
  - [x] Test `strategy='quality_weighted'` excludes episodes with `quality.overall = None`
  - [x] Test determinism: two calls with `seed=42` and same inputs → identical output list
  - [x] Test different seeds produce different orderings (probabilistic — use distinct seeds)
  - [x] Test `limit > len(episodes)`: returns all episodes, no exception
  - [x] Test invalid strategy raises `TorqComposeError`
  - [x] Test `strategy='stratified'` with unequal group sizes (e.g. pick:50, place:20, pour:10) matches AC #1

## Review Follow-ups (AI)

- [x] [AI-Review][HIGH] `_stratified` can return fewer than `limit` when a group has fewer members than its quota. E.g. pick:50, place:20, pour:3, limit=30 → quotas 10/10/10 but pour only has 3 → result=23. Unallocated slots should be redistributed to larger groups. [src/torq/compose/sampling.py:117]
- [x] [AI-Review][MEDIUM] `_quality_weighted` excludes episodes with `quality.overall=0.0` since `0.0 * rng.random() == 0.0` always ranks last. Valid scored episodes with zero quality can never be selected. Consider adding a small epsilon or using `(weight + eps) * rng.random()`. [src/torq/compose/sampling.py:155]
- [x] [AI-Review][LOW] No test for `strategy='stratified'` with a single task group — edge case where stratified should behave like `_none` with limit. [tests/unit/test_compose_sampling.py]
- [x] [AI-Review][LOW] `_none` copies the full list when no limit is set (`list(episodes)`) — unnecessary allocation for large datasets. [src/torq/compose/sampling.py:84]

## Dev Notes

### Stratified Sampling Algorithm

Group episodes by `episode.metadata.get('task', '')` then distribute `limit` evenly:

```python
import math
from collections import defaultdict
import random as _random

def _stratified(episodes, limit, seed):
    rng = _random.Random(seed)
    groups = defaultdict(list)
    for ep in episodes:
        task = ep.metadata.get("task", "") if ep.metadata else ""
        groups[task].append(ep)

    n_groups = len(groups)
    if n_groups == 0:
        return []

    base = (limit or len(episodes)) // n_groups
    remainder = (limit or len(episodes)) % n_groups

    result = []
    for i, (task, group_eps) in enumerate(sorted(groups.items())):
        quota = base + (1 if i < remainder else 0)
        rng.shuffle(group_eps)
        result.extend(group_eps[:quota])
    return result
```

Key points:
- Sort group keys for determinism (same input order regardless of dict insertion order)
- Use `random.Random(seed)` not `random.seed(seed)` — module-level seed would affect other code
- Distribute remainder episodes to first N groups (standard approach)

### Quality-Weighted Sampling Algorithm

```python
def _quality_weighted(episodes, limit, seed):
    rng = _random.Random(seed)
    scored = [(ep, ep.quality.overall) for ep in episodes
              if ep.quality is not None and ep.quality.overall is not None]
    if not scored:
        return []
    eps, weights = zip(*scored)
    k = min(limit or len(scored), len(scored))
    # random.choices samples WITH replacement; for without-replacement use different approach
    # Use weighted shuffle: assign random priority weighted by quality, sort descending
    weighted = sorted(zip(weights, [rng.random() for _ in eps], eps),
                      key=lambda x: x[0] * x[1], reverse=True)
    return [ep for _, _, ep in weighted[:k]]
```

**Note on `random.choices` vs without-replacement:** `random.choices` samples WITH replacement which is wrong for datasets (you'd get duplicate episodes). The weighted shuffle approach above gives proportional sampling WITHOUT replacement. Alternatively use `numpy.random.Generator.choice(replace=False, p=weights)` if numpy is acceptable.

### Handling `None` Quality in `quality_weighted`

Episodes with `ep.quality is None` or `ep.quality.overall is None` get weight 0 and must be excluded from the weighted sample. If ALL episodes have no quality score, return empty list (with a `logger.warning()` suggesting to run `tq.quality.score()` first).

### Valid Strategies

```python
VALID_STRATEGIES = frozenset({"none", "stratified", "quality_weighted"})
```

Error message example:
```
TorqComposeError: Unknown sampling strategy 'random'.
Valid strategies are: 'none', 'stratified', 'quality_weighted'.
```

### Import Graph Compliance

`sampling.py` imports ONLY:
- `torq.episode.Episode`
- `torq.errors.TorqComposeError`
- Standard library: `collections`, `random`, `logging`

No numpy required (pure stdlib random is sufficient). No imports from `quality`, `storage`, `ingest`, or `serve`.

### File Structure Requirements

- `src/torq/compose/sampling.py` — NEW
- `tests/unit/test_compose_sampling.py` — NEW

Note: `sampling.py` is NOT yet wired into `tq.compose()` — that happens in story 4.4. This story implements the standalone `sample()` function only.

### References

- [Source: epics.md#Epic 4 > Story 4.3] — ACs and story statement
- [Source: architecture.md#DC-02] — `compose/sampling.py`, deterministic with seed
- [Source: architecture.md#Data Flow] — sample step sits between filter and Dataset construction
- [Source: src/torq/compose/dataset.py] — `Dataset` class already exists from story 4.2
- [Source: src/torq/errors.py] — `TorqComposeError` already added in story 4.1

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6

### Debug Log References

Minor: test regex for valid-strategy error message corrected to alphabetical order (`none, quality_weighted, stratified`) — the message sorts strategies for human readability.

### Completion Notes List

- Implemented `sample()` with `_none`, `_stratified`, `_quality_weighted` as private helpers.
- Used `random.Random(seed)` (not module-level `random.seed`) to avoid side-effects on caller's RNG.
- Quality-weighted sampling uses weighted shuffle (random priority = weight × uniform) for proportional WITHOUT-replacement sampling — avoids `random.choices` which allows duplicates.
- All-unscored `quality_weighted` emits `logger.warning()` with hint to run `tq.quality.score()`.
- `sampling.py` NOT wired into `tq.compose()` yet — that is story 4.4.
- 23 tests, all pass. 412 total (0 regressions).
- ✅ Resolved review finding [HIGH]: `_stratified` now does a second pass to redistribute unallocated slots from undersized groups to groups that still have capacity. pick:50, place:20, pour:3, limit=30 now correctly returns 30 episodes.
- ✅ Resolved review finding [MEDIUM]: Added `_WEIGHT_EPS = 1e-9` to quality weights so episodes with `quality.overall=0.0` receive a small non-zero priority and can be selected. New test `test_zero_quality_episodes_are_eligible` verifies this.
- ✅ Resolved review finding [LOW]: Added `test_single_task_group_with_limit` and `test_single_task_group_no_limit` covering the single-group stratified edge case.
- ✅ Resolved review finding [LOW]: `_none` now returns `episodes` directly (no copy) when no limit is set — avoids O(n) allocation for large datasets.
- 27 tests, all pass. 416 total (0 regressions).

### File List

- `src/torq/compose/sampling.py` — NEW (modified: slot redistribution in `_stratified`, epsilon in `_quality_weighted`, no-copy `_none`)
- `tests/unit/test_compose_sampling.py` — NEW (modified: +4 tests for review findings)
