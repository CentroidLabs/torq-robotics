# Story 4.2: Dataset Class

Status: review

## Story

As a developer,
I want a Dataset object that wraps a collection of episodes with a named recipe,
so that I can pass a versioned, inspectable dataset to downstream training tools.

## Acceptance Criteria

1. **Given** a list of Episodes and a name, **When** `Dataset(episodes=episodes, name='pick_v1', recipe={...})` is constructed, **Then** `len(dataset)` returns the episode count **And** `iter(dataset)` yields Episodes one at a time **And** `repr(dataset)` shows name, episode count, and average quality score (e.g. `Dataset('pick_v1', 31 episodes, quality_avg=0.81)`).

2. **Given** a Dataset, **When** `dataset.recipe` is read, **Then** it returns the full composition query dict that created this dataset (filters, sampling config, seed).

## Tasks / Subtasks

- [x] Task 1: Create `src/torq/compose/dataset.py` — `Dataset` class (AC: #1, #2)
  - [x] Define `Dataset` dataclass or class with fields: `episodes: list[Episode]`, `name: str`, `recipe: dict`
  - [x] Implement `__len__(self) -> int` — returns `len(self.episodes)`
  - [x] Implement `__iter__(self) -> Iterator[Episode]` — yields from `self.episodes`
  - [x] Implement `__repr__(self) -> str` — format: `Dataset('pick_v1', 31 episodes, quality_avg=0.81)`
    - [x] Compute `quality_avg` as average of `ep.quality.overall` for episodes where `ep.quality` and `ep.quality.overall` are not `None`
    - [x] If no episodes have quality scores: format as `Dataset('pick_v1', 31 episodes, quality_avg=N/A)`
    - [x] Round quality_avg to 2 decimal places in repr
  - [x] `recipe` field stores the dict as-is (no validation in R1 — caller owns the contract)
  - [x] Add `__all__ = ["Dataset"]`

- [x] Task 2: Export `Dataset` from `src/torq/compose/__init__.py` and top-level (AC: #1)
  - [x] Add `from torq.compose.dataset import Dataset` to `compose/__init__.py`
  - [x] Add `"Dataset"` to `compose/__init__.__all__`
  - [x] Add `from torq.compose import Dataset` to `src/torq/__init__.py`
  - [x] Add `"Dataset"` to `torq/__init__.__all__`

- [x] Task 3: Write unit tests `tests/unit/test_compose_dataset.py` (AC: #1, #2)
  - [x] Test `len(dataset)` returns episode count
  - [x] Test `list(iter(dataset))` yields all episodes in order
  - [x] Test `repr(dataset)` contains name, episode count, and quality_avg
  - [x] Test `repr(dataset)` with scored episodes: quality_avg computed correctly (±0.01)
  - [x] Test `repr(dataset)` with unscored episodes (quality=None): shows `quality_avg=N/A`
  - [x] Test `repr(dataset)` with mix of scored and unscored: avg computed from scored only
  - [x] Test `repr(dataset)` with empty episode list: `0 episodes, quality_avg=N/A`
  - [x] Test `dataset.recipe` returns exact dict passed at construction
  - [x] Test `dataset.name` returns exact name passed at construction
  - [x] Test `dataset.episodes` is the list passed in (same object identity or equivalent)
  - [x] Test Dataset is indexable via `dataset[i]` (`__getitem__` implemented)

## Review Follow-ups (AI)

- [x] [AI-Review][DISMISSED] `Dataset` is a mutable dataclass — intentional. Mutability is needed for `tq.compose()` to build the episodes list and for users to filter/merge datasets post-construction. The `recipe` dict records provenance; immutability is not required.
- [x] [AI-Review][MEDIUM] `__getitem__` only accepts `int`, no slice support. `dataset[0:5]` raises `TypeError`. Common expectation for sequence-like objects. [src/torq/compose/dataset.py:52-54]
  - Resolution: Changed signature to `idx: int | slice` returning `Episode | list[Episode]`. Delegates to `list.__getitem__` which handles both naturally. Added 3 slice tests.
- [x] [AI-Review][LOW] No `__contains__` method — `episode in dataset` falls back to `__iter__` (O(n)). Fine for R1 but worth noting. [src/torq/compose/dataset.py]
  - Resolution: Added explicit `__contains__` delegating to `self.episodes`. O(n) is documented. Added 3 containment tests.
- [x] [AI-Review][LOW] Test mocks use bare `MagicMock()` instead of `spec=Episode` — attribute typos in production code (e.g. `ep.quality.overal`) would pass silently. Add `spec=Episode` to at least one test. [tests/unit/test_compose_dataset.py:28]
  - Resolution: Updated `_make_episode()` to use `MagicMock(spec=Episode)` across all tests.

## Dev Notes

### Architecture Contract — Non-negotiable

From `architecture.md#Dataset Interface Contract`:
```python
class Dataset:
    episodes: list[Episode]      # the actual episodes
    recipe: dict                 # the composition query that created this dataset
    name: str                    # version string e.g. 'pick_place_v3'
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Episode]: ...
    def __repr__(self) -> str: ...  # "Dataset('pick_place_v3', 31 episodes, quality_avg=0.81)"
```

The `serve/` module (story 5.1 — PyTorch DataLoader) will import `Dataset` directly. The repr format is also used in the architecture doc examples — match it exactly.

### Implementation Recommendation — Use `@dataclass`

A simple dataclass is the cleanest approach:

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterator
from torq.episode import Episode

@dataclass
class Dataset:
    episodes: list[Episode]
    name: str
    recipe: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.episodes)

    def __iter__(self) -> Iterator[Episode]:
        return iter(self.episodes)

    def __getitem__(self, idx: int) -> Episode:
        return self.episodes[idx]

    def __repr__(self) -> str:
        n = len(self.episodes)
        scored = [ep.quality.overall for ep in self.episodes
                  if ep.quality is not None and ep.quality.overall is not None]
        avg = f"{sum(scored)/len(scored):.2f}" if scored else "N/A"
        return f"Dataset('{self.name}', {n} episodes, quality_avg={avg})"
```

**Note on `__getitem__`:** Adding `__getitem__` is strongly recommended (not in AC but required by PyTorch DataLoader in story 5.1). Add it now to avoid touching this file again.

### Repr Format — Match Exactly

The architecture and story both specify:
```
Dataset('pick_v1', 31 episodes, quality_avg=0.81)
```

- Single quotes around the name
- Space after comma
- `quality_avg=` prefix (no space around `=`)
- 2 decimal places for the score

Test this exactly with `assert repr(dataset) == "Dataset('pick_v1', 31 episodes, quality_avg=0.81)"`.

### Import Graph Compliance

`dataset.py` imports ONLY:
- `torq.episode.Episode`
- Standard library (`__future__`, `dataclasses`, `typing`)

No imports from `quality`, `storage`, `ingest`, `compose._query`, or `serve`. This keeps `Dataset` as a pure data container.

### File Structure Requirements

- `src/torq/compose/dataset.py` — NEW
- `src/torq/compose/__init__.py` — MODIFY: add `Dataset` import + `__all__` entry
- `src/torq/__init__.py` — MODIFY: add `Dataset` re-export + `__all__` entry
- `tests/unit/test_compose_dataset.py` — NEW

### References

- [Source: architecture.md#Dataset Interface Contract] — Required fields, methods, repr format
- [Source: architecture.md#Return Type Conventions] — `tq.compose()` returns `Dataset`
- [Source: architecture.md#Import graph] — compose imports episode/errors/types only
- [Source: epics.md#Epic 4 > Story 4.2] — ACs and story statement
- [Source: src/torq/compose/__init__.py] — Current exports: `query` only; add `Dataset`

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6

### Debug Log References

None.

### Completion Notes List

- Implemented `Dataset` as a `@dataclass` with `episodes`, `name`, `recipe` fields.
- Added `__getitem__` (not in AC but required by PyTorch DataLoader in story 5.1).
- `__repr__` computes `quality_avg` from `ep.quality.overall` only; falls back to `N/A` when no scored episodes.
- `recipe` defaults to empty dict via `field(default_factory=dict)`.
- 20 tests, all pass. 383 total (0 regressions).

### File List

- `src/torq/compose/dataset.py` — NEW
- `src/torq/compose/__init__.py` — MODIFIED: added `Dataset` import + `__all__` entry
- `src/torq/__init__.py` — MODIFIED: added `Dataset` re-export + `__all__` entry
- `tests/unit/test_compose_dataset.py` — NEW (20 tests)
