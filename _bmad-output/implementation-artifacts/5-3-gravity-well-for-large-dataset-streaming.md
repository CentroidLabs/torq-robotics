# Story 5.3: Gravity Well for Large Dataset Streaming

Status: done

## Story

As an ML engineer,
I want a prompt directing me to cloud streaming when my local dataset exceeds 50GB,
so that I discover the cloud streaming option before storage becomes a blocker.

## Acceptance Criteria

1. **Given** a Dataset whose total on-disk size exceeds 50GB, **When** `DataLoader(dataset, store_path='/data')` is initialised, **Then** `_gravity_well()` fires with a message referencing the dataset size and the datatorq.ai streaming URL (GW-SDK-03).

2. **Given** a Dataset under 50GB, **When** `DataLoader(dataset, store_path='/data')` is initialised, **Then** no gravity well fires.

3. **Given** `tq.config.quiet = True`, **When** the 50GB threshold is exceeded, **Then** no gravity well output is printed.

## Tasks / Subtasks

- [x] Task 1: Add `store_path` parameter to `TorqDataLoader` in `src/torq/serve/torch_loader.py` (AC: #1, #2)
  - [x] Add `store_path: str | Path | None = None` as an optional parameter (keyword-only or positional after `pin_memory`)
  - [x] If `store_path` is not `None`: compute total on-disk size of the dataset
  - [x] Size computation: sum file sizes under `Path(store_path)` for all episode IDs in `dataset.recipe.get('sampled_episode_ids', [])` — both Parquet and MP4 files
  - [x] Alternatively (simpler): sum all file sizes under `Path(store_path)` recursively if `sampled_episode_ids` is not in recipe — use `sum(f.stat().st_size for f in Path(store_path).rglob('*') if f.is_file())`
  - [x] If `store_path` is `None`: skip size check entirely (no error, no gravity well)
  - [x] Define threshold: `_50GB = 50 * 1024 ** 3` (bytes)
  - [x] If total size > `_50GB`: fire `_gravity_well(f"Dataset is {size_gb:.1f} GB. Stream at scale from datatorq.ai", "GW-SDK-03")`
  - [x] Fire gravity well BEFORE `_notify_integrations()` call — at DataLoader init time
  - [x] Import `_gravity_well` from `torq._gravity_well` inside the function body (already lazy-imported in other compose modules — follow same pattern)

- [x] Task 2: Write unit tests `tests/unit/test_dataloader_gravity_well.py` (AC: #1, #2, #3)
  - [x] Test >50GB triggers gravity well: mock `Path.rglob` to return files summing to 51GB, verify stdout contains `datatorq.ai`
  - [x] Test <50GB does not trigger gravity well: mock files summing to 10GB, verify no output
  - [x] Test exactly 50GB does NOT trigger (threshold is strictly `>`): verify no output
  - [x] Test `store_path=None` (omitted): no gravity well, no error
  - [x] Test `config.quiet = True`: 51GB dataset but no output printed
  - [x] Test gravity well message contains the dataset size in GB (e.g. `"51.0 GB"`)
  - [x] Test gravity well message contains `"datatorq.ai"`

## Dev Notes

### `store_path` Parameter Design

The current `TorqDataLoader` signature:
```python
def TorqDataLoader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    **kwargs,
) -> "torch.utils.data.DataLoader":
```

Add `store_path` before `**kwargs`:
```python
def TorqDataLoader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    store_path: str | Path | None = None,
    **kwargs,
) -> "torch.utils.data.DataLoader":
```

`store_path` must NOT be forwarded to `torch.utils.data.DataLoader` via `**kwargs` — consume it before passing `**kwargs` to super.

### Size Computation

Two strategies — use whichever is simpler:

**Option A — Full store scan (simple, conservative):**
```python
from pathlib import Path

if store_path is not None:
    store = Path(store_path)
    total_bytes = sum(f.stat().st_size for f in store.rglob("*") if f.is_file())
    _50GB = 50 * 1024 ** 3
    if total_bytes > _50GB:
        size_gb = total_bytes / (1024 ** 3)
        _gravity_well(
            f"Dataset is {size_gb:.1f} GB. Stream at scale from datatorq.ai",
            "GW-SDK-03",
        )
```

**Option B — Episode-specific scan (precise):**
Look up only the Parquet and MP4 files for episode IDs in `dataset.recipe.get('sampled_episode_ids', [])`. More precise but more complex (must know the file naming convention from `storage/parquet.py`).

**Recommendation: use Option A.** It's simpler, and the store_path is the user's data directory — scanning it is fast and correct. If the user's store has non-Torq files, they contribute to the size estimate (conservative is better for this warning).

### `_gravity_well` Import Pattern

`torch_loader.py` uses lazy imports throughout. Import `_gravity_well` inside the factory function body:

```python
def TorqDataLoader(...):
    torch = _require_torch()
    ...
    if store_path is not None:
        from torq._gravity_well import _gravity_well  # noqa: PLC0415
        ...
```

OR import at module level since `_gravity_well.py` only imports from `torq._config` and has no torch dependency — this is safe:
```python
from torq._gravity_well import _gravity_well  # safe: no torch, no circular
```

Check what other serve modules do and be consistent.

### Test Mocking Strategy

Mock `Path.stat().st_size` to simulate large files without real disk usage:

```python
from unittest.mock import patch, MagicMock
import pathlib

def test_large_dataset_fires_gravity_well(tmp_path, capsys, monkeypatch):
    # Create one dummy file in tmp_path
    dummy = tmp_path / "ep_001.parquet"
    dummy.write_bytes(b"x")

    # Mock stat to return 51GB for any file
    mock_stat = MagicMock()
    mock_stat.st_size = 51 * 1024 ** 3
    monkeypatch.setattr(pathlib.Path, "stat", lambda self: mock_stat)

    loader = DataLoader(dataset, store_path=tmp_path)
    captured = capsys.readouterr()
    assert "datatorq.ai" in captured.out
```

### Gravity Well Fires Before `_notify_integrations`

Order of operations in `TorqDataLoader` after this story:
1. `torch = _require_torch()`
2. Define `_TorqTorchDataset` class
3. Create `torch_dataset` and `loader`
4. **Check size → fire GW-SDK-03 if needed** ← new
5. Call `_notify_integrations(dataset, config)`
6. Return `loader`

### File Structure Requirements

- `src/torq/serve/torch_loader.py` — MODIFY: add `store_path` param + size check + GW-SDK-03
- `tests/unit/test_dataloader_gravity_well.py` — NEW

### References

- [Source: epics.md#Epic 5 > Story 5.3] — ACs and story statement
- [Source: architecture.md#GW-SDK-03] — `serve/torch_loader.py`, fires at DataLoader init if >50GB
- [Source: src/torq/serve/torch_loader.py:96] — `TorqDataLoader` factory function to modify
- [Source: src/torq/_gravity_well.py] — `_gravity_well()` implementation; quiet suppression
- [Source: tests/unit/test_compose_gravity_well.py] — Reference test pattern for gravity well integration

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6

### Debug Log References

None — implementation was straightforward.

### Completion Notes List

- Added `store_path: str | Path | None = None` parameter to `TorqDataLoader` factory function
- Used Option A (full store scan via `rglob`) as recommended in Dev Notes — simple and correct
- Lazy-imports `_gravity_well` inside the function body to maintain zero torch-at-module-level invariant
- Gravity well fires strictly before `_notify_integrations()` call per spec
- Tests mock `torq.serve.torch_loader.Path` at module level to avoid global `pathlib.Path.stat` pollution that breaks `rglob` internals
- All 8 new tests pass; 490 total tests pass, 0 regressions

### Change Log

- Added `store_path` param and GW-SDK-03 size check to `TorqDataLoader` (2026-03-09)
- Added `tests/unit/test_dataloader_gravity_well.py` with 8 unit tests (2026-03-09)

### File List

- `src/torq/serve/torch_loader.py` — MODIFIED: added `store_path` param, `Path` import, GW-SDK-03 logic
- `tests/unit/test_dataloader_gravity_well.py` — NEW: 8 unit tests for GW-SDK-03

### Review Follow-ups (AI)

- [x] [AI-Review][HIGH] `store_path` size computation (`f.stat().st_size` in rglob loop) can raise `PermissionError`/`OSError` on unreadable files, crashing DataLoader init for a non-critical gravity well hint. Wrap the size computation in try/except and log a debug warning on failure. [src/torq/serve/torch_loader.py:189]
- [x] [AI-Review][MEDIUM] `test_51gb_fires_gravity_well` and `test_51gb_message_contains_datatorq_url` are identical — both assert `"datatorq.ai" in out`. Remove the duplicate or differentiate (e.g., test full URL format `https://www.datatorq.ai` or the `→` arrow). [tests/unit/test_dataloader_gravity_well.py:90,120]
- [x] [AI-Review][MEDIUM] No test verifies gravity well fires BEFORE `_notify_integrations()`. Story spec requires this ordering. Add a test using `unittest.mock.call_args_list` or side_effect ordering to assert GW fires first. [src/torq/serve/torch_loader.py:183-199]
- [x] [AI-Review][LOW] Redundant string annotation: `store_path: "str | Path | None"` — file already has `from __future__ import annotations`, so the quotes are unnecessary. Use `store_path: str | Path | None = None`. [src/torq/serve/torch_loader.py:103]
