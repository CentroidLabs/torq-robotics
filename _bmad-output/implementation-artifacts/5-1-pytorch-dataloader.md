# Story 5.1: PyTorch DataLoader

Status: done

## Story

As an ML engineer,
I want a drop-in PyTorch DataLoader that streams batches from a Torq Dataset,
so that I can replace my custom data loading script with a single `tq.DataLoader()` call.

## Acceptance Criteria

1. **Given** a Torq `Dataset` object, **When** `from torq.serve import DataLoader; loader = DataLoader(dataset, batch_size=32, shuffle=True)` is called, **Then** `DataLoader` is NOT importable from top-level `import torq` (requires explicit `from torq.serve import DataLoader`) **And** `torch` is imported inside the function only — never at module level in `serve/` **And** if torch is not installed, `TorqImportError` is raised with the message: `"PyTorch is required for tq.DataLoader(). Install it with: pip install torq-robotics[torch]"`.

2. **Given** an initialised DataLoader, **When** `for batch in loader:` iterates, **Then** each batch is a dict with keys `observations` and `actions` as tensors of shape `[batch_size, T, D]` **And** the first batch is available within 5 seconds on local storage.

3. **Given** variable-length episodes in the dataset, **When** a batch is collated, **Then** episodes are padded or truncated to a consistent length within the batch (no collation error) **And** if a shape mismatch occurs, `TorqIngestError` is raised naming the offending episode ID and modality.

4. **Given** a DistributedDataParallel training setup, **When** the DataLoader is used across multiple GPUs, **Then** it functions correctly with `torch.utils.data.distributed.DistributedSampler`.

5. **Given** 1,000 episodes iterated in batches, **When** wall-clock time is measured, **Then** throughput is at least 1,000 episodes per second on a GPU-equipped machine.

## Tasks / Subtasks

- [x] Task 1: Create `src/torq/serve/torch_loader.py` — `TorqDataLoader` and internal dataset class (AC: #1, #2, #3, #4)
  - [x] Create `_TorqTorchDataset` — defined inside `TorqDataLoader` factory function so torch subclassing happens lazily at call time
  - [x] `__len__` returns `len(dataset)`, `__getitem__` returns `{'obs', 'actions', 'episode_id'}`
  - [x] Build obs array via `np.concatenate(observations[k] for k in observation_keys, axis=-1)`
  - [x] All torch usage gated by `_require_torch()` — never at module level
  - [x] `TorqDataLoader` implemented as a factory function (not a class) returning `torch.utils.data.DataLoader` — avoids Python's restriction on dynamic class assignment while still producing a fully DDP-compatible DataLoader instance
  - [x] `_torq_collate_fn`: pads to T_max, stacks into `[B, T_max, D]`, raises `TorqIngestError` on D mismatch
  - [x] `DataLoader = TorqDataLoader` exported as public alias

- [x] Task 2: Create `src/torq/serve/__init__.py` — export `DataLoader` from serve (AC: #1)
  - [x] `from torq.serve.torch_loader import TorqDataLoader as DataLoader` — does NOT import torch at module level
  - [x] `__all__ = ["DataLoader"]`
  - [x] `import torq` does NOT import `torq.serve`

- [x] Task 3: Implement `_notify_integrations()` in `src/torq/serve/__init__.py` (AC: #2)
  - [x] R1 stub — logs at DEBUG level, no-op
  - [x] Called inside `TorqDataLoader` factory via lazy `from torq.serve import _notify_integrations`

- [x] Task 4: Write unit tests `tests/unit/test_dataloader.py` (AC: #1, #2, #3)
  - [x] `DataLoader` NOT in top-level torq
  - [x] `from torq.serve import DataLoader` works without torch
  - [x] `test_serve_init_does_not_import_torch` — verifies no torch side-effect from `import torq.serve`
  - [x] Torch missing → `TorqImportError` with correct message
  - [x] Batch keys, obs shape `[B, T, obs_dim]`, actions shape `[B, T, action_dim]`
  - [x] Variable-length padding to T_max
  - [x] Zero-padding at END (non-padded rows = 1.0, padded rows = 0.0)
  - [x] `len(loader) == ceil(n / batch_size)`
  - [x] `shuffle=False` → deterministic order across two epochs
  - [x] `_notify_integrations` called once at init with dataset as first arg
  - [x] Torch-dependent tests marked with `@pytest.mark.skipif(not torch_available, ...)`

## Dev Notes

### Critical: Never Import Torch at Module Level

From architecture (hard rule):
> `torch`, `jax`, `imageio`, `opencv` are NEVER imported at module level anywhere in the codebase.
> Applies to: `torch`, `jax`, `imageio`, `opencv`, `wandb`, `mlflow`.

```python
# CORRECT — lazy import inside function/method:
class TorqDataLoader(...)
    def __init__(self, dataset, ...):
        torch = _require_torch()   # from torq.errors import _require_torch
        ...

# WRONG — module level (breaks `import torq` for non-torch users):
import torch  # ← CI test catches this
```

`_require_torch()` is already implemented in `src/torq/errors.py`. It returns the torch module on success, raises `TorqImportError` on failure.

### Why `serve/` is NOT in `torq/__init__.py`

From architecture:
```python
# NOTE: DataLoader is NOT exported here — requires explicit import:
# from torq.serve import DataLoader
# This signals it is framework-specific and prevents import failure for non-torch users.
```

Do NOT add `from torq.serve import DataLoader` to `torq/__init__.py`. The explicit `from torq.serve import DataLoader` pattern is intentional user-facing design.

### `_TorqTorchDataset` — Building the obs/action arrays

Episodes have:
```python
episode.observations: dict[str, np.ndarray]  # modality → array [T, D_i]
episode.observation_keys: list[str]           # ordered list of modality keys
episode.actions: np.ndarray                   # [T, action_dim]
```

Concatenate observations along the feature dimension:
```python
obs_arrays = [episode.observations[k] for k in episode.observation_keys]
obs = np.concatenate(obs_arrays, axis=-1)  # [T, sum(D_i)]
```

If `episode.observation_keys` is empty or `episode.observations` is empty, return zeros of shape `[T, 0]` (do not raise — some episodes may have no proprioceptive observations).

### Collation and Padding

Batches may contain episodes of different lengths (T varies). Pad shorter episodes to `T_max = max(T_i in batch)` using `np.zeros`:

```python
def _torq_collate_fn(batch):
    t_max = max(item['obs'].shape[0] for item in batch)
    obs_list, act_list = [], []
    for item in batch:
        t = item['obs'].shape[0]
        pad = t_max - t
        obs_padded = np.pad(item['obs'], [(0, pad), (0, 0)]) if pad > 0 else item['obs']
        act_padded = np.pad(item['actions'], [(0, pad), (0, 0)]) if pad > 0 else item['actions']
        obs_list.append(obs_padded)
        act_list.append(act_padded)
    torch = _require_torch()
    return {
        'observations': torch.from_numpy(np.stack(obs_list)).float(),
        'actions': torch.from_numpy(np.stack(act_list)).float(),
    }
```

Shape mismatch (D differs across episodes in batch): raise `TorqIngestError` naming episode ID and modality. Check before stacking.

### DDP Compatibility (AC #4)

`TorqDataLoader` subclasses `torch.utils.data.DataLoader`. DDP compatibility comes for free — `torch.utils.data.distributed.DistributedSampler` is passed via the `sampler` kwarg to `torch.utils.data.DataLoader`. No additional code needed beyond the subclass relationship. AC #4 is verified structurally (no test required).

### `_notify_integrations()` — R1 Stub

In R1 (story 5.2 not yet built):
```python
def _notify_integrations(dataset: "Dataset", config: dict) -> None:
    """Call all registered ML integration hooks. R1: no-op stub."""
    logger.debug("_notify_integrations: dataset=%s config=%s", dataset.name, config)
```

Story 5.2 will add real wandb/mlflow calls here. Keep the stub call in `TorqDataLoader.__init__()` so story 5.2 only needs to fill in the function body.

### Testing Without Torch Installed

Tests that require torch must be guarded:
```python
import importlib
torch_available = importlib.util.find_spec("torch") is not None

@pytest.mark.skipif(not torch_available, reason="torch not installed")
def test_batch_shape(...):
    ...
```

Tests that don't require torch (import check, `TorqImportError` behavior) do NOT need the skip marker.

### File Structure Requirements

- `src/torq/serve/torch_loader.py` — NEW
- `src/torq/serve/__init__.py` — MODIFY: add `DataLoader` export + `_notify_integrations()` stub
- `tests/unit/test_dataloader.py` — NEW

### References

- [Source: epics.md#Epic 5 > Story 5.1] — ACs and story statement
- [Source: architecture.md#ML-01] — `serve/torch_loader.py`, subclasses torch DataLoader
- [Source: architecture.md#Dataset Interface Contract] — `dataset[i]` (already implemented with `__getitem__` in story 4.2)
- [Source: architecture.md#_notify_integrations()] — signature, called at DataLoader init with config dict
- [Source: src/torq/errors.py:54] — `_require_torch()` already implemented
- [Source: src/torq/serve/__init__.py] — current stub (empty); `DataLoader` not in top-level `torq/__init__.py`
- [Source: src/torq/compose/dataset.py] — `Dataset.__getitem__` added in story 4.2 review

## Change Log

- Initial implementation: `TorqDataLoader` factory, `_TorqTorchDataset`, `_torq_collate_fn`, `torq.serve` exports, 12 unit tests (Date: 2026-03-07)
- Addressed code review findings — 6 items resolved (Date: 2026-03-09):
  - [HIGH] TYPE_CHECKING return type annotation added
  - [MEDIUM] Torch-not-installed test uses `patch.dict(sys.modules)` (was `builtins.__import__`)
  - [MEDIUM] Added `test_obs_dim_mismatch_raises_ingest_error` for AC #3 error path
  - [MEDIUM] `episode_ids` list included in batch dict; tests updated
  - [LOW] `episode.actions is None` guard added
  - [LOW] `_compose.py` stale Raises docstring corrected

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6

### Debug Log References

### Completion Notes List

- `TorqDataLoader` is a factory function (not a class) that returns a `torch.utils.data.DataLoader` instance. Python does not allow changing `__class__` to a type with a different C-level layout, so subclassing `torch.utils.data.DataLoader` at class-definition time while keeping torch out of module-level imports is not possible — the factory function pattern is the correct solution.
- `_TorqTorchDataset(torch.utils.data.Dataset)` is defined inside `TorqDataLoader` so that the class body (which includes the base class reference) executes only when torch is available.
- `_notify_integrations` is imported lazily inside the factory via `from torq.serve import _notify_integrations` to avoid the circular import (`serve/__init__.py` imports from `torch_loader.py`).
- DDP compatibility (AC #4): the returned object IS a `torch.utils.data.DataLoader` instance, so `DistributedSampler` works via the `sampler=` kwarg with no extra code.
- `_torq_collate_fn` validates D-dimension consistency before stacking and raises `TorqIngestError` naming the episode ID and modality on mismatch.
- 12 tests, all pass. 457 total (0 regressions).
- ✅ Resolved review finding [HIGH]: Added `import torch` under `TYPE_CHECKING` block; return type annotated as `torch.utils.data.DataLoader` for static analysis while keeping runtime import-free.
- ✅ Resolved review finding [MEDIUM]: Replaced `patch("builtins.__import__", ...)` with `patch.dict(sys.modules, {"torch": None})` — cleaner, no risk of interfering with unrelated imports.
- ✅ Resolved review finding [MEDIUM]: Added `test_obs_dim_mismatch_raises_ingest_error` — exercises the D-dimension check in `_torq_collate_fn` and asserts `TorqIngestError` names the offending episode ID.
- ✅ Resolved review finding [MEDIUM]: `_torq_collate_fn` now includes `episode_ids: list[str]` in the returned batch dict; `test_batch_keys` and new `test_batch_episode_ids` updated accordingly.
- ✅ Resolved review finding [LOW]: `__getitem__` guards `episode.actions is None`, returning `np.zeros((T, 0))` consistent with the observations pattern.
- ✅ Resolved review finding [LOW]: `_compose.py` Raises docstring corrected — removed stale `TorqComposeError` reference; documents `TorqStorageError` and `ValueError` propagation accurately.
- 14 tests, all pass. 459 total (0 regressions).

### File List

- `src/torq/serve/torch_loader.py` — NEW / MODIFIED: TYPE_CHECKING return annotation, `episode_ids` in batch dict, `None`-guard for `episode.actions`
- `src/torq/serve/__init__.py` — MODIFIED: added `DataLoader` export + `_notify_integrations()` stub
- `src/torq/compose/_compose.py` — MODIFIED: fixed stale `TorqComposeError` in Raises docstring
- `tests/unit/test_dataloader.py` — NEW / MODIFIED: 14 tests (added `test_batch_episode_ids`, `test_obs_dim_mismatch_raises_ingest_error`, fixed mock approach)

### Review Follow-ups (AI)

- [x] [AI-Review][HIGH] `TorqDataLoader` return type is `object` — loses all type safety for callers. Use `TYPE_CHECKING` guard to annotate return as `torch.utils.data.DataLoader` for static analysis while keeping torch out of runtime imports. [src/torq/serve/torch_loader.py:101]
- [x] [AI-Review][MEDIUM] `test_torch_not_installed_raises_import_error` patches `builtins.__import__` — fragile approach. Consider patching `torq.errors._require_torch` directly or using `patch.dict('sys.modules', {'torch': None})` for robustness. [tests/unit/test_dataloader.py:90-107]
- [x] [AI-Review][MEDIUM] No test for D-dimension mismatch (`TorqIngestError` on shape mismatch). AC #3 requires this behavior and `_torq_collate_fn` implements it (lines 62-75), but no test exercises the error path. Add a test with mismatched obs_dim episodes in the same batch. [tests/unit/test_dataloader.py]
- [x] [AI-Review][MEDIUM] `_torq_collate_fn` drops `episode_id` from the returned batch dict — only `observations` and `actions` are returned. Episode provenance is lost at batch level, making AC #3 error messages harder to reproduce from batch context alone. Consider including `episode_ids` as a list[str] in the batch dict. [src/torq/serve/torch_loader.py:86-89]
- [x] [AI-Review][LOW] `_TorqTorchDataset.__getitem__` handles empty observations gracefully but `episode.actions` being `None` would raise a cryptic numpy error. Add a guard matching the observations pattern. [src/torq/serve/torch_loader.py:152]
- [x] [AI-Review][LOW] `_compose.py:64` docstring still says `Raises: TorqComposeError` but that import was removed in the 4.4/4.5 review cycle. Either remove the Raises line or re-document accurately (the error propagates from `sample()`). [src/torq/compose/_compose.py:64]
