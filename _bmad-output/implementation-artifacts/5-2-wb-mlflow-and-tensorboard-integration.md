# Story 5.2: W&B, MLflow, and TensorBoard Integration

Status: done

## Story

As an ML engineer,
I want dataset lineage automatically logged to my experiment tracker at training start,
so that I can trace any trained model back to the exact dataset version and quality statistics that produced it.

## Acceptance Criteria

1. **Given** W&B is installed and `import wandb; wandb.init()` has been called, **When** `from torq.integrations.wandb import init; init(dataset)` is called before training, **Then** dataset ID, version, composition recipe, quality statistics (mean, std, min, max), and episode count are logged to the active W&B run **And** `wandb` is imported inside the function only — never at module level.

2. **Given** MLflow is installed and an active MLflow run exists, **When** `from torq.integrations.mlflow import init; init(dataset)` is called, **Then** the same dataset metadata is logged as MLflow params and tags.

3. **Given** neither W&B nor MLflow is installed, **When** `torq.integrations` is imported, **Then** no `ImportError` is raised at import time (conditional imports only) **And** calling `init(dataset)` raises `TorqImportError` with the appropriate install instruction.

4. **Given** `tq.config.quiet = True`, **When** integration logging runs, **Then** no console output is produced by the integration layer (logging only via `logging` module).

5. **Given** a `DataLoader(dataset, ...)` is initialised (the automatic hook), **When** W&B or MLflow is installed and an active run exists, **Then** dataset metadata is logged automatically without the user calling `init()` explicitly.

## Tasks / Subtasks

- [x] Task 1: Create `src/torq/integrations/wandb.py` — W&B integration (AC: #1, #3, #4, #5)
  - [x] Implement `notify(dataset: Dataset, config: dict) -> None` — called automatically by `_notify_integrations()`
    - [x] Lazy import: `import wandb` inside function body only
    - [x] If wandb not installed: log at DEBUG, return silently (no exception — called automatically)
    - [x] If no active wandb run (`wandb.run is None`): log at DEBUG, return silently
    - [x] Log via `wandb.run.config.update({...})` or `wandb.run.log({...})`:
      - `torq_dataset_name`: `dataset.name`
      - `torq_episode_count`: `len(dataset)`
      - `torq_recipe`: `dataset.recipe` (serialised as dict)
      - `torq_quality_mean`, `torq_quality_std`, `torq_quality_min`, `torq_quality_max`: computed from `ep.quality.overall` for scored episodes
    - [x] No console output — only `logging.getLogger(__name__)` calls
  - [x] Implement `init(dataset: Dataset, config: dict | None = None) -> None` — user-facing alias
    - [x] If wandb not installed: raise `TorqImportError("wandb is required. Install with: pip install wandb")`
    - [x] Else: call `notify(dataset, config or {})`

- [x] Task 2: Create `src/torq/integrations/mlflow.py` — MLflow integration (AC: #2, #3, #4, #5)
  - [x] Implement `notify(dataset: Dataset, config: dict) -> None` — same pattern as wandb
    - [x] Lazy import: `import mlflow` inside function body
    - [x] If mlflow not installed or no active run (`mlflow.active_run() is None`): return silently
    - [x] Log via `mlflow.log_params({...})` and `mlflow.set_tags({...})`:
      - Same metadata fields as wandb (name, count, recipe, quality stats)
      - Prefix keys with `torq_` for namespacing
    - [x] No console output
  - [x] Implement `init(dataset: Dataset, config: dict | None = None) -> None`
    - [x] If mlflow not installed: raise `TorqImportError("mlflow is required. Install with: pip install mlflow")`
    - [x] Else: call `notify(dataset, config or {})`

- [x] Task 3: Update `src/torq/integrations/__init__.py` — real `_notify_integrations()` (AC: #5)
  - [x] Implement `_notify_integrations(dataset: Dataset, config: dict) -> None`
  - [x] Call `torq.integrations.wandb.notify(dataset, config)` — wrapped in try/except, log warning on unexpected error
  - [x] Call `torq.integrations.mlflow.notify(dataset, config)` — same
  - [x] Never raises — failures logged as warnings

- [x] Task 4: Update `src/torq/serve/__init__.py` — delegate to real `_notify_integrations()` (AC: #5)
  - [x] Replaced stub body with lazy import: `from torq.integrations import _notify_integrations as _real_notify; _real_notify(dataset, config)`

- [x] Task 5: Write unit tests `tests/unit/test_integrations.py` (AC: #1, #2, #3, #4, #5)
  - [x] Test `import torq.integrations` raises no exception when wandb/mlflow absent (patch sys.modules)
  - [x] Test `wandb.init(dataset)` raises `TorqImportError` when wandb not installed
  - [x] Test `mlflow.init(dataset)` raises `TorqImportError` when mlflow not installed
  - [x] Test `wandb.notify()` skips silently when wandb not installed (no exception)
  - [x] Test `wandb.notify()` skips silently when no active wandb run (`wandb.run is None`)
  - [x] Test `wandb.notify()` calls `wandb.run.config.update` with correct keys when run is active (mock wandb)
  - [x] Test quality stats computed correctly (mean/std/min/max from scored episodes only)
  - [x] Test unscored episodes excluded from quality stats
  - [x] Test `mlflow.notify()` calls `mlflow.log_params` with correct keys (mock mlflow)
  - [x] Test `_notify_integrations()` calls both wandb and mlflow notify (mock both)
  - [x] Test `config.quiet` does not suppress `logging` calls (quiet only suppresses `print()`)

## Dev Notes

### Key Architectural Points

**`integrations/` module rules:**
- ALL framework imports (`wandb`, `mlflow`) INSIDE function bodies — never at module level
- `import torq.integrations` must succeed even when neither wandb nor mlflow is installed
- `notify()` functions: silent on missing dep or no active run — called automatically, must not disrupt training
- `init()` functions: raise `TorqImportError` on missing dep — user-called, failure is appropriate

**`config.quiet` and integrations:**
- `quiet=True` suppresses `print()` / `_gravity_well()` output
- It does NOT suppress `logger.warning()` or `logger.info()` calls
- The integration layer uses ONLY `logging` module — zero `print()` calls

### Quality Stats Computation

```python
def _quality_stats(dataset: "Dataset") -> dict:
    scores = [
        ep.quality.overall
        for ep in dataset.episodes
        if ep.quality is not None and ep.quality.overall is not None
    ]
    if not scores:
        return {"torq_quality_mean": None, "torq_quality_std": None,
                "torq_quality_min": None, "torq_quality_max": None,
                "torq_quality_n_scored": 0}
    import statistics
    return {
        "torq_quality_mean": round(sum(scores) / len(scores), 4),
        "torq_quality_std": round(statistics.stdev(scores), 4) if len(scores) > 1 else 0.0,
        "torq_quality_min": round(min(scores), 4),
        "torq_quality_max": round(max(scores), 4),
        "torq_quality_n_scored": len(scores),
    }
```

### Circular Import Avoidance in `serve/__init__.py`

`serve/__init__.py` currently has the stub `_notify_integrations`. To delegate to integrations without circular imports:

```python
def _notify_integrations(dataset: "Dataset", config: dict) -> None:
    from torq.integrations import _notify_integrations as _real_notify
    _real_notify(dataset, config)
```

This lazy import is safe because `torq.integrations` does not import from `torq.serve`.

### MLflow Key Flattening

`mlflow.log_params()` only accepts scalar values (strings, numbers). The `recipe` dict must be serialised:

```python
mlflow.log_params({
    "torq_dataset_name": dataset.name,
    "torq_episode_count": len(dataset),
    "torq_recipe": str(dataset.recipe),  # serialise dict as string
    **quality_stats,
})
```

### W&B: `config.update` vs `run.log`

Use `wandb.run.config.update({...})` for dataset metadata (hyperparameter-like, one-time) rather than `wandb.run.log({...})` (time-series metrics). This appears in the W&B run's Config panel.

### Import Graph Compliance

```
integrations/  ← imports types only (all framework imports conditional)
serve/         ← imports episode, errors, compose, integrations, types
```

`integrations/wandb.py` and `integrations/mlflow.py` import ONLY:
- `torq.compose.dataset.Dataset` (TYPE_CHECKING guard to avoid circular)
- `torq.errors.TorqImportError`
- Standard library: `logging`, `statistics`
- Their respective optional framework: `wandb` / `mlflow` — INSIDE function bodies only

### File Structure Requirements

- `src/torq/integrations/wandb.py` — NEW
- `src/torq/integrations/mlflow.py` — NEW
- `src/torq/integrations/__init__.py` — MODIFY: add real `_notify_integrations()`
- `src/torq/serve/__init__.py` — MODIFY: stub delegates to `integrations._notify_integrations`
- `tests/unit/test_integrations.py` — NEW

### TensorBoard — Deferred Beyond R1

> **Note:** The story title mentions TensorBoard but no TensorBoard implementation was built in this story.
> TensorBoard integration is deferred to a post-R1 story. The `_notify_integrations()` hook is already
> in place so TensorBoard support only requires adding `src/torq/integrations/tensorboard.py` and calling
> `tensorboard.notify(dataset, config)` inside `integrations/__init__.py._notify_integrations()`.

### References

- [Source: epics.md#Epic 5 > Story 5.2] — ACs and story statement
- [Source: architecture.md#ML-03] — `integrations/wandb.py`, `integrations/mlflow.py`, via `_notify_integrations()`
- [Source: architecture.md#_notify_integrations() signature] — exact signature, called with dataset + config dict
- [Source: architecture.md#Import graph] — `integrations/` imports types only; all framework imports conditional
- [Source: src/torq/serve/__init__.py] — current `_notify_integrations` stub to replace/delegate
- [Source: src/torq/integrations/__init__.py] — currently empty stub

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6

### Debug Log References

### Completion Notes List

- `torq.integrations.wandb` and `torq.integrations.mlflow` both follow the same notify/init split: `notify()` is silent-on-failure (safe for auto-hook), `init()` raises `TorqImportError` (clear user feedback when called explicitly).
- `_quality_stats()` lives in `wandb.py` and is re-used by `mlflow.py` via import — single source of truth for quality computation.
- `_notify_integrations()` in `integrations/__init__.py` wraps each integration in try/except so one tracker's failure never blocks the other or disrupts training.
- `serve/__init__.py` stub now lazily imports `torq.integrations._notify_integrations` — avoids circular import since `integrations/` never imports from `serve/`.
- 23 tests, all pass. 482 total (0 regressions).
- ✅ Resolved review finding [MEDIUM]: Extracted `_quality_stats` to `src/torq/integrations/_utils.py`; both `wandb.py` and `mlflow.py` now import from there — decoupled, no wandb↔mlflow dependency.
- ✅ Resolved review finding [MEDIUM]: `test_notify_calls_config_update_with_correct_keys` now asserts concrete values: mean, min (0.6), max (0.9), n_scored (3), and recipe is str.
- ✅ Resolved review finding [MEDIUM]: `test_dataloader_calls_notify_integrations` inlines its own `_make_torch_episode` — no cross-module import.
- ✅ Resolved review finding [LOW]: Removed stale `import numpy as np` and `from torq.compose.dataset import Dataset` (now used properly within the inlined helper).
- ✅ Resolved review finding [LOW]: `wandb.notify()` now serialises recipe with `str()` matching mlflow — consistent across both integrations.
- ✅ Resolved review finding [LOW]: Added "TensorBoard — Deferred Beyond R1" dev note in story file; the `_notify_integrations` hook is already in place for future extension.

### File List

- `src/torq/integrations/_utils.py` — NEW: shared `_quality_stats()` helper
- `src/torq/integrations/wandb.py` — NEW / MODIFIED: imports `_quality_stats` from `_utils`, recipe serialised with `str()`
- `src/torq/integrations/mlflow.py` — NEW / MODIFIED: imports `_quality_stats` from `_utils`
- `src/torq/integrations/__init__.py` — MODIFIED: real `_notify_integrations()` calling both trackers
- `src/torq/serve/__init__.py` — MODIFIED: stub now delegates to `torq.integrations._notify_integrations`
- `tests/unit/test_integrations.py` — NEW / MODIFIED: concrete value assertions, inlined DataLoader helper, removed cross-module import (23 tests)

### Review Follow-ups (AI)

- [x] [AI-Review][MEDIUM] `mlflow.py` imports `_quality_stats` from `wandb.py` — couples two optional integrations. If wandb module has issues, mlflow breaks too. Extract `_quality_stats` to a shared `src/torq/integrations/_utils.py` that both import from. [src/torq/integrations/mlflow.py:24]
- [x] [AI-Review][MEDIUM] `test_notify_calls_config_update_with_correct_keys` asserts key presence but not actual quality stat values. If `notify()` passed wrong data to `_quality_stats`, test would still pass. Assert concrete values (mean, min, max) for the test dataset `[0.8, 0.6, 0.9]`. [tests/unit/test_integrations.py:129-136]
- [x] [AI-Review][MEDIUM] `test_dataloader_calls_notify_integrations` imports `_make_dataset` from `tests.unit.test_dataloader` — cross-test module dependency. Inline the helper or extract to a shared `conftest.py` fixture. [tests/unit/test_integrations.py:331]
- [x] [AI-Review][LOW] Unused `import numpy as np` and `from torq.compose.dataset import Dataset` in `test_dataloader_calls_notify_integrations`. [tests/unit/test_integrations.py:329-330]
- [x] [AI-Review][LOW] `wandb.notify()` passes `dataset.recipe` as raw dict while `mlflow.notify()` serialises with `str()`. Consider aligning both to use the same strategy for consistency and to avoid W&B serialisation edge cases with non-scalar values. [src/torq/integrations/wandb.py:95]
- [x] [AI-Review][LOW] Story title mentions TensorBoard but no implementation exists. Rename story or add a dev note clarifying TensorBoard is deferred beyond R1. [story title]
- [x] [AI-Review-2][HIGH] AC #2 — `mlflow.set_tags()` never called. Only `log_params` was used. Added `mlflow.set_tags()` call with dataset name, episode count, and quality n_scored. [src/torq/integrations/mlflow.py:64-68]
- [x] [AI-Review-2][HIGH] AC #4 — No test for `config.quiet=True` behavior. Added `TestConfigQuiet::test_quiet_mode_produces_no_console_output`. [tests/unit/test_integrations.py]
- [x] [AI-Review-2][MEDIUM] MLflow test only checked key presence, not concrete values. Now asserts min=0.7, max=0.9, mean, n_scored=2, and batch_size from config. [tests/unit/test_integrations.py:233]
- [x] [AI-Review-2][MEDIUM] `wandb.py.__all__` re-exported `_quality_stats` from `_utils`. Removed — only `init` and `notify` belong in wandb's public API. [src/torq/integrations/wandb.py:29]
- [x] [AI-Review-2][MEDIUM] Config dict (batch_size, etc.) passed to `notify()` but silently discarded. Both wandb and mlflow now log config keys with `torq_` prefix. [src/torq/integrations/wandb.py:62, mlflow.py:62]
