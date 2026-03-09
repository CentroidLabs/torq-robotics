# Story 3.5: Custom Metric Registry and Kinematic Feasibility Stub

Status: review

## Story

As a robotics researcher,
I want to register my own quality metrics alongside the built-in ones,
so that I can extend scoring with domain-specific criteria without modifying SDK internals.

## Acceptance Criteria

1. **Given** a callable `fn(episode) -> float` returning [0.0, 1.0], **When** `tq.quality.register('grip_force', fn, weight=0.2)` is called, **Then** existing weights are rescaled proportionally so all sum to 1.0. Example: `{s:0.40, c:0.35, co:0.25}` + new at 0.20 → `{s:0.32, c:0.28, co:0.20, grip_force:0.20}`.

2. **Given** a metric name already registered, **When** `tq.quality.register('grip_force', new_fn, weight=0.15)` is called again, **Then** the existing metric is overwritten **And** a `UserWarning` is emitted (not an error).

3. **Given** `tq.quality.register()` has been called, **When** `tq.config.reset_quality_weights()` is called, **Then** `tq.config.quality_weights` returns `DEFAULT_QUALITY_WEIGHTS` **And** all custom metrics are removed from the registry.

4. **Given** `feasibility.score(episode)` in the R1 stub, **When** called on any valid episode, **Then** `1.0` is always returned.

## Tasks / Subtasks

- [x] Task 1: Create `src/torq/quality/feasibility.py` — R1 stub returning 1.0 always (AC: #4)
  - [x] Add module docstring explaining this is an R1 stub; full URDF validation deferred to R2
  - [x] Implement `score(episode: Episode) -> float` that unconditionally returns `1.0`
  - [x] Add `__all__ = ["score"]`
  - [x] Export `feasibility` submodule from `src/torq/quality/__init__.py` (so `tq.quality.feasibility.score()` works)

- [x] Task 2: Fix `tq.config.reset_quality_weights()` to also clear the registry (AC: #3)
  - [x] In `src/torq/_config.py`, import `_registry` from `torq.quality.registry` — CAUTION: check for circular imports
  - [x] If circular import is unavoidable, use a lazy import inside the method body
  - [x] In `reset_quality_weights()`, after resetting `_quality_weights`, call `_registry.reset()`
  - [x] Alternatively: add a module-level `reset()` call in `torq.quality.__init__` that orchestrates both — whichever avoids circular imports

- [x] Task 3: Write unit tests `tests/unit/test_quality_registry.py` (AC: #1, #2, #3)
  - [x] Test weight rescaling: register one metric at 0.20, verify built-in weights scaled by 0.80
  - [x] Test exact rescaled values: smoothness=0.32, consistency=0.28, completeness=0.20, grip=0.20
  - [x] Test two custom metrics: register at 0.20 then 0.10, verify all four weights sum to 1.0
  - [x] Test re-registration emits `UserWarning` (AC #2)
  - [x] Test re-registration replaces scorer without double-scaling weights
  - [x] Test built-in name collision raises `TorqQualityError` (names: smoothness, consistency, completeness)
  - [x] Test weight=0.0 raises `TorqQualityError`
  - [x] Test weight=1.0 raises `TorqQualityError`
  - [x] Test weight outside (0.0, 1.0) raises `TorqQualityError`
  - [x] Test `get_metrics()` returns all metrics and their weights
  - [x] Test `reset()` restores DEFAULT_QUALITY_WEIGHTS and clears custom metrics
  - [x] Test `reset_quality_weights()` on config also clears registry (AC #3)
  - [x] Test `has_custom_metrics()` returns False initially, True after register, False after reset
  - [x] Always call `tq.quality.reset()` in test teardown (use `yield` fixture) to avoid cross-test pollution

- [x] Task 4: Write unit tests `tests/unit/test_quality_feasibility.py` (AC: #4)
  - [x] Test `tq.quality.feasibility.score(episode)` returns `1.0` for any valid episode
  - [x] Test return type is `float` (not `int`, not `None`)
  - [x] Test returns `1.0` for episode with < 10 timesteps (unlike other scorers, no None return)
  - [x] Test returns `1.0` for episode with NaN in actions
  - [x] Test returns `1.0` for empty/minimal episode

### Review Follow-ups (AI)

- [x] [AI-Review][MEDIUM] `_quality_reset_hooks` is exported in `__all__` — internal mutable list should not be publicly exported [`src/torq/_config.py:23`]
- [x] [AI-Review][MEDIUM] No test for re-registration with a *different* weight (e.g. 0.20 → 0.15) — AC #2 scenario with weight reversal + re-scale at a new weight is untested [`tests/unit/test_quality_registry.py:78-86`]
- [x] [AI-Review][MEDIUM] `register()` never validates that `fn` is callable — passing a non-callable succeeds silently, only failing at scoring time [`src/torq/quality/registry.py:61`]
- [x] [AI-Review][LOW] `_quality_reset_hooks` has bare `list` type — should be `list[Callable[[], None]]` [`src/torq/_config.py:27`]
- [x] [AI-Review][LOW] `test_builtin_name_collision_raises_quality_error` loops 3 names in one test — should use `@pytest.mark.parametrize` for clear failure messages [`tests/unit/test_quality_registry.py:92-96`]

## Dev Notes

### What Is Already Implemented — DO NOT REWRITE

`src/torq/quality/registry.py` is **fully implemented**. It contains:
- `_Registry` class with `register()`, `get_metrics()`, `get_built_in_weights()`, `get_custom_scorers()`, `has_custom_metrics()`, `reset()`
- Module-level singleton `_registry = _Registry()`
- Module-level functions `register()`, `get_metrics()`, `reset()` that delegate to `_registry`
- Weight rescaling logic (including re-registration reversal)
- `UserWarning` on re-registration via `warnings.warn(..., UserWarning, stacklevel=3)`

`src/torq/quality/__init__.py` already imports and re-exports `register`, `get_metrics`, `reset` from `registry.py` and adds them to `__all__`.

**The only code to write for Tasks 1 and 2 is:**
1. `feasibility.py` — a tiny stub (~15 lines)
2. Patching `reset_quality_weights()` in `_config.py` to also call `_registry.reset()`

### Feasibility Stub Implementation

```python
# src/torq/quality/feasibility.py
"""Kinematic feasibility scorer — R1 stub.

In R1, this always returns 1.0.  Full URDF-based joint limit and collision
checking is deferred to R2 (QM-06).

Usage::

    from torq.quality import feasibility
    score = feasibility.score(episode)  # always 1.0 in R1
"""
from __future__ import annotations

from torq.episode import Episode

__all__ = ["score"]


def score(episode: Episode) -> float:
    """Return the kinematic feasibility score for an episode.

    R1 stub — always returns 1.0.  Full URDF-based validation (joint limits,
    collision detection) is planned for R2.

    Args:
        episode: The Episode to score (not used in R1).

    Returns:
        1.0 always.
    """
    return 1.0
```

### Exposing `tq.quality.feasibility` as a Submodule

After creating the file, expose `feasibility` in `src/torq/quality/__init__.py`:

```python
from torq.quality import feasibility  # makes tq.quality.feasibility.score() work
```

Add `"feasibility"` to `__all__`.

### Circular Import Risk for Task 2

`_config.py` is imported by `torq.quality.registry` (via `DEFAULT_QUALITY_WEIGHTS`). If `_config.py` imports from `torq.quality.registry`, this creates a circular import.

**Solution — lazy import inside the method body:**

```python
def reset_quality_weights(self) -> None:
    """Restore quality weights to DEFAULT_QUALITY_WEIGHTS and clear custom metrics."""
    self._quality_weights = dict(DEFAULT_QUALITY_WEIGHTS)
    # Lazy import to avoid circular dependency (_config ← registry ← _config)
    from torq.quality.registry import _registry  # noqa: PLC0415
    _registry.reset()
```

This is safe because:
- `_config.py` is loaded early (imported by many modules)
- By the time `reset_quality_weights()` is _called_, `torq.quality.registry` is always loaded
- The lazy import is a standard Python pattern for circular import resolution

### Registry Weight Rescaling Contract (for test verification)

Registering `grip_force` at weight=0.20:
```
scale_factor = 1.0 - 0.20 = 0.80
smoothness:   0.40 * 0.80 = 0.32
consistency:  0.35 * 0.80 = 0.28
completeness: 0.25 * 0.80 = 0.20
grip_force:   0.20        (new, at specified weight)
sum = 0.32 + 0.28 + 0.20 + 0.20 = 1.00 ✓
```

Then registering a second `torque_quality` at weight=0.10:
```
scale_factor = 1.0 - 0.10 = 0.90
smoothness:     0.32 * 0.90 = 0.288
consistency:    0.28 * 0.90 = 0.252
completeness:   0.20 * 0.90 = 0.180
grip_force:     0.20 * 0.90 = 0.180
torque_quality: 0.10
sum = 0.288 + 0.252 + 0.180 + 0.180 + 0.10 = 1.00 ✓
```

Note: Due to floating-point rounding in registry.py (`round(..., 8)`), assert with `pytest.approx` tolerance.

### Test Isolation — CRITICAL

The registry is a module-level singleton. Tests that register metrics MUST reset afterward or they will contaminate subsequent tests. Use a pytest fixture:

```python
@pytest.fixture(autouse=True)
def reset_registry():
    yield
    tq.quality.reset()  # always runs after each test
```

### File Structure Requirements

- `src/torq/quality/feasibility.py` — NEW file
- `src/torq/quality/__init__.py` — ADD `from torq.quality import feasibility` + add to `__all__`
- `src/torq/_config.py` — MODIFY `reset_quality_weights()` to add lazy import + `_registry.reset()`
- `tests/unit/test_quality_registry.py` — NEW file
- `tests/unit/test_quality_feasibility.py` — NEW file

### References

- [Source: epics.md#Epic 3 > Story 3.5] — ACs and weight rescaling example
- [Source: architecture.md#Custom metric plugins (R1)] — Rescaling contract, idempotent re-registration
- [Source: architecture.md#FR Category > QM-06] — "quality/completeness.py | returns 1.0 in R1" (but implementation should be separate `feasibility.py`)
- [Source: src/torq/quality/registry.py] — Already fully implemented; read before touching anything
- [Source: src/torq/_config.py:87-89] — `reset_quality_weights()` currently only resets `_quality_weights`, not registry

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6

### Debug Log References

- Lazy import approach for `_config.py` was rejected: existing `test_config_module_imports` uses AST analysis and catches all `torq.*` imports including those inside method bodies. Used hook-list pattern instead.

### Completion Notes List

- Task 1: Created `src/torq/quality/feasibility.py` — R1 stub, always returns `1.0`. Exported `feasibility` submodule from `src/torq/quality/__init__.py`; added to `__all__`.
- Task 2: Used a `_quality_reset_hooks` list in `_config.py` (avoids circular import, passes `test_config_module_imports`). `registry.py` appends `_registry.reset` to the list at module load time. `reset_quality_weights()` iterates hooks. All ACs for #3 verified.
- Task 3: 20 registry tests written and passing in `tests/unit/test_quality_registry.py`. Autouse fixture resets registry after each test.
- Task 4: 6 feasibility tests written and passing in `tests/unit/test_quality_feasibility.py`.
- Full regression suite: 303 passed, 0 failures.

### File List

- `src/torq/quality/feasibility.py` — NEW
- `src/torq/quality/__init__.py` — MODIFIED (added `feasibility` import + `__all__` entry)
- `src/torq/_config.py` — MODIFIED (added `_quality_reset_hooks`, updated `reset_quality_weights()`)
- `src/torq/quality/registry.py` — MODIFIED (imports `_quality_reset_hooks`, appends reset hook)
- `tests/unit/test_quality_registry.py` — NEW (20 tests)
- `tests/unit/test_quality_feasibility.py` — NEW (6 tests)

## Change Log

- 2026-03-09: Implemented Story 3.5 — feasibility stub, reset hook pattern for config/registry sync, 26 new tests (20 registry + 6 feasibility). All 303 tests passing.
- 2026-03-09: Addressed 5 AI review findings — removed `_quality_reset_hooks` from `__all__`, typed hook list as `list[Callable[[], None]]`, added callable validation to `register()`, added re-registration-with-different-weight test, converted builtin collision test to `@pytest.mark.parametrize`. 308 tests passing.
