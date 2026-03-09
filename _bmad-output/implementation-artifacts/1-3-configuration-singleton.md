# Story 1.3: Configuration Singleton

Status: done

## Story

As a developer,
I want to configure SDK-wide behaviour (quiet mode, quality weights),
So that I can suppress output in CI and customise scoring without changing my code.

## Acceptance Criteria

1. **Given** `src/torq/_config.py` implementing the Config singleton,
   **When** `tq.config.quiet = True` is set,
   **Then** all subsequent gravity well prompts and tqdm bars are suppressed.

2. **Given** the `TORQ_QUIET=1` environment variable is set before import,
   **When** `import torq as tq` is executed,
   **Then** `tq.config.quiet` is `True` automatically.

3. **Given** `tq.config.quality_weights` is set to a custom dict,
   **When** the weights do NOT sum to 1.0 +/- 0.001,
   **Then** `TorqConfigError` is raised with a message stating the actual sum and the correction needed.

4. **Given** `tq.config.reset_quality_weights()` is called,
   **When** `tq.config.quality_weights` is read,
   **Then** it returns `DEFAULT_QUALITY_WEIGHTS` = `{smoothness: 0.40, consistency: 0.35, completeness: 0.25}`.

## Tasks / Subtasks

- [x] Task 1: Create `src/torq/_config.py` with Config singleton class (AC: #1, #2, #3, #4)
  - [x] Define `DEFAULT_QUALITY_WEIGHTS` module constant
  - [x] Implement `Config` class with `quiet` property (default: reads `TORQ_QUIET` env var)
  - [x] Implement `quality_weights` property with setter that validates sum to 1.0 +/- 0.001
  - [x] Implement `reset_quality_weights()` method
  - [x] Create module-level `config` singleton instance
  - [x] Add `__all__` export list
- [x] Task 2: Update `src/torq/__init__.py` to export `config` (AC: #1, #2)
  - [x] Add `from torq._config import config`
  - [x] Add `config` to `__all__`
- [x] Task 3: Write unit tests in `tests/unit/test_config.py` (AC: #1, #2, #3, #4)
  - [x] Test `config.quiet` defaults to `False`
  - [x] Test `config.quiet = True` sets quiet mode
  - [x] Test `TORQ_QUIET=1` env var sets quiet to `True`
  - [x] Test `TORQ_QUIET=0` or unset means quiet is `False`
  - [x] Test `config.quality_weights` returns `DEFAULT_QUALITY_WEIGHTS` by default
  - [x] Test setting valid custom weights (sum = 1.0) succeeds
  - [x] Test setting invalid weights (sum != 1.0) raises `TorqConfigError`
  - [x] Test error message contains actual sum and correction hint
  - [x] Test `config.reset_quality_weights()` restores defaults
  - [x] Test `config` is a singleton (same instance across imports)
  - [x] Test `_config.py` imports only from `torq.errors` (dependency rule)
- [x] Task 4: Run full test suite and verify no regressions (AC: #1, #2, #3, #4)
  - [x] All Story 1.1 + 1.2 tests still pass
  - [x] All new config tests pass
  - [x] `ruff check src/ && ruff format --check src/` clean

### Review Follow-ups (AI)
- [x] [AI-Review][MEDIUM] M1: Non-numeric weight values cause bare `TypeError` instead of `TorqConfigError`. Wrap `sum(weights.values())` in try/except to catch `TypeError` and re-raise as `TorqConfigError` with helpful message. [`src/torq/_config.py:55`]
- [x] [AI-Review][MEDIUM] M2: No validation of weight dimension keys — arbitrary keys like `{"foo": 0.5, "bar": 0.5}` silently accepted. Validate keys match `DEFAULT_QUALITY_WEIGHTS.keys()` or warn on unknown keys. [`src/torq/_config.py:54`]
- [x] [AI-Review][MEDIUM] M3: Missing test for non-numeric weight values. Add `test_set_non_numeric_weights_raises_config_error()` after fixing M1. [`tests/unit/test_config.py`]
- [x] [AI-Review][LOW] L1: No `__repr__` on Config class — `repr(tq.config)` returns unhelpful default. Add `__repr__` showing quiet status and current weights. [`src/torq/_config.py:25`]
- [x] [AI-Review][LOW] L2: `TORQ_QUIET` env var only recognizes `"1"` — `"true"`/`"yes"` silently ignored. Per-spec but potentially confusing. Document or broaden acceptance. [`src/torq/_config.py:36`]
- [x] [AI-Review][LOW] L3: `__all__` placement is between class and singleton — convention is near top of module after imports. [`src/torq/_config.py:69`]
- [x] [AI-Review][LOW] L4: Missing boundary-rejection test for just-outside-tolerance (e.g., sum=0.998). Add test to strengthen edge case coverage. [`tests/unit/test_config.py`]

## Dev Notes

### Exact File Contents — `src/torq/_config.py`

```python
"""Torq SDK configuration singleton.

Provides SDK-wide settings accessible via ``tq.config``.
Reads ``TORQ_QUIET`` environment variable on import.

Usage::

    import torq as tq
    tq.config.quiet = True              # suppress gravity wells and tqdm
    tq.config.quality_weights = {...}   # override scoring weights
    tq.config.reset_quality_weights()   # restore defaults
"""

import os

from torq.errors import TorqConfigError

DEFAULT_QUALITY_WEIGHTS: dict[str, float] = {
    "smoothness": 0.40,
    "consistency": 0.35,
    "completeness": 0.25,
}


class Config:
    """SDK-wide configuration singleton.

    Attributes:
        quiet: When True, suppresses gravity well prompts and tqdm progress bars.
            Initialised from ``TORQ_QUIET`` environment variable (``1`` = quiet).
        quality_weights: Dict mapping dimension names to float weights.
            Must sum to 1.0 +/- 0.001. Set via property; validated on assignment.
    """

    def __init__(self) -> None:
        self._quiet: bool = os.environ.get("TORQ_QUIET", "0") == "1"
        self._quality_weights: dict[str, float] = dict(DEFAULT_QUALITY_WEIGHTS)

    @property
    def quiet(self) -> bool:
        """Whether to suppress gravity wells and tqdm progress bars."""
        return self._quiet

    @quiet.setter
    def quiet(self, value: bool) -> None:
        self._quiet = bool(value)

    @property
    def quality_weights(self) -> dict[str, float]:
        """Current quality scoring weights. Must sum to 1.0 +/- 0.001."""
        return dict(self._quality_weights)

    @quality_weights.setter
    def quality_weights(self, weights: dict[str, float]) -> None:
        total = sum(weights.values())
        if abs(total - 1.0) > 0.001:
            raise TorqConfigError(
                f"Quality weights must sum to 1.0 (+/- 0.001), got {total:.4f}. "
                f"Adjust weights so they sum to 1.0. "
                f"Current weights: {weights}"
            )
        self._quality_weights = dict(weights)

    def reset_quality_weights(self) -> None:
        """Restore quality weights to DEFAULT_QUALITY_WEIGHTS."""
        self._quality_weights = dict(DEFAULT_QUALITY_WEIGHTS)


config = Config()
```

### CRITICAL Implementation Notes

1. **`_config.py` imports ONLY from `torq.errors`** — this is mandated by the dependency graph:
   ```
   _config.py  ← imports errors only
   ```
   Do NOT import from `torq.types`, `torq._version`, or any other `torq.*` module.

2. **The `config` object is a module-level singleton** — there is ONE instance, created at import time. All code accesses the same instance via `tq.config`.

3. **`quality_weights` getter returns a COPY** — `dict(self._quality_weights)` prevents external mutation of internal state. The setter validates and stores a copy too.

4. **`TORQ_QUIET` is read ONCE at import time** — not on every `.quiet` access. This matches the AC: "Given the TORQ_QUIET=1 environment variable is set **before import**". If set after import, use `tq.config.quiet = True` instead.

5. **Error message format** — must follow [what failed] + [why] + [what to try]:
   - What: "Quality weights must sum to 1.0"
   - Why: "got {total}"
   - What to try: "Adjust weights so they sum to 1.0"

6. **Naming convention** — `DEFAULT_QUALITY_WEIGHTS` is `UPPER_SNAKE_CASE` per architecture naming rules for module constants.

7. **`QualityConfigError` in architecture = `TorqConfigError`** — the architecture mentions "raises QualityConfigError" but the error hierarchy has `TorqConfigError`. Use `TorqConfigError` — it is the correct class from `errors.py`.

### `__init__.py` Update

Current `src/torq/__init__.py` (after Story 1.2):
```python
"""Torq — Robot Learning Data Infrastructure SDK."""
from torq._version import __version__
from torq.errors import TorqError

__all__ = ["__version__", "TorqError"]
```

Update to:
```python
"""Torq — Robot Learning Data Infrastructure SDK."""
from torq._version import __version__
from torq._config import config
from torq.errors import TorqError

__all__ = ["__version__", "config", "TorqError"]
```

This enables `tq.config.quiet = True` and `tq.config.quality_weights = {...}`.

**DO NOT** add imports for `ingest`, `quality`, `compose`, `save`, `load`, or `cloud` yet — those modules are still stubs.

### Dependency Rules — Import Graph Position

```
errors.py    ← imports NOTHING from torq (dependency leaf)
types.py     ← imports NOTHING from torq (dependency leaf)
_version.py  ← imports NOTHING from torq (dependency leaf)
_config.py   ← imports errors ONLY          ← THIS STORY
__init__.py  ← imports _version, errors, _config
```

`_config.py` sits one level above the leaf layer. It may import from `errors.py` only. Story 1.4 will add `_gravity_well.py` which imports from `_config` only.

### Project Structure Notes

#### Files created/modified in THIS story

```
src/torq/
├── _config.py         ← CREATE (Config singleton + DEFAULT_QUALITY_WEIGHTS)
└── __init__.py        ← MODIFY (add config export)

tests/unit/
└── test_config.py     ← CREATE (config singleton tests)
```

#### Files NOT touched in this story

```
src/torq/errors.py     ← Already complete from Story 1.2
src/torq/types.py      ← Already complete from Story 1.2
src/torq/_version.py   ← Already correct from Story 1.1
pyproject.toml         ← No dependency changes needed
```

### Previous Story Intelligence (Story 1.2)

**Key learnings from Story 1.2:**
- `errors.py` is complete with 7-class hierarchy and `_require_torch()`. `__all__` does NOT include `_require_torch` (private naming convention).
- `types.py` removed unused `from pathlib import Path` import (ruff F401 fix). If `_config.py` doesn't use `Path`, don't import it.
- `__init__.py` currently exports `__version__` and `TorqError`.
- `conftest.py` uses `@pytest.fixture(scope="session")` — not module constants (corrected from Story 1.1 notes).
- 20 tests pass, 1 skipped (episode.py not yet created).
- ruff config: line-length=100, select E/F/I/W.
- Python >= 3.10 — use `dict[str, float]` not `Dict[str, float]`.

**Story 1.2 review follow-ups carried forward:**
- `_require_torch()` error message omits [what failed] per AC format — acknowledged as spec issue. For `_config.py`, ensure TorqConfigError messages include all three parts.

### Architecture Compliance

| Rule | Requirement | How This Story Complies |
|---|---|---|
| No circular imports | `_config.py` imports from `errors` only | Dependency graph enforced |
| Helpful error messages | [what] + [why] + [what to try] format | TorqConfigError message includes all three parts |
| Module constants UPPER_SNAKE | `DEFAULT_QUALITY_WEIGHTS` | Follows naming convention |
| `tq.config.quiet` for tqdm | `disable=tq.config.quiet` on all tqdm calls | Config.quiet property enables this (consumers in future stories) |
| TORQ_QUIET env var | Read on import, sets `config.quiet` | `os.environ.get("TORQ_QUIET", "0") == "1"` |
| Config file stub | `~/.torq/config.toml` — R2 feature | NOT implemented in this story (R2 scope) |
| Google-style docstrings | All public classes and functions | Config class + properties + reset method |
| Type hints on public API | All public functions typed | Properties and reset_quality_weights() typed |

### Library/Framework Requirements

No new dependencies. Uses only:
- `os` (stdlib) — for `os.environ.get()`
- `torq.errors` — for `TorqConfigError`

No changes to `pyproject.toml`.

### Testing Requirements

#### `tests/unit/test_config.py`

| Test | Purpose | Notes |
|---|---|---|
| `test_quiet_defaults_to_false` | `config.quiet` is `False` when `TORQ_QUIET` not set | Use `unittest.mock.patch.dict(os.environ, {}, clear=True)` |
| `test_quiet_settable` | `config.quiet = True` then read returns `True` | |
| `test_torq_quiet_env_var` | `TORQ_QUIET=1` → `config.quiet == True` | Must create fresh Config instance with env var set |
| `test_torq_quiet_env_var_zero` | `TORQ_QUIET=0` → `config.quiet == False` | |
| `test_default_quality_weights` | Default weights match `DEFAULT_QUALITY_WEIGHTS` | |
| `test_set_valid_quality_weights` | Setting weights that sum to 1.0 succeeds | |
| `test_set_weights_near_tolerance` | Weights summing to 0.999 or 1.001 accepted | Edge case: tolerance boundary |
| `test_set_invalid_quality_weights` | Weights not summing to 1.0 raises `TorqConfigError` | |
| `test_invalid_weights_error_message` | Error message contains actual sum and correction hint | |
| `test_reset_quality_weights` | After custom set + reset, weights equal defaults | |
| `test_quality_weights_returns_copy` | Modifying returned dict doesn't change internal state | |
| `test_config_singleton` | `from torq._config import config` returns same object | |
| `test_config_module_imports` | `_config.py` source only imports from `torq.errors` | Scan source file for import lines |

**Important testing note:** When testing `TORQ_QUIET` env var, you must create a **fresh `Config()` instance** because the module-level `config` singleton reads the env var only at import time. Use:
```python
with patch.dict(os.environ, {"TORQ_QUIET": "1"}):
    fresh = Config()
    assert fresh.quiet is True
```

Do NOT rely on re-importing the module — Python caches modules in `sys.modules`.

All tests are fast (< 1s). No `@pytest.mark.slow` needed.

### References

- Configuration architecture: [Source: planning-artifacts/architecture.md#Configuration-Architecture]
- Default quality weights: [Source: planning-artifacts/architecture.md#Quality-Scoring-Architecture] — `DEFAULT_QUALITY_WEIGHTS = {smoothness: 0.40, consistency: 0.35, completeness: 0.25}`
- Weight validation: [Source: planning-artifacts/architecture.md#Quality-Scoring-Architecture] — "weights must sum to 1.0 +/- 0.001; raises QualityConfigError"
- Dependency rules: [Source: planning-artifacts/architecture.md#Dependency-Rules] — `_config.py ← imports errors only`
- Naming conventions: [Source: planning-artifacts/architecture.md#Naming-Conventions] — UPPER_SNAKE_CASE for module constants
- tqdm quiet hook: [Source: planning-artifacts/architecture.md#Progress-Bar-Pattern] — `disable=tq.config.quiet`
- Public API surface: [Source: planning-artifacts/architecture.md#Public-API-Surface] — `from torq.config import config`
- Story 1.2 completed: [Source: implementation-artifacts/1-2-core-types-errors-and-version.md]

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6

### Debug Log References

- Floating point precision issue: `0.334+0.334+0.333` marginally exceeds `1.001` in float64, triggering false rejection. Fixed test to use `{"a": 0.5, "b": 0.5005}` (sum ≈ 1.0005) for clear within-tolerance boundary testing.
- `test_config_module_imports` was text-scanning the raw source, which found `import torq as tq` inside the module docstring. Fixed to use `ast.parse()` for real import AST node detection only.

### Completion Notes List

- Implemented `src/torq/_config.py` with `Config` singleton class following exact spec from Dev Notes
- `Config.quiet` reads `TORQ_QUIET` env var once at import time (not per-access)
- `Config.quality_weights` setter validates sum within ±0.001 tolerance; getter returns a copy
- `Config.reset_quality_weights()` restores `DEFAULT_QUALITY_WEIGHTS = {smoothness: 0.40, consistency: 0.35, completeness: 0.25}`
- Updated `src/torq/__init__.py` to export `config` singleton
- 17 new tests in `tests/unit/test_config.py` — all pass
- Full suite: 37 passed, 1 skipped (episode stub) — zero regressions
- `ruff check` and `ruff format --check` clean
- ✅ Resolved review finding [MEDIUM]: M1 — wrapped `sum()` in try/except TypeError → re-raises as TorqConfigError
- ✅ Resolved review finding [MEDIUM]: M2 — unknown dimension keys now raise TorqConfigError; valid keys are smoothness/consistency/completeness
- ✅ Resolved review finding [MEDIUM]: M3 — added `test_set_non_numeric_weights_raises_config_error` and `test_set_none_weight_raises_config_error`
- ✅ Resolved review finding [LOW]: L1 — added `__repr__` showing quiet status and current weights
- ✅ Resolved review finding [LOW]: L2 — documented TORQ_QUIET "1"-only behaviour in module docstring
- ✅ Resolved review finding [LOW]: L3 — moved `__all__` above class definition (after imports)
- ✅ Resolved review finding [LOW]: L4 — added `test_weights_just_below_tolerance_rejected` and `test_weights_just_above_tolerance_rejected`
- Post-review suite: 45 passed, 1 skipped — zero regressions

### File List

- `src/torq/_config.py` — CREATED
- `src/torq/__init__.py` — MODIFIED (added config export)
- `tests/unit/test_config.py` — CREATED

### Change Log

- 2026-03-06: Implemented Story 1.3 — Config singleton with quiet mode, quality weight validation, env var support, and reset. 17 tests added, all passing.
- 2026-03-06: Code review (AI) — 0 Critical, 3 Medium, 4 Low findings. All ACs implemented, all tasks verified done. 7 action items created. Status → in-progress pending M1-M3 fixes.
- 2026-03-06: Addressed all 7 review findings (M1-M3, L1-L4). 8 new tests added (25 total in test_config.py). Full suite 45 passed, 1 skipped. Status → review.
- 2026-03-06: Re-review (AI) — All 7 fixes verified correct. No new issues. 45 passed, 1 skipped. Status → done.
