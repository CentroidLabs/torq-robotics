# Story 1.4: Gravity Well Infrastructure and Cloud Stub

Status: done

## Story

As a developer,
I want `tq.cloud()` to direct me to the cloud platform,
So that I know how to access collaborative and cloud-scale features.

## Acceptance Criteria

1. **Given** `src/torq/_gravity_well.py` is implemented,
   **When** `_gravity_well(message="...", feature="GW-01")` is called,
   **Then** output matches format: `💡 {message}\n   → https://www.datatorq.ai\n`,
   **And** no network calls are made.

2. **Given** `tq.config.quiet = True`,
   **When** `_gravity_well()` is called,
   **Then** nothing is printed.

3. **Given** `src/torq/cloud.py` is implemented,
   **When** `tq.cloud()` is called,
   **Then** the datatorq.ai URL and waitlist message are printed via `_gravity_well()`,
   **And** no exception is raised.

4. **Given** any cloud-only keyword argument is passed to a local SDK function,
   **When** the function is called,
   **Then** `_gravity_well()` fires and the function continues without raising an unhandled exception.

## Tasks / Subtasks

- [x] Task 1: Create `src/torq/_gravity_well.py` (AC: #1, #2)
  - [x] Implement `_gravity_well(message: str, feature: str)` function
  - [x] Output format: `💡 {message}\n   → https://www.datatorq.ai\n`
  - [x] Check `config.quiet` — if True, return immediately without printing
  - [x] No network calls — print-only in R1
  - [x] Use `print()` for output (this is the ONE place print() is allowed besides tqdm)
- [x] Task 2: Create `src/torq/cloud.py` (AC: #3)
  - [x] Implement `cloud()` function that calls `_gravity_well()` with GW-SDK-04 message
  - [x] Message should mention datatorq.ai waitlist/cloud platform
  - [x] No exception raised, no return value needed
- [x] Task 3: Update `src/torq/__init__.py` to export `cloud` (AC: #3)
  - [x] Add `from torq.cloud import cloud`
  - [x] Add `cloud` to `__all__`
- [x] Task 4: Write unit tests in `tests/unit/test_gravity_well.py` (AC: #1, #2)
  - [x] Test output format matches `💡 {message}\n   → https://www.datatorq.ai\n`
  - [x] Test quiet mode suppresses all output
  - [x] Test no network calls (no socket/urllib imports)
  - [x] Test `_gravity_well.py` imports only from `torq._config`
- [x] Task 5: Write unit tests in `tests/unit/test_cloud.py` (AC: #3)
  - [x] Test `tq.cloud()` produces output via `_gravity_well()`
  - [x] Test `tq.cloud()` does not raise any exception
  - [x] Test quiet mode suppresses `tq.cloud()` output
  - [x] Test `cloud.py` imports only from `torq._gravity_well`
- [x] Task 6: Run full test suite and verify no regressions (AC: #1, #2, #3, #4)
  - [x] All Story 1.1 + 1.2 + 1.3 tests still pass
  - [x] All new tests pass
  - [x] `ruff check src/ && ruff format --check src/` clean

## Dev Notes

### Exact File Contents — `src/torq/_gravity_well.py`

```python
"""Gravity well infrastructure — single owner of all gravity well output.

Gravity wells are non-intrusive prompts that fire after successful SDK operations,
directing users to the datatorq.ai cloud platform. They are print-only in R1
(no network calls). Suppressed when ``tq.config.quiet = True``.

Output format (owned exclusively by this function)::

    💡 {message}
       → https://www.datatorq.ai
"""

from torq._config import config

DATATORQ_URL = "https://www.datatorq.ai"


def _gravity_well(message: str, feature: str) -> None:
    """Print a gravity well prompt if not in quiet mode.

    Args:
        message: The user-facing message to display.
        feature: The gravity well identifier (e.g. "GW-SDK-01").
            Used for tracking/analytics in R2; ignored in R1.
    """
    if config.quiet:
        return
    print(f"💡 {message}")
    print(f"   → {DATATORQ_URL}")
```

**CRITICAL rules:**
- `_gravity_well.py` imports ONLY from `torq._config` (dependency graph rule).
- This is the SINGLE OWNER of the output format. No other module may print `💡 ...` directly.
- `feature` parameter is captured but unused in R1 — it exists for R2 telemetry.
- `print()` is correct here. This is the ONE exception to the "no print()" rule (along with tqdm).
- No `logging` needed — gravity wells are user-facing prompts, not log messages.

### Exact File Contents — `src/torq/cloud.py`

```python
"""Cloud platform stub — directs users to datatorq.ai.

Implements GW-SDK-04: explicit ``tq.cloud()`` call.
"""

from torq._gravity_well import _gravity_well


def cloud() -> None:
    """Print the datatorq.ai cloud platform prompt.

    Directs users to the cloud platform for collaborative and cloud-scale
    features. No network calls are made; this is a local-only prompt.

    Suppressed when ``tq.config.quiet = True``.
    """
    _gravity_well(
        message="Torq Cloud — collaborative datasets, cloud-scale training, "
        "and team workflows. Join the waitlist!",
        feature="GW-SDK-04",
    )
```

**CRITICAL rules:**
- `cloud.py` imports ONLY from `torq._gravity_well` (dependency graph rule).
- No network calls. No return value. No exceptions raised.
- The message text is a suggestion — the exact wording is flexible as long as it mentions datatorq.ai and cloud features.

### `__init__.py` Update

Current `src/torq/__init__.py` (after Story 1.3):
```python
"""Torq — Robot Learning Data Infrastructure SDK."""
from torq._version import __version__
from torq._config import config
from torq.errors import TorqError

__all__ = ["__version__", "config", "TorqError"]
```

Update to:
```python
"""Torq — Robot Learning Data Infrastructure SDK."""
from torq._version import __version__
from torq._config import config
from torq.cloud import cloud
from torq.errors import TorqError

__all__ = ["__version__", "cloud", "config", "TorqError"]
```

**DO NOT** add imports for `ingest`, `quality`, `compose`, `save`, or `load` yet — those modules are still stubs.

### Dependency Rules — Import Graph Position

```
errors.py        ← imports NOTHING from torq (dependency leaf)
types.py         ← imports NOTHING from torq (dependency leaf)
_version.py      ← imports NOTHING from torq (dependency leaf)
_config.py       ← imports errors only
_gravity_well.py ← imports _config only          ← THIS STORY
cloud.py         ← imports _gravity_well only     ← THIS STORY
__init__.py      ← imports _version, _config, cloud, errors
```

`_gravity_well.py` and `cloud.py` form a strict chain: `_config → _gravity_well → cloud`. Neither may import from any other `torq.*` module.

### AC #4 — Cloud-Only Keyword Arguments

AC #4 says: "Given any cloud-only keyword argument is passed to a local SDK function, When the function is called, Then `_gravity_well()` fires and the function continues."

This is implemented in FUTURE stories when those SDK functions are built (e.g., `tq.ingest()`, `tq.compose()`). The pattern will be:

```python
def ingest(path, *, cloud_sync=False, **kwargs):
    if cloud_sync:
        _gravity_well(
            message="Cloud sync requires Torq Cloud. Join the waitlist!",
            feature="GW-SDK-XX",
        )
    # ... continue with local-only logic
```

**This story only builds the infrastructure** (`_gravity_well()` + `cloud()`). The per-function cloud keyword handling is each function's story scope. No test needed for AC #4 in this story — it's a contract for future stories to implement.

### Where Each Gravity Well Fires (Reference for Future Stories)

| ID | Location | Trigger | Story |
|---|---|---|---|
| GW-SDK-01 | `quality/__init__.py` | After `tq.quality.score()` succeeds | Epic 3 |
| GW-SDK-02 | `compose/__init__.py` | After `tq.compose()` returns non-empty | Epic 4 |
| GW-SDK-03 | `serve/torch_loader.py` | DataLoader init on dataset >50GB | Epic 5 |
| GW-SDK-04 | `cloud.py` | `tq.cloud()` explicit call | **THIS STORY** |
| GW-SDK-05 | `compose/__init__.py` | `tq.compose()` returns <5 episodes | Epic 4 |

### Project Structure Notes

#### Files created/modified in THIS story

```
src/torq/
├── _gravity_well.py   ← CREATE (gravity well infrastructure)
├── cloud.py           ← CREATE (tq.cloud() stub)
└── __init__.py        ← MODIFY (add cloud export)

tests/unit/
├── test_gravity_well.py  ← CREATE
└── test_cloud.py         ← CREATE
```

#### Files NOT touched in this story

```
src/torq/_config.py    ← Already complete from Story 1.3
src/torq/errors.py     ← Already complete from Story 1.2
src/torq/types.py      ← Already complete from Story 1.2
pyproject.toml         ← No dependency changes
```

### Previous Story Intelligence (Story 1.2 + 1.3)

**Key learnings from Story 1.2:**
- `__all__` should NOT include `_`-prefixed names (review finding from 1.2).
- `_gravity_well` is private — do NOT add it to `_gravity_well.py`'s `__all__`. Export only `DATATORQ_URL` if anything.
- Actually, `_gravity_well.py` is a private module (underscore-prefixed). It doesn't need `__all__` at all — it's not part of the public API. Only `cloud.py` is public.
- Source file import scanning tests work well (used in test_errors.py and test_types.py). Reuse the pattern for `_gravity_well.py` and `cloud.py`.

**Key learnings from Story 1.3 (expected):**
- `_config.py` exposes a `config` singleton with `config.quiet` property.
- `tq.config.quiet` is the mechanism to suppress output — `_gravity_well()` must check this.
- The `config` import in `_gravity_well.py` is `from torq._config import config` (the singleton instance, not the class).

**Testing pattern from previous stories:**
- Use `capsys` (pytest fixture) to capture print output for gravity well tests.
- Use `unittest.mock.patch` to temporarily set `config.quiet = True`.
- Use source file scanning to verify import graph compliance.

### Architecture Compliance

| Rule | Requirement | How This Story Complies |
|---|---|---|
| Single owner of output format | `_gravity_well()` owns `💡 {msg}\n   → {url}\n` | Only function that prints this format |
| No network calls in R1 | Gravity wells are print-only | No socket/urllib/requests imports |
| Quiet mode suppression | `config.quiet = True` → no output | `_gravity_well()` checks `config.quiet` first |
| No circular imports | `_gravity_well.py` → `_config` only; `cloud.py` → `_gravity_well` only | Strict chain |
| `print()` only here + tqdm | All other output uses `logging` | `_gravity_well()` is the sanctioned exception |
| Google-style docstrings | All public functions | `_gravity_well()` and `cloud()` both documented |
| Gravity wells fire on success only | Never on error paths | Contract for future stories; `cloud()` always succeeds |

### Library/Framework Requirements

No new dependencies. Uses only:
- `torq._config` — for `config.quiet` check
- `torq._gravity_well` — for `cloud.py` to call

No changes to `pyproject.toml`.

### Testing Requirements

#### `tests/unit/test_gravity_well.py`

| Test | Purpose | Technique |
|---|---|---|
| `test_output_format` | Output matches `💡 {msg}\n   → https://www.datatorq.ai\n` | `capsys.readouterr()` |
| `test_quiet_mode_suppresses_output` | No output when `config.quiet = True` | Patch `config.quiet`, check `capsys` empty |
| `test_feature_parameter_accepted` | `feature` param is accepted without error | Call with various feature strings |
| `test_no_network_imports` | `_gravity_well.py` doesn't import socket/urllib/requests | Scan source file |
| `test_gravity_well_imports_only_config` | Only `torq._config` imported | Scan source for `from torq` / `import torq` lines |

#### `tests/unit/test_cloud.py`

| Test | Purpose | Technique |
|---|---|---|
| `test_cloud_produces_output` | `tq.cloud()` prints gravity well format | `capsys.readouterr()` |
| `test_cloud_no_exception` | `tq.cloud()` completes without error | Simple call assertion |
| `test_cloud_quiet_mode` | No output when `config.quiet = True` | Patch + `capsys` |
| `test_cloud_mentions_datatorq` | Output contains `datatorq.ai` | Check captured stdout |
| `test_cloud_imports_only_gravity_well` | Only `torq._gravity_well` imported | Scan source |

All tests are fast (< 1s). No `@pytest.mark.slow` needed.

### References

- Gravity well pattern: [Source: planning-artifacts/architecture.md#Gravity-Well-Pattern]
- Output format: [Source: planning-artifacts/architecture.md#Gravity-Well-Pattern] — `💡 {message}\n   → https://www.datatorq.ai\n`
- Dependency rules: [Source: planning-artifacts/architecture.md#Dependency-Rules] — `_gravity_well.py ← imports _config only`, `cloud.py ← imports _gravity_well only`
- FR19 (GW-SDK-04): [Source: planning-artifacts/epics.md#Requirements-Inventory] — `tq.cloud()` prints waitlist prompt
- FR20 (GW-SDK-06): [Source: planning-artifacts/epics.md#Requirements-Inventory] — unified `_gravity_well()` helper
- Logging vs printing rules: [Source: planning-artifacts/architecture.md#Logging-vs-Printing] — `print()` reserved for gravity wells + tqdm
- Public API surface: [Source: planning-artifacts/architecture.md#Public-API-Surface] — `from torq.cloud import cloud`
- GW trigger map: [Source: planning-artifacts/architecture.md#FR-to-File-Mapping]
- Story 1.2 completed: [Source: implementation-artifacts/1-2-core-types-errors-and-version.md]
- Story 1.3 (dependency): [Source: implementation-artifacts/1-3-configuration-singleton.md]

### Review Follow-ups (AI)

- [x] [AI-Review][Med] Refactor tests to use pytest fixture or monkeypatch for `config.quiet` save/restore instead of manual try/finally pattern (repeated ~10 times across test_gravity_well.py and test_cloud.py)
- [x] [AI-Review][Med] Add explicit `assert result is None` to `test_cloud_no_exception` to lock in the no-return-value contract (test_cloud.py:32-39)
- [x] [AI-Review][Med] Add test documenting `_gravity_well(message=None)` behavior — either add runtime type guard or test that documents the silent `"None"` string output as accepted (test_gravity_well.py)
- [x] [AI-Review][Low] Note for future: multiline messages break gravity well output format — no action needed in R1 (all messages are single-line)
- [x] [AI-Review][Low] Remove unused `import pytest` from test_gravity_well.py (line 10)

## Senior Developer Review (AI)

**Reviewer:** Claude Opus 4.6
**Date:** 2026-03-06
**Outcome:** Approve (with minor action items)

**Summary:** Implementation is correct and faithful to the story spec. All ACs are satisfied, all tasks genuinely complete, dependency graph enforced, tests are real assertions with good coverage. No critical or high-severity issues found. Five minor findings documented as action items.

### Action Items

- [x] [Med] Refactor config.quiet save/restore in tests to use pytest fixtures or monkeypatch
- [x] [Med] Assert `cloud()` returns `None` explicitly in test_cloud_no_exception
- [x] [Med] Add test or guard for `_gravity_well(message=None)` behavior
- [x] [Low] Multiline message format degradation (no action needed R1)
- [x] [Low] Remove unused `import pytest` from test_gravity_well.py

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6

### Debug Log References

No blockers encountered. Straightforward implementation following exact file contents from Dev Notes.

### Completion Notes List

- Created `src/torq/_gravity_well.py` with `_gravity_well(message, feature)` and `DATATORQ_URL` constant. Checks `config.quiet` before printing. No network calls.
- Created `src/torq/cloud.py` with `cloud()` function calling `_gravity_well()` with GW-SDK-04 message mentioning datatorq.ai waitlist.
- Updated `src/torq/__init__.py` to import and export `cloud` in `__all__` (alphabetically ordered: `["__version__", "cloud", "config", "TorqError"]`).
- Wrote 9 tests in `tests/unit/test_gravity_well.py`: output format, URL constant, message content, quiet mode (×2), feature param acceptance, no network imports, import graph compliance.
- Wrote 8 tests in `tests/unit/test_cloud.py`: output produced, no exception, quiet mode, datatorq.ai mention, gravity well format (💡 prefix), import graph compliance, `__all__` presence, callable via `tq.cloud`.
- All 62 tests pass (1 pre-existing skip). `ruff check` and `ruff format --check` clean.
- Dependency chain enforced: `_config → _gravity_well → cloud → __init__`. Verified via AST import scanning tests.

### File List

- `src/torq/_gravity_well.py` — CREATED
- `src/torq/cloud.py` — CREATED
- `src/torq/__init__.py` — MODIFIED (added `cloud` import and export)
- `tests/unit/test_gravity_well.py` — CREATED
- `tests/unit/test_cloud.py` — CREATED

## Change Log

- 2026-03-06: Implemented Story 1.4 — Gravity Well Infrastructure and Cloud Stub. Created `_gravity_well.py` and `cloud.py`, updated `__init__.py`, wrote 17 new tests. All 62 tests pass, ruff clean.
- 2026-03-06: Code review (Claude Opus 4.6) — Approved with 3 Med + 2 Low action items. No critical/high issues. All ACs verified, all tasks genuinely complete. 5 review follow-up tasks added.
