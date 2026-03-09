# Story 3.6: Gravity Well After Quality Scoring

Status: review

## Story

As a robotics researcher,
I want a prompt directing me to the cloud platform after scoring completes,
so that I know where to compare my quality scores against the community benchmark.

## Acceptance Criteria

1. **Given** `tq.quality.score(episodes)` completes successfully, **When** the function returns, **Then** `_gravity_well()` fires with a message including the computed average score and the `datatorq.ai` URL.

2. **Given** `tq.config.quiet = True`, **When** `tq.quality.score(episodes)` completes, **Then** no gravity well output is printed.

3. **Given** `tq.quality.score(episodes)` fails with an exception, **When** the exception is raised, **Then** `_gravity_well()` does NOT fire (gravity wells fire only on success).

## Tasks / Subtasks

- [x] Task 1: Write integration tests `tests/unit/test_quality_score_gravity_well.py` (AC: #1, #2, #3)
  - [x] Test gravity well fires after successful batch scoring: score 3+ episodes, verify stdout contains average score and `datatorq.ai`
  - [x] Test gravity well message contains computed average: deterministic scores → verify formatted avg in output (e.g., "0.80" for avg 0.8)
  - [x] Test gravity well contains the URL `https://www.datatorq.ai`
  - [x] Test quiet mode suppresses gravity well: `tq.config.quiet = True` → stdout is empty after `score()`
  - [x] Test gravity well does NOT fire when scorer raises: mock a scorer to throw `TorqQualityError`, verify `_gravity_well` not called and no datatorq output
  - [x] Test gravity well does NOT fire when all episodes are unscored (< 10 timesteps → all None): no output expected
  - [x] Test gravity well fires for single-episode success: `score(episode)` (not a list) → gravity well fires
  - [x] Test gravity well fires for single-episode, quiet=False: confirm output is present
  - [x] Test gravity well NOT fired on empty list: `score([])` → no output

### Review Follow-ups (AI)

- [x] [AI-Review][MEDIUM] `ensure_not_quiet` fixture and quiet-mode tests access private `_quiet` attribute via monkeypatch — should use public `config.quiet = True/False` setter [`tests/unit/test_quality_score_gravity_well.py:27,84,97`]
- [x] [AI-Review][LOW] `test_quiet_mode_suppresses_gravity_well` has redundant assertion — `captured.out == ""` already implies `"datatorq.ai" not in captured.out` [`tests/unit/test_quality_score_gravity_well.py:91-92`]
- [x] [AI-Review][LOW] No test with varying scores across episodes to verify average computation — all tests use uniform scores; a test with e.g. 0.6 and 1.0 episodes verifying "0.80" would catch averaging bugs [`tests/unit/test_quality_score_gravity_well.py`]

## Dev Notes

### What Is Already Implemented — NO CODE TO WRITE EXCEPT TESTS

The gravity well integration in `tq.quality.score()` is **fully implemented** in `src/torq/quality/__init__.py`:

```python
# Lines ~250-255 in src/torq/quality/__init__.py
# ── Gravity well (non-empty result only) ──────────────────────────────────
if scored_overalls:
    avg_score = sum(scored_overalls) / len(scored_overalls)
    _gravity_well(f"Average quality: {avg_score:.2f}", "GW-SDK-01")
```

How each AC is already satisfied by the implementation:
- **AC1**: The `if scored_overalls:` block computes average and calls `_gravity_well()` after the loop — only reachable on success ✓
- **AC2**: `_gravity_well()` internally checks `config.quiet` and returns immediately without printing ✓ (see `src/torq/_gravity_well.py:26`)
- **AC3**: If `score()` raises an exception mid-loop, execution never reaches the `_gravity_well()` call ✓

**This story's only deliverable is the test file.**

### Test Strategy

Use `capsys` to capture stdout rather than mocking `_gravity_well` directly — this tests the full end-to-end behavior including quiet suppression.

For AC3 (exception case), mock one of the internal scorers to raise:

```python
from unittest.mock import patch
from torq.errors import TorqQualityError

def test_gravity_well_does_not_fire_on_exception(make_episode, capsys, monkeypatch):
    monkeypatch.setattr("torq.quality._smoothness_score", lambda ep: (_ for _ in ()).throw(TorqQualityError("bad")))
    # OR simpler:
    with patch("torq.quality._smoothness_score", side_effect=TorqQualityError("injected")):
        with pytest.raises(TorqQualityError):
            tq.quality.score([make_episode()])
    captured = capsys.readouterr()
    assert "datatorq.ai" not in captured.out
```

### Creating Deterministic Episodes for Score Verification

To assert the exact average score in the gravity well message, build episodes with predictable scores. The easiest approach is to patch the built-in scorers:

```python
with patch("torq.quality._smoothness_score", return_value=0.8), \
     patch("torq.quality._consistency_score", return_value=0.8), \
     patch("torq.quality._completeness_score", return_value=0.8):
    tq.quality.score(episodes)
# expected overall = 0.8*0.40 + 0.8*0.35 + 0.8*0.25 = 0.8
# expected avg = 0.80
# expected message contains "0.80"
```

Alternatively, use real episodes from conftest — but patching is more deterministic.

### Quiet Mode Test Pattern

```python
def test_quiet_suppresses_gravity_well(make_episode, capsys, monkeypatch):
    monkeypatch.setattr(config, "quiet", True)  # or monkeypatch.setattr(config, "_quiet", True)
    tq.quality.score([make_episode()])
    captured = capsys.readouterr()
    assert "datatorq.ai" not in captured.out
    assert captured.out == ""
```

### Gravity Well Output Format (from `src/torq/_gravity_well.py`)

```
💡 Average quality: {avg:.2f}
   → https://www.datatorq.ai
```

### Existing Test Coverage to Be Aware Of

`tests/unit/test_gravity_well.py` — tests `_gravity_well()` in isolation (output format, quiet, URL constant). This story adds integration tests: gravity well called correctly FROM `score()`.

No existing test in `test_quality_*.py` files currently touches gravity well behavior — confirmed by grep.

### File Structure Requirements

- `tests/unit/test_quality_score_gravity_well.py` — NEW file (tests only, no production code changes)

### References

- [Source: epics.md#Epic 3 > Story 3.6] — ACs and story statement
- [Source: src/torq/quality/__init__.py:250-255] — Gravity well call site in `score()`
- [Source: src/torq/_gravity_well.py] — `_gravity_well()` implementation; output format; quiet suppression
- [Source: tests/unit/test_gravity_well.py] — Existing gravity well unit tests (do not duplicate)

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6

### Debug Log References

- tqdm writes its progress bar to stderr (not stdout). Removed `assert captured.err == ""` from the batch-scoring test — only stdout is relevant for gravity well assertions.

### Completion Notes List

- Task 1 complete: 9 integration tests written in `tests/unit/test_quality_score_gravity_well.py`. Covers all 3 ACs. No production code changes — implementation was already complete. All 317 tests passing.

### File List

- `tests/unit/test_quality_score_gravity_well.py` — NEW (9 tests)

## Change Log

- 2026-03-09: Implemented Story 3.6 — 9 gravity well integration tests for tq.quality.score(). No production code changes. 317 tests passing.
