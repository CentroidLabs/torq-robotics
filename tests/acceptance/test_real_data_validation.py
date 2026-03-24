"""Real-data pre-validation gate (SC-1.0) — Story 8.2.

Validates the full pipeline against:
  1. Real-world MCAP datasets (via env vars TORQ_TEST_ALOHA2_PATH / TORQ_TEST_FRANKA_PATH)
     — skipped gracefully when paths are not set or files absent.
  2. Existing MCAP test fixtures bundled with the repo (sample.mcap, boundary_detection.mcap)
     — always run when fixtures are present.

Run with:
    pytest tests/acceptance/test_real_data_validation.py -v
    TORQ_TEST_ALOHA2_PATH=/data/aloha2.mcap pytest tests/acceptance/test_real_data_validation.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

import torq as tq

# ── Dataset path resolution ───────────────────────────────────────────────────

FIXTURES_DATA = Path(__file__).parent.parent / "fixtures" / "data"

REAL_DATA_PATHS: dict[str, Path | None] = {
    "aloha2": Path(p) if (p := os.environ.get("TORQ_TEST_ALOHA2_PATH")) else None,
    "franka": Path(p) if (p := os.environ.get("TORQ_TEST_FRANKA_PATH")) else None,
}

BUNDLED_MCAP_FIXTURES: list[tuple[str, Path]] = [
    ("sample_mcap", FIXTURES_DATA / "sample.mcap"),
    ("boundary_detection_mcap", FIXTURES_DATA / "boundary_detection.mcap"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _run_pipeline(mcap_path: Path, tmp_path: Path) -> dict:
    """Run the full Torq pipeline on a single MCAP file.

    Exercises the complete AC#1 chain:
    ingest → score → save → compose → DataLoader → batch iteration.

    Returns a summary dict with episode count, quality scores, batch info, and timing.
    """
    start = time.perf_counter()
    batch_ok = False

    tq.config.quiet = True
    try:
        # Ingest → score (in memory) → save (writes quality to index)
        episodes = tq.ingest(mcap_path)
        tq.quality.score(episodes)
        for ep in episodes:
            tq.save(ep, path=tmp_path, quiet=True)

        # Compose → DataLoader → batch (full pipeline per AC#1)
        if len(episodes) > 0:
            dataset = tq.compose(quality_min=0.0, store_path=tmp_path, name="validation")
            if len(dataset) > 0:
                torch = pytest.importorskip("torch", reason="torch needed for DataLoader")
                from torq.serve import DataLoader

                loader = DataLoader(dataset, batch_size=min(4, len(dataset)))
                batch = next(iter(loader))
                batch_ok = "actions" in batch and "observations" in batch
    finally:
        tq.config.quiet = False

    elapsed = time.perf_counter() - start

    scores = [ep.quality.overall for ep in episodes if ep.quality is not None]
    return {
        "episode_count": len(episodes),
        "scored_count": len(scores),
        "quality_scores": scores,
        "batch_ok": batch_ok,
        "elapsed_s": elapsed,
    }


# ── Bundled fixture tests (always run) ───────────────────────────────────────


@pytest.mark.parametrize("name,mcap_path", BUNDLED_MCAP_FIXTURES)
@pytest.mark.slow
@pytest.mark.acceptance
def test_bundled_mcap_full_pipeline(name, mcap_path, tmp_path):
    """Full pipeline runs without error on bundled MCAP test fixtures.

    These fixtures are always available — they are the guaranteed baseline
    that must pass before any user validation session.
    """
    if not mcap_path.exists():
        pytest.skip(f"{name} fixture not found — run: python tests/fixtures/generate_fixtures.py")

    result = _run_pipeline(mcap_path, tmp_path)

    assert result["episode_count"] > 0, (
        f"{name}: ingest() produced 0 episodes from {mcap_path}"
    )
    assert result["scored_count"] == result["episode_count"], (
        f"{name}: scoring failed for some episodes"
    )
    assert result["batch_ok"], (
        f"{name}: compose → DataLoader → batch failed — full pipeline not working"
    )
    assert result["elapsed_s"] < 60.0, (
        f"{name}: pipeline took {result['elapsed_s']:.1f}s (limit: 60s)"
    )


@pytest.mark.slow
@pytest.mark.acceptance
def test_bundled_fixtures_quality_scores_vary(tmp_path):
    """Quality scores vary across episodes in boundary_detection fixture (>= 2 episodes).

    This guards against constant-returning scorers on real-world data.
    """
    mcap_path = FIXTURES_DATA / "boundary_detection.mcap"
    if not mcap_path.exists():
        pytest.skip(
            "boundary_detection.mcap not found — "
            "run: python tests/fixtures/generate_fixtures.py"
        )

    result = _run_pipeline(mcap_path, tmp_path)

    if result["episode_count"] < 2:
        pytest.skip(f"Only {result['episode_count']} episode(s) — need ≥ 2 to check variance")

    scores = result["quality_scores"]
    assert len(set(round(s, 6) for s in scores)) > 1, (
        f"All quality scores identical ({scores[0]:.4f}) — scorer may be returning a constant"
    )


# ── Real-world dataset tests (skipped when env vars not set) ─────────────────


@pytest.mark.slow
@pytest.mark.acceptance
@pytest.mark.real_data
@pytest.mark.parametrize("platform", ["aloha2", "franka"])
def test_real_mcap_full_pipeline(platform, tmp_path):
    """Full pipeline on real-world MCAP datasets from distinct hardware platforms.

    Set environment variables to enable:
        TORQ_TEST_ALOHA2_PATH=/path/to/aloha2.mcap
        TORQ_TEST_FRANKA_PATH=/path/to/franka.mcap

    AC#1: No unrecoverable errors, quality scores vary, episode boundaries detected.
    AC#3: 5-line workflow completes on real MCAP files.
    """
    mcap_path = REAL_DATA_PATHS.get(platform)
    if mcap_path is None:
        pytest.skip(
            f"Real {platform} dataset not configured. "
            f"Set TORQ_TEST_{platform.upper()}_PATH to enable this test."
        )
    if not mcap_path.exists():
        pytest.skip(f"Real {platform} dataset not found at {mcap_path}")

    result = _run_pipeline(mcap_path, tmp_path)

    # AC#1: ingestion produced results
    assert result["episode_count"] > 0, (
        f"{platform}: ingest() produced 0 episodes from {mcap_path.name}"
    )

    # AC#1: episode boundaries detected (real recordings typically have multiple episodes)
    if result["episode_count"] < 2:
        import warnings

        warnings.warn(
            f"{platform}: only {result['episode_count']} episode — "
            "boundary detection may not be exercised. "
            "Real recordings typically contain multiple episodes.",
            stacklevel=1,
        )

    # AC#1: quality scores vary (not constant)
    scores = result["quality_scores"]
    if len(scores) >= 2:
        assert len(set(round(s, 6) for s in scores)) > 1, (
            f"{platform}: all quality scores identical — scorer may be returning a constant"
        )

    # AC#1: full pipeline including compose → DataLoader → batch
    assert result["batch_ok"], (
        f"{platform}: compose → DataLoader → batch failed — full pipeline not working"
    )

    # Timing guard
    assert result["elapsed_s"] < 300.0, (
        f"{platform}: pipeline took {result['elapsed_s']:.1f}s (limit: 300s for real data)"
    )
