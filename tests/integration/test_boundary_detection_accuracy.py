"""Integration test: boundary detection accuracy against annotated fixture.

AC #2: boundary detection achieves >90% accuracy against boundary_detection.mcap.

Accuracy definition:
    accuracy = (# ground-truth boundaries with a matching detected boundary
                within TOLERANCE_NS) / (total ground-truth boundaries)

Ground truth is stored in the MCAP metadata record "ground_truth_boundaries"
as the field "boundary_transition_indices" (comma-separated integer indices into
the 50 Hz aligned timeline, starting at t=1_000_000_000 ns, step=20_000_000 ns).

This test is marked @pytest.mark.slow because it reads a multi-episode fixture.
"""

from pathlib import Path

import numpy as np
import pytest

FIXTURES_DATA = Path(__file__).parent.parent / "fixtures" / "data"

TOLERANCE_NS = 5 * 20_000_000  # ±5 timesteps at 50 Hz = ±100 ms

T_START_NS = 1_000_000_000
STEP_NS = 20_000_000  # 50 Hz


@pytest.mark.slow
def test_boundary_detection_accuracy_above_90_percent() -> None:
    """Auto boundary detection achieves ≥90% accuracy vs annotated fixture.

    Reads ground-truth boundary indices from MCAP metadata, converts to
    timestamps, then compares against the timestamps inferred from the
    returned episodes.
    """
    from mcap.reader import make_reader

    from torq.ingest.mcap import ingest

    fixture_path = FIXTURES_DATA / "boundary_detection.mcap"
    if not fixture_path.exists():
        pytest.skip(
            "boundary_detection.mcap not found — run: python tests/fixtures/generate_fixtures.py"
        )

    # ── Load ground truth from MCAP metadata ──────────────────────────────────
    gt_transition_indices: list[int] = []
    with open(fixture_path, "rb") as f:
        reader = make_reader(f)
        for metadata in reader.iter_metadata():
            if metadata.name == "ground_truth_boundaries":
                raw = metadata.metadata.get("boundary_transition_indices", "")
                gt_transition_indices = [int(x) for x in raw.split(",") if x.strip()]
                break

    assert gt_transition_indices, (
        "No ground-truth boundaries found in MCAP metadata — fixture may be outdated."
    )

    # Convert ground-truth indices to nanosecond timestamps
    gt_timestamps_ns = [T_START_NS + idx * STEP_NS for idx in gt_transition_indices]

    # ── Run ingest with auto boundary detection ────────────────────────────────
    episodes = ingest(fixture_path, boundary_strategy="auto")
    assert len(episodes) > 1, (
        f"Expected multiple episodes, got {len(episodes)} — boundary detection may have failed"
    )

    # ── Collect detected boundary timestamps ──────────────────────────────────
    # Each episode boundary = last timestamp of episode[i] or first timestamp of episode[i+1]
    detected_boundary_ts: list[int] = []
    for ep in episodes:
        detected_boundary_ts.append(int(ep.timestamps[0]))  # start of each episode
        detected_boundary_ts.append(int(ep.timestamps[-1]))  # end of each episode

    detected_boundary_ts = sorted(set(detected_boundary_ts))

    # ── Compute accuracy ──────────────────────────────────────────────────────
    hits = 0
    for gt_ts in gt_timestamps_ns:
        distances = [abs(gt_ts - d) for d in detected_boundary_ts]
        if min(distances) <= TOLERANCE_NS:
            hits += 1

    n_gt = len(gt_timestamps_ns)
    accuracy = hits / n_gt

    assert accuracy >= 0.90, (
        f"Boundary detection accuracy {accuracy:.1%} is below the 90% threshold "
        f"({hits}/{n_gt} ground-truth boundaries detected within ±{TOLERANCE_NS // 1_000_000} ms). "
        f"Ground-truth timestamps (ns): {gt_timestamps_ns}\n"
        f"Detected boundary timestamps (ns): {detected_boundary_ts[:20]}"
    )
