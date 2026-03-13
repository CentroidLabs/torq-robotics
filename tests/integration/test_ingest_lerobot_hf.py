"""Integration test: ingest a real LeRobot dataset from HuggingFace Hub.

Requires: pip install huggingface_hub
Skipped when huggingface_hub is not installed or network is unavailable.
"""

import numpy as np
import pytest


@pytest.mark.slow
@pytest.mark.network
class TestLerobotIngestHuggingFace:
    def test_ingest_real_hf_dataset_returns_episodes(self, tmp_path):
        """Download lerobot/pusht (meta + Parquet only) and verify ingestion."""
        hub = pytest.importorskip("huggingface_hub")
        dataset_path = hub.snapshot_download(
            repo_id="lerobot/pusht",
            repo_type="dataset",
            local_dir=tmp_path / "pusht",
            allow_patterns=["meta/**", "data/chunk-000/*"],
            revision="main",
        )
        from torq.ingest.lerobot import ingest

        episodes = ingest(dataset_path)
        assert len(episodes) > 0, "Expected at least 1 episode from lerobot/pusht"
        ep = episodes[0]
        assert ep.actions.ndim == 2 and ep.actions.shape[1] > 0
        assert len(ep.observations) > 0
        assert ep.timestamps.dtype == np.int64
        assert ep.metadata.get("embodiment", "") != ""
