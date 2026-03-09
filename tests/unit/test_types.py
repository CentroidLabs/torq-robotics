"""Tests for torq.types — type aliases for domain concepts."""

import numpy as np
import pytest


class TestTypeAliases:
    def test_episode_id_is_str(self):
        from torq.types import EpisodeID

        assert EpisodeID is str

    def test_timestamp_is_np_int64(self):
        from torq.types import Timestamp

        assert Timestamp is np.int64

    def test_quality_score_allows_none(self):
        """QualityScore is float | None — verify the annotation at runtime."""
        import types

        from torq.types import QualityScore

        # float | None produces a types.UnionType in Python 3.10+
        assert isinstance(QualityScore, types.UnionType)
        assert float in QualityScore.__args__
        assert type(None) in QualityScore.__args__

    def test_task_name_is_str(self):
        from torq.types import TaskName

        assert TaskName is str

    def test_embodiment_name_is_str(self):
        from torq.types import EmbodimentName

        assert EmbodimentName is str


class TestNoTorqImports:
    def test_types_module_has_no_torq_imports(self):
        """types.py must not import from torq.* (dependency leaf)."""
        import ast
        from pathlib import Path

        types_path = Path(__file__).parents[2] / "src" / "torq" / "types.py"
        source = types_path.read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("torq"), (
                        f"types.py imports from torq: 'import {alias.name}'"
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("torq"):
                    pytest.fail(f"types.py imports from torq: 'from {node.module} import ...'")
