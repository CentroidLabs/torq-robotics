"""Import graph CI gate — enforces architectural integrity.

These 3 tests run on every push and block merge on failure.
They enforce the dependency rules in architecture.md § "Dependency Rules".
"""

import subprocess
import sys
from pathlib import Path


def test_core_import_succeeds_without_optional_deps():
    """import torq must work with only core deps (numpy, pyarrow, mcap, h5py, tqdm).

    This enforces NFR-C05 and the lazy-loading constraint.
    Torch/jax/opencv must NOT be imported at module level anywhere in src/torq/.
    """
    result = subprocess.run(
        [sys.executable, "-c", "import torq; assert torq.__version__"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, (
        f"'import torq' failed.\nstdout: {result.stdout}\nstderr: {result.stderr}\n"
        "Check for module-level imports of torch/jax/opencv in src/torq/"
    )


def test_serve_module_does_not_import_torch_at_module_level():
    """import torq.serve must not trigger torch import.

    torq.serve is the only place torch may be imported, but only INSIDE functions,
    never at module level. This prevents import failures for non-torch users.
    """
    code = (
        "import sys; "
        "import torq.serve; "
        "assert 'torch' not in sys.modules, "
        "f'torch was imported by torq.serve at module level: {list(sys.modules)}'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, (
        f"torq.serve imports torch at module level.\nstderr: {result.stderr}\n"
        "Move all 'import torch' calls inside functions, guarded by _require_torch()."
    )


def test_episode_py_has_no_torq_imports():
    """src/torq/episode.py may only import from torq.errors (true leaf — no circular risk).

    episode.py is the dependency ROOT. It must not import from any torq module other
    than torq.errors, since torq.errors itself imports nothing from torq.
    This prevents circular import cascades through the module graph.
    """
    episode_path = Path(__file__).parent.parent / "src" / "torq" / "episode.py"
    if not episode_path.exists():
        # episode.py doesn't exist yet (created in Story 2.1) — skip
        import pytest

        pytest.skip("episode.py not yet created (Story 2.1)")
        return
    source = episode_path.read_text()
    # Allow only torq.errors — all other torq.* imports are forbidden
    torq_imports = [
        line.strip()
        for line in source.splitlines()
        if (
            "from torq" in line
            or ("import torq" in line and not line.strip().startswith("#"))
        )
        and "torq.errors" not in line
    ]
    assert not torq_imports, (
        "episode.py contains forbidden torq.* imports (only torq.errors is allowed):\n"
        + "\n".join(torq_imports)
        + "\nepisode.py may only depend on torq.errors, not other torq modules."
    )
