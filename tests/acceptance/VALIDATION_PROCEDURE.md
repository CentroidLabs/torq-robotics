# Torq R1 Alpha — Real-Data Pre-Validation Procedure (SC-1.0)

This document describes the manual validation procedure for Story 8.2.
Run this before any user validation session or public release.

---

## 1. Fresh-Install Validation (AC#2)

Verifies `pip install torq-robotics` works cleanly in a new environment.

```bash
# Create a fresh conda environment
conda create -n torq-validate python=3.12 -y
conda activate torq-validate

# Install from source (or PyPI once published)
pip install -e ".[torch]"

# Verify import succeeds
python -c "import torq as tq; print(tq.__version__)"

# Verify DataLoader is importable
python -c "from torq.serve import DataLoader; print('DataLoader OK')"
```

Expected output:
```
0.1.0a1
DataLoader OK
```

No import errors, no dependency conflicts.

---

## 2. Bundled Fixture Validation (always available)

```bash
# Generate bundled fixtures if not already present
python tests/fixtures/generate_fixtures.py

# Run bundled MCAP acceptance tests
pytest tests/acceptance/test_real_data_validation.py -v -m "acceptance"
```

Expected: all bundled fixture tests pass.

---

## 3. Real-World Dataset Validation (AC#1, AC#3)

### Acquire Datasets

Obtain MCAP recordings from 2 distinct hardware platforms. Examples:
- **ALOHA-2**: `aloha2_pick_and_place_001.mcap` (Stanford ALOHA-2 bimanual arm)
- **Franka**: `franka_panda_push_001.mcap` (Franka Panda 7-DOF arm)

Each file should contain:
- At least 2 episode recordings
- `/joint_states` or equivalent topic (sensor_msgs/JointState)
- `/action` or `/cmd_vel` or equivalent action topic
- 50+ Hz data, 30+ seconds per episode

### Configure Paths

```bash
export TORQ_TEST_ALOHA2_PATH=/path/to/aloha2_pick_and_place_001.mcap
export TORQ_TEST_FRANKA_PATH=/path/to/franka_panda_push_001.mcap
```

### Run Validation

```bash
pytest tests/acceptance/test_real_data_validation.py -v -m "real_data" -s
```

### Expected Output

```
PASSED tests/acceptance/test_real_data_validation.py::test_real_mcap_full_pipeline[aloha2]
PASSED tests/acceptance/test_real_data_validation.py::test_real_mcap_full_pipeline[franka]
```

Each test should report:
- `episode_count > 0`
- `quality_scores` not all identical
- Runtime < 300s per dataset

---

## 4. Full 5-Line README Workflow on Real Data (AC#3)

After running the validation tests, verify the 5-line workflow manually:

```python
import torq as tq
from torq.serve import DataLoader

store_path = "/tmp/torq_validate"

# Line 1: Ingest real MCAP
episodes = tq.ingest("/path/to/real_data.mcap")
print(f"Ingested {len(episodes)} episodes")

# Line 2: Score quality
tq.quality.score(episodes)
for ep in episodes:
    print(f"  {ep.episode_id}: overall={ep.quality.overall:.3f}")

# Save scored episodes (quality written to index at save time)
for ep in episodes:
    tq.save(ep, path=store_path)

# Line 3: Compose dataset
dataset = tq.compose(quality_min=0.3, store_path=store_path, name="validation_v1")
print(f"Dataset: {len(dataset)} episodes after quality filter")

# Line 4-5: DataLoader and batch
loader = DataLoader(dataset, batch_size=4)
batch = next(iter(loader))
print(f"Batch actions shape: {batch['actions'].shape}")
print(f"Batch observations shape: {batch['observations'].shape}")
```

Expected: no exceptions, valid tensor shapes printed.

---

## 5. Validation Sign-Off Checklist

Before tagging R1 Alpha:

- [ ] Fresh-install validation passes (Section 1)
- [ ] Bundled fixture tests pass: `pytest tests/acceptance/ -m acceptance`
- [ ] Real-data validation on ALOHA-2: PASSED
- [ ] Real-data validation on Franka (or equivalent): PASSED
- [ ] 5-line README workflow runs on real data without modification
- [ ] No import errors on clean Python 3.12 environment
- [ ] `pytest tests/ -q` passes (full suite, all 634+ tests)

Record results:

| Platform | Episodes | Quality Range | Runtime | Status |
|---|---|---|---|---|
| ALOHA-2  | ___ | ___–___ | ___s | PASS/FAIL |
| Franka   | ___ | ___–___ | ___s | PASS/FAIL |

Validated by: ___________________  Date: ___________
