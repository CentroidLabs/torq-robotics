# torq-robotics

**Robot Learning Data Infrastructure SDK**

```python
import torq as tq

# Ingest → Score → Compose → Train in 5 lines
episodes = tq.ingest("./recordings/")
episodes = tq.quality.score(episodes)
dataset  = tq.compose(episodes, task="pick_place", quality_min=0.7)
loader   = tq.DataLoader(dataset, batch_size=32)
# → ready for training
```

## Install

```bash
pip install torq-robotics          # core (numpy, pyarrow, mcap, h5py, tqdm)
pip install torq-robotics[torch]   # + PyTorch DataLoader
pip install torq-robotics[dev]     # + pytest, ruff (contributors)
```

## Quick Start

```bash
tq ingest ./recordings/
tq list
tq info ep_0001
tq export ./dataset/ --format lerobot
```

## Status

R1 Alpha — pre-release.

---

Made by [Centroid Foundry](https://www.datatorq.ai)
