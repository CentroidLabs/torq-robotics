<p align="center">
  <img src="git-cover-img.svg" alt="torq-robotics" width="100%">
</p>

<p align="center">
  <a href="https://github.com/CentroidLabs/torq-robotics/actions/workflows/ci.yml"><img src="https://github.com/CentroidLabs/torq-robotics/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python 3.12+">
  <a href="LICENSE"><img src="https://img.shields.io/github/license/CentroidLabs/torq-robotics" alt="License"></a>
</p>

---

**Torq** turns raw robot recordings into training-ready datasets in 5 lines of Python.

```python
import torq as tq

episodes = tq.ingest("./recordings/")           # MCAP, LeRobot, HDF5
tq.quality.score(episodes)                       # smoothness + consistency + completeness
dataset  = tq.compose(episodes, task="pick_place", quality_min=0.7)
loader   = tq.DataLoader(dataset, batch_size=32) # PyTorch-ready
```

## Why Torq?

Robot learning teams spend more time wrangling data than training models. Recordings arrive in different formats, quality varies wildly, and composing the right training mix is manual and error-prone.

Torq provides a single pipeline: **ingest** any format, **score** quality automatically, **compose** balanced datasets, and **serve** them directly to PyTorch.

## Supported Formats

| Format | Source | Status |
|--------|--------|--------|
| [MCAP](https://mcap.dev) | ROS 2 bag recordings | Supported |
| [LeRobot](https://github.com/huggingface/lerobot) | HuggingFace LeRobot v3.0 datasets | Supported |
| [HDF5](https://robomimic.github.io/) | Robomimic-style demonstration files | Supported |

## CLI

```bash
tq ingest ./recordings/          # auto-detect format, ingest to local store
tq list                          # list all ingested episodes
tq info ep_0001                  # detailed episode metadata
tq export ./out/ --format lerobot  # export to LeRobot format
```

## Quality Scoring

Every episode is scored across three dimensions:

- **Smoothness** — jerk-based trajectory analysis (are motions fluid?)
- **Consistency** — action autocorrelation (is the policy decisive?)
- **Completeness** — task completion heuristics (did it finish the job?)

Scores combine into a weighted `overall` score (0.0 - 1.0) used for filtering and dataset composition.

## Architecture

```
torq.ingest     →  Episode  →  torq.quality.score  →  torq.compose  →  torq.DataLoader
  (MCAP/LR/H5)    (unified)      (score & filter)     (balance/mix)     (PyTorch)
```

The `Episode` dataclass is the universal container — all formats ingest into it, all downstream tools consume it. No conversions, no adapters.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, and guidelines.

## License

[MIT](LICENSE) — Centroid Foundry
