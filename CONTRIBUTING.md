# Contributing to Torq

Thanks for your interest in contributing to Torq.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/CentroidLabs/torq-robotics.git
cd torq-robotics

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in editable mode with dev extras
pip install -e ".[dev]"

# Verify
python -c "import torq; print(torq.__version__)"
```

## Running Tests

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# Specific module
pytest tests/unit/test_episode.py -v

# With coverage
pytest tests/ --cov=torq --cov-report=term-missing
```

## Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
# Check
ruff check src/
ruff format --check src/

# Auto-fix
ruff check --fix src/
ruff format src/
```

Rules: `E`, `F`, `I` (isort), `W`. Line length: 100.

## Guidelines

- **Type hints** on all public functions.
- **Google-style docstrings** on all public classes and functions.
- **Test-first** when possible — write the test, watch it fail, then implement.
- **No torch at module level** — framework imports only inside `serve/` and only conditionally.
- **Helpful error messages** — every exception tells the user what went wrong and what to do about it.

## Commit Messages

```
feat(module): description
fix(module): description
test(module): description
```

## Pull Requests

1. Fork the repo and create a branch from `main`.
2. Add tests for any new functionality.
3. Ensure `ruff check src/` and `pytest tests/` pass.
4. Open a PR with a clear description of what changed and why.
