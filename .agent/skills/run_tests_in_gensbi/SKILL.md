---
name: Run Tests in GenSBI Environment
description: Instructions for running pytest inside the gensbi uv environment with parallel execution.
---

# Running Tests in GenSBI Environment

When you need to run tests in this workspace, use `uv run`. By default, tests run in parallel via `pytest-xdist` (`-n 2` in `pyproject.toml`):

```bash
uv run pytest tests/ -x --tb=short
uv run pytest tests/test_file.py -x --tb=short
```

## Other options

```bash
# With coverage
uv run pytest --cov=src tests/

# Debug mode (no parallelism, see stdout)
uv run pytest tests/test_file.py -s
```
