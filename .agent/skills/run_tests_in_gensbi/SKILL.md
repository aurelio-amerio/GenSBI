---
name: Run Tests in GenSBI Environment
description: Instructions for running pytest inside the gensbi mamba environment with parallel execution.
---

# Running Tests in GenSBI Environment

When you need to run tests in this workspace, follow this two-step approach:

## Preferred: `mamba run`

Try this first. By default, tests run in parallel via `pytest-xdist` (`-n 2` in `pyproject.toml`):

```bash
mamba run -n gensbi pytest tests/ -x --tb=short
mamba run -n gensbi pytest tests/test_file.py -x --tb=short
```

## Fallback: explicit activation

If `mamba run` fails (e.g. unrecognized arguments, wrong env), deactivate any nested environments and activate explicitly:

```bash
mamba deactivate && mamba deactivate && mamba activate gensbi
pytest tests/ -x --tb=short
```

This is less formally correct but always works.

## Other options

```bash
# With coverage
pytest --cov=src tests/

# Debug mode (no parallelism, see stdout)
pytest tests/test_file.py -s
```
