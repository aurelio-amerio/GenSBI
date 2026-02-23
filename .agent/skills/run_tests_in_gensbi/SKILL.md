---
name: Run Tests in GenSBI Environment
description: Instructions for running pytest inside the gensbi mamba environment with parallel execution.
---

# Running Tests in GenSBI Environment

When you need to run tests in this workspace, you MUST use `pytest` prefixed with `mamba run -n gensbi`. 

By default, we use `pytest-xdist` to run tests in parallel. Use the `-n 2` flag to automatically detect the number of available CPUs.

## Usage

### Running All Tests
```bash
mamba run -n gensbi pytest -n 2
```

### Running a Specific Test File
```bash
mamba run -n gensbi pytest -n 2 tests/test_file.py
```

### Running with Coverage
```bash
mamba run -n gensbi pytest -n 2 --cov=src
```

### When to Disable Parallelism
If you need to debug a test or see stdout more clearly, you can omit `-n 2`:
```bash
mamba run -n gensbi pytest tests/test_file.py -s
```
