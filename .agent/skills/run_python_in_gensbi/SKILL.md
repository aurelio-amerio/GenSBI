---
name: Run Python in Gensbi Environment
description: Instructions for running python inside the gensbi mamba environment.
---

# Running Python in GenSBI Environment

When you need to run any python command, script, or module in the terminal within this workspace, you MUST prefix the command with `mamba run -n gensbi`.

This ensures that the correct environment and dependencies are used.

## Usage

### Running Scripts
Instead of:
```bash
python script.py arg1 arg2
```
Use:
```bash
mamba run -n gensbi python script.py arg1 arg2
```

### Running Modules
Instead of:
```bash
python -m my.module
```
Use:
```bash
mamba run -n gensbi python -m my.module
```

### Running Tests
Instead of:
```bash
pytest test/
```
Use:
```bash
mamba run -n gensbi pytest test/
```
