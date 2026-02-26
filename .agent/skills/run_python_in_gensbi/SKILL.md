---
name: Run Python in Gensbi Environment
description: Instructions for running python inside the gensbi mamba environment.
---

# Running Python in GenSBI Environment

When you need to run any python command, script, or module in the terminal within this workspace, follow this two-step approach:

## Preferred: `mamba run`

Try this first:

```bash
mamba run -n gensbi python script.py arg1 arg2
mamba run -n gensbi python -m my.module
```

## Fallback: explicit activation

If `mamba run` fails or behaves unexpectedly, deactivate any nested environments and activate explicitly:

```bash
mamba deactivate && mamba deactivate && mamba activate gensbi
python script.py arg1 arg2
```

This is less formally correct but always works.
