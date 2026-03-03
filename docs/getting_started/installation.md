# Installation

## Using uv (recommended)

To add gensbi as a dependency to an existing pyproject.toml:

```bash
uv add gensbi
```

For a standalone install without a project:

```bash
uv pip install gensbi
```

For GPU support:

```bash
uv add gensbi[cuda12]
# or
uv pip install gensbi[cuda12]
```

```{info} Install uv
To install uv, run:

`curl -LsSf https://astral.sh/uv/install.sh | sh`

```


## Using pip

```bash
pip install gensbi
```

For GPU support:

```bash
pip install gensbi[cuda12]
```

If you want to run the examples, install the GenSBI-examples repository:

```bash
uv add gensbi[examples]
# or
pip install gensbi[examples]
```

To install all the optional dependencies at once, run:

```bash
uv add gensbi[cuda12,examples]
# or
pip install gensbi[cuda12,examples]
```

## Requirements

- Python 3.11+
- JAX
- Flax
- (See `pyproject.toml` for full requirements)
