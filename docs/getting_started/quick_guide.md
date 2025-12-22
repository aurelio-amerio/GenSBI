# 5-minute guide

Welcome to GenSBI! This page is a quick guide to get you started with installation and basic usage.

## Installation

GenSBI is in early development. To install, clone the repository and install dependencies:

```bash
pip install git+https://github.com/aurelio-amerio/GenSBI.git
```

If a GPU is available, it is advisable to install the cuda version of the package:

```bash
pip install "GenSBI[cuda12] @ git+https://github.com/aurelio-amerio/GenSBI.git"
```

## Requirements

- Python 3.11+
- JAX
- Flax
- (See `pyproject.toml` for full requirements)

## Basic Usage
The most basic usage of GenSBI involves defining a simulation-based inference pipeline using one of the provided recipes. Here is a minimal example of setting up a flow-based inference pipeline using `Flux1`:


```{literalinclude} /examples/conditional_flow_pipeline.py
:language: python
:linenos:
```

```{image} /examples/conditional_flow_pipeline_marginals.png
:width: 600
```

```{note}
If you plan on using multiprocessing prefetching, ensure that your script is wrapped 
in a ``if __name__ == "__main__":`` guard. 
See https://docs.python.org/3/library/multiprocessing.html
```

See the [Examples](/examples) page for practical demonstrations on common SBI benchmarks.

## Citing GenSBI

If you use this library, please consider citing this work and the original methodology papers, see [references](/references).