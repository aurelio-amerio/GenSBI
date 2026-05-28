# 15-minute quick start

Welcome to GenSBI! This page is a quick guide to get you started with installation and basic usage.

## Installation

Using [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv add gensbi
# or, for a standalone install:
uv pip install gensbi
```

Or using pip:

```bash
pip install gensbi
```

If a GPU is available, it is advisable to install the CUDA version of the package:

```bash
uv add gensbi[cuda12]
# or
pip install gensbi[cuda12]
```

See the [Installation Guide](/getting_started/installation) for more options, including how to install the examples.

## Requirements

- Python 3.11+
- JAX
- Flax
- (See `pyproject.toml` for full requirements)

## Basic Usage

To get started *fast*, use the provided recipes.

```{note}
The example below is a **minimal script** designed for copy-pasting by experienced users. If you want a step-by-step educational walkthrough that explains the concepts, please see the [My First Model Tutorial](/notebooks/my_first_model).
```

Here is a minimal example of setting up a flow-based conditional inference pipeline using `Flux1`.

This example covers:

1. **Data Generation**: Creating synthetic data for a simple linear problem.
2. **Model Configuration**: Setting up the `Flux1` parameters.
3. **Pipeline Creation**: Initializing the `Flux1FlowPipeline` which handles training and sampling.
4. **Training**: Running the training loop.
5. **Inference**: Sampling from the posterior given new observation.

The code below is a complete, runnable script:

```{literalinclude} /examples/flux1_flow_pipeline.py
:language: python
:linenos:
```

```{image} /examples/flux1_flow_pipeline_marginals.png
:width: 600
```

```{note}
If you plan on using multiprocessing prefetching, ensure that your script is wrapped 
in a ``if __name__ == "__main__":`` guard. 
See https://docs.python.org/3/library/multiprocessing.html
```

See the full example notebook [my_first_model](/notebooks/my_first_model) for a more detailed walkthrough, and the [Examples](/examples) page for practical demonstrations on common SBI benchmarks.

## Citing GenSBI

If you use this library, please consider citing this work and the original methodology papers, see [references](/references).

For the paper:
```bibtex
@article{amerio2026gensbi_paper,
      title={GenSBI: Generative Methods for Simulation-Based Inference in JAX}, 
      author={Aurelio Amerio},
      year={2026},
      eprint={2605.27499},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.27499}, 
}
```
For the software:
```bibtex
@software{amerio2026gensbi_software,
  author       = {Amerio, Aurelio},
  title        = {GenSBI: Generative Methods for Simulation-Based Inference in JAX
                  },
  month        = may,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v0.3.4},
  doi          = {10.5281/zenodo.20410084},
  url          = {https://doi.org/10.5281/zenodo.20410084},
}
```
