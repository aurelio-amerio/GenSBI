# 15-minute quick start

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
To get started *fast*, use the provided recipes. 

```{note} 
The example below is a **minimal script** designed for copy-pasting by experienced users. If you want a step-by-step educational walkthrough that explains the concepts, please see the [My First Model Tutorial](/notebooks/my_first_model).
```

Here is a minimal example of setting up a flow-based conditional inference pipeline using `Flux1`.

This example covers:
1.  **Data Generation**: Creating synthetic data for a simple linear problem.
2.  **Model Configuration**: Setting up the `Flux1` parameters.
3.  **Pipeline Creation**: Initializing the `Flux1FlowPipeline` which handles training and sampling.
4.  **Training**: Running the training loop.
5.  **Inference**: Sampling from the posterior given new observation.

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

```bibtex
@misc{GenSBI,
  author       = {Amerio, Aurelio},
  title        = "{GenSBI: Generative models for Simulation-Based Inference}",
  year         = {2025}, 
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/aurelio-amerio/GenSBI}}
}
```