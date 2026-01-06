# GenSBI
[![Build](https://github.com/aurelio-amerio/GenSBI/actions/workflows/python-app.yml/badge.svg)](https://github.com/aurelio-amerio/GenSBI/actions/workflows/python-app.yml)
![Coverage](https://raw.githubusercontent.com/aurelio-amerio/GenSBI/refs/heads/main/img/badges/coverage.svg)
![GenSBI Logo](https://raw.githubusercontent.com/aurelio-amerio/GenSBI/refs/heads/main/docs/_static/logo.png)

> [!IMPORTANT]  
> This library is at an early stage of development. The API is potentially subject to change.

## Overview

**GenSBI** is a powerful JAX-based library for Simulation-Based Inference (SBI) using state-of-the-art generative models, currently revolving around Optimal Transport Flow Matching and Diffusion Models.

It is designed for researchers and practitioners who need a flexible, high-performance toolkit to solve complex inference problems where the likelihood function is intractable.

## Key Features

- **Modern SBI Algorithms**: Implements cutting-edge techniques like **Optimal Transport Conditional Flow Matching** and **Diffusion Models** for robust and flexible posterior inference.
- **Built on JAX and Flax NNX**: Leverages the power of JAX for automatic differentiation, vectorization, and seamless execution on CPUs, GPUs, and TPUs.
- **High-Level Recipes API**: A simplified interface for common workflows, allowing you to train models and run inference with just a few lines of code.
- **Powerful Transformer Models**: Includes implementations of recent, high-performing models like **Flux1**, **Flux1Join**, and **Simformer** for handling complex, high-dimensional data.
- **Modular and Extensible**: A clean, well-structured codebase that is easy to understand, modify, and extend for your own research.

## Installation

```bash
pip install git+https://github.com/aurelio-amerio/GenSBI.git
```

For GPU support and other options, see the [Installation Guide](https://aurelio-amerio.github.io/GenSBI/getting_started/installation.html).

## Quick Start

To get started immediately, you can use the high-level API to train a model.

> [!TIP]
> Check out the **[my_first_model.ipynb](https://github.com/aurelio-amerio/GenSBI-examples/blob/main/examples/getting_started/my_first_model.ipynb)** notebook for a complete, step-by-step introductory tutorial.

```python
from flax import nnx
from gensbi.recipes import Flux1FlowPipeline
from gensbi.models import Flux1Params

train_dataset = ... # define a training dataset (infinite iterator)
val_dataset = ...   # define a validation dataset (infinite iterator)
dim_obs = ...       # dimension of the parameters (theta)
dim_cond = ...      # dimension of the simulator observations (x)
params = Flux1Params(...) # the parameters for your model

# Instantiate the pipeline
pipeline = Flux1FlowPipeline(
    train_dataset,
    val_dataset,
    dim_obs,
    dim_cond,
    params=params,
)

# Train the model
# Note: GenSBI uses Flax NNX, so we pass a random key generator
pipeline.train(rngs=nnx.Rngs(0))

# After training, get a sampler for posterior sampling
key = jax.random.PRNGKey(42)
samples = pipeline.sample(key, x_observed, num_samples=10_000)
```

## Examples
<table align="center" style="width:95%;">
  <tr>
    <td align="center">
      <img src="https://github.com/aurelio-amerio/GenSBI-examples/blob/main/examples/sbi-benchmarks/two_moons/flow_simformer/animated_plot_samples_simformer.gif?raw=true" alt="two-moons posterior sampling" height="300">
    </td>
    <td align="center">
      <img src="https://github.com/aurelio-amerio/GenSBI-examples/blob/main/examples/sbi-benchmarks/two_moons/flow_simformer/animated_plot_posterior_simformer.gif?raw=true" alt="two-moons posterior sampling" height="300">
    </td>
  </tr>
</table>




Examples for this library are available separately in the [GenSBI-examples](https://github.com/aurelio-amerio/GenSBI-examples) repository.

Some key examples include:

**Getting Started:**

- `my_first_model.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/examples/getting_started/my_first_model.ipynb) <br>
A beginner-friendly notebook introducing the core concepts of GenSBI on a simple problem.

**Unconditional Density Estimation:**

- `flow_matching_2d_unconditional.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/examples/flow_matching_2d_unconditional.ipynb) <br>
Demonstrates how to use flow matching in 2D for unconditional density estimation.
- `diffusion_2d_unconditional.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/examples/diffusion_2d_unconditional.ipynb) <br>
Demonstrates how to use diffusion models in 2D for unconditional density estimation.

**Conditional Density Estimation:**

- `two_moons_flow_simformer.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/examples/sbi-benchmarks/two_moons/flow_simformer/two_moons_flow_simformer.ipynb) <br>
Uses the Simformer model for posterior density estimation on the two-moons benchmark.
- `two_moons_flow_flux.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/examples/sbi-benchmarks/two_moons/flow_flux/two_moons_flow_flux.ipynb) <br>
Uses the Flux1 model for posterior density estimation on the two-moons benchmark.
- `gaussian_linear_flow_flux1joint.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/examples/sbi-benchmarks/gaussian_linear/flow_flux1joint/gaussian_linear_flow_flux1joint.ipynb) <br>
Uses the Flux1Joint model for posterior density estimation on the Gaussian Linear benchmark.
- `slcp_flow_simformer.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/examples/sbi-benchmarks/slcp/flow_simformer/slcp_flow_simformer.ipynb) <br>
Uses the Simformer model for posterior density estimation on the SLCP benchmark.    


> [!NOTE]
> A full list of the currently available examples is available at the [examples](https://aurelio-amerio.github.io/GenSBI/examples.html) documentation page.

## Citing GenSBI

If you use this library, please consider citing this work and the original methodology papers, see [references](https://aurelio-amerio.github.io/GenSBI/references.html).

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

### Reference implementations:
- **Facebook Flow Matching library**: [https://github.com/facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching)
- **Elucidating the Design Space of Diffusion-Based Generative Models**: [https://github.com/NVlabs/edm](https://github.com/NVlabs/edm)
- **Simformer model**: [https://github.com/mackelab/simformer](https://github.com/mackelab/simformer)
- **Flux1 model from BlackForest Lab**: [https://github.com/black-forest-labs/flux](https://github.com/black-forest-labs/flux)
- **Simulation-Based Inference Benchmark**: [https://github.com/sbi-benchmark/sbibm](https://github.com/sbi-benchmark/sbibm)

> [!NOTE]
> **AI Usage Disclosure** <br>
> This project utilized large language models, specifically Google Gemini and GitHub Copilot, to assist with code suggestions, documentation drafting, and grammar corrections. All AI-generated content has been manually reviewed and verified by human authors to ensure accuracy and adherence to scientific standards.
