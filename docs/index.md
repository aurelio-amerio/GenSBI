# GenSBI 
![Tests](../img/badges/tests.svg)
![Coverage](../img/badges/coverage.svg)

```{image} _static/logo.png
  :alt: GenSBI Logo
  :align: center
  :width: 600px
  :class: logo-transparent-bg
```

```{admonition} Project Status
:class: info
GenSBI is currently reaching the end of the **Alpha** cycle. The API is reaching stability, but may still change in the future.
```

## Getting Started

```{admonition} New to GenSBI?
:class: tip

Start here:
1. [Installation](/getting_started/installation) - Get GenSBI installed
2. [Quick Start Guide](/getting_started/quick_start) - 15-minute introduction
3. [My First Model Tutorial](/notebooks/my_first_model) - Complete step-by-step walkthrough
```

### Standard Installation (CPU / Compatible)

```bash
pip install git+https://github.com/aurelio-amerio/GenSBI.git
```

### High-Performance Installation (CUDA 12)

If you have a compatible NVIDIA GPU, install with CUDA 12 support for significantly faster training:

```bash
pip install "GenSBI[cuda12] @ git+https://github.com/aurelio-amerio/GenSBI.git"
```

For more installation options, see the [Installation Guide](/getting_started/installation).

## Key Documentation Sections

### 📚 Basics

Learn the core concepts and how to use GenSBI effectively:

- **[Conceptual Overview](/basics/overview)** - Understand how GenSBI is structured
- **[Model Cards](/basics/model_cards)** - Choose the right model for your problem
- **[Training Guide](/basics/training)** - Learn how to train models effectively
- **[Inference Guide](/basics/inference)** - Sample from posterior distributions
- **[Validation Guide](/basics/validation)** - Validate your results with SBC, TARP, and L-C2ST
- **[Troubleshooting](/basics/troubleshooting)** - Solve common issues

### 📖 Examples

See GenSBI in action with complete working examples:

- **[My First Model](/notebooks/my_first_model)** - Recommended starting tutorial
- **[SBI Benchmarks](/examples)** - Two Moons, Gaussian Linear, SLCP, and more
- **[All Examples](/examples)** - Full list of notebooks and scripts

All examples are available in the [GenSBI-examples repository](https://github.com/aurelio-amerio/GenSBI-examples).

### 🔧 API Reference

Detailed API documentation for all classes and functions:

- **[API Documentation](/api/gensbi/index)** - Auto-generated API reference

### 👥 Contributing

Want to contribute? Check out the guides:

- **[Contributing Guide](/basics/contributing)** - How to contribute to GenSBI
- **[GitHub Repository](https://github.com/aurelio-amerio/GenSBI)** - Source code and issues

## Examples

<img src="_static/animated_plot_samples_simformer.gif" alt="two-moons posterior sampling" height="400px">
<img src="_static/animated_plot_posterior_simformer.gif" alt="two-moons posterior sampling" height="400px">

Some key examples include:

**Getting Started:**

- [My First Model](/notebooks/my_first_model) - Complete beginner tutorial

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

See the [Examples](/examples) page for the complete list and detailed descriptions.

```{admonition} AI Usage Disclosure
:class: note

This project utilized large language models, specifically Google Gemini and GitHub Copilot, to assist with code suggestions, documentation drafting, and grammar corrections. All AI-generated content has been manually reviewed and verified by human authors to ensure accuracy and adherence to scientific standards.
```

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
## Similar packages
GenSBI is designed to provide a numerically efficient JAX implementation of flow and diffusion models, complementing existing SBI libraries. You might also want to check out:

- **[`sbi`](https://github.com/sbi-dev/sbi)**: A comprehensive Pytorch-based package for simulation-based inference. It implements neural posterior estimation (NPE), neural likelihood estimation (NLE), and neural ratio estimation (NRE) methods. It is an excellent choice for a wide range of SBI tasks and supports amortized as well as sequential inference.
- **[`swyft`](https://github.com/undark-lab/swyft)**: An official implementation of Truncated Marginal Neural Ratio Estimation (TMNRE). It is designed to be highly efficient for marginal posterior estimation and scales well to complex simulations, leveraging `dask` and `zarr` for handling large datasets.
- **[`ltu-ili`](https://github.com/maho3/ltu-ili)**: The "Learning the Universe" Implicit Likelihood Inference library. It unifies multiple SBI backends (including `sbi`, `pydelfi`, and `lampe`) under a single interface, making it easy to benchmark different methods. It is particularly focused on applications in astrophysics and cosmology.
- **[`sbijax`](https://github.com/dirmeier/sbijax)**: A simulation-based inference library built on top of JAX. It implements standard neural simulation-based inference methods (NPE, NLE, NRE) as well as ABC, leveraging JAX's just-in-time compilation and automatic differentiation for high-performance inference. Its API is inspired by the `sbi` package.

```{toctree}
:hidden:
:maxdepth: 1

Get Started! </documentation/index>
Examples </examples>
References </references>
```


