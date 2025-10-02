# GenSBI
[![Build](https://github.com/aurelio-amerio/GenSBI/actions/workflows/python-app.yml/badge.svg)](https://github.com/aurelio-amerio/GenSBI/actions/workflows/python-app.yml)
![Coverage](img/badges/coverage.svg)
![GenSBI Logo](docs/_static/logo.png)

> [!IMPORTANT]  
> This library is in an early stage of development. The API is potentially subject to change.

## Overview

**GenSBI** is a powerful JAX-based library for Simulation-Based Inference (SBI) using state-of-the-art generative models, currently revolving around Optimal Transport Flow Matching and Diffusion Models.

It is designed for researchers and practitioners who need a flexible, high-performance toolkit to solve complex inference problems where the likelihood function is intractable.

## Key Features

- **Modern SBI Algorithms**: Implements cutting-edge techniques like **Optimal Transport Conditional Flow Matching** and **Diffusion Models** for robust and flexible posterior inference.
- **Built on JAX**: Leverages the power of JAX for automatic differentiation, vectorization, and seamless execution on CPUs, GPUs, and TPUs.
- **High-Level Recipes API**: A simplified interface for common workflows, allowing you to train models and run inference with just a few lines of code.
- **Powerful Transformer Models**: Includes implementations of recent, high-performing models like **Flux1** and **Simformer** for handling complex, high-dimensional data.
- **Modular and Extensible**: A clean, well-structured codebase that is easy to understand, modify, and extend for your own research.

## Examples

Examples for this library are available separately in the [GenSBI-examples](https://github.com/aurelio-amerio/GenSBI-examples) repository.

Some key examples include:

**Unconditional Density Estimation:**

- `flow_matching_2d_unconditional.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/examples/flow_matching_2d_unconditional.ipynb) <br>
Demonstrates how to use flow matching in 2D for unconditional density estimation.
- `diffusion_2d_unconditional.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/examples/diffusion_2d_unconditional.ipynb) <br>
Demonstrates how to use diffusion models in 2D for unconditional density estimation.

**Conditional Density Estimation:**

- `two_moons`: \[WIP\] Showcases how to use training pipelines for the benchmark two-moons dataset using Flux1 and Simformer models.

> [!NOTE]
> A full list of the currently available examples is available at the [examples](https://aurelio-amerio.github.io/GenSBI/examples.html) documentation page.

## TODO

The following tasks are planned for future development:

- [x] Implement OT flow matching techniques.
- [x] Implement diffusion models (EDM and score matching).
- [x] Implement Transformer-based models for conditional posterior estimation (Flux1 and Simformer).
- [x] Unify the API for flow matching and diffusion models.
- [x] Implement wrappers to make training of flow matching and diffusion models similar.
- [x] Write tests for core functionalities.
- [ ] Add more examples and benchmarks.
- [ ] Improve documentation and tutorials.
- [ ] Provide better pre-trained models and checkpoints (currently the training is sub-optimal and for illustration purposes only).

## Known Issues
- Bfloat16 support is currently limited and may lead to unexpected behavior.
- Currently, it is not straight forward to load a checkpoint created on GPU on a CPU-only machine (and vice-versa). Soon the model building pipeline will become sharding-aware, which should fix the issue.

## Citation

If you use this library, please consider citing this work and the original methodology papers.

### Reference implementations:
- **Facebook Flow Matching library**: [https://github.com/facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching)
- **Elucidating the Design Space of Diffusion-Based Generative Models**: [https://github.com/NVlabs/edm](https://github.com/NVlabs/edm)
- **Simformer model**: [https://github.com/mackelab/simformer](https://github.com/mackelab/simformer)
- **Flux1 model from BlackForest Lab**: [https://github.com/black-forest-labs/flux](https://github.com/black-forest-labs/flux)

