# GenSBI
![Tests](img/badges/tests.svg)
![Coverage](img/badges/coverage.svg)
![GenSBI Logo](docs/_static/logo.png)

**Warning**: This library is in an early stage of development and will change significantly in the future.

## Overview

**GenSBI** is a library for Simulation-Based Inference (SBI) adopting Optimal Transport Flow Matching and Diffusion models in JAX. It provides tools for probabilistic modeling and simulation, inspired by cutting-edge research and implementations, including:

- **Facebook Flow Matching library**: [https://github.com/facebookresearch/flow_matching]
- **Elucidating the Design Space of Diffusion-Based Generative Models**: [https://github.com/NVlabs/edm]
- **Simformer model**: [https://github.com/mackelab/simformer]
- **Flux1 model from BlackForest Lab**: [https://github.com/black-forest-labs/flux]

## Contents

### `src/`
The `src` directory contains the core implementation of the library:

- **Flow Matching**: Implements flow matching techniques, including paths, solvers, and utilities.
- **Diffusion**: Contains diffusion models and utilities for training and evaluation.
- **Models**:
  - **Flux1**: A transformer-based architecture for flow matching on sequences.
  - **Simformer**: Implements the Simformer model for all-in-one simulation tasks.
- **Loss Functions**: Includes loss functions tailored for flow matching tasks, such as:
  - `FluxCFMLoss`
  - `SimformerCFMLoss`

### Examples
Examples for this library are avaialble separately in the [GenSBI-examples repository](https://github.com/aurelio-amerio/GenSBI-examples)

#### Flow Matching
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/examples/flow_matching_2d_unconditional.ipynb)`flow_matching_2d_unconditional.ipynb` Demonstrates how to use flow matching in 2D.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/examples/diffusion_2d_unconditional.ipynb) `diffusion_2d_unconditional.ipynb` Demonstrates how to use diffusion models in 2D.

#### SBI Benchmarks
- `two_moons`: Contains benchmarks for the two-moons dataset using Flux1 and Simformer models.

These examples showcase training, evaluation, and visualization of flow matching models.

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
- Currently, it is not straight forward to load a checkpoint created on GPU on a CPU-only machine (and vice-versa). This is an underlying issue with Flax/Orbax serialization, and the documentation will be updated once I find a solution. 
- Currently `diffrax` is not compatible with `jax >= 0.7.*`, as such the library is pinned to `jax==0.6.2`. This will be fixed as soon as `diffrax` supports `jax` 0.7.

## Citation

If you use this library, please consider citing this work and the original methodology papers.

