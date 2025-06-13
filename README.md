# gen-sbi

**Warning**: This library is in an early stage of development and will change significantly in the future.

Continuous Flow Matching and Diffusion models in JAX.

This library implements continuous flow matching and diffusion techniques for probabilistic modeling and simulation using JAX. It is inspired by the Facebook Flow Matching library ([link]) and EDM ([link]), the Simformer model from all-in-one simulation based inference ([link]), and the Flux1 model from BlackForest Lab ([link]) and the jflux implementation ([link]).

## Contents

### `src/`
The `src` directory contains the core implementation of the library:
- **Flow Matching**: Implements continuous flow matching techniques, including paths, solvers, and utilities.
- **Diffusion**: Contains diffusion models and utilities for training and evaluation.
- **Models**:
  - `flux1`: Contains the Flux1 model, a transformer-based architecture for flow matching on sequences.
  - `simformer`: Implements the Simformer model for all-in-one simulation tasks.
- **Loss Functions**: Includes loss functions tailored for flow matching tasks, such as `FluxCFMLoss` and `SimformerCFMLoss`.

### `examples/`
The `examples` directory provides Jupyter notebooks demonstrating the usage of the library:
- **Flow Matching**:
  - `flow_matching_2d.ipynb`: Demonstrates continuous flow matching in 2D.
  - `flow_matching_2d_discrete.ipynb`: Explores discrete flow matching techniques.
- **SBI Benchmarks**:
  - `sbi_example.ipynb`: Example of simulation-based inference using flow matching.
  - `two_moons/`: Contains benchmarks for the two-moons dataset using Flux1 and Simformer models.

These examples showcase training, evaluation, and visualization of flow matching models.

## TODO
- [x] Implement continuous flow matching techniques.
- [x] Implement diffusion models (EDM and score matching).
- [x] Implement Transformer-based models for conditional posterior estimation (Flux1 and Simformer).
- [ ] Unify the API for flow matching and diffusion models.
- [ ] Implemente wrappers to make training of flow matching and diffusion models similar.
- [ ] Add more examples and benchmarks.
- [ ] Improve documentation and tutorials.
- [ ] Write tests for core functionalities.

## Citation
If you use this library, please consider citing also the original sources:
- Facebook Flow Matching library ([https://github.com/facebookresearch/flow_matching])
- Elucidating the Design Space of Diffusion-Based Generative Models ([https://github.com/NVlabs/edm])
- Simformer model ([https://github.com/mackelab/simformer/tree/main])
- Flux1 model from BlackForest Lab ([https://github.com/black-forest-labs/flux])