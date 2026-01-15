# Examples

This page provides links and descriptions for example notebooks and scripts demonstrating the use of GenSBI. Every notebook is a self-contained example that can be run independently.

All examples are available in the [GenSBI-examples repository](https://github.com/aurelio-amerio/GenSBI-examples).

```{tip}
**New to GenSBI?** Start with the [Quick Start Guide](/getting_started/quick_start) and the [my_first_model notebook](/notebooks/my_first_model) for a complete walkthrough.
```

## Getting Started

### My First Model Tutorial
**Recommended starting point** for beginners. This comprehensive tutorial walks through:
- Setting up a simple simulation-based inference problem
- Training a flow matching model
- Sampling from the posterior
- Validating results with SBC, TARP, and L-C2ST

```{toctree}
:maxdepth: 1

notebooks/my_first_model
```

## Unconditional Density Estimation

These examples demonstrate how to use flow matching and diffusion models for unconditional density estimation in 2D. These are useful for understanding the basics of generative modeling before moving to conditional inference.

**What you'll learn:**
- How to train flow matching models on arbitrary 2D distributions
- How to train diffusion models on 2D data
- Basics of data preparation and visualization

```{toctree}
:maxdepth: 1

notebooks/flow_matching_2d_unconditional
notebooks/diffusion_2d_unconditional
```

## Posterior Density Estimation: SBI Benchmarks

This series of examples demonstrates how to use GenSBI for posterior density estimation on standard Simulation-Based Inference benchmarks. These benchmarks are commonly used to evaluate SBI methods.

### Two Moons Problem

The **two moons** problem is a classic 2D benchmark with a bimodal posterior. Great for visualizing how flow matching captures multimodal distributions.

**What you'll learn:**
- How to use the Simformer model on simple problems
- How to use the Flux1 model on simple problems
- Visualizing bimodal posterior distributions

```{toctree}
:maxdepth: 1

notebooks/two_moons_flow_simformer
notebooks/two_moons_flow_flux
```

### Gaussian Linear Problem

The **Gaussian linear** problem involves inferring parameters of a linear Gaussian model. This is a higher-dimensional problem that tests scalability.

**What you'll learn:**
- Using Simformer for medium-dimensional problems
- Using Flux1Joint for explicit joint modeling
- Handling higher-dimensional inference

```{toctree}
:maxdepth: 1

notebooks/gaussian_linear_simformer
notebooks/gaussian_linear_flux1joint
```

### SLCP (Simple Likelihood Complex Posterior)

The **SLCP** benchmark features a simple likelihood but a complex, multimodal posterior distribution. This tests how well models can learn intricate posterior structures.

**What you'll learn:**
- Handling complex multimodal posteriors
- Using Simformer for challenging distributions

```{toctree}
:maxdepth: 1

notebooks/slcp_flow_simformer
```

### Advanced Examples: Custom Pipelines & Embeddings

These advanced examples demonstrate how to write a custom embedding network for the Flux1 backbone using the `ConditionalFlowPipeline`. They showcase how to handle specific data structures (1D/2D) with custom architectures.

**Lensing Example (2D)**
Demonstrates how to embed 2D (mock) gravitational lensing data using a custom CNN.
```{toctree}
:maxdepth: 1
notebooks/lensing_example
```

**Gravitational Waves Example (1D)**
Demonstrates how to embed 1D (mock) gravitational waves data using a custom CNN.
```{toctree}
:maxdepth: 1
notebooks/gw_example
```

## Running the Examples

### Option 1: Google Colab (Easiest)

Most examples include a "Open in Colab" badge. Click it to run the notebook in Google Colab without any local setup.

### Option 2: Local Jupyter

1. Clone the examples repository:
   ```bash
   git clone https://github.com/aurelio-amerio/GenSBI-examples.git
   cd GenSBI-examples
   ```

2. Install dependencies (see the [Installation Guide](/getting_started/installation) for details):
   ```bash
   pip install jupyter
   pip install "GenSBI[cuda12,examples] @ git+https://github.com/aurelio-amerio/GenSBI.git"
   ```

3. Launch Jupyter:
   ```bash
   jupyter notebook
   ```

4. Open and run the notebooks in the `examples/` directory.

## Next Steps

After exploring the examples:

1. **Understand the Architecture**: Read the [Conceptual Overview](/basics/overview) to understand how GenSBI is structured.

2. **Choose Your Model**: See [Model Cards](/basics/model_cards) for guidance on selecting the right model for your problem.

3. **Dive Deeper**: Read the [Training Guide](/basics/training), [Inference Guide](/basics/inference), and [Validation Guide](/basics/validation).

4. **Apply to Your Problem**: Use GenSBI for your own simulation-based inference tasks!

## Questions or Issues?

- Check the [Troubleshooting Guide](/basics/troubleshooting) for common issues
- Open an issue on the [GitHub repository](https://github.com/aurelio-amerio/GenSBI/issues)
- Read the [API Documentation](/api/gensbi/index) for detailed function signatures

