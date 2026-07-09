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

getting_started/my_first_model
```

## Neural Density Estimators (NDE)

These examples demonstrate how to use flow matching and diffusion models for unconditional density estimation in 2D. These are useful for understanding the basics of generative modeling before moving to conditional inference.

**What you'll learn:**

- Training flow matching models on arbitrary 2D distributions
- Training diffusion models on 2D data
- Basics of data preparation and visualization

```{toctree}
:maxdepth: 1

notebooks/diffusion_EDM_2d_unconditional
notebooks/diffusion_SM_2d_unconditional
notebooks/flow_matching_2d_unconditional
notebooks/flow_matching_2d_unconditional_flux1joint
```

## SBI Benchmark Examples

This series covers standard benchmarks from the Simulation-Based Inference literature (based on `sbibm`). These tasks allow you to evaluate GenSBI methods against known ground-truth posteriors.

### Two Moons Problem

The **two moons** problem is a classic 2D benchmark with a bimodal posterior. It is excellent for visualizing how flow matching captures multimodal distributions.

**What you'll learn:**

- Using the Simformer model on simple problems
- Using the Flux1 model on simple problems
- Visualizing bimodal posterior distributions

```{toctree}
:maxdepth: 1

notebooks/two_moons_diffusion_flux
notebooks/two_moons_diffusion_flux1joint
notebooks/two_moons_diffusion_simformer
notebooks/two_moons_flow_flux
notebooks/two_moons_flow_flux1joint
notebooks/two_moons_flow_simformer
```

### Bernoulli Generalized Linear Model

The **Bernoulli Generalized Linear Model** (**GLM)** task involves inferring the parameters of a 10-dimensional Generalized Linear Model. The observations provided are the **sufficient statistics** of the process rather than raw binary data. This creates a concise 10-dimensional problem that tests the model's ability to handle GLM structures with smoothness-inducing priors.

**What you'll learn:**

- Inferring parameters for Generalized Linear Models
- Handling datasets comprised of sufficient statistics
- Comparing Flux and Flux1Joint on 10D problems

```{toctree}
:maxdepth: 1

notebooks/bernoulli_glm_flow_flux
notebooks/bernoulli_glm_flow_flux1joint
```

### Gaussian Linear Problem

The **Gaussian linear** problem involves inferring the mean of a 10D Gaussian model with a fixed covariance. While conceptually simple, it serves as a baseline for testing scalability and calibration in higher dimensions.

**What you'll learn:**

- Using Simformer for medium-dimensional problems
- Using Flux1Joint for explicit joint modeling
- verifying calibration on high-dimensional Gaussian posteriors

```{toctree}
:maxdepth: 1

notebooks/gaussian_linear_flow_flux
notebooks/gaussian_linear_flow_flux1joint
```

### Gaussian Mixture

The **Gaussian Mixture** task requires inferring the common mean of a mixture of two 2D Gaussian distributions: one with a broad covariance and one with a narrow covariance. This is a classic ABC benchmark that tests how well the inference method handles heavy-tailed noise and varying scales of variance.

**What you'll learn:**

- Handling mixture models with disparate variances
- Robustness to outlier-prone likelihoods
- Visualizing posteriors with varying sharpness

```{toctree}
:maxdepth: 1

notebooks/gaussian_mixture_flow_flux
notebooks/gaussian_mixture_flow_flux1joint
```

### SLCP (Simple Likelihood Complex Posterior)

The **SLCP** benchmark features a simple likelihood but a complex, multimodal posterior distribution (typically four symmetric modes). This tests how well models can learn intricate posterior structures from a non-linear mapping.

For this task, we train all three main models (**Flux1**, **Flux1Joint**, and **Simformer**) to enable direct performance comparison. Results highlight that while Flux1 is generally powerful, models trained to reconstruct the joint distribution (Flux1Joint and Simformer) often perform better on this task. This suggests that learning to reconstruct the likelihood alongside the posterior can be beneficial for likelihood-dominated problems.

**What you'll learn:**

- Capturing complex, symmetric multimodal posteriors
- Comparing Flux1, Flux1Joint, and Simformer performance
- Understanding when joint distribution learning is beneficial over direct posterior estimation
- Neural likelihood estimation with the experimental TarFlow normalizing flow + MCLMC posterior sampling

```{toctree}
:maxdepth: 1

notebooks/slcp_flow_flux
notebooks/slcp_flow_flux1joint
notebooks/slcp_flow_simformer
notebooks/slcp_tarflow_nle
```

## Advanced SBI examples with custom embedding networks

These examples demonstrate how to write a custom embedding network for the Flux1 model. This is necessary when using complex data types—such as time series or images—where you need a specialized architecture (like a CNN) to process the data before passing it to the flow matching model.

### Gravitational Waves (Time Series 1D)

The **Gravitational Waves** example demonstrates how to embed 1D (mock) gravitational wave data using a custom CNN.

**What you'll learn:**

- Handling 1D time-series data
- Designing custom embedding networks for Flux1

```{toctree}
:maxdepth: 1

notebooks/gw_example
```

### Gravitational Lensing (2D Images)

The **Gravitational Lensing** example demonstrates how to embed 2D (mock) gravitational lensing data using a custom CNN.

**What you'll learn:**

- Handling 2D image data
- Integrating vision-based embedding networks with Flux1

```{toctree}
:maxdepth: 1

notebooks/lensing_example
```

## Running the Examples

### Option 1: Google Colab (Easiest)

Most examples include an "Open in Colab" badge. Click it to run the notebook in Google Colab without any local setup.

### Option 2: Local Jupyter

1. Clone the examples repository:

   ```bash
   git clone https://github.com/aurelio-amerio/GenSBI-examples.git
   cd GenSBI-examples
   ```

2. Install dependencies (see the [Installation Guide](/getting_started/installation) for details):

   ```bash
   # using uv (recommended)
   uv add jupyter "gensbi[cuda12,examples]"

   # or using pip
   pip install jupyter "gensbi[cuda12,examples]"
   ```

3. Launch Jupyter:

   ```bash
   jupyter notebook
   ```