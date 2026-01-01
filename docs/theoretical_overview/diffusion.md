# EDM Diffusion Models: A Short Overview

Based on the article *Elucidating the Design Space of Diffusion-Based Generative Models* ([Karras et al., 2022](https://arxiv.org/pdf/2206.00364)). 

The **EDM (Elucidated Diffusion Model)** framework streamlines the theory and practice of diffusion models. It unifies previous approaches (like DDPM and score-based models) by focusing on a "tangible" design space that separates sampling schedules, network parameterization, and training objectives.

## Neural Density Estimation via Denoising

To understand diffusion models from a statistical perspective, imagine a process that gradually corrupts a complex data distribution (like images or simulator parameters) by adding Gaussian noise until it becomes indistinguishable from pure white noise.

Generative modeling in this framework is simply the **reverse** of this process. We start with a sample from a simple prior distribution (pure noise) and iteratively remove the noise to recover a sample from the target data distribution.

In the EDM framework, this reverse process is learned via a regression task. We train a neural network, called a **denoiser** $D_\theta(x; \sigma)$, to predict the original clean data $y$ given a noisy observation $x = y + n$, where $n \sim \mathcal{N}(0, \sigma^2 I)$. By learning to denoise at every intensity level $\sigma$, the model implicitly learns the **score function**—the gradient of the log-probability density ($\nabla_x \log p(x)$). This gradient points towards regions of higher data density, effectively guiding the transformation from unstructured noise back to structured data.

## Key Formulas

**1. The Probability Flow ODE** While diffusion processes are often described as Stochastic Differential Equations (SDEs), EDM emphasizes the deterministic **Probability Flow ODE** (PF ODE) as the theoretical backbone that shares the same marginal distributions. The data evolves according to:

$$d\mathbf{x} = -\dot{\sigma}(t)\sigma(t)\nabla_{\mathbf{x}} \log p(\mathbf{x}; \sigma(t)) dt$$

where $\sigma(t)$ is the noise schedule and $\nabla_{\mathbf{x}} \log p(\mathbf{x}; \sigma)$ is the score function.

**2. Score Parameterization and Denoiser** A key contribution of EDM is **preconditioning**. Instead of predicting the noise or the score directly, the neural network $F_\theta$ is wrapped to ensure stable training inputs and outputs across all noise levels. The denoiser $D_\theta$ is parameterized as:

$$D_\theta(\mathbf{x}; \sigma) = c_{skip}(\sigma)\mathbf{x} + c_{out}(\sigma)F_\theta(c_{in}(\sigma)\mathbf{x}; c_{noise}(\sigma))$$

where coefficients $c_{skip}, c_{out}, c_{in}, c_{noise}$ are chosen to maintain unit variance for the network inputs and training targets, ensuring effective training dynamics independent of the noise level $\sigma$.

**3. The Training Loss** The model is trained using a weighted mean squared error (MSE) objective on the denoiser output. The loss function averages over training data $y$, Gaussian noise $n$, and noise levels $\sigma$:

$$\mathcal{L} = \mathbb{E}_{\sigma, y, n} \left[ \lambda(\sigma) \| D_\theta(y + n; \sigma) - y \|^2 \right]$$

The weighting function $\lambda(\sigma)$ is typically set to $1/c_{out}(\sigma)^2$ to balance the effective learning rate across noise levels.

**4. Stochastic Sampling (Algorithm 2)** To generate samples, this library implements the stochastic sampler proposed by Karras et al. (Algorithm 2). This method combines a 2nd-order ODE solver (Heun's method) with explicit Langevin-like "churn" to correct approximation errors during the trajectory.

The update step involves:

1. **Churn:** Temporarily increase the noise level from $t_i$ to $\hat{t}_i$ by injecting fresh random noise. This helps the sampler explore the manifold and correct earlier errors.
2. **Predict & Correct:** Solve the ODE backward from $\hat{t}_i$ to $t_{i+1}$ using Heun's method (a predictor-corrector scheme).

## Applications: Simulation-Based Inference (SBI)

In the context of **Simulation-Based Inference (SBI)**, the goal is to estimate the posterior distribution of simulator parameters $\theta$ given observed data $x$, denoted as $p(\theta|x)$.

**Generating Posterior Samples** This library uses the **Stochastic Sampler (Algorithm 2)** to generate posterior samples. After training the denoiser conditionally as $D_\psi(\theta; \sigma, x)$, we sample initial latent noise $\theta_{N} \sim \mathcal{N}(0, \sigma_{max}^2 I)$ and evolve it backward to $\sigma=0$ conditioned on the observation $x$.

The addition of stochastic churn in Algorithm 2 is beneficial for SBI tasks involving complex posteriors. As noted by Karras et al., the explicit Langevin-like noise addition actively corrects approximation errors accumulated during the sampling steps. This prevents the trajectory from drifting off the target manifold, which is critical for faithfully representing the diversity of the posterior distribution compared to prior purely deterministic methods that may suffer from accumulated truncation errors.