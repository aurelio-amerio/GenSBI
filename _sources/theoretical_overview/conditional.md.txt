# Conditional Density Estimation

The three generative frameworks described in the preceding pages — [score matching](score_matching), the [EDM probability flow ODE](diffusion), and [flow matching](flow) — were presented in their unconditional form: given a dataset of samples $\{\mathbf{z}^{(i)}\}$, the network learns to generate new samples from the same distribution. Simulation-based inference, however, requires *conditional* generation: given an observation $\mathbf{x}_\mathrm{obs}$, the goal is to draw posterior samples $\boldsymbol{\theta} \sim p(\boldsymbol{\theta} \mid \mathbf{x}_\mathrm{obs})$.

## From Unconditional to Conditional Generation

The extension from unconditional to conditional generation is straightforward in all three frameworks. The key idea is to supply the conditioning variable $\mathbf{x}$ as an **additional input** to the neural network, so that the learned velocity field, score function, or denoiser becomes a function of both the noisy state and the condition:

| Framework | Unconditional | Conditional |
|---|---|---|
| Score matching | $s_\phi(\boldsymbol{\theta}_t, t)$ | $s_\phi(\boldsymbol{\theta}_t, t, \mathbf{x})$ |
| EDM | $D_\phi(\boldsymbol{\theta}_t; \sigma)$ | $D_\phi(\boldsymbol{\theta}_t; \sigma, \mathbf{x})$ |
| Flow matching | $v_\phi(\boldsymbol{\theta}_t, t)$ | $v_\phi(\boldsymbol{\theta}_t, t, \mathbf{x})$ |

The training objectives carry over with a trivial modification: one conditions on $\mathbf{x}$ drawn from the joint training set $\{(\boldsymbol{\theta}^{(i)}, \mathbf{x}^{(i)})\}$ and minimises the expected loss over both the noise level and the conditioning variable. For conditional flow matching, the loss becomes:

$$\mathcal{L}_\mathrm{CFM}(\phi) = \mathbb{E}_{t,\, q(\boldsymbol{\theta}_1, \mathbf{x}),\, p_t(\boldsymbol{\theta} \mid \boldsymbol{\theta}_1)} \| v_\phi(\boldsymbol{\theta}_t, t, \mathbf{x}) - u_t(\boldsymbol{\theta} \mid \boldsymbol{\theta}_1) \|^2,$$

where the only change relative to the unconditional case is the presence of $\mathbf{x}$ as a network input.

## Neural Posterior Estimation (NPE)

This conditional formulation is the direct mathematical basis for **Neural Posterior Estimation (NPE)**. The network is trained on simulated pairs $\{(\boldsymbol{\theta}^{(i)}, \mathbf{x}^{(i)})\}$ drawn from the joint distribution $p(\boldsymbol{\theta}, \mathbf{x}) = p(\mathbf{x} \mid \boldsymbol{\theta})\, p(\boldsymbol{\theta})$, where $p(\boldsymbol{\theta})$ is the prior and $p(\mathbf{x} \mid \boldsymbol{\theta})$ is the simulator.

After training, the model generates samples $\boldsymbol{\theta} \sim q_\phi(\boldsymbol{\theta} \mid \mathbf{x}_\mathrm{obs})$ for any new observation $\mathbf{x}_\mathrm{obs}$ by running the reverse SDE, probability flow ODE, or flow ODE with $\mathbf{x}_\mathrm{obs}$ held fixed. The inference cost is **independent of the simulator**: expensive forward simulations are performed once during dataset generation, and the trained model can be applied to as many observations as needed without additional simulations. This is the **amortization** property.

## Conditioning Mechanisms

How the conditioning information enters the network depends on the architecture. Three mechanisms are commonly used:

- **Cross-attention:** the embedding of $\mathbf{x}$ is projected into a separate key–value sequence, and the model's intermediate representations attend to it through standard multi-head attention. This allows selective extraction of relevant features from the condition at each layer.

- **Concatenation:** the embedded condition is appended to the noisy state as additional input tokens and processed jointly through self-attention. This is the simplest approach and works well when the condition and target have comparable dimensionality.

- **Adaptive Layer Normalization (adaLN):** the conditioning vector modulates the layer normalization parameters — the scale and shift are regressed from the condition rather than learned as fixed parameters. The adaLN-Zero variant additionally regresses a per-block scaling factor initialized to zero, so each transformer block starts as the identity function.

In GenSBI, **Flux1** uses cross-attention to inject the condition through double-stream transformer blocks, while **SimFormer** and **Flux1Joint** rely on concatenation with node-level embeddings.

## Joint Density Estimation

Beyond conditional generation, the same frameworks support **joint** density estimation. Instead of learning the conditional $p(\boldsymbol{\theta} \mid \mathbf{x})$, a joint model learns the full distribution $p(\boldsymbol{\theta}, \mathbf{x})$ and recovers any conditional at inference time by fixing the appropriate variables.

In GenSBI, this is implemented through a binary `condition_mask` that indicates which variables are observed and which are to be generated. During training, the mask is randomised, exposing the model to all possible conditioning patterns. At inference time:

- Fix $\mathbf{x} = \mathbf{x}_\mathrm{obs}$ and generate $\boldsymbol{\theta}$ → **posterior samples**
- Fix $\boldsymbol{\theta}$ and generate $\mathbf{x}$ → **likelihood samples**

Enabling this flexibility requires no architectural changes beyond randomising the mask — the same training loop handles all masking patterns. **SimFormer** and **Flux1Joint** support this joint mode in GenSBI.

## Unconditional Mode

GenSBI also supports **unconditional** density estimation, in which neither parameters nor observations play a distinguished role. The model learns to sample from an arbitrary target distribution $p(\mathbf{z})$ without any conditioning input. This makes GenSBI usable as a general-purpose neural density estimation library, extending its applicability beyond SBI to any problem that requires learning and sampling from complex distributions.
