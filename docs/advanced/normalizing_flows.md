# Normalizing Flows (experimental)

```{warning}
Discrete normalizing flows are **experimental** in GenSBI. The API is
functional and tested, but may change between releases.
```

Alongside its flow-matching and diffusion methods, GenSBI provides discrete
normalizing flows: conditional density models `q(obs | cond)` whose exact
log-density is available in a **single forward pass**, with no ODE
integration. This makes them natural for likelihood-dominated workflows:

- **NPE** (neural posterior estimation): model `q(theta | x)` directly and
  sample it.
- **NLE** (neural likelihood estimation): model `q(x | theta)`, then sample
  the posterior `p(theta | x_o) ∝ p(theta) q(x_o | theta)` with MCMC —
  practical because the flow's likelihood is exact and cheap to evaluate.

## Models

### MAFlow — Masked Autoregressive Flow

`MAFlow` stacks masked-MLP (MADE) autoregressive layers with affine or
rational-quadratic-spline transformers. It is small, fast to train, and a
solid default for tabular problems up to a few tens of dimensions.

```python
from flax import nnx
from gensbi.models import MAFlow, MAFlowParams

flow = MAFlow(MAFlowParams(rngs=nnx.Rngs(0), dim=dim_theta, cond_dim=dim_x))
```

### TarFlow — Transformer Autoregressive Flow

`TarFlow` ports Apple's TarFlow/STARFlow transformer autoregressive flow:
stacked causal-attention blocks with alternating token permutations. It
scales to larger problems and supports structured (image) modeled variables
and conditions.

```python
from gensbi.models import TarFlow, TarFlowParams

flow = TarFlow(TarFlowParams(rngs=nnx.Rngs(0), dim=dim_x, cond_dim=dim_theta,
                             cond="vector", num_blocks=4, layers_per_block=2))
```

The `cond` argument selects the conditioning mechanism: `"bias"` (additive
bias), `"vector"` (per-coordinate condition tokens), or `"image"` (a
patchified image condition attended to as a prefix).

## Training with ConditionalFlowPipeline

`ConditionalFlowPipeline` mirrors the flow-matching pipeline surface
(`sample` / `sample_batched` / `log_prob` / `get_sampler` /
`get_log_prob_fn`), so the diagnostics run unchanged. All tabular tensors
carry the uniform `(B, dim, C)` channel convention (`C = 1` for plain
vectors).

```python
import jax
from flax import nnx
from gensbi.recipes import ConditionalFlowPipeline

pipeline = ConditionalFlowPipeline(
    flow, train_ds, val_ds, dim_obs=dim_theta, dim_cond=dim_x)
pipeline.fit_standardization(theta_train)      # before train()
pipeline.train(nnx.Rngs(0))

x_o = x_observed.reshape(1, dim_x, 1)          # one observation: (1, dim_cond, C)
samples = pipeline.sample(jax.random.PRNGKey(0), x_o)  # (nsamples, dim_theta, 1)
```

Single-observation methods take exactly one observation of shape
`(1, dim_cond, C)`; a batch of conditions goes to `sample_batched`, and a
batched input to a single-observation method raises `ValueError`.

## NLE posterior sampling

For NLE, train a flow with `obs = x`, `cond = theta` (the flow models the
likelihood) — i.e. swap the roles from the training example above:
`dim_obs=dim_x`, `dim_cond=dim_theta`. Then wrap the trained flow in
{class}`~gensbi.inference.NLEPosterior`:

```python
from gensbi.core.prior import make_gaussian_prior
from gensbi.inference import MCLMC, NestedSampler, NLEPosterior, TemperedSMC

# nle_pipeline: a ConditionalFlowPipeline trained with dim_obs=dim_x, dim_cond=dim_theta
posterior = NLEPosterior(nle_pipeline.ema_model, prior=make_gaussian_prior((dim_theta,)))
samples = posterior.sample(jax.random.PRNGKey(0), x_o)   # adjusted MCLMC by default
samples, info = posterior.sample(jax.random.PRNGKey(0), x_o,
                                 sampler=TemperedSMC(), return_info=True)
```

The default sampler is adjusted microcanonical Langevin Monte Carlo
(blackjax MCLMC); adaptive tempered SMC is available for multimodal
posteriors. These are convenience samplers — for full control build a
`PosteriorTarget` via `posterior.build_target(x_o)` and run your own
blackjax loop.

For multimodal posteriors where you also want the **model evidence**,
{class}`~gensbi.inference.NestedSampler` runs blackjax nested slice
sampling from prior-drawn live points. Unlike the MCMC samplers it needs
no tempering to cross modes and returns the log evidence in its info
object:

```python
samples, info = posterior.sample(jax.random.PRNGKey(0), x_o,
                                 sampler=NestedSampler(num_samples=2000),
                                 return_info=True)
print(info.log_evidence, info.log_evidence_err)   # for model comparison
```

The returned {class}`~gensbi.inference.NestedSamplerInfo` also reports the
effective sample size and dead-point count. With-replacement resampling
can duplicate draws when `num_samples` approaches the run's ESS; pass
`num_rejuvenation_steps > 0` to break the duplicated atoms with
posterior-invariant slice moves.

The [Two Moons MAF NLE notebook](/notebooks/two_moons_maf_nle) works
through nested sampling end to end on a bimodal posterior, including the
log-evidence cross-check against tempered SMC.

## Saving and loading

Flows serialize like any other GenSBI model with the portable safetensors
helpers:

```python
from gensbi.utils.serialization import load_safetensors, save_safetensors

save_safetensors(pipeline.ema_model, "flow.safetensors")
flow2 = MAFlow(params)                     # rebuild the architecture from Params
load_safetensors(flow2, "flow.safetensors")
```

## End-to-end example

See the [SLCP TarFlow NLE notebook](/notebooks/slcp_tarflow_nle) for a
complete workflow: simulate, train a TarFlow likelihood, sample the
posterior with MCLMC, and check calibration. The
[Two Moons MAF NLE notebook](/notebooks/two_moons_maf_nle) covers the same
arc with a MAF likelihood and contrasts two multimodal samplers — tempered
SMC and nested sampling — on a bimodal posterior.
