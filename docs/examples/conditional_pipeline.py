# %% Imports
import os

# Set JAX backend (use 'cuda' for GPU, 'cpu' otherwise)
# os.environ["JAX_PLATFORMS"] = "cuda"

import grain
import numpy as np
import jax
from jax import numpy as jnp
from numpyro import distributions as dist
from flax import nnx

# Import the unified ConditionalPipeline and the generative method
from gensbi.recipes import ConditionalPipeline
from gensbi.core import FlowMatchingMethod

# We use the Flux1 model, but any model with the correct interface works.
# See docs/advanced/custom_models.md for how to use a custom model.
from gensbi.models import Flux1, Flux1Params

from gensbi.utils.plotting import plot_marginals
import matplotlib.pyplot as plt


# %%

theta_prior = dist.Uniform(
    low=jnp.array([-2.0, -2.0, -2.0]), high=jnp.array([2.0, 2.0, 2.0])
)

dim_obs = 3
dim_cond = 3
dim_joint = dim_obs + dim_cond


# %%
def simulator(key, nsamples):
    theta_key, sample_key = jax.random.split(key, 2)
    thetas = theta_prior.sample(theta_key, (nsamples,))

    xs = thetas + 1 + jax.random.normal(sample_key, thetas.shape) * 0.1

    thetas = thetas[..., None]
    xs = xs[..., None]

    # when making a dataset for the conditional pipeline, thetas need to come first
    data = jnp.concatenate([thetas, xs], axis=1)

    return data


# %% Define your training and validation datasets.
train_data = simulator(jax.random.PRNGKey(0), 100_000)
val_data = simulator(jax.random.PRNGKey(1), 2000)


# %% Normalize the dataset
means = jnp.mean(train_data, axis=0)
stds = jnp.std(train_data, axis=0)


def normalize(data, means, stds):
    return (data - means) / stds


def unnormalize(data, means, stds):
    return data * stds + means


# %% Prepare the data for the pipeline
# The conditional pipeline expects the data to be split into (observations, conditions).
def split_obs_cond(data):
    data = normalize(data, means, stds)
    return (
        data[:, :dim_obs],
        data[:, dim_obs:],
    )


# %% Create the input pipeline using Grain
batch_size = 256

train_dataset_grain = (
    grain.MapDataset.source(np.array(train_data))
    .shuffle(42)
    .repeat()
    .to_iter_dataset()
    .batch(batch_size)
    .map(split_obs_cond)
)

val_dataset_grain = (
    grain.MapDataset.source(np.array(val_data))
    .shuffle(42)
    .repeat()
    .to_iter_dataset()
    .batch(batch_size)
    .map(split_obs_cond)
)

# %% Define the model
# We use Flux1 as our neural network architecture.
params = Flux1Params(
    in_channels=1,
    vec_in_dim=None,
    context_in_dim=1,
    mlp_ratio=3,
    num_heads=2,
    depth=4,
    depth_single_blocks=8,
    axes_dim=[
        10,
    ],
    qkv_bias=True,
    dim_obs=dim_obs,
    dim_cond=dim_cond,
    id_embedding_strategy=("absolute", "absolute"),
    theta=10 * dim_joint,
    rngs=nnx.Rngs(default=42),
    param_dtype=jnp.float32,
)

model = Flux1(params)

# %% Choose the generative method
# The unified ConditionalPipeline is parameterized by a GenerativeMethod.
# Here we use FlowMatchingMethod (recommended).
method = FlowMatchingMethod()

# Alternative methods (uncomment to use):
# from gensbi.core import DiffusionEDMMethod
# method = DiffusionEDMMethod()             # EDM diffusion (default EDM scheduler)
# method = DiffusionEDMMethod(sde="VP")     # EDM with VP scheduler
# method = DiffusionEDMMethod(sde="VE")     # EDM with VE scheduler
#
# from gensbi.core import ScoreMatchingMethod
# method = ScoreMatchingMethod()             # Score matching (default VP SDE)
# method = ScoreMatchingMethod(sde_type="VE")  # Score matching with VE SDE

# %% Instantiate the pipeline
# The ConditionalPipeline is model-agnostic: it works with any model that follows
# the standard interface (see docs/advanced/custom_models.md).
training_config = ConditionalPipeline.get_default_training_config()
training_config["nsteps"] = 10000

pipeline = ConditionalPipeline(
    model,
    train_dataset_grain,
    val_dataset_grain,
    dim_obs=dim_obs,
    dim_cond=dim_cond,
    method=method,
    training_config=training_config,
)

# %% Train the model
rngs = nnx.Rngs(42)
pipeline.train(
    rngs, save_model=False
)  # if you want to save the model, set save_model=True

# %% Sample from the posterior (default ODE solver)
new_sample = simulator(jax.random.PRNGKey(20), 1)
true_theta = new_sample[:, :dim_obs, :]

new_sample = normalize(new_sample, means, stds)
x_o = new_sample[:, dim_obs:, :]

samples = pipeline.sample(rngs.sample(), x_o, nsamples=100_000)
samples = unnormalize(samples, means[:dim_obs], stds[:dim_obs])

# %% Plot the samples
plot_marginals(
    np.array(samples[..., 0]),
    gridsize=30,
    true_param=np.array(true_theta[0, :, 0]),
    range=[(1, 3), (1, 3), (-0.6, 0.5)],
)
plt.savefig("conditional_pipeline_marginals.png", dpi=100, bbox_inches="tight")
plt.show()

# %% Alternative: sample with ZeroEndsSolver (SDE-based flow matching sampler)
# Instead of the default deterministic ODE solver, you can use the ZeroEndsSolver
# for stochastic sampling. This can sometimes improve sample diversity.
# The SDE solver requires mu0 (prior mean) and sigma0 (prior std) matching the
# data shape, plus an alpha parameter controlling diffusion strength.
# For a full list of available solvers, see docs/advanced/samplers.md.
from gensbi.flow_matching.solver import ZeroEndsSolver

solver_kwargs = {
    "mu0": jnp.zeros((dim_obs, 1)),  # prior mean (data is normalized)
    "sigma0": jnp.ones((dim_obs, 1)),  # prior std
    "alpha": 0.2,  # diffusion strength
}

samples_sde = pipeline.sample(
    rngs.sample(),
    x_o,
    nsamples=100_000,
    solver=(ZeroEndsSolver, solver_kwargs),
)
samples_sde = unnormalize(samples_sde, means[:dim_obs], stds[:dim_obs])

plot_marginals(
    np.array(samples_sde[..., 0]),
    gridsize=30,
    true_param=np.array(true_theta[0, :, 0]),
    range=[(1, 3), (1, 3), (-0.6, 0.5)],
)
plt.savefig("conditional_pipeline_sde_marginals.png", dpi=100, bbox_inches="tight")
plt.show()

# %%
