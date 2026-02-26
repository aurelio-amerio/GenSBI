# %% Imports
import os

# Set JAX backend (use 'cuda' for GPU, 'cpu' otherwise)
# os.environ["JAX_PLATFORMS"] = "cuda"

import grain
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx

# Import the unified UnconditionalPipeline and the generative method
from gensbi.recipes import UnconditionalPipeline
from gensbi.core import FlowMatchingMethod

# For unconditional estimation, we demonstrate a custom MLP model.
# See docs/advanced/custom_models.md for requirements on the model interface.
from gensbi.utils.model_wrapping import _expand_dims, _expand_time
from gensbi.utils.plotting import plot_marginals
import matplotlib.pyplot as plt


# %% Define a simulator
def simulator(key, nsamples):
    return 3 + jax.random.normal(key, (nsamples, 2)) * jnp.array([0.5, 1]).reshape(
        1, 2
    )  # a simple 2D gaussian


# %% Define your training and validation datasets.
train_data = simulator(jax.random.PRNGKey(0), 100_000).reshape(-1, 2, 1)
val_data = simulator(jax.random.PRNGKey(1), 2000).reshape(-1, 2, 1)

# %% Normalize the dataset
means = jnp.mean(train_data, axis=0)
stds = jnp.std(train_data, axis=0)


def normalize(data, means, stds):
    return (data - means) / stds


def unnormalize(data, means, stds):
    return data * stds + means


def process_data(data):
    return normalize(data, means, stds)


# %% Create the input pipeline using Grain
batch_size = 256

train_dataset_grain = (
    grain.MapDataset.source(np.array(train_data))
    .shuffle(42)
    .repeat()
    .to_iter_dataset()
    .batch(batch_size)
    .map(process_data)
)

val_dataset_grain = (
    grain.MapDataset.source(np.array(val_data))
    .shuffle(42)
    .repeat()
    .to_iter_dataset()
    .batch(batch_size)
    .map(process_data)
)


# %% Define a custom model
# This is a simple MLP velocity field model for unconditional estimation.
# It only works for inputs of shape (batch, dim, 1).
# For more complex models, use the transformer-based models in gensbi.models.
#
# Custom models must accept the same arguments as the built-in models:
#   __call__(self, t, obs, node_ids, *args, **kwargs)
# Non-transformer models should use *args and **kwargs to absorb unused arguments
# like node_ids, obs_ids, etc. See docs/advanced/custom_models.md for details.
class MLP(nnx.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 128, *, rngs: nnx.Rngs):

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        din = input_dim + 1

        self.linear1 = nnx.Linear(din, self.hidden_dim, rngs=rngs)
        self.linear2 = nnx.Linear(self.hidden_dim, self.hidden_dim, rngs=rngs)
        self.linear3 = nnx.Linear(self.hidden_dim, self.hidden_dim, rngs=rngs)
        self.linear4 = nnx.Linear(self.hidden_dim, self.hidden_dim, rngs=rngs)
        self.linear5 = nnx.Linear(self.hidden_dim, self.input_dim, rngs=rngs)

    def __call__(self, t: jax.Array, obs: jax.Array, *args, **kwargs):
        # *args and **kwargs absorb node_ids and other unused arguments
        obs = _expand_dims(obs)[
            ..., 0
        ]  # for this model, we use (batch, dim) internally
        t = _expand_time(t)
        t = jnp.broadcast_to(t, (obs.shape[0], 1))

        h = jnp.concatenate([obs, t], axis=-1)

        x = self.linear1(h)
        x = jax.nn.gelu(x)

        x = self.linear2(x)
        x = jax.nn.gelu(x)

        x = self.linear3(x)
        x = jax.nn.gelu(x)

        x = self.linear4(x)
        x = jax.nn.gelu(x)

        x = self.linear5(x)

        return x[..., None]  # return shape (batch, dim, 1)


model = MLP(rngs=nnx.Rngs(42))

# %% Choose the generative method
# The unified UnconditionalPipeline is parameterized by a GenerativeMethod.
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
# The UnconditionalPipeline is model-agnostic: it works with any model that follows
# the standard interface (see docs/advanced/custom_models.md).
training_config = UnconditionalPipeline.get_default_training_config()
training_config["nsteps"] = 10000

dim_obs = 2  # Dimension of the parameter space
ch_obs = 1  # Number of channels

pipeline = UnconditionalPipeline(
    model,
    train_dataset_grain,
    val_dataset_grain,
    dim_obs,
    method=method,
    ch_obs=ch_obs,
    training_config=training_config,
)

# %% Train the model
rngs = nnx.Rngs(42)
pipeline.train(
    rngs, save_model=False
)  # if you want to save the model, set save_model=True

# %% Sample (default ODE solver)
samples = pipeline.sample(rngs.sample(), nsamples=100_000)
samples = unnormalize(samples, means, stds)

# %% Plot the samples
plot_marginals(
    np.array(samples[..., 0]), true_param=[3, 3], gridsize=30, range=[(-2, 8), (-2, 8)]
)
plt.savefig("unconditional_pipeline_marginals.png", dpi=300, bbox_inches="tight")
plt.show()

# %% Alternative: sample with ZeroEndsSolver (SDE-based flow matching sampler)
# Instead of the default deterministic ODE solver, you can use the ZeroEndsSolver
# for stochastic sampling. This can sometimes improve sample diversity.
# The SDE solver requires mu0 (prior mean) and sigma0 (prior std) matching the
# data shape, plus an alpha parameter controlling diffusion strength.
# For a full list of available solvers, see docs/advanced/samplers.md.
from gensbi.flow_matching.solver import ZeroEndsSolver

solver_kwargs = {
    "mu0": jnp.zeros((dim_obs, 1)),    # prior mean (data is normalized)
    "sigma0": jnp.ones((dim_obs, 1)),  # prior std
    "alpha": 1.0,                       # diffusion strength
}

samples_sde = pipeline.sample(
    rngs.sample(), nsamples=100_000,
    solver=(ZeroEndsSolver, solver_kwargs),
)
samples_sde = unnormalize(samples_sde, means, stds)

plot_marginals(
    np.array(samples_sde[..., 0]),
    true_param=[3, 3],
    gridsize=30,
    range=[(-2, 8), (-2, 8)],
)
plt.savefig("unconditional_pipeline_sde_marginals.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
