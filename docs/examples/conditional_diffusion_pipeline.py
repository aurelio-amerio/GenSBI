# %% Imports
import os

# Set JAX backend (use 'cuda' for GPU, 'cpu' otherwise)
os.environ["JAX_PLATFORMS"] = "cuda"

import grain
import numpy as np
import jax
from jax import numpy as jnp
from numpyro import distributions as dist
from flax import nnx

from gensbi.recipes import ConditionalDiffusionPipeline
from gensbi.models import Flux1, Flux1Params

from gensbi.utils.plotting import plot_marginals
import matplotlib.pyplot as plt




# %%

theta_prior = dist.Uniform(
    low=jnp.array([-2.0, -2.0, -2.0]), high=jnp.array([2.0, 2.0, 2.0])
)

obs_dim = 3
cond_dim = 3
joint_dim = obs_dim + cond_dim


# %%
def simulator(key, nsamples):
    theta_key, sample_key = jax.random.split(key, 2)
    thetas = theta_prior.sample(theta_key, (nsamples,))

    xs = thetas + 1 + jax.random.normal(sample_key, thetas.shape) * 0.1

    thetas = thetas[..., None]
    xs = xs[..., None]

    # when making a dataset for the joint pipeline, thetas need to come first
    data = jnp.concatenate([thetas, xs], axis=1)

    return data


# %% Define your training and validation datasets.
train_data = simulator(jax.random.PRNGKey(0), 10_000)
val_data = simulator(jax.random.PRNGKey(1), 2000)
# %%
def split_obs_cond(data):
    return data[:, :obs_dim], data[:, obs_dim:]  # assuming first dim_obs are obs, last dim_cond are cond


# %%

batch_size = 128

train_dataset_grain = (
    grain.MapDataset.source(np.array(train_data))
    .shuffle(42)
    .repeat()
    .to_iter_dataset()
    .batch(batch_size)
    .map(split_obs_cond)
    # .mp_prefetch() # Uncomment if you want to use multiprocessing prefetching
)

val_dataset_grain = (
    grain.MapDataset.source(np.array(val_data))
    .shuffle(42)
    .repeat()
    .to_iter_dataset()
    .batch(batch_size)
    .map(split_obs_cond)
    # .mp_prefetch() # Uncomment if you want to use multiprocessing prefetching
)

# %% Define your model
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
    obs_dim=obs_dim,
    cond_dim=cond_dim,
    theta=10*joint_dim,
    rngs=nnx.Rngs(default=42),
    param_dtype=jnp.float32,
)

model = Flux1(params)

# %% Instantiate the pipeline

pipeline = ConditionalDiffusionPipeline(
    model,
    train_dataset_grain,
    val_dataset_grain,
    obs_dim,
    cond_dim,
)

# %% Train the model
rngs = nnx.Rngs(42)
pipeline.train(
    rngs, nsteps=5000, save_model=False
)  # if you want to save the model, set save_model=True

# %% Sample from the posterior

new_sample = simulator(jax.random.PRNGKey(20), 1)
true_theta = new_sample[:, :obs_dim, :]  # extract observation from the joint sample
x_o = new_sample[:, obs_dim:, :]  # extract condition from the joint sample

samples = pipeline.sample(rngs.sample(), x_o, nsamples=100_000)
# %% Plot the samples
plot_marginals(
    np.array(samples[..., 0]), gridsize=30, true_param=np.array(true_theta[0, :, 0]), range = [(1, 3), (1, 3), (-0.6, 0.5)]
)

plt.savefig("conditional_diffusion_pipeline_marginals.png", dpi=100, bbox_inches="tight")
plt.show()

# %%
