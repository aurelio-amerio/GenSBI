# %% Imports
import os

# Set JAX backend (use 'cuda' for GPU, 'cpu' otherwise)
os.environ["JAX_PLATFORMS"] = "cuda"

import grain
import numpy as np
import jax
from jax import numpy as jnp
from gensbi.recipes import SimformerDiffusionPipeline
from gensbi.utils.plotting import plot_marginals

from gensbi.models import Simformer, SimformerParams
import matplotlib.pyplot as plt

from numpyro import distributions as dist


from flax import nnx

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

    # when making a dataset for the joint pipeline, thetas need to come first
    data = jnp.concatenate([thetas, xs], axis=1)

    return data


# %% Define your training and validation datasets.
train_data = simulator(jax.random.PRNGKey(0), 10_000)
val_data = simulator(jax.random.PRNGKey(1), 2000)
# %%
train_data.shape

# %%

batch_size = 256

train_dataset_grain = (
    grain.MapDataset.source(np.array(train_data))
    .shuffle(42)
    .repeat()
    .to_iter_dataset()
    .batch(batch_size)
    # .mp_prefetch() # Uncomment if you want to use multiprocessing prefetching
)

val_dataset_grain = (
    grain.MapDataset.source(np.array(val_data))
    .shuffle(42)
    .repeat()
    .to_iter_dataset()
    .batch(batch_size)
    # .mp_prefetch() # Uncomment if you want to use multiprocessing prefetching
)

# %% Define your model
params = SimformerParams(
    rngs=nnx.Rngs(0),
    in_channels=1,
    dim_value=40,
    dim_id=40,
    dim_condition=10,
    dim_joint=dim_joint,
    fourier_features=256,
    num_heads=4,
    num_layers=8,
    widening_factor=3,
    qkv_features=40,
    num_hidden_layers=1,
)

# %% Instantiate the pipeline
training_config = SimformerDiffusionPipeline.get_default_training_config()
training_config["nsteps"] = 10000

pipeline = SimformerDiffusionPipeline(
    train_dataset_grain,
    val_dataset_grain,
    dim_obs,
    dim_cond,
    params=params,
    condition_mask_kind="posterior",
    training_config=training_config,
)

# %% Train the model
rngs = nnx.Rngs(42)
pipeline.train(
    rngs, save_model=False
)  # if you want to save the model, set save_model=True

# %% Sample from the posterior

new_sample = simulator(jax.random.PRNGKey(20), 1)
true_theta = new_sample[:, :dim_obs, :]  # extract observation from the joint sample
x_o = new_sample[:, dim_obs:, :]  # extract condition from the joint sample

samples = pipeline.sample(rngs.sample(), x_o, nsamples=100_000)
# %% Plot the samples
plot_marginals(
    np.array(samples[..., 0]),
    gridsize=30,
    true_param=np.array(true_theta[0, :, 0]),
    range=[(1, 3), (1, 3), (-0.6, 0.5)],
)
plt.savefig("simformer_diffusion_pipeline_marginals.png", dpi=100, bbox_inches="tight")
plt.show()

# %%
