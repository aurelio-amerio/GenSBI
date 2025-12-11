# %% Imports
import os

# Set JAX backend (use 'cuda' for GPU, 'cpu' otherwise)
os.environ["JAX_PLATFORMS"] = "cpu"

import grain
import numpy as np
import jax
from jax import numpy as jnp
from gensbi.recipes import JointFlowPipeline
from gensbi.utils.model_wrapping import _expand_dims, _expand_time
from gensbi.utils.plotting import plot_marginals

from gensbi.models import Flux1Joint, Flux1JointParams
import matplotlib.pyplot as plt

from numpyro import distributions as dist


from flax import nnx

#%%

mu_prior = dist.Uniform(low=jnp.array([-1.0, -1.0]), high=jnp.array([3.0, 3.0]))
std_prior = dist.Uniform(low=jnp.array([0.1, 0.1]), high=jnp.array([2.0, 2.0]))


# %%
def simulator(key, nsamples):
    mu_key, std_key, sample_key = jax.random.split(key, 3)
    mus = mu_prior.sample(mu_key, (nsamples,))
    stds = std_prior.sample(std_key, (nsamples,))

    thetas = jnp.concatenate([mus, stds], axis=-1)
    xs = mus + jax.random.normal(sample_key, (nsamples, 2)) * stds

    thetas = thetas.reshape(-1, 4, 1)
    xs = xs.reshape(-1, 2, 1)

    # when making a dataset for the joint pipeline, thetas need to come first
    data = jnp.concatenate([thetas, xs], axis=1)

    return data


# %% Define your training and validation datasets.
train_data = simulator(jax.random.PRNGKey(0), 10_000)
val_data = simulator(jax.random.PRNGKey(1), 2000)
# %%
train_data.shape
# %%

batch_size = 128

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
params = Flux1JointParams(
    in_channels=1,
    vec_in_dim=None,
    mlp_ratio=3.0,
    num_heads=2,
    depth_single_blocks=8,
    axes_dim=[4],
    condition_dim=[2],
    qkv_bias=True,
    rngs=nnx.Rngs(0),
    joint_dim=6,
    theta=50,
    guidance_embed=False,
    param_dtype=jnp.float32,
)

model = Flux1Joint(params)

# %% Instantiate the pipeline
dim_obs = 4  # dimension of the parameter space (thetas)
dim_cond = 2  # dimension of the observation space (xs)
pipeline = JointFlowPipeline(model, train_dataset_grain, val_dataset_grain, dim_obs, dim_cond)

# %% Train the model
rngs = nnx.Rngs(42)
pipeline.train(
    rngs, nsteps=5000, save_model=False
)  # if you want to save the model, set save_model=True

# %% Sample from the posterior

new_sample = simulator(jax.random.PRNGKey(1234), 1)
true_theta = new_sample[:, :4, :]  # extract observation from the joint sample
x_o = new_sample[:, 4:, :] # extract condition from the joint sample

samples = pipeline.sample(rngs.sample(), x_o, nsamples=10_000)
# %% Plot the samples
plot_marginals(
    np.array(samples[..., 0]), gridsize=30, true_param=np.array(true_theta[0,:, 0])
)
plt.show()

# %%
