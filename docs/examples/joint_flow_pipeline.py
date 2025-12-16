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

from gensbi.recipes import JointFlowPipeline
from gensbi.models import Simformer, SimformerParams

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
params = SimformerParams(
    rngs=nnx.Rngs(0),
    in_channels=1,
    dim_value=20,
    dim_id=10,
    dim_condition=10,
    dim_joint=joint_dim,
    fourier_features=128,
    num_heads=4,
    num_layers=6,
    widening_factor=3,
    qkv_features=40,
    num_hidden_layers=1,
)

model = Simformer(params)

# %% Instantiate the pipeline

pipeline = JointFlowPipeline(
    model,
    train_dataset_grain,
    val_dataset_grain,
    obs_dim,
    cond_dim,
    condition_mask_kind="posterior",
)

# %% Train the model
rngs = nnx.Rngs(42)
pipeline.train(
    rngs, nsteps=10000, save_model=False
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
plt.savefig("joint_flow_pipeline_marginals.png", dpi=100, bbox_inches="tight")
plt.show()

# %%
