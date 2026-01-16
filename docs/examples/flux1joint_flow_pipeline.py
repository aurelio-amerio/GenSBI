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

from gensbi.recipes import Flux1JointFlowPipeline
from gensbi.models import Flux1JointParams

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

    # when making a dataset for the joint pipeline, thetas need to come first
    data = jnp.concatenate([thetas, xs], axis=1)

    return data


# %% Define your training and validation datasets.
# We generate a training dataset and a validation dataset using the simulator.
# The simulator is a simple function that generates parameters (theta) and data (x).
# In this example, we use a simple Gaussian simulator.
train_data = simulator(jax.random.PRNGKey(0), 100_000)
val_data = simulator(jax.random.PRNGKey(1), 2000)
# %% Normalize the dataset
# It is important to normalize the data to have zero mean and unit variance.
# This helps the model training process.
means = jnp.mean(train_data, axis=0)
stds = jnp.std(train_data, axis=0)


def normalize(data, means, stds):
    return (data - means) / stds


def unnormalize(data, means, stds):
    return data * stds + means


# %% Prepare the data for the pipeline
# The pipeline expects the data to be normalized but not split (for joint pipelines).
def process_data(data):
    return normalize(data, means, stds)


# %%
train_data.shape

# %%

# %% Create the input pipeline using Grain
# We use Grain to create an efficient input pipeline.
# This involves shuffling, repeating for multiple epochs, and batching the data.
# We also map the process_data function to prepare (normalize) the data for the model.
batch_size = 256

train_dataset_grain = (
    grain.MapDataset.source(np.array(train_data))
    .shuffle(42)
    .repeat()
    .to_iter_dataset()
    .batch(batch_size)
    .map(process_data)
    # .mp_prefetch() # Uncomment if you want to use multiprocessing prefetching
)

val_dataset_grain = (
    grain.MapDataset.source(np.array(val_data))
    .shuffle(42)
    .repeat()
    .to_iter_dataset()
    .batch(batch_size)
    .map(process_data)
    # .mp_prefetch() # Uncomment if you want to use multiprocessing prefetching
)

# %% Define your model
# specific model parameters are defined here.
# For Flux1Joint, we need to specify dimensions, embedding strategies, and other architecture details.
params = Flux1JointParams(
    in_channels=1,
    vec_in_dim=None,
    mlp_ratio=3.0,
    num_heads=2,
    depth_single_blocks=8,
    axes_dim=[10],
    condition_dim=[2],
    qkv_bias=True,
    rngs=nnx.Rngs(0),
    dim_joint=dim_joint,
    theta=10 * dim_joint,
    id_embedding_strategy="absolute",
    guidance_embed=False,
    param_dtype=jnp.float32,
)

# %% Instantiate the pipeline
# The Flux1JointFlowPipeline handles the training loop and sampling.
# We configure it with the model parameters, datasets, dimensions using a default training configuration.
# We also specify the condition_mask_kind, which determines how conditioning is handled during training.
training_config = Flux1JointFlowPipeline.get_default_training_config()
training_config["nsteps"] = 10000

pipeline = Flux1JointFlowPipeline(
    train_dataset_grain,
    val_dataset_grain,
    dim_obs,
    dim_cond,
    params=params,
    condition_mask_kind="posterior",
    training_config=training_config,
)

# %% Train the model
# We create a random key for training and start the training process.
rngs = nnx.Rngs(42)
pipeline.train(
    rngs, save_model=False
)  # if you want to save the model, set save_model=True

# %% Sample from the posterior
# To generate samples, we first need an observation (and its corresponding condition).
# We generate a new sample from the simulator, normalize it, and extract the condition x_o.

new_sample = simulator(jax.random.PRNGKey(20), 1)
true_theta = new_sample[:, :dim_obs, :]  # extract observation from the joint sample

new_sample = normalize(new_sample, means, stds)
x_o = new_sample[:, dim_obs:, :]  # extract condition from the joint sample

# Then we invoke the pipeline's sample method.
samples = pipeline.sample(rngs.sample(), x_o, nsamples=100_000)
# Finally, we unnormalize the samples to get them back to the original scale.
samples = unnormalize(samples, means[:dim_obs], stds[:dim_obs])

# %% Plot the samples
# We verify the model's performance by plotting the marginal distributions of the generated samples
# against the true parameters.
plot_marginals(
    np.array(samples[..., 0]),
    gridsize=30,
    true_param=np.array(true_theta[0, :, 0]),
    range=[(1, 3), (1, 3), (-0.6, 0.5)],
)
plt.savefig("flux1joint_flow_pipeline_marginals.png", dpi=100, bbox_inches="tight")
plt.show()

# %%
