# %% [markdown]
# ### Library Imports and JAX Backend Selection


# %%
# Import libraries and set JAX backend
import os

os.environ["JAX_PLATFORMS"] = "cuda"  # select cpu instead if no gpu is available
# os.environ['JAX_PLATFORMS']="cpu"

from flax import nnx
import jax
import jax.numpy as jnp
import optax
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt
from matplotlib import cm

# %%
# Set training and model restoration flags
overwrite_model = False
restore_model = True  # Use pretrained model if available
train_model = False  # Set to True to train from scratch

# %%
# Specify the checkpoint directory for saving/restoring models
import orbax.checkpoint as ocp

checkpoint_dir = f"{os.getcwd()}/checkpoints/diffusion_2d_example"

import os

os.makedirs(checkpoint_dir, exist_ok=True)

if overwrite_model:
    checkpoint_dir = ocp.test_utils.erase_and_create_empty(checkpoint_dir)

# %% [markdown]
# ## 2. Data Generation
#
# We generate a synthetic 2D dataset using JAX. This section defines the data generation functions and visualizes the data distribution.

# %%
# Define a function to generate 2D box data using JAX
import jax
import jax.numpy as jnp
from jax import random
from functools import partial
import grain


@partial(jax.jit, static_argnums=[1])  # type: ignore
def make_boxes_jax(key, batch_size: int = 200):
    """
    Generates a batch of 2D data points similar to the original PyTorch function
    using JAX.

    Args:
        key: A JAX PRNG key for random number generation.
        batch_size: The number of data points to generate.

    Returns:
        A JAX array of shape (batch_size, 2) with generated data,
        with dtype float32.
    """
    # Split the key for different random operations
    keys = jax.random.split(key, 3)
    x1 = jax.random.uniform(keys[0], batch_size) * 4 - 2
    x2_ = (
        jax.random.uniform(keys[1], batch_size)
        - jax.random.randint(keys[2], batch_size, 0, 2) * 2
    )
    x2 = x2_ + (jnp.floor(x1) % 2)

    data = 1.0 * jnp.concatenate([x1[:, None], x2[:, None]], axis=1) / 0.45

    return data


# %%
# # Infinite data generator for training batches
# @partial(jax.jit, static_argnums=[1])  # type: ignore
# def inf_train_gen(key, batch_size: int = 200):
#     x = make_boxes_jax(key, batch_size)

#     return x

batch_size = 8

data = make_boxes_jax(jax.random.PRNGKey(0), 500_000)

train_dataset_grain = (
    grain.MapDataset.source(np.array(data)[..., None])
    .shuffle(42)
    .repeat()
    .to_iter_dataset()
)


train_dataset_batched = train_dataset_grain.batch(batch_size)

train_iter = iter(train_dataset_batched)

data_val = make_boxes_jax(jax.random.PRNGKey(1), 1000)

val_dataset_batched = (
    grain.MapDataset.source(np.array(data_val)[..., None])
    .shuffle(42)
    .repeat()
    .to_iter_dataset()
    .batch(512)
)

# %%
# Visualize the generated data distribution
# samples = np.array(data)

# H = plt.hist2d(samples[:, 0], samples[:, 1], 300, range=((-5, 5), (-5, 5)))
# cmin = 0.0
# cmax = jnp.quantile(jnp.array(H[0]), 0.99).item()
# norm = cm.colors.Normalize(vmax=cmax, vmin=cmin)

# _ = plt.hist2d(
#     samples[:, 0],
#     samples[:, 1],
#     300,
#     range=((-5, 5), (-5, 5)),
#     norm=norm,
#     cmap="viridis",
# )

# # set equal ratio of axes
# plt.gca().set_aspect("equal", adjustable="box")


# plt.show()

# %% [markdown]
# ## 3. Model and Loss Definition
#
# We define the velocity field model (an MLP), the loss function, and the optimizer for training the score-matching model.

# %%
# Import diffusion components and utilities
from gensbi.recipes import UnconditionalSMPipeline


# %%
# Define the MLP velocity field model
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

    def __call__(self, t: jax.Array, obs: jax.Array, **kwargs):
        assert (
            obs.ndim == 3
        ), f"Input obs must have shape (batch_size, input_dim, 1), got {obs.shape}"

        # we need to massage the data a bit to make it compatible with the pipeline format
        t = jnp.atleast_1d(t)
        x = jnp.squeeze(obs, axis=-1)

        if t.ndim < 2:
            t = t[..., None]

        if t.ndim == 3:
            t = t[..., 0]

        t = jnp.broadcast_to(t, (x.shape[0], t.shape[-1]))

        # now everything whould have the right dimension, we can proceed

        h = jnp.concatenate([x, t], axis=-1)

        x = self.linear1(h)
        x = jax.nn.gelu(x)

        x = self.linear2(x)
        x = jax.nn.gelu(x)

        x = self.linear3(x)
        x = jax.nn.gelu(x)

        x = self.linear4(x)
        x = jax.nn.gelu(x)

        x = self.linear5(x)

        return x[..., None]


# %%
# Initialize the velocity field model
hidden_dim = 512

# velocity field model init
model = MLP(input_dim=2, hidden_dim=hidden_dim, rngs=nnx.Rngs(0))

training_config = UnconditionalSMPipeline.get_default_training_config()
training_config["checkpoint_dir"] = (
    "/home/aure/Documents/GitHub/GenSBI/wip-score/checkpoints/diffusion_2d_example"
)
training_config["nsteps"] = 10_000


pipeline = UnconditionalSMPipeline(
    model,
    train_dataset_batched,
    val_dataset_batched,
    2,
    training_config=training_config,
)

# %%
# Restore the model from checkpoint if requested
if restore_model:
    pipeline.restore_model()

# %%
# pipeline.train(nnx.Rngs(0))

# %%
key = jax.random.PRNGKey(42)
# time steps: 20,
sol_ = pipeline.sample(key, nsamples=20, nsteps=10, return_intermediates=False)
jax.block_until_ready(sol_)
