# %% Imports
import os

# Set JAX backend (use 'cuda' for GPU, 'cpu' otherwise)
# os.environ["JAX_PLATFORMS"] = "cuda"

import grain
import numpy as np
import jax
from jax import numpy as jnp
from gensbi.recipes import UnconditionalFlowPipeline
from gensbi.utils.model_wrapping import _expand_dims, _expand_time
from gensbi.utils.plotting import plot_marginals
import matplotlib.pyplot as plt


from flax import nnx


# %% define a simulator
def simulator(key, nsamples):
    return 3 + jax.random.normal(key, (nsamples, 2)) * jnp.array([0.5, 1]).reshape(
        1, 2
    )  # a simple 2D gaussian


# %% Define your training and validation datasets.
# We generate a training dataset and a validation dataset using the simulator.
# The simulator generates samples from a 2D Gaussian distribution.
train_data = simulator(jax.random.PRNGKey(0), 100_000).reshape(-1, 2, 1)
val_data = simulator(jax.random.PRNGKey(1), 2000).reshape(-1, 2, 1)

# %% Normalize the dataset
# It is important to normalize the data to have zero mean and unit variance.
# This helps the model training process.
means = jnp.mean(train_data, axis=0)
stds = jnp.std(train_data, axis=0)


def normalize(data, means, stds):
    return (data - means) / stds


def unnormalize(data, means, stds):
    return data * stds + means


def process_data(data):
    return normalize(data, means, stds)


# %% Create the input pipeline using Grain
# We use Grain to create an efficient input pipeline.
# This involves shuffling, repeating for multiple epochs, and batching the data.
# We also map the process_data function to prepare (normalize) the data.
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
    .shuffle(
        42
    )  # Use a different seed/strategy for validation if needed, but shuffling is fine
    .repeat()
    .to_iter_dataset()
    .batch(batch_size)
    .map(process_data)
    # .mp_prefetch() # Uncomment if you want to use multiprocessing prefetching
)


# %% Define your model
# Here we define a MLP velocity field model,
# this model only works for inputs of shape (batch, dim, 1).
# For more complex models, please refer to the transformer-based models in gensbi.models.
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

    def __call__(self, t: jax.Array, obs: jax.Array, node_ids, *args, **kwargs):
        obs = _expand_dims(obs)[
            ..., 0
        ]  # for this specific model, we use samples of shape (batch, dim), while for transformer models we use (batch, dim, c)
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


model = MLP(
    rngs=nnx.Rngs(42)
)  # your nnx.Module model here, e.g., a simple MLP, or the Simformer model
# if you define a custom model, it should take as input the following arguments:
#    t: Array,
#    obs: Array,
#    node_ids: Array (optional, if your model is a transformer-based model)
#    *args
#    **kwargs

# the obs input should have shape (batch_size, dim_joint, c), and the output will be of the same shape

# %% Instantiate the pipeline
# The UnconditionalFlowPipeline handles the training loop and sampling.
# We configure it with the model, datasets, dimensions using a default training configuration.
training_config = UnconditionalFlowPipeline.get_default_training_config()
training_config["nsteps"] = 10000

dim_obs = 2  # Dimension of the parameter space
ch_obs = 1  # Number of channels of the parameter space

pipeline = UnconditionalFlowPipeline(
    model,
    train_dataset_grain,
    val_dataset_grain,
    dim_obs,
    ch_obs,
    training_config=training_config,
)

# %% Train the model
# We create a random key for training and start the training process.
rngs = nnx.Rngs(42)
pipeline.train(
    rngs, save_model=False
)  # if you want to save the model, set save_model=True

# %% Sample from the posterior
# We generate new samples using the trained model.
samples = pipeline.sample(rngs.sample(), nsamples=100_000)
# Finally, we unnormalize the samples to get them back to the original scale.
samples = unnormalize(samples, means, stds)

# %% Plot the samples
# We verify the model's performance by plotting the marginal distributions of the generated samples.
plot_marginals(
    np.array(samples[..., 0]), true_param=[3, 3], gridsize=30, range=[(-2, 8), (-2, 8)]
)
plt.savefig("unconditional_flow_samples.png", dpi=300, bbox_inches="tight")
plt.show()
# %%
