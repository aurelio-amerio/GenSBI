#%% Imports
import os

# Set JAX backend (use 'cuda' for GPU, 'cpu' otherwise)
os.environ["JAX_PLATFORMS"] = "cpu" 

import grain
import numpy as np
import jax
from jax import numpy as jnp
from gensbi.recipes import UnconditionalDiffusionPipeline
from gensbi.utils.model_wrapping import _expand_dims, _expand_time
from gensbi.utils.plotting import plot_marginals
import matplotlib.pyplot as plt
from gensbi.models import Simformer, SimformerParams


from flax import nnx

#%% define a simulator
def simulator(key, nsamples):
    return 3 + jax.random.normal(key, (nsamples,2))*jnp.array([0.5, 1]).reshape(1,2) # a simple 2D gaussian


#%%


#%% Define your training and validation datasets.
train_data = simulator(jax.random.PRNGKey(0), 10_000).reshape(-1,2,1)
val_data = simulator(jax.random.PRNGKey(1), 2000).reshape(-1,2,1)
#%%
# it is advisable to normalize the inference parameter space to zero mean and unit variance for better training performance
# mean_train = jnp.mean(train_data, axis=0)
# std_train = jnp.std(train_data, axis=0)

# train_data_ = (train_data - mean_train) / std_train
# val_data_ = (val_data - mean_train) / std_train

train_data_ = train_data
val_data_ = val_data

batch_size = 128

train_dataset_grain = (
    grain.MapDataset.source(np.array(train_data_))
    .shuffle(42)
    .repeat()
    .to_iter_dataset()
    .batch(batch_size)
    # .mp_prefetch() # Uncomment if you want to use multiprocessing prefetching
)

val_dataset_grain = (
    grain.MapDataset.source(np.array(val_data_))
    .shuffle(42)
    .repeat()
    .to_iter_dataset()
    .batch(batch_size)
    # .mp_prefetch() # Uncomment if you want to use multiprocessing prefetching
)
#%% Define your model
# Here we define a MLP velocity field model, 
# this model only works for inputs of shape (batch, dim, 1). 
# For more complex models, please refer to the transformer-based models in gensbi.models.

class MLP(nnx.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 512, *, rngs: nnx.Rngs):

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        din = input_dim + 1

        self.linear1 = nnx.Linear(din, self.hidden_dim, rngs=rngs)
        self.linear2 = nnx.Linear(self.hidden_dim, self.hidden_dim, rngs=rngs)
        self.linear3 = nnx.Linear(self.hidden_dim, self.hidden_dim, rngs=rngs)
        self.linear4 = nnx.Linear(self.hidden_dim, self.hidden_dim, rngs=rngs)
        self.linear5 = nnx.Linear(self.hidden_dim, self.input_dim, rngs=rngs)

    def __call__(self, t: jax.Array, obs: jax.Array, node_ids, *args, **kwargs):
        obs = _expand_dims(obs)[...,0] # for this specific model, we use samples of shape (batch, dim), while for transformer models we use (batch, dim, c)
        t = _expand_time(t)
        if t.ndim == 3:
            t = t.reshape(t.shape[0], t.shape[1])
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



model = MLP(rngs=nnx.Rngs(42))  # your nnx.Module model here, e.g., a simple MLP, or the Simformer model
# if you define a custom model, it should take as input the following arguments:
#    t: Array,
#    obs: Array,
#    node_ids: Array (optional, if your model is a transformer-based model)
#    *args
#    **kwargs

# the obs input should have shape (batch_size, dim_joint, c), and the output will be of the same shape
#%% Instantiate the pipeline
dim_obs = 2  # Dimension of the parameter space

pipeline = UnconditionalDiffusionPipeline(model, train_dataset_grain, val_dataset_grain, dim_obs)

#%% Train the model
rngs = nnx.Rngs(42)
pipeline.train(rngs, nsteps=1500, save_model=False) # if you want to save the model, set save_model=True

#%% Sample from the posterior
samples = pipeline.sample(rngs.sample(), nsamples=100_000)

# if you normalized the data before training, remember to unnormalize the samples
# samples = samples * std_train + mean_train

#%% Plot the samples
samples.mean(axis=0), samples.std(axis=0)
#%%

plot_marginals(np.array(samples[...,0]), true_param=[3,3], gridsize=20, range = [(-2, 8), (-2, 8)])
plt.show()



# %%
