# Getting Started

Welcome to GenSBI! This page will help you get started with installation and basic usage.

## Installation

GenSBI is in early development. To install, clone the repository and install dependencies:

```bash
pip install git+https://github.com/aurelio-amerio/GenSBI.git
```

If a GPU is available, it is advisable to install the cuda version of the package:

```bash
pip install git+https://github.com/aurelio-amerio/GenSBI.git[cuda12]
```

## Requirements

- Python 3.11+
- JAX
- Flax
- (See `pyproject.toml` for full requirements)

## Basic Usage
The most basic usage of GenSBI involves defining a simulation-based inference pipeline using one of the provided recipes. Here is a minimal example of setting up a flow-based inference pipeline using Simformer:

```python
import jax 
from jax import numpy as jnp
import numpy as np

from gensbi.recipes import SimformerFlowPipeline
from gensbi.utils.plotting import plot_marginals

import grain # data loaders for jax

import matplotlib.pyplot as plt # for plotting


thetas_train = ... # inference parameters for training, shape (N_train, dim_theta, 1)
thetas_val = ... # inference parameters for validation, shape (N_val, dim_theta, 1)

xs_train = ... # observed/simulated data for training, shape (N_train, dim_data, 1)
xs_val = ...  # observed/simulated data for validation, shape (N_val, dim_data, 1)

# concatenate thetas and xs for dataset creation
train_data = np.concatenate((thetas_train, xs_train), axis=1)
val_data = np.concatenate((thetas_val, xs_val), axis=1)

# define a batch size for training, and create a batched dataset using grain
batch_size = 256

train_dataset = (
    grain.MapDataset.source(np.array(train_data))
    .shuffle(42)
    .repeat()
    .to_iter_dataset()
    .batch(batch_size)
)

val_dataset = (
    grain.MapDataset.source(np.array(val_data))
    .shuffle(42)
    .repeat()
    .to_iter_dataset()
    .batch(batch_size)
)

# define JAX datasets
dim_theta = ... # number of inference parameters, that is thetas_train.shape[1]
dim_data = ... # dimensionality of observed data, that is xs_train.shape[1]

# define the pipeline
pipeline = SimformerFlowPipeline(train_dataset, val_dataset, dim_theta, dim_data)

# train the model
pipeline.train(num_steps=10000)

rng = jax.random.PRNGKey(0)
nsamples = 10_000
observation = ... # target observation, shape (1, dim_data, 1)
samples = pipeline.sample(rng, observation, nsamples)

# plot the posterior distributions 
plot_marginals(samples, gridsize=50)
plt.show()

```

See the [Examples](examples.md) page for practical demonstrations on common SBI benchmarks.

## Citing GenSBI

If you use this library, please consider citing this work and the original methodology papers.
