import os

os.environ["JAX_PLATFORMS"] = "cpu"

import tempfile
import warnings

import grain
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from gensbi.recipes import ConditionalDiffusionPipeline, ConditionalFlowPipeline

nsamples = 1000
key = jax.random.PRNGKey(0)

dim_obs = 2
dim_cond = 7
dim_joint = dim_obs + dim_cond


theta = jax.random.normal(key, (nsamples, dim_obs, 2))
x = jax.random.normal(key, (nsamples, dim_cond, 2))

data = jnp.concatenate([theta, x], axis=1)


def split_obs_cond(data):
    return (
        data[:, :dim_obs],
        data[:, dim_obs:],
    )  # assuming first dim_obs are obs, last dim_cond are cond


train_dataset_cond = (
    grain.MapDataset.source(np.array(data)[:800])
    .shuffle(42)
    .repeat()
    .to_iter_dataset()
    .batch(32)
    .map(split_obs_cond)
    # .mp_prefetch() # Uncomment if you want to use multiprocessing prefetching
)

val_dataset_cond = (
    grain.MapDataset.source(np.array(data)[800:])
    .shuffle(42)
    .repeat()
    .to_iter_dataset()
    .batch(32)
    .map(split_obs_cond)
    # .mp_prefetch() # Uncomment if you want to use multiprocessing prefetching
)


@pytest.mark.parametrize(
    "pipeline_cls, params",
    [
        (ConditionalFlowPipeline, None),
        (ConditionalDiffusionPipeline, None),
    ],
)
def test_model_pipeline(pipeline_cls, params):
    # initialize the pipeline
    pipeline = pipeline_cls(
        model=None,
        train_dataset=train_dataset_cond,
        val_dataset=val_dataset_cond,
        dim_obs=dim_obs,
        dim_cond=dim_cond,
        id_embedding_strategy=("absolute", "absolute"),
    )
    pipeline = pipeline_cls(
        model=None,
        train_dataset=train_dataset_cond,
        val_dataset=val_dataset_cond,
        dim_obs=dim_obs,
        dim_cond=dim_cond,
        id_embedding_strategy=("pos1d", "pos1d"),
    )
    pipeline = pipeline_cls(
        model=None,
        train_dataset=train_dataset_cond,
        val_dataset=val_dataset_cond,
        dim_obs=(dim_obs, dim_obs),
        dim_cond=(dim_cond, dim_cond),
        id_embedding_strategy=("pos2d", "pos2d"),
    )
    pipeline = pipeline_cls(
        model=None,
        train_dataset=train_dataset_cond,
        val_dataset=val_dataset_cond,
        dim_obs=dim_obs,
        dim_cond=dim_cond,
        id_embedding_strategy=("rope1d", "rope1d"),
    )
    pipeline = pipeline_cls(
        model=None,
        train_dataset=train_dataset_cond,
        val_dataset=val_dataset_cond,
        dim_obs=(dim_obs, dim_obs),
        dim_cond=(dim_cond, dim_cond),
        id_embedding_strategy=("rope2d", "rope2d"),
    )
