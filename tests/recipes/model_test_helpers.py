"""Shared fixtures for recipe model integration tests."""
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx

import pytest
import tempfile

import grain
import numpy as np


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
    )


train_dataset_joint = (
    grain.MapDataset.source(np.array(data)[:800])
    .shuffle(42)
    .repeat()
    .to_iter_dataset()
    .batch(32)
)

val_dataset_joint = (
    grain.MapDataset.source(np.array(data)[800:])
    .shuffle(42)
    .repeat()
    .to_iter_dataset()
    .batch(32)
)

train_dataset_cond = (
    grain.MapDataset.source(np.array(data)[:800])
    .shuffle(42)
    .repeat()
    .to_iter_dataset()
    .batch(32)
    .map(split_obs_cond)
)

val_dataset_cond = (
    grain.MapDataset.source(np.array(data)[800:])
    .shuffle(42)
    .repeat()
    .to_iter_dataset()
    .batch(32)
    .map(split_obs_cond)
)


def run_model_pipeline_test(pipeline_cls, params, is_conditional):
    """Shared integration test: init, train, wrap, evaluate, sample."""
    if is_conditional:
        train_dataset = train_dataset_cond
        val_dataset = val_dataset_cond
    else:
        train_dataset = train_dataset_joint
        val_dataset = val_dataset_joint

    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        training_config = pipeline_cls.get_default_training_config()
        training_config["checkpoint_dir"] = model_dir
        training_config["val_every"] = 1

        # Init with defaults
        default_pipeline = pipeline_cls(train_dataset, val_dataset, dim_obs, dim_cond)
        assert isinstance(default_pipeline, pipeline_cls)

        # Init with explicit params
        kwargs = dict(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            ch_obs=2,
            params=params,
            training_config=training_config,
        )
        if is_conditional:
            kwargs["ch_cond"] = 2
        pipeline = pipeline_cls(**kwargs)

        assert model_dir == pipeline.training_config["checkpoint_dir"]

        batch_size = 3
        t = jnp.linspace(0, 1, batch_size)
        obs = jnp.ones((batch_size, dim_obs, 2))
        cond = jnp.ones((batch_size, dim_cond, 2))

        obs_ids = pipeline.obs_ids
        cond_ids = pipeline.cond_ids

        # Train + wrap
        pipeline.train(nnx.Rngs(0), nsteps=2, save_model=True)
        pipeline._wrap_model()

        # Evaluate
        out = pipeline.model_wrapped(t, obs, obs_ids, cond, cond_ids)
        out_ema = pipeline.ema_model_wrapped(t, obs, obs_ids, cond, cond_ids)
        assert out.shape == (batch_size, dim_obs, 2)
        assert out_ema.shape == (batch_size, dim_obs, 2)

        # Sample with default solver
        x_o = cond[:1]  # single conditioning observation
        samples = pipeline.sample(
            jax.random.PRNGKey(1), x_o, nsamples=3, use_ema=False
        )
        assert samples.shape == (3, dim_obs, 2)

        # Sample batched with multiple conditions
        x_o_batched = cond[:3]  # 3 conditioning observations
        samples_batched = pipeline.sample_batched(
            jax.random.PRNGKey(2), x_o_batched, nsamples=4, use_ema=False
        )
        assert samples_batched.shape == (4, 3, dim_obs, 2)

        # Log prob (only supported for FlowMatchingMethod)
        from gensbi.core import FlowMatchingMethod
        if isinstance(pipeline.method, FlowMatchingMethod):
            x_1 = jnp.zeros((batch_size, dim_obs, 2))
            # Exact divergence
            lp = pipeline.log_prob(x_1, x_o, use_ema=False, exact_divergence=True)
            assert lp.shape == (batch_size,)
            lp_ema = pipeline.log_prob(x_1, x_o, use_ema=True, exact_divergence=True)
            assert lp_ema.shape == (batch_size,)
            # Hutchinson divergence
            lp_hutch = pipeline.log_prob(
                x_1, x_o, use_ema=False,
                key=jax.random.PRNGKey(42), exact_divergence=False,
            )
            assert lp_hutch.shape == (batch_size,)
            lp_hutch_ema = pipeline.log_prob(
                x_1, x_o, use_ema=True,
                key=jax.random.PRNGKey(42), exact_divergence=False,
            )
            assert lp_hutch_ema.shape == (batch_size,)


def run_load_config_test(pipeline_cls, config_path, is_conditional):
    """Shared config loading test."""
    if is_conditional:
        train_dataset = train_dataset_cond
        val_dataset = val_dataset_cond
    else:
        train_dataset = train_dataset_joint
        val_dataset = val_dataset_joint

    checkpoint_dir = tempfile.mkdtemp()
    pipeline = pipeline_cls.init_pipeline_from_config(
        train_dataset,
        val_dataset,
        dim_obs,
        dim_cond,
        config_path=config_path,
        checkpoint_dir=checkpoint_dir,
    )
    assert isinstance(pipeline, pipeline_cls)

