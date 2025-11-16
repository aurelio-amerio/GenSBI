# %%
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx

import warnings

import pytest

import tempfile

from gensbi.recipes import (
    ConditionalFlowPipeline,
    ConditionalDiffusionPipeline,
    UnconditionalFlowPipeline,
    UnconditionalDiffusionPipeline,
    JointFlowPipeline,
    JointDiffusionPipeline,
)

from gensbi.models import Simformer, SimformerParams, Flux1, Flux1Params

import itertools


nsamples = 1000
rng = jax.random.PRNGKey(0)

dim_theta = 2
dim_data = 7
dim_joint = dim_theta + dim_data


theta = jax.random.normal(rng, (nsamples, dim_theta, 1))
x = jax.random.normal(rng, (nsamples, dim_data, 1))

data = jnp.concatenate([theta, x], axis=1)

train_data = data[:800].reshape(10, -1, dim_joint, 1)
val_data = data[800:].reshape(10, -1, dim_joint, 1)

train_dataset = itertools.cycle(train_data)
val_dataset = itertools.cycle(val_data)

# we define a conditional and a joint model for testing

params_simf = SimformerParams(
    rngs=nnx.Rngs(0),
    in_channels=1,
    dim_value=4,
    dim_id=2,
    dim_condition=2,
    dim_joint=dim_joint,
    fourier_features=128,
    num_heads=2,
    num_layers=2,
    widening_factor=2,
    qkv_features=10,
    num_hidden_layers=1,
)

model_joint = Simformer(params_simf)

params = Flux1Params(
    in_channels=1,
    vec_in_dim=None,
    context_in_dim=1,
    mlp_ratio=4,
    num_heads=4,
    depth=1,
    depth_single_blocks=2,
    axes_dim=[
        2,
    ],
    obs_dim=dim_theta,
    cond_dim=dim_data,
    qkv_bias=True,
    guidance_embed=False,
    rngs=nnx.Rngs(0),
    param_dtype=jnp.float32,
)

model_conditional = Flux1(params)


# %%

def get_model(pipeline_cls):
    if pipeline_cls in [
        ConditionalFlowPipeline,
        ConditionalDiffusionPipeline,
    ]:
        return model_conditional
    else:
        return model_joint


@pytest.mark.parametrize(
    "pipeline_cls",
    [
        ConditionalFlowPipeline,
        ConditionalDiffusionPipeline,
        JointFlowPipeline,
        JointDiffusionPipeline,
    ],
)
def test_model_general_conditional(pipeline_cls):
    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        training_config = pipeline_cls._get_default_training_config()
        training_config["checkpoint_dir"] = model_dir
        training_config["val_every"] = 1  # validate every epoch

        model = get_model(pipeline_cls)

        # first we try to initialize a default pipeline, to make sure it works
        default_pipeline = pipeline_cls(
            model, train_dataset, val_dataset, dim_theta, dim_data
        )

        assert isinstance(
            default_pipeline, pipeline_cls
        ), f"Expected {pipeline_cls}, got {type(default_pipeline)}"

        pipeline = pipeline_cls(
            model,
            train_dataset,
            val_dataset,
            dim_theta,
            dim_data,
            training_config=training_config,
        )

        batch_size = 3
        t = jnp.linspace(0, 1, batch_size)
        obs = jnp.ones((batch_size, dim_theta, 1))
        cond = jnp.ones((batch_size, dim_data, 1))

        obs_ids = pipeline.obs_ids
        cond_ids = pipeline.cond_ids

        # try training the model
        pipeline.train(nnx.Rngs(0), nsteps=2, save_model=True)
        # wrap the model
        pipeline._wrap_model()

        # try evaluating the model, and save the result
        out = pipeline.model_wrapped(t, obs, obs_ids, cond, cond_ids)
        out_ema = pipeline.ema_model_wrapped(t, obs, obs_ids, cond, cond_ids)
        assert out.shape == (
            batch_size,
            dim_theta,
            1,
        ), f"Expected shape {(batch_size, dim_theta, 1)}, got {out.shape}"
        assert out_ema.shape == (
            batch_size,
            dim_theta,
            1,
        ), f"Expected shape {(batch_size, dim_theta, 1)}, got {out_ema.shape}"

        # try restoring the model from the checkpoint
        # ignore warnings about sharding for the next line

        pipeline2 = pipeline_cls(
            model,
            train_dataset,
            val_dataset,
            dim_theta,
            dim_data,
            training_config=training_config,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline2.restore_model()

        # we evaluate again the model, and check that the output is the same as before
        out_restored = pipeline2.model_wrapped(t, obs, obs_ids, cond, cond_ids)
        out_ema_restored = pipeline2.ema_model_wrapped(t, obs, obs_ids, cond, cond_ids)
        assert jnp.allclose(out, out_restored), "Restored model output does not match"
        assert jnp.allclose(
            out_ema, out_ema_restored
        ), "Restored EMA model output does not match"

        # try sampling from the model
        sample = pipeline.sample(
            jax.random.PRNGKey(1),
            jnp.arange(dim_data)[None, ...],
            nsamples=32,
            use_ema=False,
        )
        assert sample.shape == (
            32,
            dim_theta,
        ), f"Expected shape (32, {dim_theta}), got {sample.shape}"

        sample = pipeline.sample(
            jax.random.PRNGKey(1),
            jnp.arange(dim_data)[None, ...],
            nsamples=32,
            use_ema=True,
        )
        assert sample.shape == (
            32,
            dim_theta,
        ), f"Expected shape (32, {dim_theta}), got {sample.shape}"

        # sample from the restored model
        sample_restored = pipeline2.sample(
            jax.random.PRNGKey(1),
            jnp.arange(dim_data)[None, ...],
            nsamples=32,
            use_ema=True,
        )
        assert jnp.allclose(
            sample, sample_restored
        ), "Restored model samples do not match"


########

@pytest.mark.parametrize(
    "pipeline_cls",
    [
        UnconditionalFlowPipeline,
        UnconditionalDiffusionPipeline,
    ],
)
def test_model_general_unconditional(pipeline_cls):
    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        training_config = pipeline_cls._get_default_training_config()
        training_config["checkpoint_dir"] = model_dir
        training_config["val_every"] = 1  # validate every epoch

        model = get_model(pipeline_cls)

        # first we try to initialize a default pipeline, to make sure it works

        default_pipeline = pipeline_cls(model, train_dataset, val_dataset, dim_joint)

        assert isinstance(
            default_pipeline, pipeline_cls
        ), f"Expected {pipeline_cls}, got {type(default_pipeline)}"

        # then we use a real pipeline

        pipeline = pipeline_cls(
            model,
            train_dataset,
            val_dataset,
            dim_joint,
            training_config=training_config,
        )

        batch_size = 3
        t = jnp.linspace(0, 1, batch_size)
        obs = jnp.ones((batch_size, dim_joint, 1))

        obs_ids = pipeline.obs_ids

        # try training the model
        pipeline.train(nnx.Rngs(0), nsteps=2, save_model=True)
        # wrap the model
        pipeline._wrap_model()

        # try evaluating the model, and save the result
        out = pipeline.model_wrapped(t, obs, obs_ids)
        out_ema = pipeline.ema_model_wrapped(t, obs, obs_ids)
        assert out.shape == (
            batch_size,
            dim_joint,
            1,
        ), f"Expected shape {(batch_size, dim_joint, 1)}, got {out.shape}"
        assert out_ema.shape == (
            batch_size,
            dim_joint,
            1,
        ), f"Expected shape {(batch_size, dim_joint, 1)}, got {out_ema.shape}"

        # try restoring the model from the checkpoint
        # ignore warnings about sharding for the next line

        pipeline2 = pipeline_cls(
            model,
            train_dataset,
            val_dataset,
            dim_joint,
            training_config=training_config,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline2.restore_model()

        # we evaluate again the model, and check that the output is the same as before
        out_restored = pipeline2.model_wrapped(t, obs, obs_ids)
        out_ema_restored = pipeline2.ema_model_wrapped(t, obs, obs_ids)
        assert jnp.allclose(out, out_restored), "Restored model output does not match"
        assert jnp.allclose(
            out_ema, out_ema_restored
        ), "Restored EMA model output does not match"

        # try sampling from the model
        sample = pipeline.sample(
            jax.random.PRNGKey(1),
            nsamples=32,
            use_ema=False,
        )
        assert sample.shape == (
            32,
            dim_joint,
        ), f"Expected shape (32, {dim_joint}), got {sample.shape}"

        sample = pipeline.sample(
            jax.random.PRNGKey(1),
            nsamples=32,
            use_ema=True,
        )
        assert sample.shape == (
            32,
            dim_joint,
        ), f"Expected shape (32, {dim_joint}), got {sample.shape}"

        # sample from the restored model
        sample_restored = pipeline2.sample(
            jax.random.PRNGKey(1),
            nsamples=32,
            use_ema=True,
        )
        assert jnp.allclose(
            sample, sample_restored
        ), "Restored model samples do not match"
