# %%
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx

import warnings

import pytest

import tempfile

from gensbi.models import SimformerParams, Flux1JointParams, Flux1Params
from gensbi.recipes import (
    SimformerFlowPipeline,
    SimformerDiffusionPipeline,
    Flux1JointFlowPipeline,
    Flux1JointDiffusionPipeline,
    Flux1FlowPipeline,
    Flux1DiffusionPipeline,
)

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

params_simf = SimformerParams(
    rngs=nnx.Rngs(0),
    in_channels=1,
    dim_value=2,
    dim_id=2,
    dim_condition=2,
    dim_joint=dim_joint,
    fourier_features=32,
    num_heads=2,
    num_layers=1,
    widening_factor=3,
    qkv_features=4,
    num_hidden_layers=1,
)

params_simf2 = Flux1JointParams(
    in_channels=1,
    vec_in_dim=None,
    mlp_ratio=3.0,
    num_heads=2,
    depth_single_blocks=2,
    axes_dim=[4],
    condition_dim=[2],
    qkv_bias=True,
    rngs=nnx.Rngs(0),
    joint_dim=4,
    theta=16,
    guidance_embed=False,
    param_dtype=jnp.float32,
)

params_flux = Flux1Params(
    in_channels=1,
    vec_in_dim=None,
    context_in_dim=1,
    mlp_ratio=1,
    num_heads=2,
    depth=2,
    depth_single_blocks=2,
    axes_dim=[
        2,
    ],
    qkv_bias=True,
    obs_dim=dim_theta,
    cond_dim=dim_data,
    theta=20,
    rngs=nnx.Rngs(default=42),
    param_dtype=jnp.float32,
)

# %%

config_diff_flux = "test/recipes/configs/config_diffusion_flux.yaml"
config_flow_flux = "test/recipes/configs/config_flow_flux.yaml"
config_diff_simformer = "test/recipes/configs/config_diffusion_simformer.yaml"
config_flow_simformer = "test/recipes/configs/config_flow_simformer.yaml"
config_diff_flux1joint = "test/recipes/configs/config_diffusion_flux1joint.yaml"
config_flow_flux1joint = "test/recipes/configs/config_flow_flux1joint.yaml"


@pytest.mark.parametrize(
    "pipeline_cls, config_path",
    [
        (SimformerFlowPipeline, config_flow_simformer),
        (SimformerDiffusionPipeline, config_diff_simformer),
        (Flux1JointFlowPipeline, config_flow_flux1joint),
        (Flux1JointDiffusionPipeline, config_diff_flux1joint),
        (Flux1FlowPipeline, config_flow_flux),
        (Flux1DiffusionPipeline, config_diff_flux),
    ],
)
def test_load_configs(pipeline_cls, config_path):

    checkpoint_dir = tempfile.mkdtemp()

    pipeline = pipeline_cls.init_pipeline_from_config(
        train_dataset,
        val_dataset,
        dim_theta,
        dim_data,
        config_path=config_path,
        checkpoint_dir=checkpoint_dir,
    )

    assert isinstance(
        pipeline, pipeline_cls
    ), f"Expected {pipeline_cls}, got {type(pipeline)}"

    return


@pytest.mark.parametrize(
    "pipeline_cls, params",
    [
        (SimformerFlowPipeline, params_simf),
        (SimformerDiffusionPipeline, params_simf),
        (Flux1JointFlowPipeline, params_simf2),
        (Flux1JointDiffusionPipeline, params_simf2),
        (Flux1FlowPipeline, params_flux),
        (Flux1DiffusionPipeline, params_flux),
    ],
)
def test_model_pipeline(pipeline_cls, params):
    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        training_config = pipeline_cls._get_default_training_config()
        training_config["checkpoint_dir"] = model_dir
        training_config["val_every"] = 1  # validate every epoch

        # first we try to initialize a default pipeline, to make sure it works
        default_pipeline = pipeline_cls(train_dataset, val_dataset, dim_theta, dim_data)

        assert isinstance(
            default_pipeline, pipeline_cls
        ), f"Expected {pipeline_cls}, got {type(default_pipeline)}"

        # then we use a real pipeline

        pipeline = pipeline_cls(
            train_dataset,
            val_dataset,
            dim_theta,
            dim_data,
            params,
            training_config=training_config,
        )

        batch_size = 3
        t = jnp.linspace(0, 1, batch_size)
        obs = jnp.ones((batch_size, dim_theta, 1))
        cond = jnp.ones((batch_size, dim_data, 1))

        obs_ids = pipeline.obs_ids
        cond_ids = pipeline.cond_ids

        # try getting the default parameters
        default_params = pipeline._get_default_params()
        # make sure the default parameters are of the same class as params
        assert isinstance(
            default_params, type(params)
        ), f"Expected {type(params)}, got {type(default_params)}"

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
            train_dataset,
            val_dataset,
            dim_theta,
            dim_data,
            params,
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

