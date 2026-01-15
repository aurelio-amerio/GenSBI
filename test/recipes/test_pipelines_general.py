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
    )  # assuming first dim_obs are obs, last dim_cond are cond


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
# we define a conditional and a joint model for testing

params_simf = SimformerParams(
    rngs=nnx.Rngs(0),
    in_channels=2,
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
    in_channels=2,
    vec_in_dim=None,
    context_in_dim=2,
    mlp_ratio=4,
    num_heads=4,
    depth=1,
    depth_single_blocks=2,
    axes_dim=[2, 2],
    dim_obs=dim_obs,
    dim_cond=dim_cond,
    qkv_bias=True,
    guidance_embed=False,
    rngs=nnx.Rngs(0),
    id_embedding_strategy=("pos1d", "pos1d"),
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

    if pipeline_cls in [
        ConditionalFlowPipeline,
        ConditionalDiffusionPipeline,
    ]:
        train_dataset = train_dataset_cond
        val_dataset = val_dataset_cond
    else:
        train_dataset = train_dataset_joint
        val_dataset = val_dataset_joint

    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        training_config = pipeline_cls.get_default_training_config()
        training_config["checkpoint_dir"] = model_dir
        training_config["val_every"] = 1  # validate every epoch

        model = get_model(pipeline_cls)

        # first we try to initialize a default pipeline, to make sure it works
        if pipeline_cls in [
            ConditionalFlowPipeline,
            ConditionalDiffusionPipeline,
        ]:
            default_pipeline = pipeline_cls(
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                dim_obs=dim_obs,
                dim_cond=dim_cond,
                ch_obs=2,
                ch_cond=2,
            )
        else:
            default_pipeline = pipeline_cls(
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                dim_obs=dim_obs,
                dim_cond=dim_cond,
                ch_obs=2,
            )

        assert isinstance(
            default_pipeline, pipeline_cls
        ), f"Expected {pipeline_cls}, got {type(default_pipeline)}"

        if pipeline_cls in [
            ConditionalFlowPipeline,
            ConditionalDiffusionPipeline,
        ]:
            pipeline = pipeline_cls(
                model,
                train_dataset,
                val_dataset,
                dim_obs,
                dim_cond,
                ch_obs=2,
                ch_cond=2,
                training_config=training_config,
            )
        else:
            pipeline = pipeline_cls(
                model,
                train_dataset,
                val_dataset,
                dim_obs,
                dim_cond,
                ch_obs=2,
                training_config=training_config,
            )

        batch_size = 3
        t = jnp.linspace(0, 1, batch_size)
        obs = jnp.ones((batch_size, dim_obs, 2))
        cond = jnp.ones((batch_size, dim_cond, 2))

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
            dim_obs,
            2,
        ), f"Expected shape {(batch_size, dim_obs, 2)}, got {out.shape}"
        assert out_ema.shape == (
            batch_size,
            dim_obs,
            2,
        ), f"Expected shape {(batch_size, dim_obs, 2)}, got {out_ema.shape}"

        # try restoring the model from the checkpoint
        # ignore warnings about sharding for the next line
        if pipeline_cls in [
            ConditionalFlowPipeline,
            ConditionalDiffusionPipeline,
        ]:
            pipeline2 = pipeline_cls(
                model,
                train_dataset,
                val_dataset,
                dim_obs,
                dim_cond,
                ch_obs=2,
                ch_cond=2,
                training_config=training_config,
            )
        else:
            pipeline2 = pipeline_cls(
                model,
                train_dataset,
                val_dataset,
                dim_obs,
                dim_cond,
                ch_obs=2,
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

        cond = jnp.zeros((32, dim_cond, 2))
        # try sampling from the model
        sample = pipeline.sample(
            jax.random.PRNGKey(1),
            cond,
            nsamples=32,
            use_ema=False,
        )
        assert sample.shape == (
            32,
            dim_obs,
            2,
        ), f"Expected shape (32, {dim_obs}, 2), got {sample.shape}"

        sample = pipeline.sample(
            jax.random.PRNGKey(1),
            cond,
            nsamples=32,
            use_ema=True,
        )
        assert sample.shape == (
            32,
            dim_obs,
            2,
        ), f"Expected shape (32, {dim_obs}, 2), got {sample.shape}"

        # sample from the restored model
        sample_restored = pipeline2.sample(
            jax.random.PRNGKey(1),
            cond,
            nsamples=32,
            use_ema=True,
        )
        assert jnp.allclose(
            sample, sample_restored
        ), "Restored model samples do not match"

        # test batched sampling
        cond = jnp.zeros((3, dim_cond, 2))
        sample = pipeline.sample_batched(
            jax.random.PRNGKey(1),
            cond,
            nsamples=4,
            chunk_size=2,
            show_progress_bars=False,
        )
        assert sample.shape == (
            4,
            3,
            dim_obs,
            2,
        ), f"Expected shape (4, 3, {dim_obs}, 2), got {sample.shape}"


########


@pytest.mark.parametrize(
    "pipeline_cls",
    [
        UnconditionalFlowPipeline,
        UnconditionalDiffusionPipeline,
    ],
)
def test_model_general_unconditional(pipeline_cls):

    train_dataset = train_dataset_joint
    val_dataset = val_dataset_joint

    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        training_config = pipeline_cls.get_default_training_config()
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
            ch_obs=2,
            training_config=training_config,
        )

        batch_size = 3
        t = jnp.linspace(0, 1, batch_size)
        obs = jnp.ones((batch_size, dim_joint, 2))

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
            2,
        ), f"Expected shape {(batch_size, dim_joint, 2)}, got {out.shape}"
        assert out_ema.shape == (
            batch_size,
            dim_joint,
            2,
        ), f"Expected shape {(batch_size, dim_joint, 2)}, got {out_ema.shape}"

        # try restoring the model from the checkpoint
        # ignore warnings about sharding for the next line

        pipeline2 = pipeline_cls(
            model,
            train_dataset,
            val_dataset,
            dim_joint,
            ch_obs=2,
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
            2,
        ), f"Expected shape (32, {dim_joint}, 2), got {sample.shape}"

        sample = pipeline.sample(
            jax.random.PRNGKey(1),
            nsamples=32,
            use_ema=True,
        )
        assert sample.shape == (
            32,
            dim_joint,
            2,
        ), f"Expected shape (32, {dim_joint}, 2), got {sample.shape}"

        # sample from the restored model
        sample_restored = pipeline2.sample(
            jax.random.PRNGKey(1),
            nsamples=32,
            use_ema=True,
        )
        assert jnp.allclose(
            sample, sample_restored
        ), "Restored model samples do not match"

        # test batched sampling, should return NotImplementedError
        cond = jnp.zeros((32, dim_cond, 2))
        with pytest.raises(NotImplementedError):
            sample = pipeline.sample_batched(
                jax.random.PRNGKey(1),
                cond,
                nsamples=20,
                chunk_size=8,
            )
