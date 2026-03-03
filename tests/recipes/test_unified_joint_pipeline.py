import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx

import pytest
import tempfile

import grain
import numpy as np

from gensbi.recipes import JointPipeline
from gensbi.core import FlowMatchingMethod, DiffusionEDMMethod, ScoreMatchingMethod

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from mock_models import MockJointModel

nsamples = 1000
key = jax.random.PRNGKey(0)

dim_obs = 2
dim_cond = 7
dim_joint = dim_obs + dim_cond

theta = jax.random.normal(key, (nsamples, dim_obs, 2))
x = jax.random.normal(key, (nsamples, dim_cond, 2))
data = jnp.concatenate([theta, x], axis=1)

train_dataset = (
    grain.MapDataset.source(np.array(data)[:800])
    .shuffle(42)
    .repeat()
    .to_iter_dataset()
    .batch(32)
)

val_dataset = (
    grain.MapDataset.source(np.array(data)[800:])
    .shuffle(42)
    .repeat()
    .to_iter_dataset()
    .batch(32)
)


@pytest.mark.parametrize(
    "method",
    [
        FlowMatchingMethod(),
        DiffusionEDMMethod(),
        ScoreMatchingMethod(),
    ],
    ids=["flow", "diffusion", "score"],
)
@pytest.mark.parametrize("condition_mask_kind", ["structured", "posterior"])
def test_unified_joint_init(method, condition_mask_kind):
    pipeline = JointPipeline(
        model=MockJointModel(),
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        dim_obs=dim_obs,
        dim_cond=dim_cond,
        method=method,
        condition_mask_kind=condition_mask_kind,
    )
    assert isinstance(pipeline, JointPipeline)
    assert hasattr(pipeline, "node_ids")
    assert hasattr(pipeline, "obs_ids")
    assert hasattr(pipeline, "cond_ids")
    assert hasattr(pipeline, "path")
    assert hasattr(pipeline, "loss_obj")
    assert pipeline.dim_joint == dim_obs + dim_cond


@pytest.mark.parametrize(
    "method",
    [
        FlowMatchingMethod(),
        DiffusionEDMMethod(),
        ScoreMatchingMethod(),
    ],
    ids=["flow", "diffusion", "score"],
)
def test_unified_joint_train_and_sample(method):
    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        training_config = JointPipeline.get_default_training_config()
        training_config["checkpoint_dir"] = model_dir
        training_config["val_every"] = 1

        pipeline = JointPipeline(
            model=MockJointModel(),
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            method=method,
            ch_obs=2,
            training_config=training_config,
        )

        # Train 2 steps
        pipeline.train(nnx.Rngs(0), nsteps=2, save_model=False)
        pipeline._wrap_model()

        # Direct model evaluation
        batch_size = 3
        t = jnp.linspace(0, 1, batch_size)
        obs = jnp.zeros((batch_size, dim_obs, 2))
        cond_eval = jnp.zeros((batch_size, dim_cond, 2))
        out = pipeline.model_wrapped(t, obs, pipeline.obs_ids, cond_eval, pipeline.cond_ids)
        assert out.shape == (batch_size, dim_obs, 2)
        out_ema = pipeline.ema_model_wrapped(t, obs, pipeline.obs_ids, cond_eval, pipeline.cond_ids)
        assert out_ema.shape == (batch_size, dim_obs, 2)

        # Sample
        cond_single = jnp.zeros((1, dim_cond, 2))
        sample = pipeline.sample(
            jax.random.PRNGKey(1), cond_single, nsamples=32, use_ema=False
        )
        assert sample.shape == (32, dim_obs, 2)

        sample_ema = pipeline.sample(
            jax.random.PRNGKey(1), cond_single, nsamples=32, use_ema=True
        )
        assert sample_ema.shape == (32, dim_obs, 2)


@pytest.mark.parametrize(
    "method",
    [
        FlowMatchingMethod(),
        DiffusionEDMMethod(),
        ScoreMatchingMethod(),
    ],
    ids=["flow", "diffusion", "score"],
)
def test_unified_joint_loss_fn(method):
    pipeline = JointPipeline(
        model=MockJointModel(),
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        dim_obs=dim_obs,
        dim_cond=dim_cond,
        method=method,
        ch_obs=2,
    )
    pipeline._wrap_model()

    loss_fn = pipeline.get_loss_fn()
    mock_batch = jnp.zeros((32, dim_joint, 2))
    loss = loss_fn(pipeline.model, mock_batch, key=jax.random.PRNGKey(1))
    assert loss.shape == ()


@pytest.mark.parametrize("use_ema", [False, True], ids=["no_ema", "ema"])
@pytest.mark.parametrize("exact_divergence", [True, False], ids=["exact", "hutchinson"])
def test_unified_joint_log_prob(use_ema, exact_divergence):
    # log_prob is only supported for FlowMatchingMethod
    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        training_config = JointPipeline.get_default_training_config()
        training_config["checkpoint_dir"] = model_dir
        training_config["val_every"] = 1

        pipeline = JointPipeline(
            model=MockJointModel(),
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            method=FlowMatchingMethod(),
            ch_obs=2,
            training_config=training_config,
        )

        pipeline.train(nnx.Rngs(0), nsteps=2, save_model=False)
        pipeline._wrap_model()

        batch_size = 3
        x_1 = jnp.zeros((batch_size, dim_obs, 2))
        x_o = jnp.zeros((1, dim_cond, 2))

        log_prob_key = jax.random.PRNGKey(42) if not exact_divergence else None
        lp = pipeline.log_prob(
            x_1, x_o,
            use_ema=use_ema,
            key=log_prob_key,
            exact_divergence=exact_divergence,
        )
        assert lp.shape == (batch_size,)


@pytest.mark.parametrize(
    "method",
    [
        DiffusionEDMMethod(),
        ScoreMatchingMethod(),
    ],
    ids=["diffusion", "score"],
)
def test_unified_joint_log_prob_not_implemented(method):
    # EDM and ScoreMatching do not support log_prob
    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        training_config = JointPipeline.get_default_training_config()
        training_config["checkpoint_dir"] = model_dir
        training_config["val_every"] = 1

        pipeline = JointPipeline(
            model=MockJointModel(),
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            method=method,
            ch_obs=2,
            training_config=training_config,
        )

        pipeline.train(nnx.Rngs(0), nsteps=2, save_model=False)
        pipeline._wrap_model()

        x_1 = jnp.zeros((3, dim_obs, 2))
        x_o = jnp.zeros((1, dim_cond, 2))

        with pytest.raises(NotImplementedError):
            pipeline.log_prob(x_1, x_o, use_ema=True)
