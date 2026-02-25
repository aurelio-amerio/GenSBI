# Tests for ScoreMatching solver variations (mirror test_diffusion_scheduler.py)

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx

import pytest
import tempfile

import grain
import numpy as np

from gensbi.recipes import (
    UnconditionalPipeline,
    ConditionalPipeline,
    JointPipeline,
)

from gensbi.core import ScoreMatchingMethod
from gensbi.diffusion.solver import SMSolver, SMPFSolver

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from mock_models import MockUnconditionalModel, MockConditionalModel, MockJointModel


nsamples = 100
key = jax.random.PRNGKey(0)

dim_obs = 2
dim_cond = 2
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
    grain.MapDataset.source(np.array(data)[:80])
    .shuffle(42)
    .repeat()
    .to_iter_dataset()
    .batch(32)
)

val_dataset_joint = (
    grain.MapDataset.source(np.array(data)[80:])
    .shuffle(42)
    .repeat()
    .to_iter_dataset()
    .batch(32)
)

train_dataset_cond = (
    grain.MapDataset.source(np.array(data)[:80])
    .shuffle(42)
    .repeat()
    .to_iter_dataset()
    .batch(32)
    .map(split_obs_cond)
)

val_dataset_cond = (
    grain.MapDataset.source(np.array(data)[80:])
    .shuffle(42)
    .repeat()
    .to_iter_dataset()
    .batch(32)
    .map(split_obs_cond)
)


# --- SMSolver tests (default, different SDE types) ---


@pytest.mark.parametrize("sde_type", ["VP", "VE"])
def test_unconditional_sm_default_solver(sde_type):
    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        method = ScoreMatchingMethod(sde_type=sde_type)
        training_config = UnconditionalPipeline.get_default_training_config()
        training_config["checkpoint_dir"] = model_dir
        training_config["val_every"] = 1

        pipeline = UnconditionalPipeline(
            MockUnconditionalModel(),
            train_dataset_joint,
            val_dataset_joint,
            dim_joint,
            method=method,
            ch_obs=2,
            training_config=training_config,
        )

        pipeline.ema_model = pipeline.model
        pipeline._wrap_model()

        sample = pipeline.sample(
            jax.random.PRNGKey(1),
            nsamples=10,
            use_ema=False,
        )
        assert sample.shape == (10, dim_joint, 2)


@pytest.mark.parametrize("sde_type", ["VP", "VE"])
def test_conditional_sm_default_solver(sde_type):
    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        method = ScoreMatchingMethod(sde_type=sde_type)
        training_config = ConditionalPipeline.get_default_training_config()
        training_config["checkpoint_dir"] = model_dir
        training_config["val_every"] = 1

        pipeline = ConditionalPipeline(
            MockConditionalModel(),
            train_dataset_cond,
            val_dataset_cond,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            method=method,
            ch_obs=2,
            ch_cond=2,
            training_config=training_config,
        )

        pipeline.ema_model = pipeline.model
        pipeline._wrap_model()

        x_o = jax.random.normal(jax.random.PRNGKey(2), (1, dim_cond, 2))

        sample = pipeline.sample(
            jax.random.PRNGKey(1),
            x_o=x_o,
            nsamples=5,
            use_ema=False,
        )
        assert sample.shape == (5, dim_obs, 2)


@pytest.mark.parametrize("sde_type", ["VP", "VE"])
def test_joint_sm_default_solver(sde_type):
    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        method = ScoreMatchingMethod(sde_type=sde_type)
        training_config = JointPipeline.get_default_training_config()
        training_config["checkpoint_dir"] = model_dir
        training_config["val_every"] = 1

        pipeline = JointPipeline(
            MockJointModel(),
            train_dataset_joint,
            val_dataset_joint,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            method=method,
            ch_obs=2,
            training_config=training_config,
            condition_mask_kind="structured",
        )

        pipeline.ema_model = pipeline.model
        pipeline._wrap_model()

        x_o = jax.random.normal(jax.random.PRNGKey(2), (1, dim_cond, 2))

        sample = pipeline.sample(
            jax.random.PRNGKey(1),
            x_o=x_o,
            nsamples=5,
            use_ema=False,
        )
        assert sample.shape == (5, dim_obs, 2)


# --- SMPFSolver tests (probability flow ODE) ---


@pytest.mark.parametrize("sde_type", ["VP", "VE"])
def test_unconditional_sm_pf_solver(sde_type):
    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        method = ScoreMatchingMethod(sde_type=sde_type)
        training_config = UnconditionalPipeline.get_default_training_config()
        training_config["checkpoint_dir"] = model_dir
        training_config["val_every"] = 1

        pipeline = UnconditionalPipeline(
            MockUnconditionalModel(),
            train_dataset_joint,
            val_dataset_joint,
            dim_joint,
            method=method,
            ch_obs=2,
            training_config=training_config,
        )

        pipeline.ema_model = pipeline.model
        pipeline._wrap_model()

        sample = pipeline.sample(
            jax.random.PRNGKey(1),
            nsamples=10,
            use_ema=False,
            solver=(SMPFSolver, {}),
        )
        assert sample.shape == (10, dim_joint, 2)


@pytest.mark.parametrize("sde_type", ["VP", "VE"])
def test_conditional_sm_pf_solver(sde_type):
    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        method = ScoreMatchingMethod(sde_type=sde_type)
        training_config = ConditionalPipeline.get_default_training_config()
        training_config["checkpoint_dir"] = model_dir
        training_config["val_every"] = 1

        pipeline = ConditionalPipeline(
            MockConditionalModel(),
            train_dataset_cond,
            val_dataset_cond,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            method=method,
            ch_obs=2,
            ch_cond=2,
            training_config=training_config,
        )

        pipeline.ema_model = pipeline.model
        pipeline._wrap_model()

        x_o = jax.random.normal(jax.random.PRNGKey(2), (1, dim_cond, 2))

        sample = pipeline.sample(
            jax.random.PRNGKey(1),
            x_o=x_o,
            nsamples=5,
            use_ema=False,
            solver=(SMPFSolver, {}),
        )
        assert sample.shape == (5, dim_obs, 2)


@pytest.mark.parametrize("sde_type", ["VP", "VE"])
def test_joint_sm_pf_solver(sde_type):
    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        method = ScoreMatchingMethod(sde_type=sde_type)
        training_config = JointPipeline.get_default_training_config()
        training_config["checkpoint_dir"] = model_dir
        training_config["val_every"] = 1

        pipeline = JointPipeline(
            MockJointModel(),
            train_dataset_joint,
            val_dataset_joint,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            method=method,
            ch_obs=2,
            training_config=training_config,
            condition_mask_kind="structured",
        )

        pipeline.ema_model = pipeline.model
        pipeline._wrap_model()

        x_o = jax.random.normal(jax.random.PRNGKey(2), (1, dim_cond, 2))

        sample = pipeline.sample(
            jax.random.PRNGKey(1),
            x_o=x_o,
            nsamples=5,
            use_ema=False,
            solver=(SMPFSolver, {}),
        )
        assert sample.shape == (5, dim_obs, 2)
