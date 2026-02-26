# Tests for EDM diffusion pipeline schedulers (unified pipelines)

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx

import warnings

import pytest

import tempfile

from gensbi.recipes import (
    UnconditionalPipeline,
    ConditionalPipeline,
    JointPipeline,
)

from gensbi.core import DiffusionEDMMethod

import grain
import numpy as np
from gensbi.diffusion.path.scheduler import EDMScheduler, VEEdmScheduler, VPEdmScheduler
from gensbi.diffusion.solver import EDMSolver

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
    )  # assuming first dim_obs are obs, last dim_cond are cond


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

@pytest.mark.parametrize("sde_type", ["EDM", "VE", "VP"])
def test_unconditional_diffusion_sde_types(sde_type):
    train_dataset = train_dataset_joint
    val_dataset = val_dataset_joint

    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        method = DiffusionEDMMethod(sde=sde_type)
        training_config = UnconditionalPipeline.get_default_training_config()
        extra = method.get_extra_training_config()
        for k, v in extra.items():
            training_config.setdefault(k, v)
        training_config["checkpoint_dir"] = model_dir
        training_config["val_every"] = 1

        pipeline = UnconditionalPipeline(
            MockUnconditionalModel(),
            train_dataset,
            val_dataset,
            dim_joint,
            method=method,
            ch_obs=2,
            training_config=training_config,
        )

        if sde_type == "EDM":
            assert isinstance(pipeline.path.scheduler, EDMScheduler)
        elif sde_type == "VE":
            assert isinstance(pipeline.path.scheduler, VEEdmScheduler)
        if sde_type == "VP":
            assert isinstance(pipeline.path.scheduler, VPEdmScheduler)

        # Initialize model wrappers for testing
        pipeline.ema_model = pipeline.model
        pipeline._wrap_model()

        # try sampling
        sample = pipeline.sample(
            jax.random.PRNGKey(1),
            nsamples=10,
            use_ema=False,
        )
        assert sample.shape == (10, dim_joint, 2)


def test_unconditional_diffusion_solver_scheduler():
    train_dataset = train_dataset_joint
    val_dataset = val_dataset_joint
    sde_type = "EDM"

    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        method = DiffusionEDMMethod(sde=sde_type)
        training_config = UnconditionalPipeline.get_default_training_config()
        extra = method.get_extra_training_config()
        for k, v in extra.items():
            training_config.setdefault(k, v)
        training_config["checkpoint_dir"] = model_dir
        training_config["val_every"] = 1

        pipeline = UnconditionalPipeline(
            MockUnconditionalModel(),
            train_dataset,
            val_dataset,
            dim_joint,
            method=method,
            ch_obs=2,
            training_config=training_config,
        )

        # Initialize model wrappers for testing
        pipeline.ema_model = pipeline.model
        pipeline._wrap_model()

        # Create a custom scheduler (e.g., different parameters)
        custom_scheduler = EDMScheduler(sigma_min=0.1, sigma_max=50.0)

        # Sample with the custom scheduler via the solver tuple
        sample = pipeline.sample(
            jax.random.PRNGKey(1),
            nsamples=10,
            use_ema=False,
            solver_scheduler=custom_scheduler,
        )
        assert sample.shape == (10, dim_joint, 2)

        # Verify that we can pass a VE scheduler to an EDM pipeline (unusual but allowed by code)
        ve_scheduler = VEEdmScheduler(sigma_min=0.1, sigma_max=20.0)
        sample_ve = pipeline.sample(
            jax.random.PRNGKey(1),
            nsamples=10,
            use_ema=False,
            solver_scheduler=ve_scheduler,
        )
        assert sample_ve.shape == (10, dim_joint, 2)


@pytest.mark.parametrize("sde_type", ["EDM", "VE", "VP"])
def test_conditional_diffusion_sde_types(sde_type):
    train_dataset = train_dataset_cond
    val_dataset = val_dataset_cond

    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        method = DiffusionEDMMethod(sde=sde_type)
        training_config = ConditionalPipeline.get_default_training_config()
        extra = method.get_extra_training_config()
        for k, v in extra.items():
            training_config.setdefault(k, v)
        training_config["checkpoint_dir"] = model_dir
        training_config["val_every"] = 1

        pipeline = ConditionalPipeline(
            MockConditionalModel(),
            train_dataset,
            val_dataset,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            method=method,
            ch_obs=2,
            ch_cond=2,
            training_config=training_config,
        )

        if sde_type == "EDM":
            assert isinstance(pipeline.path.scheduler, EDMScheduler)
        elif sde_type == "VE":
            assert isinstance(pipeline.path.scheduler, VEEdmScheduler)
        if sde_type == "VP":
            assert isinstance(pipeline.path.scheduler, VPEdmScheduler)

        # Initialize model wrappers for testing
        pipeline.ema_model = pipeline.model
        pipeline._wrap_model()

        # try sampling
        x_o = jax.random.normal(jax.random.PRNGKey(2), (1, dim_cond, 2))

        sample = pipeline.sample(
            jax.random.PRNGKey(1),
            x_o=x_o,
            nsamples=10,
            use_ema=False,
        )
        assert sample.shape == (10, dim_obs, 2)


@pytest.mark.parametrize("sde_type", ["EDM", "VE", "VP"])
def test_joint_diffusion_sde_types(sde_type):
    train_dataset = train_dataset_joint
    val_dataset = val_dataset_joint

    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        method = DiffusionEDMMethod(sde=sde_type)
        training_config = JointPipeline.get_default_training_config()
        extra = method.get_extra_training_config()
        for k, v in extra.items():
            training_config.setdefault(k, v)
        training_config["checkpoint_dir"] = model_dir
        training_config["val_every"] = 1

        # Joint pipeline needs condition_mask_kind
        pipeline = JointPipeline(
            MockJointModel(),
            train_dataset,
            val_dataset,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            method=method,
            ch_obs=2,
            training_config=training_config,
            condition_mask_kind="structured",
        )

        if sde_type == "EDM":
            assert isinstance(pipeline.path.scheduler, EDMScheduler)
        elif sde_type == "VE":
            assert isinstance(pipeline.path.scheduler, VEEdmScheduler)
        if sde_type == "VP":
            assert isinstance(pipeline.path.scheduler, VPEdmScheduler)

        # Initialize model wrappers for testing
        pipeline.ema_model = pipeline.model
        pipeline._wrap_model()

        # try sampling
        x_o = jax.random.normal(jax.random.PRNGKey(2), (1, dim_cond, 2))

        sample = pipeline.sample(
            jax.random.PRNGKey(1),
            x_o=x_o,
            nsamples=10,
            use_ema=False,
        )
        assert sample.shape == (10, dim_obs, 2)


def test_conditional_diffusion_solver_scheduler():
    train_dataset = train_dataset_cond
    val_dataset = val_dataset_cond
    sde_type = "EDM"

    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        method = DiffusionEDMMethod(sde=sde_type)
        training_config = ConditionalPipeline.get_default_training_config()
        extra = method.get_extra_training_config()
        for k, v in extra.items():
            training_config.setdefault(k, v)
        training_config["checkpoint_dir"] = model_dir
        training_config["val_every"] = 1

        pipeline = ConditionalPipeline(
            MockConditionalModel(),
            train_dataset,
            val_dataset,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            method=method,
            ch_obs=2,
            ch_cond=2,
            training_config=training_config,
        )

        # Initialize model wrappers for testing
        pipeline.ema_model = pipeline.model
        pipeline._wrap_model()

        x_o = jax.random.normal(jax.random.PRNGKey(2), (1, dim_cond, 2))

        # Create a custom scheduler (e.g., different parameters)
        custom_scheduler = EDMScheduler(sigma_min=0.1, sigma_max=50.0)

        # Sample with the custom scheduler via the solver tuple
        sample = pipeline.sample(
            jax.random.PRNGKey(1),
            x_o=x_o,
            nsamples=10,
            use_ema=False,
            solver_scheduler=custom_scheduler,
        )
        assert sample.shape == (10, dim_obs, 2)

        # Verify that we can pass a VE scheduler to an EDM pipeline (unusual but allowed by code)
        ve_scheduler = VEEdmScheduler(sigma_min=0.1, sigma_max=20.0)
        sample_ve = pipeline.sample(
            jax.random.PRNGKey(1),
            x_o=x_o,
            nsamples=10,
            use_ema=False,
            solver_scheduler=ve_scheduler,
        )
        assert sample_ve.shape == (10, dim_obs, 2)


def test_joint_diffusion_solver_scheduler():
    train_dataset = train_dataset_joint
    val_dataset = val_dataset_joint
    sde_type = "EDM"

    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        method = DiffusionEDMMethod(sde=sde_type)
        training_config = JointPipeline.get_default_training_config()
        extra = method.get_extra_training_config()
        for k, v in extra.items():
            training_config.setdefault(k, v)
        training_config["checkpoint_dir"] = model_dir
        training_config["val_every"] = 1

        pipeline = JointPipeline(
            MockJointModel(),
            train_dataset,
            val_dataset,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            method=method,
            ch_obs=2,
            training_config=training_config,
            condition_mask_kind="structured",
        )

        # Initialize model wrappers for testing
        pipeline.ema_model = pipeline.model
        pipeline._wrap_model()

        x_o = jax.random.normal(jax.random.PRNGKey(2), (1, dim_cond, 2))

        # Create a custom scheduler (e.g., different parameters)
        custom_scheduler = EDMScheduler(sigma_min=0.1, sigma_max=50.0)

        # Sample with the custom scheduler via the solver tuple
        sample = pipeline.sample(
            jax.random.PRNGKey(1),
            x_o=x_o,
            nsamples=10,
            use_ema=False,
            solver_scheduler=custom_scheduler,
        )
        assert sample.shape == (10, dim_obs, 2)

        # Verify that we can pass a VE scheduler to an EDM pipeline (unusual but allowed by code)
        ve_scheduler = VEEdmScheduler(sigma_min=0.1, sigma_max=20.0)
        sample_ve = pipeline.sample(
            jax.random.PRNGKey(1),
            x_o=x_o,
            nsamples=10,
            use_ema=False,
            solver_scheduler=ve_scheduler,
        )
        assert sample_ve.shape == (10, dim_obs, 2)


# --- sample_batched tests ---


@pytest.mark.parametrize("sde_type", ["EDM", "VE", "VP"])
def test_conditional_diffusion_sample_batched(sde_type):
    """sample_batched with multiple conditions for ConditionalPipeline + EDM."""
    train_dataset = train_dataset_cond
    val_dataset = val_dataset_cond

    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        method = DiffusionEDMMethod(sde=sde_type)
        training_config = ConditionalPipeline.get_default_training_config()
        extra = method.get_extra_training_config()
        for k, v in extra.items():
            training_config.setdefault(k, v)
        training_config["checkpoint_dir"] = model_dir
        training_config["val_every"] = 1

        pipeline = ConditionalPipeline(
            MockConditionalModel(),
            train_dataset,
            val_dataset,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            method=method,
            ch_obs=2,
            ch_cond=2,
            training_config=training_config,
        )

        pipeline.ema_model = pipeline.model
        pipeline._wrap_model()

        x_o = jax.random.normal(jax.random.PRNGKey(2), (3, dim_cond, 2))

        samples = pipeline.sample_batched(
            jax.random.PRNGKey(1),
            x_o=x_o,
            nsamples=5,
            use_ema=False,
        )
        assert samples.shape == (5, 3, dim_obs, 2)


@pytest.mark.parametrize("sde_type", ["EDM", "VE", "VP"])
def test_joint_diffusion_sample_batched(sde_type):
    """sample_batched with multiple conditions for JointPipeline + EDM."""
    train_dataset = train_dataset_joint
    val_dataset = val_dataset_joint

    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        method = DiffusionEDMMethod(sde=sde_type)
        training_config = JointPipeline.get_default_training_config()
        extra = method.get_extra_training_config()
        for k, v in extra.items():
            training_config.setdefault(k, v)
        training_config["checkpoint_dir"] = model_dir
        training_config["val_every"] = 1

        pipeline = JointPipeline(
            MockJointModel(),
            train_dataset,
            val_dataset,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            method=method,
            ch_obs=2,
            training_config=training_config,
            condition_mask_kind="structured",
        )

        pipeline.ema_model = pipeline.model
        pipeline._wrap_model()

        x_o = jax.random.normal(jax.random.PRNGKey(2), (3, dim_cond, 2))

        samples = pipeline.sample_batched(
            jax.random.PRNGKey(1),
            x_o=x_o,
            nsamples=5,
            use_ema=False,
        )
        assert samples.shape == (5, 3, dim_obs, 2)
