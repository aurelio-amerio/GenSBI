# Tests for EDM diffusion solver across all pipeline types.
# Verifies that every (pipeline × SDE-type) combination produces correct sample shapes,
# and that model_extras flow correctly through sample_batched (dynamic extras).

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp

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


# ---------------------------------------------------------------------------
# sample: one observation, default EDMSolver
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sde_type", ["EDM", "VE", "VP"])
def test_unconditional_edm_sample(sde_type):
    """Unconditional sampling with each SDE noise schedule (EDM/VE/VP)."""
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
            train_dataset_joint,
            val_dataset_joint,
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

        pipeline.ema_model = pipeline.model
        pipeline._wrap_model()

        sample = pipeline.sample(
            jax.random.PRNGKey(1),
            nsamples=10,
            use_ema=False,
        )
        assert sample.shape == (10, dim_joint, 2)


@pytest.mark.parametrize("sde_type", ["EDM", "VE", "VP"])
def test_conditional_edm_sample(sde_type):
    """Conditional sampling with each SDE noise schedule (EDM/VE/VP)."""
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
            train_dataset_cond,
            val_dataset_cond,
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

        pipeline.ema_model = pipeline.model
        pipeline._wrap_model()

        x_o = jax.random.normal(jax.random.PRNGKey(2), (1, dim_cond, 2))

        sample = pipeline.sample(
            jax.random.PRNGKey(1),
            x_o=x_o,
            nsamples=10,
            use_ema=False,
        )
        assert sample.shape == (10, dim_obs, 2)


@pytest.mark.parametrize("sde_type", ["EDM", "VE", "VP"])
def test_joint_edm_sample(sde_type):
    """Joint sampling with each SDE noise schedule (EDM/VE/VP)."""
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
            train_dataset_joint,
            val_dataset_joint,
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

        pipeline.ema_model = pipeline.model
        pipeline._wrap_model()

        x_o = jax.random.normal(jax.random.PRNGKey(2), (1, dim_cond, 2))

        sample = pipeline.sample(
            jax.random.PRNGKey(1),
            x_o=x_o,
            nsamples=10,
            use_ema=False,
        )
        assert sample.shape == (10, dim_obs, 2)


# ---------------------------------------------------------------------------
# sample_batched: multiple observations, verifies dynamic model_extras
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sde_type", ["EDM", "VE", "VP"])
def test_conditional_edm_sample_batched(sde_type):
    """Batched conditional sampling — model_extras must change per condition."""
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

        x_o = jax.random.normal(jax.random.PRNGKey(2), (3, dim_cond, 2))

        samples = pipeline.sample_batched(
            jax.random.PRNGKey(1),
            x_o=x_o,
            nsamples=5,
            use_ema=False,
        )
        assert samples.shape == (5, 3, dim_obs, 2)


@pytest.mark.parametrize("sde_type", ["EDM", "VE", "VP"])
def test_joint_edm_sample_batched(sde_type):
    """Batched joint sampling — model_extras must change per condition."""
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

        x_o = jax.random.normal(jax.random.PRNGKey(2), (3, dim_cond, 2))

        samples = pipeline.sample_batched(
            jax.random.PRNGKey(1),
            x_o=x_o,
            nsamples=5,
            use_ema=False,
        )
        assert samples.shape == (5, 3, dim_obs, 2)


# ---------------------------------------------------------------------------
# sample with custom scheduler: override the noise schedule at sampling time
# ---------------------------------------------------------------------------


def test_unconditional_edm_sample_custom_scheduler():
    """Override the noise schedule at sampling time (uncond pipeline)."""
    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        method = DiffusionEDMMethod(sde="EDM")
        training_config = UnconditionalPipeline.get_default_training_config()
        extra = method.get_extra_training_config()
        for k, v in extra.items():
            training_config.setdefault(k, v)
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

        # Custom EDM scheduler with different sigma bounds
        custom_scheduler = EDMScheduler(sigma_min=0.1, sigma_max=50.0)
        sample = pipeline.sample(
            jax.random.PRNGKey(1),
            nsamples=10,
            use_ema=False,
            solver_scheduler=custom_scheduler,
        )
        assert sample.shape == (10, dim_joint, 2)

        # Cross-family scheduler swap: VE scheduler on an EDM pipeline
        ve_scheduler = VEEdmScheduler(sigma_min=0.1, sigma_max=20.0)
        sample_ve = pipeline.sample(
            jax.random.PRNGKey(1),
            nsamples=10,
            use_ema=False,
            solver_scheduler=ve_scheduler,
        )
        assert sample_ve.shape == (10, dim_joint, 2)


def test_conditional_edm_sample_custom_scheduler():
    """Override the noise schedule at sampling time (cond pipeline)."""
    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        method = DiffusionEDMMethod(sde="EDM")
        training_config = ConditionalPipeline.get_default_training_config()
        extra = method.get_extra_training_config()
        for k, v in extra.items():
            training_config.setdefault(k, v)
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

        custom_scheduler = EDMScheduler(sigma_min=0.1, sigma_max=50.0)
        sample = pipeline.sample(
            jax.random.PRNGKey(1),
            x_o=x_o,
            nsamples=10,
            use_ema=False,
            solver_scheduler=custom_scheduler,
        )
        assert sample.shape == (10, dim_obs, 2)

        ve_scheduler = VEEdmScheduler(sigma_min=0.1, sigma_max=20.0)
        sample_ve = pipeline.sample(
            jax.random.PRNGKey(1),
            x_o=x_o,
            nsamples=10,
            use_ema=False,
            solver_scheduler=ve_scheduler,
        )
        assert sample_ve.shape == (10, dim_obs, 2)


def test_joint_edm_sample_custom_scheduler():
    """Override the noise schedule at sampling time (joint pipeline)."""
    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        method = DiffusionEDMMethod(sde="EDM")
        training_config = JointPipeline.get_default_training_config()
        extra = method.get_extra_training_config()
        for k, v in extra.items():
            training_config.setdefault(k, v)
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

        custom_scheduler = EDMScheduler(sigma_min=0.1, sigma_max=50.0)
        sample = pipeline.sample(
            jax.random.PRNGKey(1),
            x_o=x_o,
            nsamples=10,
            use_ema=False,
            solver_scheduler=custom_scheduler,
        )
        assert sample.shape == (10, dim_obs, 2)

        ve_scheduler = VEEdmScheduler(sigma_min=0.1, sigma_max=20.0)
        sample_ve = pipeline.sample(
            jax.random.PRNGKey(1),
            x_o=x_o,
            nsamples=10,
            use_ema=False,
            solver_scheduler=ve_scheduler,
        )
        assert sample_ve.shape == (10, dim_obs, 2)
