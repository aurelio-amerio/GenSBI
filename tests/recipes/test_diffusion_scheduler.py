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

from gensbi.models import Simformer, SimformerParams, Flux1, Flux1Params

import grain
import numpy as np
from gensbi.diffusion.path.scheduler import EDMScheduler, VEEdmScheduler, VPEdmScheduler
from gensbi.diffusion.solver import EDMSolver


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

# we define a conditional and a joint model for testing

params_simf = SimformerParams(
    rngs=nnx.Rngs(0),
    in_channels=2,
    val_emb_dim=2,
    id_emb_dim=2,
    cond_emb_dim=2,
    dim_joint=dim_joint,
    fourier_features=32,
    num_heads=2,
    depth=1,
    mlp_ratio=3,
    qkv_features=4,
    num_hidden_layers=1,
)

model_joint = Simformer(params_simf)


params = Flux1Params(
    in_channels=2,
    vec_in_dim=None,
    context_in_dim=2,
    mlp_ratio=1,
    num_heads=2,
    depth=2,
    depth_single_blocks=2,
    axes_dim=[
        2,
    ],
    qkv_bias=True,
    dim_obs=dim_obs,
    dim_cond=dim_cond,
    theta=20,
    id_merge_mode="sum",
    id_embedding_strategy=("absolute", "absolute"),
    rngs=nnx.Rngs(default=42),
    param_dtype=jnp.float32,
)

model_conditional = Flux1(params)


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
            model_joint,
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
            model_joint,
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
            solver=(EDMSolver, {"solver_scheduler": custom_scheduler}),
        )
        assert sample.shape == (10, dim_joint, 2)

        # Verify that we can pass a VE scheduler to an EDM pipeline (unusual but allowed by code)
        ve_scheduler = VEEdmScheduler(sigma_min=0.1, sigma_max=20.0)
        sample_ve = pipeline.sample(
            jax.random.PRNGKey(1),
            nsamples=10,
            use_ema=False,
            solver=(EDMSolver, {"solver_scheduler": ve_scheduler}),
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
            model_conditional,
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
        x_o = jax.random.normal(jax.random.PRNGKey(2), (10, dim_cond, 2))

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
            model_joint,
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
        x_o = jax.random.normal(jax.random.PRNGKey(2), (10, dim_cond, 2))

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
            model_conditional,
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

        x_o = jax.random.normal(jax.random.PRNGKey(2), (10, dim_cond, 2))

        # Create a custom scheduler (e.g., different parameters)
        custom_scheduler = EDMScheduler(sigma_min=0.1, sigma_max=50.0)

        # Sample with the custom scheduler via the solver tuple
        sample = pipeline.sample(
            jax.random.PRNGKey(1),
            x_o=x_o,
            nsamples=10,
            use_ema=False,
            solver=(EDMSolver, {"solver_scheduler": custom_scheduler}),
        )
        assert sample.shape == (10, dim_obs, 2)

        # Verify that we can pass a VE scheduler to an EDM pipeline (unusual but allowed by code)
        ve_scheduler = VEEdmScheduler(sigma_min=0.1, sigma_max=20.0)
        sample_ve = pipeline.sample(
            jax.random.PRNGKey(1),
            x_o=x_o,
            nsamples=10,
            use_ema=False,
            solver=(EDMSolver, {"solver_scheduler": ve_scheduler}),
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
            model_joint,
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

        x_o = jax.random.normal(jax.random.PRNGKey(2), (10, dim_cond, 2))

        # Create a custom scheduler (e.g., different parameters)
        custom_scheduler = EDMScheduler(sigma_min=0.1, sigma_max=50.0)

        # Sample with the custom scheduler via the solver tuple
        sample = pipeline.sample(
            jax.random.PRNGKey(1),
            x_o=x_o,
            nsamples=10,
            use_ema=False,
            solver=(EDMSolver, {"solver_scheduler": custom_scheduler}),
        )
        assert sample.shape == (10, dim_obs, 2)

        # Verify that we can pass a VE scheduler to an EDM pipeline (unusual but allowed by code)
        ve_scheduler = VEEdmScheduler(sigma_min=0.1, sigma_max=20.0)
        sample_ve = pipeline.sample(
            jax.random.PRNGKey(1),
            x_o=x_o,
            nsamples=10,
            use_ema=False,
            solver=(EDMSolver, {"solver_scheduler": ve_scheduler}),
        )
        assert sample_ve.shape == (10, dim_obs, 2)
