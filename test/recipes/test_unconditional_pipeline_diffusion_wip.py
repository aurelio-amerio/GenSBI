# WIP not working

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx

import warnings

import pytest

import tempfile

from gensbi.recipes import UnconditionalDiffusionPipeline

from gensbi.models import Simformer, SimformerParams

import grain
import numpy as np
from gensbi.diffusion.path.scheduler import EDMScheduler, VEEdmScheduler, VPEdmScheduler


nsamples = 100
key = jax.random.PRNGKey(0)

dim_obs = 2
dim_cond = 0
dim_joint = dim_obs + dim_cond


theta = jax.random.normal(key, (nsamples, dim_obs, 2))
data = theta


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


def get_model():
    return model_joint


@pytest.mark.parametrize("sde_type", ["EDM", "VE", "VP"])
def test_unconditional_diffusion_sde_types(sde_type):
    pipeline_cls = UnconditionalDiffusionPipeline
    train_dataset = train_dataset_joint
    val_dataset = val_dataset_joint

    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        training_config = pipeline_cls.get_default_training_config(sde=sde_type)
        training_config["checkpoint_dir"] = model_dir
        training_config["val_every"] = 1

        model = get_model()

        pipeline = pipeline_cls(
            model,
            train_dataset,
            val_dataset,
            dim_joint,
            ch_obs=2,
            sde=sde_type,
            training_config=training_config,
        )

        assert pipeline.sde == sde_type
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
    pipeline_cls = UnconditionalDiffusionPipeline
    train_dataset = train_dataset_joint
    val_dataset = val_dataset_joint
    sde_type = "EDM"

    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        training_config = pipeline_cls.get_default_training_config(sde=sde_type)
        training_config["checkpoint_dir"] = model_dir
        training_config["val_every"] = 1

        model = get_model()

        pipeline = pipeline_cls(
            model,
            train_dataset,
            val_dataset,
            dim_joint,
            ch_obs=2,
            sde=sde_type,
            training_config=training_config,
        )

        # Initialize model wrappers for testing
        pipeline.ema_model = pipeline.model
        pipeline._wrap_model()

        # Create a custom scheduler (e.g., different parameters)
        custom_scheduler = EDMScheduler(sigma_min=0.1, sigma_max=50.0)

        # Sample with the custom scheduler
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
