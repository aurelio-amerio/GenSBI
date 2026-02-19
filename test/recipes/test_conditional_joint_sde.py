#WIP not working


import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx
import pytest
import tempfile
import grain
import numpy as np

from gensbi.recipes import ConditionalDiffusionPipeline, JointDiffusionPipeline
from gensbi.models import Simformer, SimformerParams
from gensbi.diffusion.path.scheduler import EDMScheduler, VEEdmScheduler, VPEdmScheduler

nsamples = 100
key = jax.random.PRNGKey(0)

dim_obs = 2
dim_cond = 2
dim_joint = dim_obs + dim_cond

theta = jax.random.normal(key, (nsamples, dim_obs, 1))  # ch_obs=1
cond = jax.random.normal(key, (nsamples, dim_cond, 1))  # ch_cond=1


def get_dataset_conditional():
    ds = grain.MapDataset.source((np.array(theta), np.array(cond)))
    ds = ds.shuffle(42).repeat().to_iter_dataset().batch(10)
    return ds


params_simf = SimformerParams(
    rngs=nnx.Rngs(0),
    in_channels=1,
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


@pytest.mark.parametrize("sde_type", ["EDM", "VE", "VP"])
def test_conditional_diffusion_sde_types(sde_type):
    pipeline_cls = ConditionalDiffusionPipeline
    train_dataset = get_dataset_conditional()
    val_dataset = get_dataset_conditional()

    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        training_config = pipeline_cls.get_default_training_config(sde=sde_type)
        training_config["checkpoint_dir"] = model_dir
        training_config["val_every"] = 1

        model_test = Simformer(params_simf)

        pipeline = pipeline_cls(
            model_test,
            train_dataset,
            val_dataset,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            ch_obs=1,
            ch_cond=1,
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

        pipeline.ema_model = pipeline.model
        pipeline._wrap_model()

        # Fix: Provide x_o with shape (1, dim_cond, 1) or ensure broadcasting works
        # If pipeline expands dims, we want final to be broadcastable against (batch, dim, 1)
        # If we provide (1, dim, 1), _expand_dims keeps it (ndim=3).
        # Broadcast (1, dim, 1) vs (10, dim, 1) -> OK.
        x_o = jax.random.normal(jax.random.PRNGKey(2), (1, dim_cond, 1))

        sample = pipeline.sample(
            jax.random.PRNGKey(1),
            x_o=x_o,
            nsamples=10,
            use_ema=False,
        )
        assert sample.shape == (10, dim_obs, 1)

        custom_scheduler = EDMScheduler(sigma_min=0.1, sigma_max=20.0)
        sample = pipeline.sample(
            jax.random.PRNGKey(1),
            x_o=x_o,
            nsamples=10,
            use_ema=False,
            solver_scheduler=custom_scheduler,
        )
        assert sample.shape == (10, dim_obs, 1)


def get_dataset_joint():
    joint_data = np.concatenate([theta, cond], axis=1)  # (100, 4, 1)
    ds = grain.MapDataset.source(joint_data)
    ds = ds.shuffle(42).repeat().to_iter_dataset().batch(10)
    return ds


@pytest.mark.parametrize("sde_type", ["EDM", "VE", "VP"])
def test_joint_diffusion_sde_types(sde_type):
    pipeline_cls = JointDiffusionPipeline
    train_dataset = get_dataset_joint()
    val_dataset = get_dataset_joint()

    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        training_config = pipeline_cls.get_default_training_config(sde=sde_type)
        training_config["checkpoint_dir"] = model_dir
        training_config["val_every"] = 1

        model_test = Simformer(params_simf)

        pipeline = pipeline_cls(
            model_test,
            train_dataset,
            val_dataset,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            ch_obs=1,
            sde=sde_type,
            training_config=training_config,
            condition_mask_kind="structured",
        )

        assert pipeline.sde == sde_type
        if sde_type == "EDM":
            assert isinstance(pipeline.path.scheduler, EDMScheduler)
        elif sde_type == "VE":
            assert isinstance(pipeline.path.scheduler, VEEdmScheduler)
        if sde_type == "VP":
            assert isinstance(pipeline.path.scheduler, VPEdmScheduler)

        pipeline.ema_model = pipeline.model
        pipeline._wrap_model()

        # Fix: Provide x_o with shape (1, dim, 1)
        x_o = jax.random.normal(jax.random.PRNGKey(2), (1, dim_cond, 1))

        sample = pipeline.sample(
            jax.random.PRNGKey(1),
            x_o=x_o,
            nsamples=10,
            use_ema=False,
        )
        assert sample.shape == (10, dim_obs, 1)

        custom_scheduler = EDMScheduler(sigma_min=0.1, sigma_max=20.0)
        sample = pipeline.sample(
            jax.random.PRNGKey(1),
            x_o=x_o,
            nsamples=10,
            use_ema=False,
            solver_scheduler=custom_scheduler,
        )
        assert sample.shape == (10, dim_obs, 1)
