import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx

import warnings

import pytest

import tempfile

import grain
import numpy as np

from gensbi.recipes import JointFlowPipeline, JointDiffusionPipeline, JointSMPipeline
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


@pytest.mark.parametrize(
    "pipeline_cls, condition_mask_kind",
    [
        (JointFlowPipeline, "structured"),
        (JointFlowPipeline, "posterior"),
        (JointDiffusionPipeline, "structured"),
        (JointDiffusionPipeline, "posterior"),
        (JointSMPipeline, "structured"),
        (JointSMPipeline, "posterior"),
    ],
)
def test_joint_pipeline_init(pipeline_cls, condition_mask_kind):
    pipeline = pipeline_cls(
        model=MockJointModel(),
        train_dataset=train_dataset_joint,
        val_dataset=val_dataset_joint,
        dim_obs=dim_obs,
        dim_cond=dim_cond,
        condition_mask_kind=condition_mask_kind,
    )
    assert isinstance(pipeline, pipeline_cls)


@pytest.mark.parametrize(
    "pipeline_cls",
    [
        JointFlowPipeline,
        JointDiffusionPipeline,
        JointSMPipeline,
    ],
)
def test_joint_pipeline_methods(pipeline_cls):
    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        training_config = pipeline_cls.get_default_training_config()
        training_config["checkpoint_dir"] = model_dir
        training_config["val_every"] = 1

        model = MockJointModel()

        pipeline = pipeline_cls(
            model=model,
            train_dataset=train_dataset_joint,
            val_dataset=val_dataset_joint,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            ch_obs=2,
            training_config=training_config,
        )

        batch_size = 3
        t = jnp.linspace(0, 1, batch_size)
        obs = jnp.ones((batch_size, dim_obs, 2))
        cond = jnp.ones((batch_size, dim_cond, 2))

        obs_ids = pipeline.obs_ids
        cond_ids = pipeline.cond_ids

        # Train 2 steps
        pipeline.train(nnx.Rngs(0), nsteps=2, save_model=True)
        pipeline._wrap_model()

        # Check inference shapes
        out = pipeline.model_wrapped(t, obs, obs_ids, cond, cond_ids)
        out_ema = pipeline.ema_model_wrapped(t, obs, obs_ids, cond, cond_ids)
        assert out.shape == (batch_size, dim_obs, 2)
        assert out_ema.shape == (batch_size, dim_obs, 2)

        # Restore formulation
        pipeline2 = pipeline_cls(
            model=model,
            train_dataset=train_dataset_joint,
            val_dataset=val_dataset_joint,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            ch_obs=2,
            training_config=training_config,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline2.restore_model()

        # Check equality of restored model outputs vs fresh outputs
        out_restored = pipeline2.model_wrapped(t, obs, obs_ids, cond, cond_ids)
        assert jnp.allclose(out, out_restored), "Restored model output discrepancy"

        cond_single = jnp.zeros((1, dim_cond, 2))
        sample = pipeline.sample(
            jax.random.PRNGKey(1), cond_single, nsamples=32, use_ema=False
        )
        assert sample.shape == (32, dim_obs, 2)

        sample_ema = pipeline.sample(
            jax.random.PRNGKey(1), cond_single, nsamples=32, use_ema=True
        )
        assert sample_ema.shape == (32, dim_obs, 2)

        # Batch Sampling Test
        cond_single = jnp.zeros((1, dim_cond, 2))
        sample = pipeline.sample_batched(
            jax.random.PRNGKey(1),
            cond_single,
            nsamples=4,
            chunk_size=2,
            show_progress_bars=False,
        )
        assert sample.shape == (4, 1, dim_obs, 2)

        # get_sampler advanced args testing
        if isinstance(pipeline, JointFlowPipeline):
            from gensbi.flow_matching.solver import ZeroEnds

            solver = (
                ZeroEnds,
                {
                    "mu0": jnp.zeros((dim_obs, 2)),
                    "sigma0": jnp.ones((dim_obs, 2)),
                    "alpha": 1.0,
                },
            )
            time_grid = jnp.linspace(0, 1, 10)
            sampler = pipeline.get_sampler(
                cond_single,
                step_size=0.1,
                time_grid=time_grid,
                solver=solver,
                use_ema=False,
            )
            samples = sampler(jax.random.PRNGKey(2), nsamples=4)
            assert samples.shape == (len(time_grid), 4, dim_obs, 2)

        elif isinstance(pipeline, JointDiffusionPipeline):
            sampler = pipeline.get_sampler(
                cond_single, nsteps=10, return_intermediates=True, use_ema=False
            )
            samples = sampler(jax.random.PRNGKey(2), nsamples=4)
            assert samples.shape == (10, 4, dim_obs, 2)

        elif isinstance(pipeline, JointSMPipeline):
            from gensbi.diffusion.solver import SMPFSolver

            solver = (SMPFSolver, {})
            sampler = pipeline.get_sampler(
                cond_single,
                nsteps=10,
                return_intermediates=True,
                solver=solver,
                use_ema=False,
            )
            samples = sampler(jax.random.PRNGKey(2), nsamples=4)
            assert samples.shape == (11, 4, dim_obs, 2)

        # loss_fn custom execution to ensure coverage inside get_loss_fn
        loss_fn = pipeline.get_loss_fn()
        mock_batch = jnp.zeros((32, dim_joint, 2))
        loss = loss_fn(pipeline.model, mock_batch, key=jax.random.PRNGKey(1))
        assert loss.shape == ()
