import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx

import pytest

import grain
import numpy as np

from gensbi.recipes.joint_pipeline import sample_condition_mask

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from mock_models import MockJointModel, MockConditionalModel

from gensbi.recipes import (
    JointFlowPipeline,
    JointDiffusionPipeline,
    ConditionalFlowPipeline,
)

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
    )


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


# --- Tests for sample_condition_mask edge cases ---


def test_condition_mask_likelihood():
    """Test the 'likelihood' kind for sample_condition_mask."""
    mask = sample_condition_mask(
        jax.random.PRNGKey(0),
        num_samples=5,
        theta_dim=dim_obs,
        x_dim=dim_cond,
        kind="likelihood",
    )
    assert mask.shape == (5, dim_joint, 1)
    # likelihood: theta is True, x is False
    assert jnp.all(mask[:, :dim_obs, :] == True)
    assert jnp.all(mask[:, dim_obs:, :] == False)


def test_condition_mask_joint():
    """Test the 'joint' kind for sample_condition_mask."""
    mask = sample_condition_mask(
        jax.random.PRNGKey(0),
        num_samples=5,
        theta_dim=dim_obs,
        x_dim=dim_cond,
        kind="joint",
    )
    assert mask.shape == (5, dim_joint, 1)
    # joint: all False
    assert jnp.all(mask == False)


def test_condition_mask_invalid_kind():
    """Test that an invalid kind raises ValueError."""
    with pytest.raises(ValueError, match="Unknown kind"):
        sample_condition_mask(
            jax.random.PRNGKey(0),
            num_samples=5,
            theta_dim=dim_obs,
            x_dim=dim_cond,
            kind="invalid_kind",
        )


# --- Tests for JointFlowPipeline validation errors ---


def test_joint_flow_dim_cond_zero():
    """Test that dim_cond=0 raises ValueError for JointFlowPipeline."""
    with pytest.raises(ValueError, match="dim_cond=0"):
        JointFlowPipeline(
            model=MockJointModel(),
            train_dataset=train_dataset_joint,
            val_dataset=val_dataset_joint,
            dim_obs=dim_joint,
            dim_cond=0,
        )


def test_joint_flow_invalid_condition_mask_kind():
    """Test that invalid condition_mask_kind raises ValueError."""
    with pytest.raises(ValueError, match="condition_mask_kind"):
        JointFlowPipeline(
            model=MockJointModel(),
            train_dataset=train_dataset_joint,
            val_dataset=val_dataset_joint,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            condition_mask_kind="invalid_kind",
        )


def test_joint_diffusion_dim_cond_zero():
    """Test that dim_cond=0 raises ValueError for JointDiffusionPipeline."""
    with pytest.raises(ValueError, match="dim_cond=0"):
        JointDiffusionPipeline(
            model=MockJointModel(),
            train_dataset=train_dataset_joint,
            val_dataset=val_dataset_joint,
            dim_obs=dim_joint,
            dim_cond=0,
        )


def test_joint_diffusion_invalid_condition_mask_kind():
    """Test that invalid condition_mask_kind raises ValueError."""
    with pytest.raises(ValueError, match="condition_mask_kind"):
        JointDiffusionPipeline(
            model=MockJointModel(),
            train_dataset=train_dataset_joint,
            val_dataset=val_dataset_joint,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            condition_mask_kind="invalid_kind",
        )


# --- Tests for pipeline.py edge cases ---


def test_update_training_config():
    """Test update_training_config correctly updates and recalculates min_scale."""
    pipeline = JointFlowPipeline(
        model=MockJointModel(),
        train_dataset=train_dataset_joint,
        val_dataset=val_dataset_joint,
        dim_obs=dim_obs,
        dim_cond=dim_cond,
    )
    new_config = {"max_lr": 1e-3, "min_lr": 1e-5}
    pipeline.update_training_config(new_config)
    assert pipeline.training_config["max_lr"] == 1e-3
    assert pipeline.training_config["min_lr"] == 1e-5
    expected_min_scale = 1e-5 / 1e-3
    assert np.isclose(pipeline.training_config["min_scale"], expected_min_scale)


def test_update_training_config_zero_max_lr():
    """Test update_training_config handles max_lr=0 without division error."""
    pipeline = JointFlowPipeline(
        model=MockJointModel(),
        train_dataset=train_dataset_joint,
        val_dataset=val_dataset_joint,
        dim_obs=dim_obs,
        dim_cond=dim_cond,
    )
    new_config = {"max_lr": 0.0, "min_lr": 0.0}
    pipeline.update_training_config(new_config)
    assert pipeline.training_config["min_scale"] == 0.0


def test_batch_sampler_with_progress_bars():
    """Test sample_batched with show_progress_bars=True to cover progress bar branch."""
    import tempfile

    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        training_config = JointFlowPipeline.get_default_training_config()
        training_config["checkpoint_dir"] = model_dir
        training_config["val_every"] = 1

        pipeline = JointFlowPipeline(
            model=MockJointModel(),
            train_dataset=train_dataset_joint,
            val_dataset=val_dataset_joint,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            ch_obs=2,
            training_config=training_config,
        )

        pipeline.train(nnx.Rngs(0), nsteps=2, save_model=True)
        pipeline._wrap_model()

        cond_single = jnp.zeros((1, dim_cond, 2))
        # Use show_progress_bars=True to hit the tqdm branch
        sample = pipeline.sample_batched(
            jax.random.PRNGKey(1),
            cond_single,
            nsamples=4,
            chunk_size=2,
            show_progress_bars=True,
        )
        assert sample.shape == (4, 1, dim_obs, 2)


def test_multistep_optimizer():
    """Test pipeline with multistep > 1 to cover the multistep optimizer branch."""
    import tempfile

    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        training_config = JointFlowPipeline.get_default_training_config()
        training_config["checkpoint_dir"] = model_dir
        training_config["val_every"] = 1
        training_config["multistep"] = 2  # cover multistep > 1 branch

        pipeline = JointFlowPipeline(
            model=MockJointModel(),
            train_dataset=train_dataset_joint,
            val_dataset=val_dataset_joint,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            ch_obs=2,
            training_config=training_config,
        )

        pipeline.train(nnx.Rngs(0), nsteps=4, save_model=False)


# --- Tests for conditional_pipeline.py SDE branches ---


def test_conditional_sm_ve_sde():
    """Test ConditionalSMPipeline with VE SDE type to cover VE branch."""
    from gensbi.recipes import ConditionalSMPipeline

    pipeline = ConditionalSMPipeline(
        model=MockConditionalModel(),
        train_dataset=train_dataset_cond,
        val_dataset=val_dataset_cond,
        dim_obs=dim_obs,
        dim_cond=dim_cond,
        sde_type="VE",
    )
    assert isinstance(pipeline, ConditionalSMPipeline)
    assert pipeline.sde_type == "VE"


def test_conditional_sm_invalid_sde():
    """Test ConditionalSMPipeline with invalid SDE type raises ValueError."""
    from gensbi.recipes import ConditionalSMPipeline

    with pytest.raises(ValueError, match="sde_type"):
        ConditionalSMPipeline(
            model=MockConditionalModel(),
            train_dataset=train_dataset_cond,
            val_dataset=val_dataset_cond,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            sde_type="INVALID",
        )


def test_conditional_sm_default_config_ve():
    """Test get_default_training_config with VE SDE type."""
    from gensbi.recipes import ConditionalSMPipeline

    config = ConditionalSMPipeline.get_default_training_config(sde_type="VE")
    assert "sigma_min" in config
    assert "sigma_max" in config


def test_conditional_diffusion_invalid_sde():
    """Test ConditionalDiffusionPipeline with invalid SDE type raises ValueError."""
    from gensbi.recipes import ConditionalDiffusionPipeline

    with pytest.raises(ValueError, match="Unknown sde type"):
        ConditionalDiffusionPipeline(
            model=MockConditionalModel(),
            train_dataset=train_dataset_cond,
            val_dataset=val_dataset_cond,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            sde="INVALID",
        )


def test_conditional_diffusion_default_config_ve():
    """Test get_default_training_config with VE and VP SDE types."""
    from gensbi.recipes import ConditionalDiffusionPipeline

    config_ve = ConditionalDiffusionPipeline.get_default_training_config(sde="VE")
    assert "sigma_min" in config_ve
    config_vp = ConditionalDiffusionPipeline.get_default_training_config(sde="VP")
    assert "beta_min" in config_vp


def test_conditional_flow_invalid_id_embedding_obs():
    """Test ConditionalFlowPipeline with invalid obs id embedding strategy."""
    with pytest.raises(ValueError, match="Unknown id embedding strategy"):
        ConditionalFlowPipeline(
            model=MockConditionalModel(),
            train_dataset=train_dataset_cond,
            val_dataset=val_dataset_cond,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            id_embedding_strategy=("invalid", "absolute"),
        )


def test_conditional_flow_invalid_id_embedding_cond():
    """Test ConditionalFlowPipeline with invalid cond id embedding strategy."""
    with pytest.raises(ValueError, match="Unknown id embedding strategy"):
        ConditionalFlowPipeline(
            model=MockConditionalModel(),
            train_dataset=train_dataset_cond,
            val_dataset=val_dataset_cond,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            id_embedding_strategy=("absolute", "invalid"),
        )


# --- Tests for joint_pipeline.py SDE branches ---


def test_joint_diffusion_invalid_sde():
    """Test JointDiffusionPipeline with invalid SDE type raises ValueError."""
    with pytest.raises(ValueError, match="Unknown sde type"):
        JointDiffusionPipeline(
            model=MockJointModel(),
            train_dataset=train_dataset_joint,
            val_dataset=val_dataset_joint,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            sde="INVALID",
        )


def test_joint_diffusion_default_config_ve():
    """Test JointDiffusionPipeline.get_default_training_config with VE and VP SDE types."""
    config_ve = JointDiffusionPipeline.get_default_training_config(sde="VE")
    assert "sigma_min" in config_ve
    config_vp = JointDiffusionPipeline.get_default_training_config(sde="VP")
    assert "beta_min" in config_vp
