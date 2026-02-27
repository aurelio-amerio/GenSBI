import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx

import pytest
import tempfile

import grain
import numpy as np

from gensbi.recipes.joint_pipeline import sample_condition_mask
from gensbi.recipes.pipeline import _get_batch_sampler

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from mock_models import MockJointModel, MockConditionalModel, MockUnconditionalModel

from gensbi.recipes import (
    ConditionalPipeline,
    JointPipeline,
    UnconditionalPipeline,
)
from gensbi.core import FlowMatchingMethod, DiffusionEDMMethod, ScoreMatchingMethod

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


# --- Tests for JointPipeline validation errors ---


def test_joint_flow_dim_cond_zero():
    """Test that dim_cond=0 raises ValueError for JointPipeline."""
    with pytest.raises(ValueError, match="dim_cond=0"):
        JointPipeline(
            model=MockJointModel(),
            train_dataset=train_dataset_joint,
            val_dataset=val_dataset_joint,
            dim_obs=dim_joint,
            dim_cond=0,
            method=FlowMatchingMethod(),
        )


def test_joint_flow_invalid_condition_mask_kind():
    """Test that invalid condition_mask_kind raises ValueError."""
    with pytest.raises(ValueError, match="condition_mask_kind"):
        JointPipeline(
            model=MockJointModel(),
            train_dataset=train_dataset_joint,
            val_dataset=val_dataset_joint,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            method=FlowMatchingMethod(),
            condition_mask_kind="invalid_kind",
        )


def test_joint_diffusion_dim_cond_zero():
    """Test that dim_cond=0 raises ValueError for JointPipeline with DiffusionEDMMethod."""
    with pytest.raises(ValueError, match="dim_cond=0"):
        JointPipeline(
            model=MockJointModel(),
            train_dataset=train_dataset_joint,
            val_dataset=val_dataset_joint,
            dim_obs=dim_joint,
            dim_cond=0,
            method=DiffusionEDMMethod(),
        )


def test_joint_diffusion_invalid_condition_mask_kind():
    """Test that invalid condition_mask_kind raises ValueError."""
    with pytest.raises(ValueError, match="condition_mask_kind"):
        JointPipeline(
            model=MockJointModel(),
            train_dataset=train_dataset_joint,
            val_dataset=val_dataset_joint,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            method=DiffusionEDMMethod(),
            condition_mask_kind="invalid_kind",
        )


# --- Tests for pipeline.py edge cases ---


def test_update_training_config():
    """Test update_training_config correctly updates and recalculates min_scale."""
    pipeline = JointPipeline(
        model=MockJointModel(),
        train_dataset=train_dataset_joint,
        val_dataset=val_dataset_joint,
        dim_obs=dim_obs,
        dim_cond=dim_cond,
        method=FlowMatchingMethod(),
    )
    new_config = {"max_lr": 1e-3, "min_lr": 1e-5}
    pipeline.update_training_config(new_config)
    assert pipeline.training_config["max_lr"] == 1e-3
    assert pipeline.training_config["min_lr"] == 1e-5
    expected_min_scale = 1e-5 / 1e-3
    assert np.isclose(pipeline.training_config["min_scale"], expected_min_scale)


def test_update_training_config_zero_max_lr():
    """Test update_training_config handles max_lr=0 without division error."""
    pipeline = JointPipeline(
        model=MockJointModel(),
        train_dataset=train_dataset_joint,
        val_dataset=val_dataset_joint,
        dim_obs=dim_obs,
        dim_cond=dim_cond,
        method=FlowMatchingMethod(),
    )
    new_config = {"max_lr": 0.0, "min_lr": 0.0}
    pipeline.update_training_config(new_config)
    assert pipeline.training_config["min_scale"] == 0.0


def test_batch_sampler_with_progress_bars():
    """Test sample_batched with show_progress_bars=True to cover progress bar branch."""
    import tempfile

    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        training_config = JointPipeline.get_default_training_config()
        training_config["checkpoint_dir"] = model_dir
        training_config["val_every"] = 1

        pipeline = JointPipeline(
            model=MockJointModel(),
            train_dataset=train_dataset_joint,
            val_dataset=val_dataset_joint,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            method=FlowMatchingMethod(),
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
        training_config = JointPipeline.get_default_training_config()
        training_config["checkpoint_dir"] = model_dir
        training_config["val_every"] = 1
        training_config["multistep"] = 2  # cover multistep > 1 branch

        pipeline = JointPipeline(
            model=MockJointModel(),
            train_dataset=train_dataset_joint,
            val_dataset=val_dataset_joint,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            method=FlowMatchingMethod(),
            ch_obs=2,
            training_config=training_config,
        )

        pipeline.train(nnx.Rngs(0), nsteps=4, save_model=False)


# --- Tests for ConditionalPipeline edge cases ---


def test_conditional_sm_ve_sde():
    """Test ConditionalPipeline with ScoreMatchingMethod VE SDE type."""
    pipeline = ConditionalPipeline(
        model=MockConditionalModel(),
        train_dataset=train_dataset_cond,
        val_dataset=val_dataset_cond,
        dim_obs=dim_obs,
        dim_cond=dim_cond,
        method=ScoreMatchingMethod(sde_type="VE"),
    )
    assert isinstance(pipeline, ConditionalPipeline)


def test_conditional_sm_invalid_sde():
    """Test ScoreMatchingMethod with invalid SDE type raises ValueError."""
    with pytest.raises(ValueError, match="sde_type must be one of"):
        ScoreMatchingMethod(sde_type="INVALID")


def test_conditional_flow_invalid_id_embedding_obs():
    """Test ConditionalPipeline with invalid obs id embedding strategy."""
    with pytest.raises(ValueError, match="Unknown id embedding strategy"):
        ConditionalPipeline(
            model=MockConditionalModel(),
            train_dataset=train_dataset_cond,
            val_dataset=val_dataset_cond,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            method=FlowMatchingMethod(),
            id_embedding_strategy=("invalid", "absolute"),
        )


def test_conditional_flow_invalid_id_embedding_cond():
    """Test ConditionalPipeline with invalid cond id embedding strategy."""
    with pytest.raises(ValueError, match="Unknown id embedding strategy"):
        ConditionalPipeline(
            model=MockConditionalModel(),
            train_dataset=train_dataset_cond,
            val_dataset=val_dataset_cond,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            method=FlowMatchingMethod(),
            id_embedding_strategy=("absolute", "invalid"),
        )


# --- Tests for JointPipeline SDE branches ---


def test_joint_diffusion_invalid_sde():
    """Test DiffusionEDMMethod with invalid SDE type raises ValueError."""
    with pytest.raises(ValueError, match="sde must be one of"):
        DiffusionEDMMethod(sde="INVALID")


# --- Tests for deprecated pipeline stubs ---


def test_deprecated_conditional_pipelines():
    """Test that deprecated conditional pipeline classes raise RuntimeError."""
    from gensbi.recipes import (
        ConditionalFlowPipeline,
        ConditionalDiffusionPipeline,
        ConditionalSMPipeline,
    )

    with pytest.raises(RuntimeError, match="has been removed"):
        ConditionalFlowPipeline()

    with pytest.raises(RuntimeError, match="has been removed"):
        ConditionalDiffusionPipeline()

    with pytest.raises(RuntimeError, match="has been removed"):
        ConditionalSMPipeline()


def test_deprecated_joint_pipelines():
    """Test that deprecated joint pipeline classes raise RuntimeError."""
    from gensbi.recipes import (
        JointFlowPipeline,
        JointDiffusionPipeline,
        JointSMPipeline,
    )

    with pytest.raises(RuntimeError, match="has been removed"):
        JointFlowPipeline()

    with pytest.raises(RuntimeError, match="has been removed"):
        JointDiffusionPipeline()

    with pytest.raises(RuntimeError, match="has been removed"):
        JointSMPipeline()


def test_deprecated_unconditional_pipelines():
    """Test that deprecated unconditional pipeline classes raise RuntimeError."""
    from gensbi.recipes import (
        UnconditionalFlowPipeline,
        UnconditionalDiffusionPipeline,
        UnconditionalSMPipeline,
    )

    with pytest.raises(RuntimeError, match="has been removed"):
        UnconditionalFlowPipeline()

    with pytest.raises(RuntimeError, match="has been removed"):
        UnconditionalDiffusionPipeline()

    with pytest.raises(RuntimeError, match="has been removed"):
        UnconditionalSMPipeline()


def test_deprecated_pipelines_get_default_training_config():
    """Test that get_default_training_config on deprecated classes raises RuntimeError."""
    from gensbi.recipes import (
        ConditionalFlowPipeline,
        JointFlowPipeline,
        UnconditionalFlowPipeline,
    )

    with pytest.raises(RuntimeError, match="has been removed"):
        ConditionalFlowPipeline.get_default_training_config()

    with pytest.raises(RuntimeError, match="has been removed"):
        JointFlowPipeline.get_default_training_config()

    with pytest.raises(RuntimeError, match="has been removed"):
        UnconditionalFlowPipeline.get_default_training_config()


# ---------------------------------------------------------------------------
# Tests for pipeline.py uncovered branches
# ---------------------------------------------------------------------------


def test_init_with_model_none():
    """Pipeline __init__ with model=None sets ema_model=None (L232-233)."""
    pipeline = ConditionalPipeline(
        model=None,
        train_dataset=train_dataset_cond,
        val_dataset=val_dataset_cond,
        dim_obs=dim_obs,
        dim_cond=dim_cond,
        method=FlowMatchingMethod(),
    )
    assert pipeline.model is None
    assert pipeline.ema_model is None


def test_init_with_default_training_config():
    """Pipeline __init__ without training_config uses defaults (L218-219)."""
    pipeline = ConditionalPipeline(
        model=MockConditionalModel(),
        train_dataset=train_dataset_cond,
        val_dataset=val_dataset_cond,
        dim_obs=dim_obs,
        dim_cond=dim_cond,
        method=FlowMatchingMethod(),
        # training_config omitted — defaults used
    )
    assert pipeline.training_config is not None
    assert "nsteps" in pipeline.training_config
    assert "max_lr" in pipeline.training_config


def test_get_batch_sampler_no_progress_bars():
    """_get_batch_sampler with show_progress_bars=False (L137)."""
    def mock_sampler_fn(key, ncond):
        return jnp.ones((ncond, 1))

    ncond = 5
    chunk_size = 10
    n_samples = 30

    batched_sampler = _get_batch_sampler(
        mock_sampler_fn, ncond, chunk_size, show_progress_bars=False
    )
    keys = jax.random.split(jax.random.PRNGKey(0), n_samples)
    result = batched_sampler(keys)
    assert result.shape == (n_samples, ncond, 1)


def test_save_and_restore_model():
    """save_model + restore_model roundtrip (L466-562)."""
    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        training_config = UnconditionalPipeline.get_default_training_config()
        training_config["checkpoint_dir"] = model_dir
        training_config["val_every"] = 1

        pipeline = UnconditionalPipeline(
            model=MockUnconditionalModel(),
            train_dataset=train_dataset_joint,
            val_dataset=val_dataset_joint,
            dim_obs=dim_joint,
            method=FlowMatchingMethod(),
            ch_obs=2,
            training_config=training_config,
        )
        pipeline.train(nnx.Rngs(0), nsteps=2, save_model=True)

        # Restore from the saved checkpoint (uses experiment_id from training_config)
        pipeline.restore_model()
        assert pipeline.model is not None
        assert pipeline.ema_model is not None


def test_restore_best_state():
    """_restore_best_state merges states back into model and ema (L581-583)."""
    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        training_config = UnconditionalPipeline.get_default_training_config()
        training_config["checkpoint_dir"] = model_dir
        training_config["val_every"] = 1

        pipeline = UnconditionalPipeline(
            model=MockUnconditionalModel(),
            train_dataset=train_dataset_joint,
            val_dataset=val_dataset_joint,
            dim_obs=dim_joint,
            method=FlowMatchingMethod(),
            ch_obs=2,
            training_config=training_config,
        )

        best_state = nnx.state(pipeline.model)
        best_state_ema = nnx.state(pipeline.ema_model)
        pipeline._restore_best_state(best_state, best_state_ema)
        # Model should still be functional after state restoration
        assert pipeline.model is not None
        assert pipeline.ema_model is not None


def test_run_validation_counter_and_best():
    """_run_validation updates counter and best state (L634-645)."""
    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        training_config = UnconditionalPipeline.get_default_training_config()
        training_config["checkpoint_dir"] = model_dir
        training_config["val_every"] = 1

        pipeline = UnconditionalPipeline(
            model=MockUnconditionalModel(),
            train_dataset=train_dataset_joint,
            val_dataset=val_dataset_joint,
            dim_obs=dim_joint,
            method=FlowMatchingMethod(),
            ch_obs=2,
            training_config=training_config,
        )

        loss_fn = pipeline.get_loss_fn()
        val_step = pipeline.get_val_step_fn(loss_fn)
        rng_val = jax.random.PRNGKey(42)
        batch_val = next(pipeline.val_dataset_iter)

        best_state = nnx.state(pipeline.model)
        best_state_ema = nnx.state(pipeline.ema_model)

        # Case 1: l_val < min_val → best state updated
        l_val, ratio, min_val, best_state, best_state_ema, counter = (
            pipeline._run_validation(
                val_step, batch_val, rng_val,
                min_val=1e10,  # large initial min_val
                best_state=best_state,
                best_state_ema=best_state_ema,
                counter=0,
                val_error_ratio=1.3,
                loss_array=[],
                val_loss_array=[],
                l_train=0.1,
            )
        )
        assert min_val < 1e10  # should have been updated
        assert counter == 0  # ratio < 1.3

        # Case 2: ratio > val_error_ratio → counter incremented
        l_val2, ratio2, min_val2, best_state2, best_state_ema2, counter2 = (
            pipeline._run_validation(
                val_step, batch_val, rng_val,
                min_val=1e-20,  # tiny min_val forces large ratio
                best_state=best_state,
                best_state_ema=best_state_ema,
                counter=5,
                val_error_ratio=1.3,
                loss_array=[],
                val_loss_array=[],
                l_train=0.1,
            )
        )
        assert counter2 == 6  # counter should have incremented


def test_train_uses_nsteps_from_config():
    """train(nsteps=None) uses nsteps from training_config (L693-694)."""
    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        training_config = UnconditionalPipeline.get_default_training_config()
        training_config["checkpoint_dir"] = model_dir
        training_config["val_every"] = 1
        training_config["nsteps"] = 100  # must be large enough for cosine scheduler

        pipeline = UnconditionalPipeline(
            model=MockUnconditionalModel(),
            train_dataset=train_dataset_joint,
            val_dataset=val_dataset_joint,
            dim_obs=dim_joint,
            method=FlowMatchingMethod(),
            ch_obs=2,
            training_config=training_config,
        )
        # nsteps=None should use config["nsteps"] = 2
        loss_array, val_loss_array = pipeline.train(
            nnx.Rngs(0), nsteps=None, save_model=False
        )
        assert isinstance(loss_array, list)


def test_train_early_stopping():
    """Early stopping triggers _restore_best_state (L706-709)."""
    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        training_config = UnconditionalPipeline.get_default_training_config()
        training_config["checkpoint_dir"] = model_dir
        training_config["val_every"] = 1
        training_config["early_stopping"] = True

        pipeline = UnconditionalPipeline(
            model=MockUnconditionalModel(),
            train_dataset=train_dataset_joint,
            val_dataset=val_dataset_joint,
            dim_obs=dim_joint,
            method=FlowMatchingMethod(),
            ch_obs=2,
            training_config=training_config,
        )
        # Train enough steps that validation happens and early stopping may trigger
        # With val_every=1, validation runs every step
        loss_array, val_loss_array = pipeline.train(
            nnx.Rngs(0), nsteps=50, save_model=False
        )
        # The model should train (at least partially) with early stopping enabled
        assert isinstance(loss_array, list)


def test_train_no_save():
    """Train with save_model=False skips save_model call (L746)."""
    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        training_config = UnconditionalPipeline.get_default_training_config()
        training_config["checkpoint_dir"] = model_dir
        training_config["val_every"] = 1

        pipeline = UnconditionalPipeline(
            model=MockUnconditionalModel(),
            train_dataset=train_dataset_joint,
            val_dataset=val_dataset_joint,
            dim_obs=dim_joint,
            method=FlowMatchingMethod(),
            ch_obs=2,
            training_config=training_config,
        )
        loss_array, val_loss_array = pipeline.train(
            nnx.Rngs(0), nsteps=2, save_model=False
        )
        # No checkpoint files should exist
        import glob
        checkpoints = glob.glob(os.path.join(model_dir, "*"))
        assert len(checkpoints) == 0

