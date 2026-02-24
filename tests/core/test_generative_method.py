"""
Tests for the ``gensbi.core`` module — GenerativeMethod ABC and implementations.

Tests cover:
- ABC interface enforcement
- Path construction
- Loss creation and evaluation
- Batch preparation shapes
- Solver construction
- Initial sample generation
- Sampler function building
- Extra training config
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from gensbi.core import (
    GenerativeMethod,
    FlowMatchingMethod,
    DiffusionEDMMethod,
    ScoreMatchingMethod,
)
from gensbi.flow_matching.path import AffineProbPath
from gensbi.flow_matching.solver import ODESolver, BaseFmSDESolver
from gensbi.diffusion.solver import EDMSolver, SMSolver, SMPFSolver
from gensbi.diffusion.path import EDMPath
from gensbi.diffusion.path.sm_path import SMPath


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DIM = 4
CH = 1
BATCH_SIZE = 8
SHAPE = (BATCH_SIZE, DIM, CH)


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def x_1(key):
    return jax.random.normal(key, SHAPE)


@pytest.fixture
def dummy_model():
    """A trivial Flax model that returns its input (identity)."""

    class Identity(nnx.Module):
        def __call__(self, *args, **kwargs):
            # Return something the same shape as the obs input
            if "obs" in kwargs:
                return kwargs["obs"]
            return args[0]

    return Identity()


# ---------------------------------------------------------------------------
# ABC enforcement
# ---------------------------------------------------------------------------


class TestGenerativeMethodABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            GenerativeMethod()

    def test_subclass_must_implement_all(self):
        class Incomplete(GenerativeMethod):
            pass

        with pytest.raises(TypeError):
            Incomplete()


# ---------------------------------------------------------------------------
# FlowMatchingMethod
# ---------------------------------------------------------------------------


class TestFlowMatchingMethod:
    @pytest.fixture
    def method(self):
        return FlowMatchingMethod()

    def test_build_path(self, method):
        path = method.build_path({})
        assert isinstance(path, AffineProbPath)

    def test_build_loss(self, method):
        from gensbi.flow_matching.loss import FMLoss

        path = method.build_path({})
        loss = method.build_loss(path)
        assert isinstance(loss, FMLoss)

    def test_prepare_batch_shapes(self, method, key, x_1):
        path = method.build_path({})
        x_0, x_1_out, t = method.prepare_batch(key, x_1, path)
        assert x_0.shape == SHAPE
        assert x_1_out.shape == SHAPE
        assert t.shape == (BATCH_SIZE,)
        # t should be in [0, 1)
        assert jnp.all(t >= 0.0)
        assert jnp.all(t < 1.0)

    def test_get_default_solver(self, method):
        cls, kwargs = method.get_default_solver()
        assert cls is ODESolver
        assert kwargs == {}

    def test_sample_init(self, method, key):
        path = method.build_path({})
        x = method.sample_init(key, SHAPE, path)
        assert x.shape == SHAPE

    def test_extra_training_config_empty(self, method):
        assert method.get_extra_training_config() == {}


# ---------------------------------------------------------------------------
# DiffusionEDMMethod
# ---------------------------------------------------------------------------


class TestDiffusionEDMMethod:
    @pytest.fixture(params=["EDM", "VP", "VE"])
    def method(self, request):
        return DiffusionEDMMethod(sde=request.param)

    def test_invalid_sde(self):
        with pytest.raises(ValueError, match="sde must be"):
            DiffusionEDMMethod(sde="INVALID")

    def test_build_path(self, method):
        config = method.get_extra_training_config()
        path = method.build_path(config)
        assert isinstance(path, EDMPath)

    def test_build_loss(self, method):
        from gensbi.diffusion.loss import EDMLoss

        config = method.get_extra_training_config()
        path = method.build_path(config)
        loss = method.build_loss(path)
        assert isinstance(loss, EDMLoss)

    def test_prepare_batch_shapes(self, method, key, x_1):
        config = method.get_extra_training_config()
        path = method.build_path(config)
        x_0, x_1_out, sigma = method.prepare_batch(key, x_1, path)
        assert x_0.shape == SHAPE
        assert x_1_out.shape == SHAPE
        assert sigma.shape == (BATCH_SIZE, 1, 1)

    def test_get_default_solver(self, method):
        cls, kwargs = method.get_default_solver()
        assert cls is EDMSolver
        assert kwargs == {}

    def test_sample_init(self, method, key):
        config = method.get_extra_training_config()
        path = method.build_path(config)
        x = method.sample_init(key, SHAPE, path)
        assert x.shape == SHAPE

    def test_extra_training_config(self, method):
        config = method.get_extra_training_config()
        assert isinstance(config, dict)
        assert len(config) > 0
        if method.sde == "EDM":
            assert "sigma_min" in config
            assert "sigma_max" in config
        elif method.sde == "VP":
            assert "beta_min" in config
            assert "beta_max" in config
        elif method.sde == "VE":
            assert "sigma_min" in config
            assert "sigma_max" in config


# ---------------------------------------------------------------------------
# ScoreMatchingMethod
# ---------------------------------------------------------------------------


class TestScoreMatchingMethod:
    @pytest.fixture(params=["VP", "VE"])
    def method(self, request):
        return ScoreMatchingMethod(sde_type=request.param)

    def test_invalid_sde_type(self):
        with pytest.raises(ValueError, match="sde_type must be"):
            ScoreMatchingMethod(sde_type="INVALID")

    def test_build_path(self, method):
        config = method.get_extra_training_config()
        path = method.build_path(config)
        assert isinstance(path, SMPath)

    def test_build_loss(self, method):
        from gensbi.diffusion.loss import SMLoss

        config = method.get_extra_training_config()
        path = method.build_path(config)
        loss = method.build_loss(path)
        assert isinstance(loss, SMLoss)

    def test_prepare_batch_shapes(self, method, key, x_1):
        config = method.get_extra_training_config()
        path = method.build_path(config)
        x_0, x_1_out, t = method.prepare_batch(key, x_1, path)
        assert x_0.shape == SHAPE
        assert x_1_out.shape == SHAPE
        assert t.shape == (BATCH_SIZE, 1, 1)

    def test_get_default_solver(self, method):
        cls, kwargs = method.get_default_solver()
        assert cls is SMSolver
        assert kwargs == {}

    def test_sample_init(self, method, key):
        config = method.get_extra_training_config()
        path = method.build_path(config)
        x = method.sample_init(key, SHAPE, path)
        assert x.shape == SHAPE

    def test_extra_training_config(self, method):
        config = method.get_extra_training_config()
        assert isinstance(config, dict)
        assert len(config) > 0
        if method.sde_type == "VP":
            assert "beta_min" in config
            assert "beta_max" in config
        elif method.sde_type == "VE":
            assert "sigma_min" in config
            assert "sigma_max" in config


# ---------------------------------------------------------------------------
# Loss evaluation (integration tests)
# ---------------------------------------------------------------------------


class TestLossEvaluation:
    """Test that loss objects produce valid scalar outputs."""

    def test_flow_loss_evaluates(self, key, x_1, dummy_model):
        method = FlowMatchingMethod()
        path = method.build_path({})
        loss_obj = method.build_loss(path)
        batch = method.prepare_batch(key, x_1, path)
        # ContinuousFMLoss.__call__(vf, batch, args=None, **kwargs)
        loss = loss_obj(dummy_model, batch)
        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_edm_loss_evaluates(self, key, x_1, dummy_model):
        method = DiffusionEDMMethod(sde="EDM")
        config = method.get_extra_training_config()
        path = method.build_path(config)
        loss_obj = method.build_loss(path)
        batch = method.prepare_batch(key, x_1, path)
        loss = loss_obj(dummy_model, batch)
        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_sm_loss_evaluates(self, key, x_1, dummy_model):
        method = ScoreMatchingMethod(sde_type="VP")
        config = method.get_extra_training_config()
        path = method.build_path(config)
        loss_obj = method.build_loss(path)
        batch = method.prepare_batch(key, x_1, path)
        loss = loss_obj(dummy_model, batch)
        assert loss.shape == ()
        assert jnp.isfinite(loss)
