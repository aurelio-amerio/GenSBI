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
from gensbi.flow_matching.solver import FMODESolver
from gensbi.core.sde_solver import SDESolver
from gensbi.diffusion.solver import EDMSolver, SMSDESolver, SMODESolver
from gensbi.diffusion.path import EDMPath
from gensbi.diffusion.path.sm_path import SMPath
from gensbi.utils.model_wrapping import ModelWrapper


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
        path = method.build_path({}, event_shape=(DIM, CH))
        assert isinstance(path, AffineProbPath)

    def test_build_loss(self, method):
        from gensbi.flow_matching.loss import FMLoss

        path = method.build_path({}, event_shape=(DIM, CH))
        loss = method.build_loss(path)
        assert isinstance(loss, FMLoss)

    def test_prepare_batch_shapes(self, method, key, x_1):
        path = method.build_path({}, event_shape=(DIM, CH))
        x_0, x_1_out, t = method.prepare_batch(key, x_1, path)
        assert x_0.shape == SHAPE
        assert x_1_out.shape == SHAPE
        assert t.shape == (BATCH_SIZE,)
        # t should be in [0, 1)
        assert jnp.all(t >= 0.0)
        assert jnp.all(t < 1.0)

    def test_get_default_solver(self, method):
        cls, kwargs = method.get_default_solver()
        assert cls is FMODESolver
        assert kwargs == {}

    def test_sample_init(self, method, key):
        path = method.build_path({}, event_shape=(DIM, CH))
        x = method.sample_init(key, BATCH_SIZE)
        assert x.shape == SHAPE

    def test_extra_training_config_empty(self, method):
        assert method.get_extra_training_config() == {}

    def test_build_sampler_fn_with_custom_time_grid(self, method, dummy_model):
        """Custom time_grid triggers return_intermediates=True (L197)."""
        path = method.build_path({}, event_shape=(DIM, CH))
        wrapped = ModelWrapper(dummy_model)
        time_grid = jnp.array([0.0, 0.5, 1.0])
        sampler_fn = method.build_sampler_fn(
            wrapped, path, model_extras={},
            time_grid=time_grid,
        )
        key = jax.random.PRNGKey(99)
        x_init = jax.random.normal(key, SHAPE)
        result = sampler_fn(key, x_init)
        assert result is not None

    def test_build_sampler_fn_with_sde_solver(self, method, dummy_model):
        """SDE solver triggers pass_key branch (L209-210)."""
        from unittest.mock import MagicMock
        path = method.build_path({}, event_shape=(DIM, CH))

        mock_solver = MagicMock(spec=SDESolver)
        mock_sampler = MagicMock(return_value=jnp.zeros(SHAPE))
        mock_solver.get_sampler.return_value = mock_sampler

        original_build_solver = method.build_solver
        method.build_solver = lambda *a, **kw: mock_solver

        try:
            sampler_fn = method.build_sampler_fn(
                dummy_model, path, model_extras={},
            )
            key = jax.random.PRNGKey(99)
            x_init = jax.random.normal(key, SHAPE)
            result = sampler_fn(key, x_init)
            assert mock_sampler.called
            assert len(mock_sampler.call_args[0]) == 2  # (x_init, key_sampler)
            assert "model_extras" in mock_sampler.call_args[1]  # model_extras passed as kwarg
        finally:
            method.build_solver = original_build_solver

    def test_build_solver_default_fallback(self, method, dummy_model):
        """build_solver with solver=None falls back to get_default_solver."""
        wrapped = ModelWrapper(dummy_model)
        solver = method.build_solver(wrapped, path=None, solver=None)
        assert isinstance(solver, FMODESolver)


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
        path = method.build_path(config, event_shape=(DIM, CH))
        assert isinstance(path, EDMPath)

    def test_build_loss(self, method):
        from gensbi.diffusion.loss import EDMLoss

        config = method.get_extra_training_config()
        path = method.build_path(config, event_shape=(DIM, CH))
        loss = method.build_loss(path)
        assert isinstance(loss, EDMLoss)

    def test_prepare_batch_shapes(self, method, key, x_1):
        config = method.get_extra_training_config()
        path = method.build_path(config, event_shape=(DIM, CH))
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
        path = method.build_path(config, event_shape=(DIM, CH))
        x = method.sample_init(key, BATCH_SIZE)
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
        path = method.build_path(config, event_shape=(DIM, CH))
        assert isinstance(path, SMPath)

    def test_build_loss(self, method):
        from gensbi.diffusion.loss import SMLoss

        config = method.get_extra_training_config()
        path = method.build_path(config, event_shape=(DIM, CH))
        loss = method.build_loss(path)
        assert isinstance(loss, SMLoss)

    def test_prepare_batch_shapes(self, method, key, x_1):
        config = method.get_extra_training_config()
        path = method.build_path(config, event_shape=(DIM, CH))
        x_0, x_1_out, t = method.prepare_batch(key, x_1, path)
        assert x_0.shape == SHAPE
        assert x_1_out.shape == SHAPE
        assert t.shape == (BATCH_SIZE, 1, 1)

    def test_get_default_solver(self, method):
        cls, kwargs = method.get_default_solver()
        assert cls is SMSDESolver
        assert kwargs == {}

    def test_sample_init(self, method, key):
        config = method.get_extra_training_config()
        path = method.build_path(config, event_shape=(DIM, CH))
        x = method.sample_init(key, BATCH_SIZE)
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
        path = method.build_path({}, event_shape=(DIM, CH))
        loss_obj = method.build_loss(path)
        batch = method.prepare_batch(key, x_1, path)
        # ContinuousFMLoss.__call__(vf, batch, args=None, **kwargs)
        loss = loss_obj(dummy_model, batch)
        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_edm_loss_evaluates(self, key, x_1, dummy_model):
        method = DiffusionEDMMethod(sde="EDM")
        config = method.get_extra_training_config()
        path = method.build_path(config, event_shape=(DIM, CH))
        loss_obj = method.build_loss(path)
        batch = method.prepare_batch(key, x_1, path)
        loss = loss_obj(dummy_model, batch)
        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_sm_loss_evaluates(self, key, x_1, dummy_model):
        method = ScoreMatchingMethod(sde_type="VP")
        config = method.get_extra_training_config()
        path = method.build_path(config, event_shape=(DIM, CH))
        loss_obj = method.build_loss(path)
        batch = method.prepare_batch(key, x_1, path)
        loss = loss_obj(dummy_model, batch)
        assert loss.shape == ()
        assert jnp.isfinite(loss)


# ---------------------------------------------------------------------------
# Method parameter forwarding
# ---------------------------------------------------------------------------


class TestMethodForwarding:
    """Verify that ``build_sampler_fn`` forwards the ``method`` parameter.

    For each generative method we mock the solver so that
    ``solver.get_sampler()`` records its kwargs. We then call
    ``build_sampler_fn(method=<custom>)`` and assert that the custom
    method reaches the solver, rather than being hardcoded.
    """

    def test_flow_matching_forwards_method(self, dummy_model):
        from unittest.mock import MagicMock

        method = FlowMatchingMethod()
        path = method.build_path({}, event_shape=(DIM, CH))
        wrapped = ModelWrapper(dummy_model)

        mock_solver = MagicMock(spec=FMODESolver)
        mock_sampler = MagicMock(return_value=jnp.zeros(SHAPE))
        mock_solver.get_sampler.return_value = mock_sampler

        original = method.build_solver
        method.build_solver = lambda *a, **kw: mock_solver

        try:
            # Default: should forward "Euler"
            method.build_sampler_fn(wrapped, path, model_extras={})
            call_kwargs = mock_solver.get_sampler.call_args[1]
            assert call_kwargs["method"] == "Euler"

            # Custom: should forward "Dopri5"
            method.build_sampler_fn(
                wrapped, path, model_extras={}, method="Dopri5",
            )
            call_kwargs = mock_solver.get_sampler.call_args[1]
            assert call_kwargs["method"] == "Dopri5"
        finally:
            method.build_solver = original

    def test_score_matching_pf_ode_forwards_method(self, dummy_model):
        from unittest.mock import MagicMock

        method = ScoreMatchingMethod(sde_type="VP")
        config = method.get_extra_training_config()
        path = method.build_path(config, event_shape=(DIM, CH))
        wrapped = ModelWrapper(dummy_model)

        # Mock an SMODESolver (PF-ODE branch)
        mock_solver = MagicMock(spec=SMODESolver)
        mock_sampler = MagicMock(return_value=jnp.zeros(SHAPE))
        mock_solver.get_sampler.return_value = mock_sampler

        original = method.build_solver
        method.build_solver = lambda *a, **kw: mock_solver

        try:
            # Default: should forward "Euler"
            method.build_sampler_fn(
                wrapped, path, model_extras={},
                solver=(SMODESolver, {}),
            )
            call_kwargs = mock_solver.get_sampler.call_args[1]
            assert call_kwargs["method"] == "Euler"

            # Custom: should forward "Dopri5"
            method.build_sampler_fn(
                wrapped, path, model_extras={},
                solver=(SMODESolver, {}), method="Dopri5",
            )
            call_kwargs = mock_solver.get_sampler.call_args[1]
            assert call_kwargs["method"] == "Dopri5"
        finally:
            method.build_solver = original

    def test_edm_forwards_method(self, dummy_model):
        from unittest.mock import MagicMock

        method = DiffusionEDMMethod(sde="EDM")
        config = method.get_extra_training_config()
        path = method.build_path(config, event_shape=(DIM, CH))
        wrapped = ModelWrapper(dummy_model)

        mock_solver = MagicMock(spec=EDMSolver)
        mock_sampler = MagicMock(return_value=jnp.zeros(SHAPE))
        mock_solver.get_sampler.return_value = mock_sampler

        original = method.build_solver
        method.build_solver = lambda *a, **kw: mock_solver

        try:
            # Default: should forward "Heun"
            method.build_sampler_fn(wrapped, path, model_extras={})
            call_kwargs = mock_solver.get_sampler.call_args[1]
            assert call_kwargs["method"] == "Heun"

            # Custom: should forward "Euler"
            method.build_sampler_fn(
                wrapped, path, model_extras={}, method="Euler",
            )
            call_kwargs = mock_solver.get_sampler.call_args[1]
            assert call_kwargs["method"] == "Euler"
        finally:
            method.build_solver = original

