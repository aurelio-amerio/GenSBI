import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
import pytest
from gensbi.flow_matching.solver.ode_solver import ODESolver
from gensbi.utils.model_wrapping import ModelWrapper

from flax import nnx
from numpyro import distributions as dist

import diffrax

from gensbi.utils.math import _expand_dims, _expand_time


class DummyModel(nnx.Module):
    def __call__(self, x, t, *args, **kwargs):
        x = _expand_dims(x)
        t = _expand_time(t)
        if t.ndim < 3:
            t = t[..., None]
        res = jnp.ones_like(x) * 3.0 * t**2
        return res


@pytest.fixture
def solver():
    dummy_model = DummyModel()
    dummy_wrapped_model = ModelWrapper(dummy_model)
    return ODESolver(velocity_model=dummy_wrapped_model)


def test_sample_shape(solver):
    x_init = jnp.ones((5, 2, 1))
    time_grid = jnp.array([0.0, 1.0])
    sol = solver.sample(
        time_grid=time_grid,
        x_init=x_init,
        method="Dopri5",
        step_size=0.1,
        return_intermediates=False,
    )
    assert sol.shape == x_init.shape

    sol = solver.sample(
        time_grid=time_grid,
        x_init=x_init,
        method=diffrax.Euler(),
        step_size=0.1,
        return_intermediates=False,
    )
    assert sol.shape == x_init.shape

    time_grid = jnp.linspace(0, 1, 10)
    sol = solver.sample(
        time_grid=time_grid,
        x_init=x_init,
        method="Dopri5",
        step_size=0.1,
        return_intermediates=True,
    )
    assert sol.shape == (10, *x_init.shape)


def test_unnorm_logprob_shape(solver):

    x_1 = jnp.ones((5, 2, 3))

    p0_cond = dist.Independent(
        dist.Normal(loc=jnp.zeros((2,3)), scale=jnp.ones((2,3))),
        reinterpreted_batch_ndims=2,
    )

    time_grid = jnp.array([1.0, 0.0])
    logp = solver.unnormalized_logprob(
        x_1=x_1,
        log_p0=p0_cond.log_prob,
        time_grid=time_grid,
        method="Dopri5",
        step_size=0.01,
        return_intermediates=False,
    )
    assert logp.shape == (1,x_1.shape[0])

    logp = solver.unnormalized_logprob(
        x_1=x_1,
        log_p0=p0_cond.log_prob,
        time_grid=time_grid,
        method=diffrax.Euler(),
        step_size=0.01,
        return_intermediates=False,
    )
    assert logp.shape == (1,x_1.shape[0])

    time_grid = jnp.linspace(1, 0, 10)
    logp = solver.unnormalized_logprob(
        x_1=x_1,
        log_p0=p0_cond.log_prob,
        time_grid=time_grid,
        method="Dopri5",
        step_size=0.01,
        return_intermediates=True,
    )
    assert logp.shape == (
        10,
        x_1.shape[0],
    )
