import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
import jax
import pytest
from gensbi.flow_matching.solver.sde_solver import ZeroEnds, NonSingular
from gensbi.utils.model_wrapping import ModelWrapper

from flax import nnx
from numpyro import distributions as dist

import diffrax


class DummyModel(nnx.Module):
    def __call__(self, obs, t, *args, **kwargs):
        res = jnp.ones_like(obs) 
        return res
    
@pytest.mark.parametrize("solver_cls", [ZeroEnds, NonSingular])
def test_f_tilde(solver_cls):
    model = DummyModel()
    wrapped_model = ModelWrapper(model)

    solver = solver_cls(velocity_model=wrapped_model, mu0=jnp.zeros(2), sigma0=jnp.ones(2), alpha=0.5)

    f_tilde = solver.get_f_tilde()
    assert f_tilde is not None

    t = jnp.array([0.5, 0.6])
    x = jnp.ones((3, 2))
    args = None

    res = f_tilde(t, x, args)
    assert res.shape == x.shape, f"Expected shape {x.shape}, but got {res.shape}"


@pytest.mark.parametrize("solver_cls", [ZeroEnds, NonSingular])
def test_g_tilde(solver_cls):
    model = DummyModel()
    wrapped_model = ModelWrapper(model)

    solver = solver_cls(velocity_model=wrapped_model, mu0=jnp.zeros(2), sigma0=jnp.ones(2), alpha=0.5)

    g_tilde = solver.get_g_tilde()
    assert g_tilde is not None

    t = jnp.array([0.5])
    x = jnp.ones((3, 2))
    args = None

    res = g_tilde(t, x, args)
    assert res.shape == (x.shape[0], x.shape[1], x.shape[1]), f"Expected shape {(x.shape[0], x.shape[1], x.shape[1])}, but got {res.shape}"

    t = jnp.array([0.5, 0.6, 0.7])
    x = jnp.ones((3, 2))
    args = None

    res = g_tilde(t, x, args)
    assert res.shape == (x.shape[0], x.shape[1], x.shape[1]), f"Expected shape {(x.shape[0], x.shape[1], x.shape[1])}, but got {res.shape}"

@pytest.mark.parametrize("solver_cls", [ZeroEnds, NonSingular])
def test_sample_shape(solver_cls):
    model = DummyModel()
    wrapped_model = ModelWrapper(model)

    solver = solver_cls(velocity_model=wrapped_model, mu0=jnp.zeros(2), sigma0=jnp.ones(2), alpha=0.5)

    x_init = jnp.ones((5, 2))

    sol = solver.sample(
        key=jax.random.PRNGKey(0),
        nsamples=5,
        nsteps=300,
        method="Euler",
        adaptive=True,
    ) 
    assert sol.shape == x_init.shape, f"Expected shape {x_init.shape}, but got {sol.shape}"

 
    sol = solver.sample(
        key=jax.random.PRNGKey(0),
        nsamples=5,
        nsteps=300,
        method="SEA",
        adaptive=True,
    ) 
    assert sol.shape == x_init.shape, f"Expected shape {x_init.shape}, but got {sol.shape}"

    sol = solver.sample(
        key=jax.random.PRNGKey(0),
        nsamples=5,
        nsteps=300,
        method="ShARK",
        adaptive=True,
    ) 
    assert sol.shape == x_init.shape, f"Expected shape {x_init.shape}, but got {sol.shape}"

    # test error
    with pytest.raises(ValueError):
        sol = solver.sample(
            key=jax.random.PRNGKey(0),
            nsamples=5,
            nsteps=300,
            method="InvalidMethod",
            adaptive=True,
        )