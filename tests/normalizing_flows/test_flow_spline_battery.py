import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx
from scipy.integrate import trapezoid

from gensbi.normalizing_flows import make_maf
from gensbi.normalizing_flows.bijections.transformers import RQSpline


def _spline_flow(dim, cond_dim, **kw):
    return make_maf(nnx.Rngs(0), dim=dim, cond_dim=cond_dim, n_layers=4,
                    nn_width=32, nn_depth=2,
                    transformer=RQSpline(num_bins=8, range_bound=6.0),
                    standardize=True, zero_init=False, **kw)


def test_spline_flow_invertibility():
    dim, cond_dim = 3, 2
    flow = _spline_flow(dim, cond_dim)
    cond = jax.random.normal(jax.random.PRNGKey(1), (cond_dim,))
    u = jax.random.normal(jax.random.PRNGKey(2), (dim,))
    x, _ = flow.chain.forward(u, cond)
    u_rec, _ = flow.chain.inverse(x, cond)
    assert jnp.allclose(u_rec, u, atol=1e-4)


def test_spline_flow_logdet_matches_autodiff():
    dim, cond_dim = 4, 2
    flow = _spline_flow(dim, cond_dim)
    cond = jax.random.normal(jax.random.PRNGKey(1), (cond_dim,))
    x = jax.random.normal(jax.random.PRNGKey(3), (dim,)) * 0.5

    _, logdet = flow.chain.inverse(x, cond)
    jac = jax.jacobian(lambda z: flow.chain.inverse(z, cond)[0])(x)
    sign, logabsdet = jnp.linalg.slogdet(jac)
    assert jnp.allclose(logdet, logabsdet, atol=1e-4), (logdet, logabsdet)


def test_spline_flow_1d_density_integrates_to_one():
    flow = make_maf(nnx.Rngs(0), dim=1, cond_dim=0, n_layers=4, nn_width=32,
                    transformer=RQSpline(num_bins=8, range_bound=6.0),
                    standardize=True, zero_init=False)
    grid = jnp.linspace(-8.0, 8.0, 4001)[:, None]       # (G, 1)
    dens = jnp.exp(flow.log_prob(grid))                 # (G,)
    integral = trapezoid(dens, grid[:, 0])
    assert jnp.allclose(integral, 1.0, atol=1e-2), integral


def test_spline_made_autoregression_preserved():
    # output dim d depends on x_d (through the transformer) and x_{<d} (through
    # the MADE params), but MUST have zero Jacobian w.r.t. x_{>d}. The strict
    # MADE mask is unchanged by the wider spline output, so this must still hold.
    dim, cond_dim = 4, 2
    flow = _spline_flow(dim, cond_dim)
    cond = jax.random.normal(jax.random.PRNGKey(1), (cond_dim,))
    x = jax.random.normal(jax.random.PRNGKey(5), (dim,)) * 0.5
    jac = jax.jacobian(lambda z: flow.chain.bijections[0].inverse(z, cond)[0])(x)
    # bijections[0] is the first MaskedAutoregressive layer (no permutation yet)
    for d in range(dim):
        for j in range(d + 1, dim):                     # strictly greater than d
            assert jnp.allclose(jac[d, j], 0.0, atol=1e-6), (d, j, jac[d, j])
