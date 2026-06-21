import jax
import jax.numpy as jnp
from flax import nnx
from scipy.integrate import trapezoid

from gensbi.normalizing_flows import make_maf
from gensbi.normalizing_flows.bijections.base import Mask


def test_log_prob_shape_and_finiteness():
    flow = make_maf(rngs=nnx.Rngs(0), dim=3, cond_dim=2, n_layers=3,
                    nn_width=32, nn_depth=2)
    x = jax.random.normal(jax.random.PRNGKey(1), (16, 3))
    cond = jax.random.normal(jax.random.PRNGKey(2), (16, 2))
    lp = flow.log_prob(x, cond)
    assert lp.shape == (16,)
    assert jnp.all(jnp.isfinite(lp))


def test_sample_shape_and_roundtrip_consistency():
    flow = make_maf(rngs=nnx.Rngs(0), dim=3, cond_dim=2, n_layers=2,
                    nn_width=16, nn_depth=1)
    cond = jnp.zeros((5, 2))
    samples = flow.sample(jax.random.PRNGKey(3), cond=cond, nsamples=5)
    assert samples.shape == (5, 3)
    # log_prob of samples is finite (forward then inverse must be consistent)
    assert jnp.all(jnp.isfinite(flow.log_prob(samples, cond)))


def test_density_integrates_to_one_1d():
    """Spec §11 #4 — 1D normalization via trapezoid (better than nothing)."""
    flow = make_maf(rngs=nnx.Rngs(0), dim=1, cond_dim=1, n_layers=3,
                    nn_width=32, nn_depth=2)
    cond = jnp.zeros((1,))
    grid = jnp.linspace(-12.0, 12.0, 4001)[:, None]   # (N, dim=1)
    cond_b = jnp.broadcast_to(cond, (grid.shape[0], 1))
    dens = jnp.exp(flow.log_prob(grid, cond_b))
    integral = trapezoid(dens, grid[:, 0])
    assert jnp.allclose(integral, 1.0, atol=1e-2)


def test_full_flow_logdet_matches_autodiff():
    flow = make_maf(rngs=nnx.Rngs(0), dim=4, cond_dim=2, n_layers=3,
                    nn_width=32, nn_depth=2)
    cond = jnp.array([0.3, -0.4])
    x = jnp.array([0.5, -1.0, 0.3, 0.8])

    def inv_only(x):
        return flow.chain.inverse(x, cond)[0]

    _, ad_logdet = jnp.linalg.slogdet(jax.jacobian(inv_only)(x))
    _, analytic_logdet = flow.chain.inverse(x, cond)
    assert jnp.allclose(ad_logdet, analytic_logdet, atol=1e-4)


def test_masks_are_not_params():
    """Spec §11 #5 — masks/buffers excluded from Param state."""
    flow = make_maf(rngs=nnx.Rngs(0), dim=4, cond_dim=2, n_layers=3,
                    nn_width=16, nn_depth=2)
    params = nnx.state(flow, nnx.Param)
    param_leaves = jax.tree_util.tree_leaves(params)
    assert all(leaf.dtype != bool for leaf in param_leaves)
    # masks ARE present as buffers
    masks = nnx.state(flow, Mask)
    assert len(jax.tree_util.tree_leaves(masks)) > 0
