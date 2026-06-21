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


def test_set_standardization_sets_buffers():
    import jax.numpy as jnp
    from flax import nnx
    from gensbi.normalizing_flows import make_maf
    from gensbi.normalizing_flows.bijections.standardize import Standardize

    flow = make_maf(nnx.Rngs(0), dim=3, cond_dim=2, n_layers=2, standardize=True)
    mean = jnp.array([1.0, -2.0, 0.5])
    std = jnp.array([2.0, 0.5, 3.0])
    flow.set_standardization(mean, std)

    std_bij = [b for b in flow.chain.bijections if isinstance(b, Standardize)][0]
    assert jnp.allclose(std_bij.mean.value, mean)
    assert jnp.allclose(std_bij.std.value, std)


def test_set_standardization_raises_without_bijection():
    import pytest
    from flax import nnx
    from gensbi.normalizing_flows import make_maf

    flow = make_maf(nnx.Rngs(0), dim=2, cond_dim=1, n_layers=2, standardize=False)
    with pytest.raises(ValueError):
        flow.set_standardization([0.0, 0.0], [1.0, 1.0])


def test_zero_init_spline_flow_is_standard_normal():
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from gensbi.normalizing_flows import make_maf
    from gensbi.normalizing_flows.bijections.transformers import RQSpline
    from gensbi.core.prior import make_gaussian_prior

    dim, cond_dim = 3, 2
    flow = make_maf(nnx.Rngs(0), dim=dim, cond_dim=cond_dim, n_layers=4,
                    transformer=RQSpline(num_bins=8, range_bound=5.0),
                    standardize=True, zero_init=True)
    base = make_gaussian_prior((dim,))

    x = jax.random.normal(jax.random.PRNGKey(1), (16, dim))
    cond = jax.random.normal(jax.random.PRNGKey(2), (16, cond_dim))
    lp = flow.log_prob(x, cond)
    lp_base = jax.vmap(base.log_prob)(x)
    # zero-init spline is the identity => flow density == base density
    assert jnp.allclose(lp, lp_base, atol=1e-4)
