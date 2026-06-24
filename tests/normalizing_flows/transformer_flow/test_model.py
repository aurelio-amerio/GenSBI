# tests/normalizing_flows/transformer_flow/test_model.py
import jax
import jax.numpy as jnp
from flax import nnx
from scipy.integrate import trapezoid
import pytest

from gensbi.normalizing_flows.transformer_flow.model import (
    TransformerFlow, make_tarflow,
)
from gensbi.core.prior import make_gaussian_prior


def _flow(dim=4, cond_dim=2, num_blocks=4, **kw):
    return make_tarflow(nnx.Rngs(0), dim=dim, cond_dim=cond_dim, channels=16,
                        num_blocks=num_blocks, layers_per_block=2, head_dim=8, **kw)


def test_log_prob_shape_and_finite():
    flow = _flow()
    x = jax.random.normal(jax.random.PRNGKey(1), (8, 4))
    cond = jax.random.normal(jax.random.PRNGKey(2), (8, 2))
    lp = flow.log_prob(x, cond)
    assert lp.shape == (8,)
    assert jnp.all(jnp.isfinite(lp))


def test_zero_init_flow_is_standard_normal():
    dim, cond_dim = 4, 2
    flow = _flow(dim=dim, cond_dim=cond_dim, zero_init=True)
    base = make_gaussian_prior((dim,))
    x = jax.random.normal(jax.random.PRNGKey(1), (8, dim))
    cond = jax.random.normal(jax.random.PRNGKey(2), (8, cond_dim))
    lp = flow.log_prob(x, cond)
    lp_base = jax.vmap(base.log_prob)(x)
    assert jnp.allclose(lp, lp_base, atol=1e-4)


def test_full_flow_logdet_matches_autodiff():
    dim, cond_dim = 4, 2
    # The composition logdet is exactly the sum of per-block (-Sum a); per-block
    # correctness is verified in Task 4. This cross-check builds the ASSEMBLED
    # composition Jacobian and takes its slogdet, which is only well-conditioned
    # at shallow depth -- at 4 random-init blocks the assembled 4x4 Jacobian is
    # near-singular and float32 slogdet drifts ~10 nats. 2 blocks stays exact.
    flow = _flow(dim=dim, cond_dim=cond_dim, zero_init=False, num_blocks=2)
    cond = jnp.array([0.3, -0.4])
    x = jnp.array([0.5, -1.0, 0.3, 0.8])
    base = make_gaussian_prior((dim,))

    def to_noise(x):
        # reproduce the data→noise map (no standardization set => identity)
        z = flow.tokenizer.tokenize(x[None])
        for blk in flow.blocks:
            z, _ = blk.inverse(z, cond[None])
        return z.reshape(-1)

    _, ad = jnp.linalg.slogdet(jax.jacobian(to_noise)(x))
    # analytic: log_prob = base.log_prob(z) + logdet  =>  logdet = lp - base
    z = to_noise(x)
    lp = flow.log_prob(x[None], cond[None])[0]
    analytic = lp - base.log_prob(z)
    assert jnp.allclose(ad, analytic, atol=3e-4)


def test_sample_shape_and_roundtrip_finite():
    # Sampling runs the sequential noise->data scan. The flow is faithful to
    # TarFlow (no affine-scale clamp), so on an UNTRAINED flow exp(a) compounds
    # through the scan and a deep random-init stack overflows float32. One block
    # is non-trivial and robustly finite at init; depth is exercised once trained.
    flow = _flow(zero_init=False, num_blocks=1)
    cond = jnp.zeros((5, 2))
    s = flow.sample(jax.random.PRNGKey(3), cond=cond)
    assert s.shape == (5, 4)
    assert jnp.all(jnp.isfinite(flow.log_prob(s, cond)))


def test_density_integrates_to_one_2d():
    # A normalized change-of-variables integrates to 1 exactly. An UNTRAINED
    # non-trivial flow places mass arbitrarily (tails escape any fixed grid), so
    # we verify normalization on the zero_init identity flow (an exact standard
    # normal). The logdet's contribution to normalization is covered by
    # test_full_flow_logdet_matches_autodiff.
    flow = make_tarflow(nnx.Rngs(0), dim=2, cond_dim=1, channels=16,
                        num_blocks=4, layers_per_block=2, head_dim=8,
                        zero_init=True)
    g = jnp.linspace(-8.0, 8.0, 161)
    xx, yy = jnp.meshgrid(g, g)
    grid = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)        # (N, 2)
    cond = jnp.zeros((grid.shape[0], 1))
    dens = jnp.exp(flow.log_prob(grid, cond)).reshape(161, 161)
    integral = trapezoid(trapezoid(dens, g, axis=1), g)
    assert jnp.allclose(integral, 1.0, atol=2e-2)


def test_log_prob_depends_on_condition():
    flow = _flow(zero_init=False)
    x = jax.random.normal(jax.random.PRNGKey(1), (5, 4))
    lp_a = flow.log_prob(x, jnp.zeros((5, 2)))
    lp_b = flow.log_prob(x, jnp.ones((5, 2)))
    assert not jnp.allclose(lp_a, lp_b)


def test_unconditional_flow():
    flow = make_tarflow(nnx.Rngs(0), dim=3, cond_dim=0, channels=16,
                        num_blocks=2, layers_per_block=1, head_dim=8)
    x = jax.random.normal(jax.random.PRNGKey(1), (6, 3))
    assert flow.log_prob(x).shape == (6,)


def test_set_standardization():
    flow = _flow()
    mean = jnp.array([1.0, -2.0, 0.5, 0.0])
    std = jnp.array([2.0, 0.5, 3.0, 1.0])
    flow.set_standardization(mean, std)
    assert jnp.allclose(flow.mean[...], mean)
    assert jnp.allclose(flow.std[...], std)


def test_set_standardization_raises_when_disabled():
    flow = _flow(standardize=False)
    with pytest.raises(ValueError):
        flow.set_standardization(jnp.zeros(4), jnp.ones(4))
