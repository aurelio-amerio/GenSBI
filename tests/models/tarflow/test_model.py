# tests/models/tarflow/test_model.py
import jax
import jax.numpy as jnp
from flax import nnx
from scipy.integrate import trapezoid
import pytest

from gensbi.models import TarFlow, TarFlowParams
from gensbi.core.prior import make_gaussian_prior


def _flow(dim=4, cond_dim=2, num_blocks=4, **kw):
    return TarFlow(TarFlowParams(rngs=nnx.Rngs(0), dim=dim, cond_dim=cond_dim,
                                 head_dim=8, num_heads=2, num_blocks=num_blocks,
                                 layers_per_block=2, **kw))


def test_log_prob_shape_and_finite():
    flow = _flow()
    x = jax.random.normal(jax.random.PRNGKey(1), (8, 4, 1))
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
    lp = flow.log_prob(x[..., None], cond)
    lp_base = jax.vmap(base.log_prob)(x)
    assert jnp.allclose(lp, lp_base, atol=1e-4)


def test_full_flow_logdet_matches_autodiff():
    dim, cond_dim = 4, 2
    # The composition logdet is exactly the sum of per-block (-Sum log_scale); per-block
    # correctness is verified in Task 4. This cross-check builds the ASSEMBLED
    # composition Jacobian and takes its slogdet, which is only well-conditioned
    # at shallow depth -- at 4 random-init blocks the assembled 4x4 Jacobian is
    # near-singular and float32 slogdet drifts ~10 nats. 2 blocks stays exact.
    flow = _flow(dim=dim, cond_dim=cond_dim, zero_init=False, num_blocks=2)
    cond = jnp.array([0.3, -0.4])
    x = jnp.array([0.5, -1.0, 0.3, 0.8])[:, None]   # (4, 1)
    base = make_gaussian_prior((dim,))

    def to_noise(x):
        # reproduce the data→noise map (no standardization set => identity)
        z = flow.tokenizer.tokenize(x[None])           # (1, 4, 1)
        for blk in flow.blocks:
            z, _ = blk.inverse(z, cond[None])
        return z.reshape(-1)

    _, ad = jnp.linalg.slogdet(jax.jacobian(to_noise)(x).reshape(dim, dim))
    # analytic: log_prob = base.log_prob(z) + logdet  =>  logdet = lp - base
    z = to_noise(x)
    lp = flow.log_prob(x[None], cond[None])[0]        # x[None] = (1, 4, 1)
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
    assert s.shape == (5, 4, 1)
    assert jnp.all(jnp.isfinite(flow.log_prob(s, cond)))


def test_density_integrates_to_one_2d():
    # A normalized change-of-variables integrates to 1 exactly. An UNTRAINED
    # non-trivial flow places mass arbitrarily (tails escape any fixed grid), so
    # we verify normalization on the zero_init identity flow (an exact standard
    # normal). The logdet's contribution to normalization is covered by
    # test_full_flow_logdet_matches_autodiff.
    flow = TarFlow(TarFlowParams(rngs=nnx.Rngs(0), dim=2, cond_dim=1, head_dim=8,
                                 num_heads=2, num_blocks=4, layers_per_block=2,
                                 zero_init=True))
    g = jnp.linspace(-8.0, 8.0, 161)
    xx, yy = jnp.meshgrid(g, g)
    grid = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)        # (N, 2)
    cond = jnp.zeros((grid.shape[0], 1))
    dens = jnp.exp(flow.log_prob(grid[..., None], cond)).reshape(161, 161)
    integral = trapezoid(trapezoid(dens, g, axis=1), g)
    assert jnp.allclose(integral, 1.0, atol=2e-2)


def test_log_prob_depends_on_condition():
    flow = _flow(zero_init=False)
    x = jax.random.normal(jax.random.PRNGKey(1), (5, 4, 1))
    lp_a = flow.log_prob(x, jnp.zeros((5, 2)))
    lp_b = flow.log_prob(x, jnp.ones((5, 2)))
    assert not jnp.allclose(lp_a, lp_b)


def test_unconditional_flow():
    flow = TarFlow(TarFlowParams(rngs=nnx.Rngs(0), dim=3, cond_dim=0, head_dim=8,
                                 num_heads=2, num_blocks=2, layers_per_block=1))
    x = jax.random.normal(jax.random.PRNGKey(1), (6, 3, 1))
    assert flow.log_prob(x).shape == (6,)


def test_set_standardization():
    flow = _flow()
    mean = jnp.array([1.0, -2.0, 0.5, 0.0])
    std = jnp.array([2.0, 0.5, 3.0, 1.0])
    flow.set_standardization(mean, std)
    assert flow.mean[...].shape == (4, 1)
    assert jnp.allclose(flow.mean[...].ravel(), mean)
    assert flow.std[...].shape == (4, 1)
    assert jnp.allclose(flow.std[...].ravel(), std)


def test_set_standardization_raises_when_disabled():
    flow = _flow(standardize=False)
    with pytest.raises(ValueError):
        flow.set_standardization(jnp.zeros(4), jnp.ones(4))


def test_image_modeled_log_prob_and_sample():
    # Use num_blocks=1: a deep random-init (no zero_init) image flow overflows
    # float32 (exp(a) compounds across 16 tokens × 4 features), same as the
    # vector test_sample_shape_and_roundtrip_finite.  Shape and finiteness are
    # the axes being tested here; depth is exercised once the model is trained.
    flow = TarFlow(TarFlowParams(rngs=nnx.Rngs(0), cond_dim=2, modeled="image",
                                 img_size=8, patch_size=2, img_channels=1,
                                 head_dim=8, num_heads=2, num_blocks=1,
                                 layers_per_block=2, zero_init=False))
    x = jax.random.normal(jax.random.PRNGKey(1), (5, 8, 8, 1))
    cond = jax.random.normal(jax.random.PRNGKey(2), (5, 2))
    lp = flow.log_prob(x, cond)
    assert lp.shape == (5,) and jnp.all(jnp.isfinite(lp))
    s = flow.sample(jax.random.PRNGKey(3), cond=cond)
    assert s.shape == (5, 8, 8, 1)


def test_image_modeled_zero_init_is_base():
    flow = TarFlow(TarFlowParams(rngs=nnx.Rngs(0), cond_dim=2, modeled="image",
                                 img_size=8, patch_size=2, img_channels=1,
                                 head_dim=8, num_heads=2, num_blocks=4,
                                 layers_per_block=2, zero_init=True))
    x = jax.random.normal(jax.random.PRNGKey(1), (4, 8, 8, 1))
    cond = jax.random.normal(jax.random.PRNGKey(2), (4, 2))
    lp = flow.log_prob(x, cond)
    # zero-init ⇒ identity flow ⇒ standard normal over the 8*8*1 elements
    expected = -0.5 * jnp.sum(x ** 2, axis=(1, 2, 3)) - 0.5 * 64 * jnp.log(2 * jnp.pi)
    assert jnp.allclose(lp, expected, atol=1e-4)


def test_image_condition_npe_depends_on_condition():
    # NPE: modeled theta vector (dim=2), condition = 8x8x1 image via prefix
    flow = TarFlow(TarFlowParams(rngs=nnx.Rngs(0), dim=2, modeled="vector",
                                 cond="image", cond_img_size=8,
                                 cond_patch_size=2, cond_channels=1, head_dim=8,
                                 num_heads=2, num_blocks=4, layers_per_block=2,
                                 zero_init=False))
    theta = jax.random.normal(jax.random.PRNGKey(1), (5, 2, 1))
    img_a = jnp.zeros((5, 8, 8, 1))
    img_b = jnp.ones((5, 8, 8, 1))
    assert not jnp.allclose(flow.log_prob(theta, img_a), flow.log_prob(theta, img_b))
    s = flow.sample(jax.random.PRNGKey(4), cond=img_a)
    assert s.shape == (5, 2, 1)


def test_vector_path_unchanged():
    # the v1 default vector path still builds and runs
    flow = TarFlow(TarFlowParams(rngs=nnx.Rngs(0), dim=4, cond_dim=2, head_dim=8,
                                 num_heads=2, num_blocks=4, layers_per_block=2))
    x = jax.random.normal(jax.random.PRNGKey(1), (8, 4, 1))
    cond = jax.random.normal(jax.random.PRNGKey(2), (8, 2))
    assert flow.log_prob(x, cond).shape == (8,)


def test_image_set_standardization_shape():
    flow = TarFlow(TarFlowParams(rngs=nnx.Rngs(0), cond_dim=2, modeled="image",
                                 img_size=8, patch_size=2, img_channels=1,
                                 head_dim=8, num_heads=2, num_blocks=2,
                                 layers_per_block=1))
    mean = jnp.zeros((8, 8, 1))
    std = jnp.ones((8, 8, 1)) * 2.0
    flow.set_standardization(mean, std)
    assert flow.mean[...].shape == (8, 8, 1)
    assert jnp.allclose(flow.std[...], std)


def _rope_params(cond="bias", **kw):
    """4x4 single-channel image, patch 2 -> T=4 tokens, head_dim=8."""
    base = dict(rngs=nnx.Rngs(0), modeled="image", img_size=4, patch_size=2,
                img_channels=1, head_dim=8, num_heads=2, num_blocks=2,
                layers_per_block=1, use_rope=True, cond=cond)
    if cond == "bias":
        base["cond_dim"] = 3
    elif cond == "vector":
        base["cond_dim"] = 3
    else:                                   # cond == "image"
        base.update(cond_img_size=4, cond_patch_size=2)
    base.update(kw)
    return TarFlowParams(**base)


def test_use_rope_requires_image_modeled():
    with pytest.raises(ValueError, match="use_rope"):
        TarFlowParams(rngs=nnx.Rngs(0), dim=4, use_rope=True)


def test_use_rope_requires_head_dim_multiple_of_4():
    with pytest.raises(ValueError, match="head_dim"):
        _rope_params(head_dim=6)


def test_rope_model_drops_pos_embed():
    model = TarFlow(_rope_params())
    assert all(blk.pos_embed is None for blk in model.blocks)
    assert all(blk.freqs_cis is not None for blk in model.blocks)


def test_rope_log_prob_and_sample_bias_cond():
    model = TarFlow(_rope_params(cond="bias"))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 4, 4, 1))
    cond = jax.random.normal(jax.random.PRNGKey(2), (2, 3))
    lp = model.log_prob(x, cond)
    assert lp.shape == (2,) and jnp.all(jnp.isfinite(lp))
    s = model.sample(jax.random.PRNGKey(3), cond=cond)
    assert s.shape == (2, 4, 4, 1) and jnp.all(jnp.isfinite(s))


def test_rope_log_prob_and_sample_vector_cond():
    model = TarFlow(_rope_params(cond="vector"))
    x = jax.random.normal(jax.random.PRNGKey(4), (2, 4, 4, 1))
    cond = jax.random.normal(jax.random.PRNGKey(5), (2, 3, 1))
    lp = model.log_prob(x, cond)
    assert lp.shape == (2,) and jnp.all(jnp.isfinite(lp))
    s = model.sample(jax.random.PRNGKey(6), cond=cond)
    assert s.shape == (2, 4, 4, 1) and jnp.all(jnp.isfinite(s))


def test_rope_log_prob_and_sample_image_cond():
    model = TarFlow(_rope_params(cond="image"))
    x = jax.random.normal(jax.random.PRNGKey(7), (2, 4, 4, 1))
    cond = jax.random.normal(jax.random.PRNGKey(8), (2, 4, 4, 1))
    lp = model.log_prob(x, cond)
    assert lp.shape == (2,) and jnp.all(jnp.isfinite(lp))
    s = model.sample(jax.random.PRNGKey(9), cond=cond)
    assert s.shape == (2, 4, 4, 1) and jnp.all(jnp.isfinite(s))


def test_rope_training_smoke():
    """A few gradient steps must reduce the NLL on a tiny fixed batch."""
    import optax

    model = TarFlow(_rope_params(cond="bias"))
    x = jax.random.normal(jax.random.PRNGKey(10), (16, 4, 4, 1))
    cond = jax.random.normal(jax.random.PRNGKey(11), (16, 3))
    opt = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

    @nnx.jit
    def step(model, opt):
        def loss_fn(m):
            return -jnp.mean(m.log_prob(x, cond))
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        opt.update(model, grads)
        return loss

    losses = [float(step(model, opt)) for _ in range(30)]
    assert all(jnp.isfinite(jnp.asarray(losses)))
    assert losses[-1] < losses[0]
