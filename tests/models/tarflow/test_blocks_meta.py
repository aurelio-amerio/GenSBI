import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from gensbi.models.tarflow.blocks import MetaBlock
from gensbi.models.tarflow.conditioners import AdditiveBiasConditioner
from gensbi.models.tarflow.conditioners import VectorConditioner
from gensbi.models.tarflow.pe import VisionRotaryEmbedding
from gensbi.models import TarFlow, TarFlowParams


def _make(T=4, F=1, channels=8, cond_dim=2, zero_init=True, rngs=None, perm=None):
    rngs = rngs or nnx.Rngs(0)
    if perm is None:
        perm = jnp.arange(T)                 # identity perm
    cond = AdditiveBiasConditioner(cond_dim, channels, rngs=rngs)
    return MetaBlock(F=F, channels=channels, T=T, perm=perm,
                     conditioner=cond, num_layers=2, num_heads=2, expansion=2,
                     rngs=rngs, zero_init=zero_init)


def test_zero_init_is_identity():
    blk = _make(zero_init=True)
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 4, 1))
    cond = jax.random.normal(jax.random.PRNGKey(2), (3, 2))
    z, logdet = blk.inverse(x, cond)
    assert jnp.allclose(z, x, atol=1e-6)
    assert jnp.allclose(logdet, 0.0, atol=1e-6)


def test_inverse_is_triangular():
    """z[i] must not depend on x[j] for j > i (F=1 ⇒ clean (T,T) Jacobian)."""
    blk = _make(F=1, zero_init=False)
    T = 4
    x0 = jax.random.normal(jax.random.PRNGKey(3), (T, 1))
    cond = jnp.array([0.3, -0.4])

    def f(x):
        z, _ = blk.inverse(x[None], cond[None])     # (1, T, 1)
        return z[0, :, 0]                            # (T,)

    J = jax.jacrev(f)(x0[:, 0])                      # (T, T)
    for i in range(T):
        for j in range(i + 1, T):
            assert jnp.allclose(J[i, j], 0.0, atol=1e-6), (i, j)


def test_inverse_logdet_matches_autodiff():
    blk = _make(F=1, zero_init=False)
    x0 = jax.random.normal(jax.random.PRNGKey(4), (4, 1))
    cond = jnp.array([0.3, -0.4])

    def f(x):
        z, _ = blk.inverse(x[None], cond[None])
        return z[0, :, 0]

    _, ad = jnp.linalg.slogdet(jax.jacobian(f)(x0[:, 0]))
    _, analytic = blk.inverse(x0[None], cond[None])
    assert jnp.allclose(ad, analytic[0], atol=1e-4)


def test_forward_inverse_roundtrip():
    blk = _make(F=1, zero_init=False)
    x = jax.random.normal(jax.random.PRNGKey(5), (3, 4, 1))
    cond = jax.random.normal(jax.random.PRNGKey(6), (3, 2))
    z, _ = blk.inverse(x, cond)
    x_rt, _ = blk.forward(z, cond)
    assert jnp.allclose(x_rt, x, atol=1e-4)


def test_forward_logdet_matches_inverse():
    """forward logdet (accumulated in the scan) must equal -inverse logdet at the
    round-trip point, for F>1 (guards the in-scan logdet accumulation)."""
    blk = _make(F=2, channels=8, zero_init=False)
    z = jax.random.normal(jax.random.PRNGKey(11), (3, 4, 2))
    cond = jax.random.normal(jax.random.PRNGKey(12), (3, 2))
    x, fwd_ld = blk.forward(z, cond)
    _, inv_ld = blk.inverse(x, cond)
    assert jnp.allclose(fwd_ld, -inv_ld, atol=1e-4)


def _make_prefix(T=4, F=1, channels=8, cond_dim=2, cond_channels=1, zero_init=False,
                 rngs=None):
    rngs = rngs or nnx.Rngs(0)
    perm = jnp.arange(T)
    cond = VectorConditioner(cond_dim, cond_channels, channels, rngs=rngs)
    return MetaBlock(F=F, channels=channels, T=T, perm=perm,
                     conditioner=cond, num_layers=2,
                     num_heads=2, expansion=2, rngs=rngs, zero_init=zero_init)


def test_prefix_inverse_is_triangular():
    """z[i] must not depend on x[j] for j > i, with a prefix condition."""
    blk = _make_prefix(F=1, zero_init=False)
    T = 4
    x0 = jax.random.normal(jax.random.PRNGKey(3), (T, 1))
    cond = jnp.array([0.3, -0.4])[:, None]   # (2, 1)

    def f(x):
        z, _ = blk.inverse(x[None], cond[None])  # (1, 2, 1)
        return z[0, :, 0]

    J = jax.jacrev(f)(x0[:, 0])
    for i in range(T):
        for j in range(i + 1, T):
            assert jnp.allclose(J[i, j], 0.0, atol=1e-6), (i, j)


def test_prefix_logdet_matches_autodiff():
    blk = _make_prefix(F=1, zero_init=False)
    x0 = jax.random.normal(jax.random.PRNGKey(4), (4, 1))
    cond = jnp.array([0.3, -0.4])[:, None]   # (2, 1)

    def f(x):
        z, _ = blk.inverse(x[None], cond[None])  # (1, 2, 1)
        return z[0, :, 0]

    _, ad = jnp.linalg.slogdet(jax.jacobian(f)(x0[:, 0]))
    _, analytic = blk.inverse(x0[None], cond[None])
    assert jnp.allclose(ad, analytic[0], atol=1e-4)


def test_prefix_roundtrip():
    blk = _make_prefix(F=1, zero_init=False)
    x = jax.random.normal(jax.random.PRNGKey(5), (3, 4, 1))
    cond = jax.random.normal(jax.random.PRNGKey(6), (3, 2, 1))
    z, _ = blk.inverse(x, cond)
    x_rt, _ = blk.forward(z, cond)
    assert jnp.allclose(x_rt, x, atol=1e-4)


def test_prefix_forward_logdet_matches_inverse():
    blk = _make_prefix(F=2, channels=8, zero_init=False)
    z = jax.random.normal(jax.random.PRNGKey(13), (3, 4, 2))
    cond = jax.random.normal(jax.random.PRNGKey(14), (3, 2, 1))
    x, fwd_ld = blk.forward(z, cond)
    _, inv_ld = blk.inverse(x, cond)
    assert jnp.allclose(fwd_ld, -inv_ld, atol=1e-4)


def test_prefix_zero_init_identity():
    blk = _make_prefix(zero_init=True)
    x = jax.random.normal(jax.random.PRNGKey(7), (3, 4, 1))
    cond = jax.random.normal(jax.random.PRNGKey(8), (3, 2, 1))
    z, logdet = blk.inverse(x, cond)
    assert jnp.allclose(z, x, atol=1e-6)
    assert jnp.allclose(logdet, 0.0, atol=1e-6)


def test_prefix_conditions_output():
    """The prefix must actually condition (a,b): different conditions must give
    different z. Guards against a regression that silently ignores the prefix."""
    blk = _make_prefix(F=1, zero_init=False)
    x = jax.random.normal(jax.random.PRNGKey(9), (2, 4, 1))
    cond1 = jnp.broadcast_to(jnp.array([0.3, -0.4])[:, None], (2, 2, 1))
    cond2 = jnp.broadcast_to(jnp.array([-0.7, 0.9])[:, None], (2, 2, 1))
    z1, _ = blk.inverse(x, cond1)
    z2, _ = blk.inverse(x, cond2)
    assert not jnp.allclose(z1, z2, atol=1e-6)


def test_tarflow_vector_channels_one_unchanged():
    m = TarFlow(TarFlowParams(rngs=nnx.Rngs(0), modeled="vector", dim=4,
                              num_blocks=2, head_dim=8, num_heads=2))
    x = jnp.zeros((3, 4, 1))
    assert m.log_prob(x).shape == (3,)
    assert m.sample(jax.random.PRNGKey(0), nsamples=3).shape == (3, 4, 1)


def test_tarflow_vector_multichannel():
    m = TarFlow(TarFlowParams(rngs=nnx.Rngs(0), modeled="vector", dim=4,
                              vec_channels=2, num_blocks=2, head_dim=8,
                              num_heads=2))
    x = jnp.zeros((3, 4, 2))
    assert m.log_prob(x).shape == (3,)                 # scalar per sample
    assert m.sample(jax.random.PRNGKey(0), nsamples=3).shape == (3, 4, 2)


def test_tarflow_set_standardization_per_channel():
    m = TarFlow(TarFlowParams(rngs=nnx.Rngs(0), modeled="vector", dim=4,
                              vec_channels=2, num_blocks=2, head_dim=8,
                              num_heads=2, standardize=True))
    m.set_standardization(jnp.array([1.0, 2.0]), jnp.array([1.0, 1.0]))  # (C,)
    assert m.log_prob(jnp.zeros((2, 4, 2))).shape == (2,)


def test_metablock_derives_inverse_permutation():
    perm = jax.random.permutation(jax.random.PRNGKey(3), 4)
    block = _make(perm=perm)
    assert jnp.array_equal(block.inv_perm[...], jnp.argsort(perm))


def _rope_meta_block(cond_dim=0, zero_init=False, seed=0):
    """2x2 image grid (T=4), F=2, channels=8, head_dim=4, 1 layer."""
    rngs = nnx.Rngs(seed)
    if cond_dim > 0:
        conditioner = VectorConditioner(cond_dim, 1, 8, rngs=rngs)
    else:
        conditioner = AdditiveBiasConditioner(0, 8, rngs=rngs)
    rope = VisionRotaryEmbedding(dim=2, pt_seq_len=2)   # head_dim=8/2heads=4
    return MetaBlock(F=2, channels=8, T=4, perm=jnp.arange(4),
                     conditioner=conditioner, num_layers=1, num_heads=2,
                     expansion=2, rngs=rngs, zero_init=zero_init,
                     rope=rope, grid=(2, 2))


def test_rope_drops_learned_pos_embed():
    blk = _rope_meta_block()
    assert blk.pos_embed is None
    assert blk.freqs_cis is not None
    assert blk.freqs_cis[...].shape == (4, 4)           # (T, head_dim), M=0


def test_rope_freqs_layout_with_prefix():
    """Prefix slots must be identity rows (all-zero angles)."""
    blk = _rope_meta_block(cond_dim=3)
    freqs = blk.freqs_cis[...]
    assert freqs.shape == (3 + 4, 4)                     # (M+T, head_dim)
    assert jnp.allclose(freqs[:3], 0.0)                  # prefix at identity
    assert not jnp.allclose(freqs[3:], 0.0)              # image slots rotated


def test_no_rope_is_unchanged():
    """rope=None keeps pos_embed and freqs_cis=None (regression guard)."""
    rngs = nnx.Rngs(0)
    blk = MetaBlock(F=2, channels=8, T=4, perm=jnp.arange(4),
                    conditioner=AdditiveBiasConditioner(0, 8, rngs=rngs),
                    num_layers=1, num_heads=2, expansion=2, rngs=rngs)
    assert blk.pos_embed is not None
    assert blk.freqs_cis is None


def test_rope_round_trip():
    """forward(inverse(x)) == x with rope on (random init)."""
    blk = _rope_meta_block(zero_init=False)
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 4, 2))
    z, ld_inv = blk.inverse(x)
    x2, ld_fwd = blk.forward(z)
    assert jnp.allclose(x2, x, atol=1e-4)
    assert jnp.allclose(ld_inv + ld_fwd, 0.0, atol=1e-4)


def test_rope_round_trip_with_prefix_cond():
    blk = _rope_meta_block(cond_dim=3, zero_init=False)
    x = jax.random.normal(jax.random.PRNGKey(2), (2, 4, 2))
    cond = jax.random.normal(jax.random.PRNGKey(3), (2, 3, 1))
    z, _ = blk.inverse(x, cond)
    x2, _ = blk.forward(z, cond)
    assert jnp.allclose(x2, x, atol=1e-4)


def test_rope_inverse_is_causal():
    """z_i must not depend on x_j for j > i, with rope on."""
    blk = _rope_meta_block(zero_init=False)
    x0 = jax.random.normal(jax.random.PRNGKey(4), (4, 2))

    def f(x):
        return blk.inverse(x[None])[0][0]                # (T, F)

    J = jax.jacrev(f)(x0)                                # (T, F, T, F)
    for i in range(4):
        for j in range(i + 1, 4):
            assert jnp.allclose(J[i, :, j, :], 0.0, atol=1e-5), (i, j)


def test_rope_zero_init_is_identity():
    blk = _rope_meta_block(zero_init=True)
    x = jax.random.normal(jax.random.PRNGKey(5), (2, 4, 2))
    z, ld = blk.inverse(x)
    assert jnp.allclose(z, x, atol=1e-6)
    assert jnp.allclose(ld, 0.0, atol=1e-6)


def test_rope_requires_grid():
    rngs = nnx.Rngs(0)
    rope = VisionRotaryEmbedding(dim=2, pt_seq_len=2)
    with pytest.raises(ValueError, match="rope requires grid"):
        MetaBlock(F=2, channels=8, T=4, perm=jnp.arange(4),
                  conditioner=AdditiveBiasConditioner(0, 8, rngs=rngs),
                  num_layers=1, num_heads=2, expansion=2, rngs=rngs,
                  rope=rope, grid=None)


def test_rope_grid_must_match_token_count():
    """grid=(h, w) with h*w != T must raise, not silently mis-lay positions."""
    rngs = nnx.Rngs(0)
    rope = VisionRotaryEmbedding(dim=2, pt_seq_len=3)
    with pytest.raises(ValueError, match=r"grid.*T"):
        MetaBlock(F=2, channels=8, T=4, perm=jnp.arange(4),
                  conditioner=AdditiveBiasConditioner(0, 8, rngs=rngs),
                  num_layers=1, num_heads=2, expansion=2, rngs=rngs,
                  rope=rope, grid=(3, 3))


@pytest.mark.parametrize("use_rope", [False, True])
@pytest.mark.parametrize("cond_kind", ["none", "bias", "prefix"])
def test_cached_forward_matches_reference(use_rope, cond_kind):
    """The KV-cached forward must equal the retained full-recompute scan."""
    rngs = nnx.Rngs(0)
    if cond_kind == "prefix":
        conditioner = VectorConditioner(3, 1, 8, rngs=rngs)
        cond = jax.random.normal(jax.random.PRNGKey(1), (2, 3, 1))
    elif cond_kind == "bias":
        conditioner = AdditiveBiasConditioner(3, 8, rngs=rngs)
        cond = jax.random.normal(jax.random.PRNGKey(1), (2, 3))
    else:
        conditioner = AdditiveBiasConditioner(0, 8, rngs=rngs)
        cond = None
    rope = (VisionRotaryEmbedding(dim=2, pt_seq_len=2), (2, 2)) if use_rope \
        else (None, None)
    blk = MetaBlock(F=2, channels=8, T=4, perm=jnp.arange(4)[::-1],
                    conditioner=conditioner, num_layers=2, num_heads=2,
                    expansion=2, rngs=rngs, zero_init=False,
                    rope=rope[0], grid=rope[1])
    z = jax.random.normal(jax.random.PRNGKey(2), (2, 4, 2))
    x_ref, ld_ref = blk._forward_reference(z, cond)
    x_new, ld_new = blk.forward(z, cond)
    assert jnp.allclose(x_new, x_ref, atol=1e-5), jnp.abs(x_new - x_ref).max()
    assert jnp.allclose(ld_new, ld_ref, atol=1e-4)


def test_cached_forward_still_inverts_inverse():
    """End-to-end: forward(inverse(x)) == x through the cached path."""
    rngs = nnx.Rngs(0)
    blk = MetaBlock(F=2, channels=8, T=4, perm=jnp.arange(4),
                    conditioner=AdditiveBiasConditioner(0, 8, rngs=rngs),
                    num_layers=1, num_heads=2, expansion=2, rngs=rngs,
                    zero_init=False)
    x = jax.random.normal(jax.random.PRNGKey(3), (2, 4, 2))
    z, _ = blk.inverse(x)
    x2, _ = blk.forward(z)
    assert jnp.allclose(x2, x, atol=1e-4)
