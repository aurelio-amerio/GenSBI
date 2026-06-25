# tests/normalizing_flows/transformer_flow/test_stability.py
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from gensbi.normalizing_flows.transformer_flow.model import make_tarflow
from gensbi.normalizing_flows.transformer_flow.blocks import MetaBlock, INV_SOFTPLUS_1


class _NullCond(nnx.Module):
    def embed(self, cond):
        return (None, None)


def _block(use_softplus=True, soft_clip=4.0, F=1, channels=16, zero_init=True):
    return MetaBlock(
        F=F, channels=channels, T=4, perm=jnp.arange(4), inv_perm=jnp.arange(4),
        conditioner=_NullCond(), num_layers=1, head_dim=8, expansion=4,
        rngs=nnx.Rngs(0), zero_init=zero_init, use_softplus=use_softplus, soft_clip=soft_clip,
    )


def test_affine_exp_mode_exact():
    blk = _block(use_softplus=False)
    a = jnp.array([[-1.5, 0.0, 2.0]])
    scale, inv_scale, log_scale = blk._affine(a)
    assert jnp.allclose(scale, jnp.exp(a))
    assert jnp.allclose(inv_scale, jnp.exp(-a))
    assert jnp.allclose(log_scale, a)


def test_affine_softplus_mode_exact():
    blk = _block(use_softplus=True)
    a = jnp.array([[-1.5, 0.0, 2.0]])
    s = jax.nn.softplus(a + INV_SOFTPLUS_1)
    scale, inv_scale, log_scale = blk._affine(a)
    assert jnp.allclose(scale, s)
    assert jnp.allclose(inv_scale, 1.0 / s)
    assert jnp.allclose(log_scale, jnp.log(s))


def test_affine_softplus_identity_at_zero():
    blk = _block(use_softplus=True)
    scale, inv_scale, log_scale = blk._affine(jnp.zeros((1, 3)))
    assert jnp.allclose(scale, 1.0, atol=1e-6)
    assert jnp.allclose(inv_scale, 1.0, atol=1e-6)
    assert jnp.allclose(log_scale, 0.0, atol=1e-6)


def test_affine_is_float32():
    blk = _block(use_softplus=True)
    scale, inv_scale, log_scale = blk._affine(jnp.zeros((1, 3), dtype=jnp.float32))
    assert scale.dtype == jnp.float32


def test_soft_clip_bounds_params():
    blk = _block(use_softplus=True, soft_clip=4.0)
    # blow up proj_out so the raw output is far outside [-4, 4]
    blk.proj_out.kernel[...] = blk.proj_out.kernel[...] + 50.0
    blk.proj_out.bias[...] = blk.proj_out.bias[...] + 50.0
    xp = jax.random.normal(jax.random.PRNGKey(3), (4, 4, 1))
    a, b = blk._params(xp, None)
    assert jnp.max(jnp.abs(a)) <= 4.0 + 1e-4
    assert jnp.max(jnp.abs(b)) <= 4.0 + 1e-4


def test_block_inverse_uses_softplus_when_enabled():
    # RED DRIVER: before the rewire, inverse uses exp(-a); after, softplus.
    blk = _block(use_softplus=True, soft_clip=0.0, zero_init=False)  # no clip: isolate softplus
    xp = jax.random.normal(jax.random.PRNGKey(1), (6, 4, 1))
    a, b = blk._params(xp, None)
    s = jax.nn.softplus(a + INV_SOFTPLUS_1)
    z_ref = (xp - b) / s                        # perm is identity here (arange)
    z, ld = blk.inverse(xp, None)
    assert jnp.allclose(z, z_ref, atol=1e-5)
    assert jnp.allclose(ld, -jnp.sum(jnp.log(s), axis=(1, 2)), atol=1e-5)


def test_block_exp_inverse_matches_bare_exp():
    # GUARD: use_softplus=False + soft_clip=0 still equals literal (xp-b)*exp(-a).
    blk = _block(use_softplus=False, soft_clip=0.0, zero_init=False)
    xp = jax.random.normal(jax.random.PRNGKey(1), (6, 4, 1))
    a, b = blk._params(xp, None)
    z, ld = blk.inverse(xp, None)
    assert jnp.allclose(z, (xp - b) * jnp.exp(-a), atol=1e-5)
    assert jnp.allclose(ld, -jnp.sum(a, axis=(1, 2)), atol=1e-5)


def test_block_roundtrip_softplus():
    # GUARD: forward and inverse stay mutually consistent (fails if only one is rewired).
    blk = _block(use_softplus=True, soft_clip=4.0, zero_init=False)
    z = jax.random.normal(jax.random.PRNGKey(7), (5, 4, 1))
    x, _ = blk.forward(z, None)
    z2, _ = blk.inverse(x, None)
    assert jnp.allclose(z2.reshape(5, 4, 1), z, atol=1e-4)


def test_make_tarflow_defaults_and_override():
    flow = make_tarflow(nnx.Rngs(0), dim=4, cond_dim=2, channels=16, num_blocks=3,
                        layers_per_block=2, head_dim=8)
    for blk in flow.blocks:
        assert blk.use_softplus is True
        assert blk.soft_clip == 4.0
    flow2 = make_tarflow(nnx.Rngs(0), dim=4, cond_dim=2, channels=16, num_blocks=2,
                         layers_per_block=2, head_dim=8,
                         use_softplus=False, soft_clip=0.0)
    for blk in flow2.blocks:
        assert blk.use_softplus is False
        assert blk.soft_clip == 0.0
