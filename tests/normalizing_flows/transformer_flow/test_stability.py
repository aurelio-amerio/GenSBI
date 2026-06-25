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
