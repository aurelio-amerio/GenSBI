import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx

from gensbi.experimental.models.fielddit.blocks import (
    _safe_groups,
    ConvModulation,
    ModulatedResBlock2D,
)


def test_safe_groups_divides():
    assert _safe_groups(8, 4) == 4
    assert _safe_groups(8, 32) == 8  # cannot exceed num_features
    assert _safe_groups(7, 8) == 1   # odd channels fall back to 1 group


def test_conv_modulation_zero_init():
    mod = ConvModulation(vec_dim=16, channels=8, rngs=nnx.Rngs(0), param_dtype=jnp.float32)
    scale, shift, gate = mod(jnp.ones((2, 16)))
    assert scale.shape == (2, 1, 1, 8)
    assert shift.shape == (2, 1, 1, 8)
    assert gate.shape == (2, 1, 1, 8)
    # zero-initialized linear => neutral modulation at init
    assert jnp.allclose(scale, 0.0)
    assert jnp.allclose(shift, 0.0)
    assert jnp.allclose(gate, 0.0)


def test_modulated_resblock_identity_at_init():
    block = ModulatedResBlock2D(
        in_channels=8, out_channels=8, vec_dim=16, norm_groups=4,
        rngs=nnx.Rngs(0), param_dtype=jnp.float32,
    )
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 12, 12, 8))
    vec = jnp.ones((2, 16))
    out = block(x, vec)
    assert out.shape == (2, 12, 12, 8)
    # gate is zero at init => out == residual == x (in_channels == out_channels)
    assert jnp.allclose(out, x, atol=1e-5)


def test_modulated_resblock_channel_change():
    block = ModulatedResBlock2D(
        in_channels=8, out_channels=16, vec_dim=16, norm_groups=4,
        rngs=nnx.Rngs(0), param_dtype=jnp.float32,
    )
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 12, 12, 8))
    out = block(x, jnp.ones((2, 16)))
    assert out.shape == (2, 12, 12, 16)
    assert jnp.all(jnp.isfinite(out))
