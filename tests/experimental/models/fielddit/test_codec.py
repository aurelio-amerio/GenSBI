import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx

from gensbi.experimental.models.fielddit.codec import (
    Downsample2D,
    Upsample2D,
    ObsEncoder,
    ObsDecoder,
    Tokenizer,
    Untokenizer,
)


def test_downsample_halves_and_changes_channels():
    down = Downsample2D(in_channels=8, out_channels=16, rngs=nnx.Rngs(0), param_dtype=jnp.float32)
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 16, 16, 8))
    out = down(x)
    assert out.shape == (2, 8, 8, 16)
    assert jnp.all(jnp.isfinite(out))


def test_upsample_doubles_and_changes_channels():
    up = Upsample2D(in_channels=16, out_channels=8, rngs=nnx.Rngs(0), param_dtype=jnp.float32)
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 8, 8, 16))
    out = up(x)
    assert out.shape == (2, 16, 16, 8)
    assert jnp.all(jnp.isfinite(out))


def test_obs_encoder_shapes_and_skips():
    widths = (8, 16, 32)  # D = 2 downsamples
    enc = ObsEncoder(
        in_channels=1, widths=widths, res_blocks=2, vec_dim=16, norm_groups=4,
        rngs=nnx.Rngs(0), param_dtype=jnp.float32,
    )
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 32, 32, 1))
    time_vec = jnp.ones((2, 16))
    feat, pos_skips, neg_skips = enc(x, time_vec)

    # bottleneck: 32 -> 16 -> 8, width 32
    assert feat.shape == (2, 8, 8, 32)
    assert len(pos_skips) == 2 and len(neg_skips) == 2
    # pos_skips captured pre-downsample at stage widths/resolutions
    assert pos_skips[0].shape == (2, 32, 32, 8)
    assert pos_skips[1].shape == (2, 16, 16, 16)
    # neg_skips captured post-downsample
    assert neg_skips[0].shape == (2, 16, 16, 16)
    assert neg_skips[1].shape == (2, 8, 8, 32)
    # the returned feature is exactly the last neg_skip
    assert jnp.allclose(feat, neg_skips[-1])
    assert jnp.all(jnp.isfinite(feat))


def test_obs_decoder_reconstructs_field_shape_and_zero_init():
    widths = (8, 16, 32)
    enc = ObsEncoder(
        in_channels=1, widths=widths, res_blocks=2, vec_dim=16, norm_groups=4,
        rngs=nnx.Rngs(0), param_dtype=jnp.float32,
    )
    dec = ObsDecoder(
        out_channels=1, widths=widths, res_blocks=2, vec_dim=16, norm_groups=4,
        rngs=nnx.Rngs(1), param_dtype=jnp.float32,
    )
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 32, 32, 1))
    vec = jnp.ones((2, 16))
    feat, pos_skips, neg_skips = enc(x, vec)
    out = dec(feat, vec, pos_skips, neg_skips)

    assert out.shape == (2, 32, 32, 1)
    # zero-init final conv => exactly-zero velocity at init (also catches any
    # upstream NaN, since 0 * NaN == NaN would break this).
    assert jnp.allclose(out, 0.0)


def test_tokenizer_untokenizer_roundtrip_shape():
    c_bottleneck, p, hidden = 32, 2, 24  # hidden != N(=16) so the shape assert validates both axes
    tok = Tokenizer(c_bottleneck, p, hidden, rngs=nnx.Rngs(0), param_dtype=jnp.float32)
    untok = Untokenizer(c_bottleneck, p, hidden, rngs=nnx.Rngs(1), param_dtype=jnp.float32)

    feat = jax.random.normal(jax.random.PRNGKey(0), (2, 8, 8, c_bottleneck))
    tokens = tok(feat)
    # token grid = (8/2, 8/2) = (4, 4) => 16 tokens, hidden=16
    assert tokens.shape == (2, 16, hidden)

    back = untok(tokens, grid=(4, 4))
    # shape round-trip (values differ; the two Linears are not inverses)
    assert back.shape == feat.shape


def test_untokenizer_normalizes_residual_stream():
    """A 100x token-magnitude blowup must not reach the conv decoder:
    LayerNorm before the projection makes the output scale-invariant at init."""
    untok = Untokenizer(
        out_channels=4, patch_size=2, hidden_size=16, rngs=nnx.Rngs(0), param_dtype=jnp.float32
    )
    tokens = jax.random.normal(jax.random.PRNGKey(0), (2, 16, 16))
    grid = (4, 4)
    out_small = untok(tokens, grid)
    out_big = untok(tokens * 100.0, grid)
    assert jnp.allclose(out_small, out_big, atol=1e-4)
