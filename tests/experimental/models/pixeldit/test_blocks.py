import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from gensbi.experimental.models.pixeldit.blocks import MMDiTBlock, PiTBlock
from gensbi.experimental.models.pixeldit.rope import (
    precompute_freqs_cis_1d,
    precompute_freqs_cis_2d,
)

HIDDEN = 64
HEADS = 4
HEAD_DIM = HIDDEN // HEADS


def _make_block(zero_init=True, dtype=jnp.float32):
    return MMDiTBlock(
        HIDDEN,
        HEADS,
        zero_init=zero_init,
        rngs=nnx.Rngs(0),
        param_dtype=dtype,
    )


def _inputs(dtype=jnp.float32):
    kx, ky, kc = jax.random.split(jax.random.key(1), 3)
    x = jax.random.normal(kx, (2, 12, HIDDEN), dtype=dtype)
    y = jax.random.normal(ky, (2, 3, HIDDEN), dtype=dtype)
    c = jax.random.normal(kc, (2, 1, HIDDEN), dtype=dtype)
    # obs is a 4x3 grid -> 12 tokens; cond is length 3
    pe_x = precompute_freqs_cis_2d(HEAD_DIM, 4, 3)
    pe_y = precompute_freqs_cis_1d(HEAD_DIM, 3)
    return x, y, c, pe_x, pe_y


def test_output_shapes_preserved():
    block = _make_block(zero_init=False)
    x, y, c, pe_x, pe_y = _inputs()
    out_x, out_y = block(x, y, c, pe_x, pe_y)
    assert out_x.shape == x.shape
    assert out_y.shape == y.shape


def test_zero_init_is_exact_identity():
    block = _make_block(zero_init=True)
    x, y, c, pe_x, pe_y = _inputs()
    out_x, out_y = block(x, y, c, pe_x, pe_y)
    assert jnp.array_equal(out_x, x)
    assert jnp.array_equal(out_y, y)


def test_non_zero_init_is_not_identity():
    block = _make_block(zero_init=False)
    x, y, c, pe_x, pe_y = _inputs()
    out_x, out_y = block(x, y, c, pe_x, pe_y)
    assert not jnp.allclose(out_x, x)
    assert not jnp.allclose(out_y, y)


def test_pe_y_reaches_cond_stream():
    block = _make_block(zero_init=False)
    x, y, c, pe_x, pe_y = _inputs()
    out_none = block(x, y, c, pe_x, pe_y=None)
    out_pe = block(x, y, c, pe_x, pe_y=pe_y)
    # Rotating the cond stream changes both stream outputs (joint attention).
    assert not jnp.allclose(out_none[0], out_pe[0])
    assert not jnp.allclose(out_none[1], out_pe[1])


def test_gradient_flows_to_qkv_x():
    block = _make_block(zero_init=False)
    x, y, c, pe_x, pe_y = _inputs()

    graphdef, params, rest = nnx.split(block, nnx.Param, ...)

    def loss_fn(params):
        model = nnx.merge(graphdef, params, rest)
        out_x, out_y = model(x, y, c, pe_x, pe_y)
        return jnp.sum(out_x**2) + jnp.sum(out_y**2)

    grads = jax.grad(loss_fn)(params)
    g = grads["qkv_x"]["kernel"].get_value()
    assert jnp.any(g != 0)


# ---------------------------------------------------------------------------
# PiTBlock
# ---------------------------------------------------------------------------

# B=2, 3x4 patch grid -> L=12, p=2 -> p^2=4 pixels/patch
PIT_B = 2
PIT_GH, PIT_GW = 3, 4
PIT_L = PIT_GH * PIT_GW
PIT_P = 2
PIT_P2 = PIT_P * PIT_P
PIT_DPIX = 8
PIT_D = 64
PIT_ATTN = 64
PIT_HEADS = 4


def _make_pit_block(zero_init=True, post_modulation=False):
    return PiTBlock(
        PIT_DPIX,
        PIT_D,
        PIT_P,
        PIT_ATTN,
        PIT_HEADS,
        post_modulation=post_modulation,
        zero_init=zero_init,
        rngs=nnx.Rngs(0),
        param_dtype=jnp.float32,
    )


def _pit_inputs():
    kx, ks = jax.random.split(jax.random.key(2))
    x = jax.random.normal(kx, (PIT_B * PIT_L, PIT_P2, PIT_DPIX), dtype=jnp.float32)
    s_cond = jax.random.normal(ks, (PIT_B * PIT_L, PIT_D), dtype=jnp.float32)
    pe = precompute_freqs_cis_2d(PIT_ATTN // PIT_HEADS, PIT_GH, PIT_GW)
    return x, s_cond, pe


@pytest.mark.parametrize("post_modulation", [False, True])
def test_pit_output_shape_preserved(post_modulation):
    block = _make_pit_block(zero_init=False, post_modulation=post_modulation)
    x, s_cond, pe = _pit_inputs()
    out = block(x, s_cond, pe, PIT_B)
    assert out.shape == x.shape


def test_pit_pre_variant_zero_init_is_exact_identity():
    block = _make_pit_block(zero_init=True, post_modulation=False)
    x, s_cond, pe = _pit_inputs()
    out = block(x, s_cond, pe, PIT_B)
    assert jnp.array_equal(out, x)


def test_pit_post_variant_zero_init_is_not_identity():
    # Post variant has no gates: x + attn_exp * (1 + 0) + 0 != x.
    block = _make_pit_block(zero_init=True, post_modulation=True)
    x, s_cond, pe = _pit_inputs()
    out = block(x, s_cond, pe, PIT_B)
    assert not jnp.allclose(out, x)


def test_pit_cross_patch_mixing():
    # Compaction attention is global over the patch grid: perturbing patch 0's
    # pixels must change the output at another patch.
    block = _make_pit_block(zero_init=False)
    x, s_cond, pe = _pit_inputs()
    out_base = block(x, s_cond, pe, PIT_B)
    x_mod = x.at[0].add(10.0)  # patch (b=0, l=0)
    out_mod = block(x_mod, s_cond, pe, PIT_B)
    # Patch index 5 of the same batch element changed without its own input changing.
    assert jnp.array_equal(x[5], x_mod[5])
    assert not jnp.allclose(out_base[5], out_mod[5])
    # Other batch element (rows L..2L-1) untouched: no cross-batch leakage.
    assert jnp.allclose(out_base[PIT_L:], out_mod[PIT_L:])


def test_pit_s_cond_sensitivity():
    block = _make_pit_block(zero_init=False)
    x, s_cond, pe = _pit_inputs()
    s_cond2 = s_cond + 1.0
    out1 = block(x, s_cond, pe, PIT_B)
    out2 = block(x, s_cond2, pe, PIT_B)
    assert not jnp.allclose(out1, out2)


def test_pit_input_guard_pixel_dim():
    block = _make_pit_block(zero_init=True)
    x, s_cond, pe = _pit_inputs()
    bad_x = jnp.ones((PIT_B * PIT_L, PIT_P2, PIT_DPIX + 1), dtype=jnp.float32)
    with pytest.raises(ValueError, match="pixel_dim"):
        block(bad_x, s_cond, pe, PIT_B)


def test_pit_input_guard_batch_divisibility():
    block = _make_pit_block(zero_init=True)
    x, s_cond, pe = _pit_inputs()
    # BL=24, bad_batch=5 -> 24 % 5 != 0
    with pytest.raises(ValueError, match="not divisible"):
        block(x, s_cond, pe, 5)


def test_pit_per_pixel_modulation_layout():
    """Per-pixel modulation layout pin: only pixel 0's gate_mlp and shift_mlp
    are nonzero, so output differs from input ONLY at pixel row 0 of every
    patch; other pixel rows are untouched exactly (their gates remain zero).

    adaLN bias shape: (n_mod * D_pix * p^2,) reshaped to (p^2, n_mod * D_pix).
    Chunk order (pre-variant): 0=shift_msa, 1=scale_msa, 2=gate_msa,
    3=shift_mlp, 4=scale_mlp, 5=gate_mlp.
    Pixel k, chunk c spans flat indices [k * n_mod*D_pix + c*D_pix,
                                          k * n_mod*D_pix + (c+1)*D_pix).
    Pixel 0's shift_mlp (c=3): [24, 32); gate_mlp (c=5): [40, 48).
    """
    n_mod = 6
    bias_len = n_mod * PIT_DPIX * PIT_P2  # 6 * 8 * 4 = 192

    block = _make_pit_block(zero_init=True, post_modulation=False)

    # Hand-set only pixel 0's shift_mlp and gate_mlp entries.
    bias = jnp.zeros(bias_len, dtype=jnp.float32)
    # pixel 0, chunk 3 (shift_mlp): indices [24, 32)
    bias = bias.at[jnp.arange(24, 32)].set(1.0)
    # pixel 0, chunk 5 (gate_mlp): indices [40, 48)
    bias = bias.at[jnp.arange(40, 48)].set(1.0)
    block.adaLN.bias.set_value(bias)

    x, s_cond, pe = _pit_inputs()
    out = block(x, s_cond, pe, PIT_B)

    # Pixel rows k=1,2,3 must be unchanged exactly (gates and shifts are zero).
    for k in range(1, PIT_P2):
        assert jnp.array_equal(out[:, k, :], x[:, k, :]), (
            f"pixel row {k} changed but its modulation params are zero"
        )

    # Pixel row 0 must differ (gate_mlp nonzero, shift_mlp nonzero -> MLP fires).
    assert not jnp.array_equal(out[:, 0, :], x[:, 0, :]), (
        "pixel row 0 unchanged even though its gate_mlp/shift_mlp are nonzero"
    )
