import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx

from gensbi.experimental.models.autoencoders.autoencoder_2d import AutoEncoder2D
from gensbi.experimental.models.autoencoders import AutoEncoderParams
from tests.precision_utils import assert_tree_dtype


def _make(dtype=jnp.bfloat16):
    # Smallest existing 2D fixture (mirrors test_autoencoder_2d.py).
    params = AutoEncoderParams(
        resolution=16,
        in_channels=3,
        ch=32,
        out_ch=3,
        ch_mult=[1, 1, 1],
        num_res_blocks=2,
        z_channels=8,
        scale_factor=1.0,
        shift_factor=0.0,
        rngs=nnx.Rngs(4),
        param_dtype=jnp.float32,
        dtype=dtype,
    )
    return AutoEncoder2D(params)


def _inputs():
    x = jnp.ones((4, 16, 16, 3))
    return dict(x=x, key=jax.random.PRNGKey(0))


def test_master_weights_fp32_with_bf16_compute():
    model = _make(jnp.bfloat16)
    assert_tree_dtype(nnx.to_pure_dict(nnx.state(model, nnx.Param)), jnp.float32)


def test_output_is_fp32():
    assert _make(jnp.bfloat16)(**_inputs()).dtype == jnp.float32


def test_grads_are_fp32():
    model = _make(jnp.bfloat16)

    def loss_fn(m):
        return jnp.mean(jnp.square(m(**_inputs())))

    grads = nnx.grad(loss_fn)(model)
    assert_tree_dtype(nnx.to_pure_dict(grads), jnp.float32)


def test_bf16_close_to_fp32():
    m32, m16 = _make(jnp.float32), _make(jnp.bfloat16)
    # same rngs seed -> identical fp32 master weights
    o32, o16 = m32(**_inputs()), m16(**_inputs())
    assert o16.dtype == jnp.float32
    err = jnp.max(jnp.abs(o32 - o16)) / (jnp.max(jnp.abs(o32)) + 1e-6)
    # Looser tolerance than the single-block precision tests elsewhere: this
    # fixture stacks ~9 ResnetBlock2D/AttnBlock2D layers across encoder +
    # decoder (3 resolution levels x 2 res blocks + mid blocks), so per-layer
    # bf16 rounding compounds more than in a shallow 2-4 layer model.
    assert err < 5e-2, f"bf16 compute deviates {err} from fp32"


def test_latent_stream_is_bf16_between_encode_and_decode():
    """Regression test: DiagonalGaussian is a designated fp32 island (its
    exp/log sampling math is numerically sensitive), and its output is then
    combined with ``scale_factor``/``shift_factor`` -- plain ``nnx.Param``
    scalars, not a Linear/Conv whose ``promote_dtype`` would downcast for
    free. Without an explicit cast at the use site, JAX's bf16+fp32
    promotion rule on that scale/shift arithmetic would silently re-promote
    the entire latent -- and therefore the whole decoder's input stream --
    back to fp32, defeating the bf16 compute knob for the decoder half of
    the model.
    """
    model = _make(jnp.bfloat16)
    z = model.encode(_inputs()["x"], key=_inputs()["key"])
    assert z.dtype == jnp.bfloat16, (
        f"expected the latent leaving encode() to stay bf16, got {z.dtype} "
        f"(DiagonalGaussian fp32-island output or scale_factor/shift_factor "
        f"likely leaked fp32 into the latent stream)"
    )
