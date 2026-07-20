import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx

from gensbi.experimental.models.pixeldit.model import PixelDiT, PixelDiTParams
from gensbi.experimental.models.pixeldit.blocks import MMDiTBlock
from tests.precision_utils import assert_tree_dtype


def _make(dtype=jnp.bfloat16):
    params = PixelDiTParams(
        in_channels=1,
        field_shape=(16, 16),
        cond_dim=2,
        rngs=nnx.Rngs(0),
        hidden_size=64,
        num_heads=4,
        patch_depth=2,
        pixel_depth=2,
        patch_size=4,
        pixel_hidden_size=8,
        dtype=dtype,
    )
    return PixelDiT(params)


def _inputs():
    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 16, 16, 1))
    cond = jax.random.normal(jax.random.PRNGKey(2), (2, 2, 1))
    t = jnp.full((2,), 0.5)
    return dict(t=t, obs=obs, cond=cond)


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
    assert err < 2e-2, f"bf16 compute deviates {err} from fp32"


def test_cond_stream_is_bf16_entering_first_patch_block(monkeypatch):
    """Regression test: CondTokenEmbedder's RMSNorm island used to leak fp32
    into the cond-token stream (``y``) passed to the patch-level MMDiTBlocks.

    CondTokenEmbedder.__call__ does ``x = self.proj(cond); x = self.norm(x)``.
    The norm is a designated fp32 island (``dtype=jnp.float32``), so its raw
    output is fp32 regardless of the compute-dtype knob. Because that fp32
    value is returned directly (and later added to a learned id embedding)
    rather than being fed straight into another compute-dtype Linear, nothing
    downcasts it back -- unlike the flux1/flux1joint pattern where the norm
    output feeds a Linear whose own ``promote_dtype`` performs the downcast
    for free. Once the leak reaches ``cond_tokens`` (the ``y`` argument to
    MMDiTBlock), JAX's bf16+fp32 promotion rule means every subsequent
    ``y = y + gate * proj_y(...)`` residual re-promotes the whole cond stream
    back to fp32, silently defeating the bf16 compute knob for it.

    We patch ``MMDiTBlock.__call__`` at the class level (instance-level
    ``__call__`` overrides are not honored by Python's implicit call protocol)
    to capture the dtype of ``y`` reaching the very first patch block.
    """
    model = _make(jnp.bfloat16)
    captured = {}
    orig_call = MMDiTBlock.__call__

    def spy(self, x, y, c, pe_x, pe_y=None):
        captured.setdefault("y_dtype", y.dtype)
        captured.setdefault("x_dtype", x.dtype)
        return orig_call(self, x, y, c, pe_x, pe_y)

    monkeypatch.setattr(MMDiTBlock, "__call__", spy)
    model(**_inputs())
    assert captured["x_dtype"] == jnp.bfloat16, (
        f"expected the patch-token stream to stay bf16, got {captured['x_dtype']}"
    )
    assert captured["y_dtype"] == jnp.bfloat16, (
        f"expected the cond-token stream to stay bf16 entering the first "
        f"MMDiTBlock, got {captured['y_dtype']} (CondTokenEmbedder's RMSNorm "
        f"output likely leaked fp32 into the cond stream)"
    )
