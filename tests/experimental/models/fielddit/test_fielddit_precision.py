import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx

from gensbi.experimental.models.fielddit.model import FieldDiT, FieldDiTParams
from gensbi.experimental.models.fielddit.core import MMDiTCore
from gensbi.models.flux1.layers import DoubleStreamBlock
from tests.precision_utils import assert_tree_dtype


def _make(dtype=jnp.bfloat16):
    params = FieldDiTParams(
        in_channels=1,
        field_shape=(32, 32),
        encoder_widths=(8, 16, 32),  # D = 2
        cond_dim=3,
        rngs=nnx.Rngs(0),
        num_heads=2,
        axes_dim=[2, 2, 4],          # sum 8 -> hidden 16
        patch_size=2,
        dtype=dtype,
    )
    return FieldDiT(params)


def _inputs():
    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 32, 32, 1))
    cond = jax.random.normal(jax.random.PRNGKey(2), (2, 3, 1))
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
    # both are zero-init (conv_out) at init -> open the gates on both models
    # identically so a genuine non-trivial signal reaches the output.
    for m in (m32, m16):
        k = m.decoder.conv_out.kernel
        k[...] = jnp.ones_like(k[...])
        mod = m.encoder.down.layers[0].block.layers[0].mod.lin
        mod.kernel[...] = 0.01 * jnp.ones_like(mod.kernel[...])
    o32, o16 = m32(**_inputs()), m16(**_inputs())
    err = jnp.max(jnp.abs(o32 - o16)) / (jnp.max(jnp.abs(o32)) + 1e-6)
    assert err < 2e-2, f"bf16 compute deviates {err} from fp32"


def test_cond_stream_is_bf16_entering_first_double_block(monkeypatch):
    """Regression test: MMDiTCore's absolute cond-id embedding (FeatureEmbedder,
    kind="absolute") defaults to fp32 output. It is added directly to
    ``cond_tokens * sqrt(hidden_size)`` (``cond_tokens = cond_tokens *
    sqrt(hidden) + self.cond_ids_embedder(cond_ids)``) rather than being fed
    into a compute-dtype Linear first, so nothing self-heals the dtype: if the
    id embedder isn't given ``dtype=dtype`` explicitly, its fp32 output leaks
    into the joint cond+obs stream and JAX's bf16+fp32 promotion rule silently
    defeats the bf16 compute knob for the whole MMDiT core (same failure mode
    as the flux1joint / pixeldit cond-stream leaks).
    """
    model = _make(jnp.bfloat16)
    captured = {}
    orig_call = DoubleStreamBlock.__call__

    def spy(self, obs, cond, vec, pe=None, mask=None):
        captured.setdefault("cond_dtype", cond.dtype)
        captured.setdefault("obs_dtype", obs.dtype)
        return orig_call(self, obs, cond, vec, pe, mask)

    monkeypatch.setattr(DoubleStreamBlock, "__call__", spy)
    model(**_inputs())
    assert captured["obs_dtype"] == jnp.bfloat16, (
        f"expected the obs-token stream to stay bf16, got {captured['obs_dtype']}"
    )
    assert captured["cond_dtype"] == jnp.bfloat16, (
        f"expected the cond-token stream to stay bf16 entering the first "
        f"DoubleStreamBlock, got {captured['cond_dtype']} (MMDiTCore's absolute "
        f"cond-id embedding likely leaked fp32 into the cond stream)"
    )
