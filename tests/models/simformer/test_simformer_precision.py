import jax.numpy as jnp
from flax import nnx

from gensbi.models.simformer.model import Simformer, SimformerParams
from gensbi.models.simformer.transformer import AttentionBlock
from tests.precision_utils import assert_tree_dtype


def _make(dtype=jnp.bfloat16):
    params = SimformerParams(
        rngs=nnx.Rngs(0),
        in_channels=1,
        val_emb_dim=2,
        id_emb_dim=2,
        cond_emb_dim=2,
        dim_joint=4,
        fourier_features=8,
        num_heads=2,
        depth=2,
        mlp_ratio=2,
        qkv_features=4,
        num_hidden_layers=1,
        dtype=dtype,
    )
    return Simformer(params)


def _inputs():
    x = jnp.ones((1, 4, 1))
    t = jnp.ones((1, 1))
    node_ids = jnp.arange(4).reshape(1, 4)
    condition_mask = jnp.zeros((1, 4, 1))
    return dict(t=t, obs=x, node_ids=node_ids, condition_mask=condition_mask)


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


def test_inter_block_residual_stream_is_bf16(monkeypatch):
    """Regression test: AttentionBlock/DenseBlock used to capture their skip
    value (`x_in`) post fp32-LayerNorm and add it back unconditionally. JAX's
    bf16+fp32 promotion on that skip-add silently re-promoted every block's
    output -- and therefore the whole inter-block residual stream -- back to
    fp32, defeating the bf16 compute knob even though the model's own final
    output correctly measured fp32 (via the separate emit-fp32 contract) and
    params/grads correctly measured fp32 in isolation.

    We patch AttentionBlock.__call__ at the class level (instance-level
    __call__ overrides are not honored by Python's implicit call protocol,
    same as the flux1joint precision regression test) to capture the dtype
    of the activation reaching the *second* attention block -- i.e. the
    stream that has already passed through one DenseBlock's fp32-island
    LayerNorm and skip connection.
    """
    model = _make(jnp.bfloat16)
    captured = []
    orig_call = AttentionBlock.__call__

    def spy(self, x, mask):
        captured.append(x.dtype)
        return orig_call(self, x, mask)

    monkeypatch.setattr(AttentionBlock, "__call__", spy)
    model(**_inputs())
    assert len(captured) == 2, f"expected 2 attention blocks, got {len(captured)}"
    assert captured[1] == jnp.bfloat16, (
        f"expected the activation entering the second attention block to "
        f"stay bf16, got {captured[1]} (fp32 skip-connection residual "
        f"likely re-promoted the inter-block stream)"
    )
