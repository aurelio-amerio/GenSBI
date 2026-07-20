import jax
import jax.numpy as jnp
from flax import nnx

from gensbi.models.flux1joint.model import Flux1Joint, Flux1JointParams
from gensbi.models.flux1.layers import SingleStreamBlock
from tests.precision_utils import assert_tree_dtype


def _make(dtype=jnp.bfloat16):
    params = Flux1JointParams(
        in_channels=1,
        vec_in_dim=None,
        mlp_ratio=3.0,
        num_heads=2,
        depth_single_blocks=2,
        val_emb_dim=4,
        cond_emb_dim=2,
        id_emb_dim=4,
        qkv_bias=True,
        rngs=nnx.Rngs(0),
        dim_joint=4,
        id_merge_mode="sum",
        id_embedding_strategy="absolute",
        guidance_embed=False,
        dtype=dtype,
    )
    return Flux1Joint(params)


def _inputs():
    obs = jnp.ones((1, 4, 1), jnp.float32)
    t = jnp.ones((1, 1))
    node_ids = jnp.arange(4).reshape(1, -1, 1)
    condition_mask = jnp.zeros((1, 4, 1))
    return dict(t=t, obs=obs, node_ids=node_ids, condition_mask=condition_mask)


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


def test_single_block_stream_is_bf16_after_condition_merge(monkeypatch):
    """Regression test: the fp32-stored condition_embedding nnx.Param used to be
    mixed directly (uncast) into the compute-dtype obs/ids_embedding streams at
    the id_merge_mode="sum" merge point. JAX's bf16+fp32 promotion silently
    upgraded that merged stream -- and therefore every downstream
    SingleStreamBlock residual (`x + gate * output` re-promotes each time) -- to
    fp32, defeating the bf16 compute knob for the whole single-block stack even
    though params/output correctly measured fp32/bf16 in isolation.

    We patch SingleStreamBlock.__call__ at the class level (instance-level
    __call__ overrides are not honored by Python's implicit call protocol) to
    capture the dtype of the activation actually reaching the first block.
    """
    model = _make(jnp.bfloat16)
    captured = {}
    orig_call = SingleStreamBlock.__call__

    def spy(self, x, vec, pe=None, mask=None):
        captured.setdefault("dtype", x.dtype)
        return orig_call(self, x, vec=vec, pe=pe, mask=mask)

    monkeypatch.setattr(SingleStreamBlock, "__call__", spy)
    model(**_inputs())
    assert captured["dtype"] == jnp.bfloat16, (
        f"expected the activation entering the single-block stack to stay "
        f"bf16, got {captured['dtype']} (condition_embedding merge likely "
        f"promoted it back to fp32)"
    )
