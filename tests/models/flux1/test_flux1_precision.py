import jax
import jax.numpy as jnp
from flax import nnx

from gensbi.models import Flux1, Flux1Params
from tests.precision_utils import assert_tree_dtype


def _make(dtype=jnp.bfloat16):
    params = Flux1Params(
        in_channels=1, vec_in_dim=None, context_in_dim=1, mlp_ratio=2.0,
        num_heads=2, depth=1, depth_single_blocks=1, qkv_bias=False,
        rngs=nnx.Rngs(0), dim_obs=3, dim_cond=4, axes_dim=[4],
        dtype=dtype,
    )
    return Flux1(params)


def _inputs():
    obs = jnp.ones((2, 3, 1), jnp.float32)
    cond = jnp.ones((2, 4, 1), jnp.float32)
    obs_ids = jnp.tile(jnp.arange(3)[None, :, None], (2, 1, 1))
    cond_ids = jnp.tile(jnp.arange(4)[None, :, None], (2, 1, 1))
    t = jnp.full((2,), 0.5)
    return dict(t=t, obs=obs, obs_ids=obs_ids, cond=cond, cond_ids=cond_ids)


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
