import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from gensbi.models import TarFlow, TarFlowParams
from tests.precision_utils import assert_tree_dtype


def _make(**kw):
    return TarFlow(TarFlowParams(rngs=nnx.Rngs(0), dim=3, cond_dim=2,
                                  num_blocks=2, layers_per_block=1, **kw))


def test_params_fp32():
    m = _make()
    assert_tree_dtype(nnx.to_pure_dict(nnx.state(m, nnx.Param)), jnp.float32)


def test_log_prob_fp32_and_finite():
    m = _make()
    x = jax.random.normal(jax.random.PRNGKey(1), (5, 3, 1))
    cond = jnp.ones((5, 2))
    lp = m.log_prob(x, cond)
    assert lp.dtype == jnp.float32
    assert jnp.all(jnp.isfinite(lp))


def test_bf16_compute_keeps_fp32_weights_and_output():
    # zero_init=False so the flow is not the identity (with the default zero_init the
    # log-prob is just the base N(0, I) density and the dtype knob has nothing to act on).
    m = _make(zero_init=False, dtype=jnp.bfloat16)
    assert_tree_dtype(nnx.to_pure_dict(nnx.state(m, nnx.Param)), jnp.float32)
    x = jax.random.normal(jax.random.PRNGKey(1), (5, 3, 1))
    lp = m.log_prob(x, jnp.ones((5, 2)))
    assert lp.dtype == jnp.float32
    assert jnp.all(jnp.isfinite(lp))


def test_bf16_compute_is_wired_and_close_to_fp32():
    # Same rngs seed -> identical fp32 master weights; only the compute dtype differs.
    x = jax.random.normal(jax.random.PRNGKey(1), (5, 3, 1))
    cond = jnp.ones((5, 2))
    lp32 = _make(zero_init=False, dtype=jnp.float32).log_prob(x, cond)
    lp16 = _make(zero_init=False, dtype=jnp.bfloat16).log_prob(x, cond)
    assert not jnp.array_equal(lp16, lp32), "TarFlow dtype knob does not reach the compute"
    err = jnp.max(jnp.abs(lp16 - lp32)) / jnp.max(jnp.abs(lp32))
    assert err < 2e-2, f"bf16 compute deviates {err} from fp32"


def test_dtype_kwarg_accepted():
    # Pre-refactor this raises TypeError (no such field). Post-refactor it
    # must construct cleanly with the fp32 default knob.
    m = _make(param_dtype=jnp.float32, dtype=jnp.float32)
    assert_tree_dtype(nnx.to_pure_dict(nnx.state(m, nnx.Param)), jnp.float32)
