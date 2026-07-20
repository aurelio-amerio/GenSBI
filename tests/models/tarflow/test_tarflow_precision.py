import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from gensbi.models import TarFlow, TarFlowParams
from tests.precision_utils import assert_tree_dtype

# Captured from UNMODIFIED code (before this task's refactor) with:
#   PYTHONPATH=.../src JAX_PLATFORMS=cpu python -c "
#   import jax, jax.numpy as jnp
#   from flax import nnx
#   from gensbi.models import TarFlow, TarFlowParams
#   m = TarFlow(TarFlowParams(rngs=nnx.Rngs(0), dim=3, cond_dim=2, num_blocks=2,
#                              layers_per_block=1))
#   x = jax.random.normal(jax.random.PRNGKey(1), (5, 3, 1))
#   cond = jnp.ones((5, 2))
#   print(repr(m.log_prob(x, cond)))"
GOLDEN_LOG_PROBS = [-2.7815738, -3.5820527, -5.594781, -3.1017501, -3.853031]


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


def test_refactor_is_bit_identical():
    # Golden values computed from the pre-refactor code at the start of this
    # task; regenerate with the command in the module docstring above.
    m = _make()
    x = jax.random.normal(jax.random.PRNGKey(1), (5, 3, 1))
    cond = jnp.ones((5, 2))
    lp = m.log_prob(x, cond)
    expected = jnp.asarray(GOLDEN_LOG_PROBS)
    assert jnp.array_equal(lp, expected), "TarFlow refactor must be a bit-exact no-op"


def test_dtype_kwarg_accepted():
    # Pre-refactor this raises TypeError (no such field). Post-refactor it
    # must construct cleanly with the fp32 default knob.
    m = _make(param_dtype=jnp.float32, dtype=jnp.float32)
    assert_tree_dtype(nnx.to_pure_dict(nnx.state(m, nnx.Param)), jnp.float32)
