import jax
import jax.numpy as jnp
from flax import nnx

from gensbi.models import MAFlow, MAFlowParams
from tests.precision_utils import assert_tree_dtype

# Captured from UNMODIFIED code (before this task's refactor) with:
#   PYTHONPATH=.../src JAX_PLATFORMS=cpu python -c "
#   import jax, jax.numpy as jnp
#   from flax import nnx
#   from gensbi.models import MAFlow, MAFlowParams
#   m = MAFlow(MAFlowParams(rngs=nnx.Rngs(0), dim=3, cond_dim=2))
#   x = jax.random.normal(jax.random.PRNGKey(1), (5, 3))
#   print(repr(m.log_prob(x, jnp.ones((5, 2)))))"
GOLDEN_LOG_PROBS = [-2.781574, -3.5820527, -5.5947814, -3.1017504, -3.853031]


def _make(**kw):
    return MAFlow(MAFlowParams(rngs=nnx.Rngs(0), dim=3, cond_dim=2, **kw))


def test_params_fp32():
    m = _make()
    assert_tree_dtype(nnx.to_pure_dict(nnx.state(m, nnx.Param)), jnp.float32)


def test_log_prob_fp32_and_finite():
    m = _make()
    x = jax.random.normal(jax.random.PRNGKey(1), (5, 3))
    cond = jnp.ones((5, 2))
    lp = m.log_prob(x, cond)
    assert lp.dtype == jnp.float32
    assert jnp.all(jnp.isfinite(lp))


def test_refactor_is_bit_identical():
    # Golden values computed from the pre-refactor code at the start of this
    # task; regenerate with the command in the plan and paste here.
    m = _make()
    x = jax.random.normal(jax.random.PRNGKey(1), (5, 3))
    cond = jnp.ones((5, 2))
    lp = m.log_prob(x, cond)
    expected = jnp.asarray(GOLDEN_LOG_PROBS)  # paste from Step 1 output
    assert jnp.array_equal(lp, expected), "MAF refactor must be a bit-exact no-op"
