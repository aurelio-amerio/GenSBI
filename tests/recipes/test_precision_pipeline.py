import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from gensbi.recipes.pipeline import _warn_if_not_fp32_master_weights


def test_warns_on_bf16_master_weights():
    model = nnx.Linear(2, 3, rngs=nnx.Rngs(0), param_dtype=jnp.bfloat16)
    with pytest.warns(UserWarning, match="fp32"):
        _warn_if_not_fp32_master_weights(model)


def test_silent_on_fp32_master_weights(recwarn):
    model = nnx.Linear(2, 3, rngs=nnx.Rngs(0), param_dtype=jnp.float32)
    _warn_if_not_fp32_master_weights(model)
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


def test_ema_integrates_small_updates_in_fp32():
    # The bug-shaped test: decay=0.999 EMA must integrate 0.1%-scale updates.
    decay = 0.999
    tx = optax.ema(decay)
    w = jnp.ones((64,), jnp.float32)
    state = tx.init(w)
    for _ in range(500):
        w = w * 1.001
        _, state = tx.update(w, state)
    # fp32 reference computed in float64. optax.ema's raw (un-debiased)
    # accumulator is zero-initialized (see optax EmaState.init_fn), not
    # seeded with the initial weight, so the reference recursion must start
    # at 0 too to be comparable to `state.ema`.
    import numpy as np
    w64, ema64 = np.ones(1), np.zeros(1)
    for _ in range(500):
        w64 = w64 * 1.001
        ema64 = decay * ema64 + (1 - decay) * w64
    # optax debiases; compare the raw accumulator
    assert abs(float(state.ema[0]) - float(ema64[0])) < 1e-3


def test_adamw_moments_are_fp32_for_fp32_params():
    # Spec 5(d): optimizer state must be fp32 once master weights are fp32.
    from tests.precision_utils import assert_tree_dtype
    model = nnx.Linear(2, 3, rngs=nnx.Rngs(0), param_dtype=jnp.float32,
                       dtype=jnp.bfloat16)
    opt = nnx.Optimizer(model, optax.adamw(1e-3), wrt=nnx.Param)
    mu = jax.tree.map(lambda x: x, opt.opt_state)  # traverse whole opt state
    leaves = [l for l in jax.tree.leaves(mu)
              if hasattr(l, "dtype") and jnp.issubdtype(l.dtype, jnp.floating)]
    assert leaves and all(l.dtype == jnp.float32 for l in leaves)


def test_ema_bf16_demonstrates_the_old_bug():
    # Documents WHY master weights must be fp32: the same accumulation in
    # bf16 diverges badly (increment below mantissa resolution + rounded decay).
    decay = 0.999
    tx = optax.ema(decay)
    w = jnp.ones((64,), jnp.bfloat16)
    state = tx.init(w)
    for _ in range(500):
        w = (w.astype(jnp.float32) * 1.001).astype(jnp.bfloat16)
        _, state = tx.update(w, state)
    import numpy as np
    w64, ema64 = np.ones(1), np.ones(1)
    for _ in range(500):
        w64 = w64 * 1.001
        ema64 = decay * ema64 + (1 - decay) * w64
    err = abs(float(state.ema.astype(jnp.float32)[0]) - float(ema64[0]))
    assert err > 1e-2, "bf16 EMA unexpectedly accurate — did optax change?"
