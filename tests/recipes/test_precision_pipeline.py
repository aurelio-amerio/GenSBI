import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

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
    # Zero-init reference, matching optax.ema's actual raw (un-debiased)
    # accumulator semantics -- see the comment in
    # test_ema_integrates_small_updates_in_fp32 above. With the (incorrect)
    # ones-init reference this test passed regardless of the bf16 bug,
    # since the reference-offset alone (~0.6) swamped the real bf16 error;
    # with the correct zero-init reference: err_bf16~=0.0215 vs
    # err_fp32~=8.7e-6, so this is now genuinely diagnostic.
    import numpy as np
    w64, ema64 = np.ones(1), np.zeros(1)
    for _ in range(500):
        w64 = w64 * 1.001
        ema64 = decay * ema64 + (1 - decay) * w64
    err = abs(float(state.ema.astype(jnp.float32)[0]) - float(ema64[0]))
    assert err > 1e-2, "bf16 EMA unexpectedly accurate — did optax change?"


def test_restore_model_casts_bf16_checkpoint_to_fp32(tmp_path):
    """orbax restore_model must dtype-cast an old bf16-param_dtype checkpoint
    into a model whose current param_dtype is fp32 -- the orbax-restore
    sibling of test_bf16_checkpoint_loads_into_fp32_model (safetensors) in
    tests/utils/test_serialization_dtype.py. Exercises
    AbstractPipeline._cast_state_to_target_dtypes for both the model and
    EMA restore paths: since save_model() is called before any training
    step, ema_model is still an untrained nnx.clone(model), so both
    checkpoints (model/ and model/ema/) carry the same bf16 values and both
    restore paths are covered by the single round trip below.
    """
    from gensbi.core import FlowMatchingMethod
    from gensbi.recipes import UnconditionalPipeline

    class _MockParams:
        def __init__(self, dtype):
            self.param_dtype = dtype

    class _MockModel(nnx.Module):
        """Minimal model matching UnconditionalWrapper's call signature."""

        def __init__(self, dtype, value):
            super().__init__()
            self.params = _MockParams(dtype)
            self.kernel = nnx.Param(jnp.full((4,), value, dtype=dtype))

        def __call__(self, t, obs, node_ids, condition_mask=None, edge_mask=None):
            return jnp.zeros_like(obs)

    training_config = UnconditionalPipeline.get_default_training_config()
    training_config["checkpoint_dir"] = str(tmp_path)

    with pytest.warns(UserWarning, match="fp32"):
        old_pipeline = UnconditionalPipeline(
            model=_MockModel(jnp.bfloat16, 0.6),
            train_dataset=[],
            val_dataset=[],
            dim_obs=2,
            method=FlowMatchingMethod(),
            ch_obs=1,
            training_config=training_config,
        )
    old_pipeline.save_model()

    new_pipeline = UnconditionalPipeline(
        model=_MockModel(jnp.float32, 0.0),
        train_dataset=[],
        val_dataset=[],
        dim_obs=2,
        method=FlowMatchingMethod(),
        ch_obs=1,
        training_config=training_config,
    )
    new_pipeline.restore_model()

    expected = jnp.full((4,), 0.6, dtype=jnp.bfloat16).astype(jnp.float32)

    assert new_pipeline.model.kernel[...].dtype == jnp.float32
    assert new_pipeline.ema_model.kernel[...].dtype == jnp.float32
    assert jnp.allclose(new_pipeline.model.kernel[...], expected)
    assert jnp.allclose(new_pipeline.ema_model.kernel[...], expected)
