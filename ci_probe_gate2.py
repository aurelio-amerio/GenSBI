"""Loss ratio of the gate-2 overfit test vs. seed and step count.

Usage: python gate2_sweep.py SEED [MAX_STEPS]
Prints the ratio loss_init/loss_final at every 500 steps from 2000 to MAX_STEPS
(default 4000), using the same pipeline, schedule (cosine over 3000 steps,
exactly as the test) and eval as the test. Steps beyond 3000 run at the
schedule's end value (1e-5).
"""
import os
import sys
import tempfile

os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402
from flax import nnx  # noqa: E402

from tests.experimental.models.pixeldit import test_training as T  # noqa: E402

SEED = int(sys.argv[1])
MAX_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
NSTEPS = 3000
with tempfile.TemporaryDirectory() as model_dir:
    pipeline = T._make_gate2_pipeline(model_dir, seed=SEED)
    pipeline._wrap_model()
    loss_fn = pipeline.get_loss_fn()

    @nnx.jit
    def eval_loss(model, batch, key):
        keys = jax.random.split(key, 64)
        return jnp.mean(jax.vmap(lambda k: loss_fn(model, batch, k))(keys))

    @nnx.jit
    def train_step(model, optimizer, batch, key):
        loss, grads = nnx.value_and_grad(loss_fn)(model, batch, key)
        optimizer.update(model, grads)
        return loss

    obs, cond = T._make_gate2_dataset()
    batch = (obs, cond)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=5e-3, warmup_steps=50,
        decay_steps=NSTEPS, end_value=1e-5,
    )
    optimizer = nnx.Optimizer(pipeline.model, optax.adam(schedule), wrt=nnx.Param)
    eval_key = jax.random.PRNGKey(99)
    loss_init = float(eval_loss(pipeline.model, batch, eval_key))
    key = jax.random.PRNGKey(42)
    out = []
    for step in range(1, MAX_STEPS + 1):
        key, subkey = jax.random.split(key)
        train_step(pipeline.model, optimizer, batch, subkey)
        if step >= 2000 and step % 500 == 0:
            lf = float(eval_loss(pipeline.model, batch, eval_key))
            out.append(f"{step}:{loss_init / lf:.0f}x")
    print(f"jax {jax.__version__} seed={SEED} init={loss_init:.4f} " + " ".join(out))
