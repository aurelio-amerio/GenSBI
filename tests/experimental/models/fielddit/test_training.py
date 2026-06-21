"""Learning gates: prove FieldDiT can actually train (handoff B1).

Phase-1 tests prove the model is well-formed; these prove it is ALIVE:
gradients reach every subtree after the zero-init gates open, and conditioning
genuinely shapes the output after a tiny overfit.
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from gensbi.experimental.models import FieldDiT, FieldDiTParams

H = W = 16
COND_DIM = 1


def _tiny_model(seed=0):
    return FieldDiT(FieldDiTParams(
        in_channels=1,
        field_shape=(H, W),
        encoder_widths=(4, 8),
        cond_dim=COND_DIM,
        rngs=nnx.Rngs(seed),
        res_blocks_down=1,
        res_blocks_up=1,
        patch_size=2,
        num_heads=2,
        axes_dim=[2, 2, 4],
        depth=1,
        depth_single_blocks=1,
        param_dtype=jnp.float32,
    ))


def _subtree_grad_nonzero(grads_param_state, key):
    leaves = jax.tree_util.tree_leaves(grads_param_state[key])
    return any(bool(jnp.any(jnp.abs(leaf) > 0)) for leaf in leaves)


def test_one_step_revives_gradients_everywhere():
    """At init every block is AdaLN-zero gated and conv_out is zero-init, so the
    output is identically zero and only conv_out has a nonzero gradient. One
    optimizer step revives the output; gradient then reaches every subtree once
    the zero-init modulation gates have opened (two steps for the time/cond
    subtrees, whose signal reaches the output only through those gates)."""
    model = _tiny_model()
    optimizer = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)

    obs = jax.random.normal(jax.random.PRNGKey(1), (4, H, W, 1))
    cond = jax.random.normal(jax.random.PRNGKey(2), (4, COND_DIM, 1))
    t = jnp.full((4,), 0.5)
    target = jax.random.normal(jax.random.PRNGKey(3), (4, H, W, 1))

    def loss_fn(m):
        return jnp.mean((m(t, obs, cond) - target) ** 2)

    # step 0: only the zero-init output conv has nonzero grads (by design); the
    # update moves conv_out off zero.
    loss0, grads0 = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads0)

    # after one step the output is no longer identically zero ...
    v = model(t, obs, cond)
    assert not jnp.allclose(v, 0.0)

    # ... but time_vec and the cond summary reach the output ONLY through the
    # AdaLN modulation linears, which are themselves zero-init: they acquire
    # nonzero values from this first update, so the time/cond subtrees only get
    # gradient on the SECOND step. After it, gradient reaches EVERY subtree.
    grads1 = nnx.grad(loss_fn)(model)
    optimizer.update(model, grads1)

    grads2 = nnx.grad(loss_fn)(model)
    gstate = nnx.state(grads2, nnx.Param)
    for subtree in ("encoder", "tokenizer", "core", "untokenizer", "decoder",
                    "cond_embedder", "time_in"):
        assert _subtree_grad_nonzero(gstate, subtree), (
            f"no gradient reaches '{subtree}' after the gates open — "
            "a dead path the zero-init design would otherwise mask"
        )


def test_tiny_overfit_and_cond_sensitivity():
    """Overfit a cond-dependent target: loss must drop >=10x and different
    conds must produce different outputs (a dead cond path fails this)."""
    model = _tiny_model(seed=1)
    optimizer = nnx.Optimizer(model, optax.adam(3e-3), wrt=nnx.Param)

    # 4 samples, cond alternating 0/2; target = cond value painted everywhere
    obs = jax.random.normal(jax.random.PRNGKey(1), (4, H, W, 1))
    cond_vals = jnp.array([0.0, 2.0, 0.0, 2.0])
    cond = cond_vals[:, None, None]                      # (4, 1, 1)
    target = jnp.broadcast_to(cond_vals[:, None, None, None], (4, H, W, 1))
    t = jnp.full((4,), 0.5)

    @nnx.jit
    def train_step(model, optimizer):
        def loss_fn(m):
            return jnp.mean((m(t, obs, cond) - target) ** 2)
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    loss_init = train_step(model, optimizer)
    loss = loss_init
    for _ in range(300):
        loss = train_step(model, optimizer)

    assert loss < loss_init / 10.0, (
        f"loss did not drop 10x: {loss_init:.4f} -> {loss:.4f}"
    )

    # conditioning must shape the output: same obs, different cond
    v_lo = model(t[:1], obs[:1], jnp.zeros((1, 1, 1)))
    v_hi = model(t[:1], obs[:1], jnp.full((1, 1, 1), 2.0))
    gap = jnp.mean(jnp.abs(v_hi - v_lo))
    assert gap > 0.5, f"cond barely changes the output (mean |dv| = {gap:.4f})"


@pytest.mark.skipif(
    not os.environ.get("GENSBI_RUN_BIG_SMOKE"),
    reason="opt-in: set GENSBI_RUN_BIG_SMOKE=1 (slow, ~GBs of RAM)",
)
def test_realistic_256_config_smoke():
    """256^2 field, hidden 768 (defaults): instantiate, check the derived
    token budget, run one forward pass. Records what a real config costs."""
    params = FieldDiTParams(
        in_channels=1,
        field_shape=(256, 256),
        encoder_widths=(64, 128, 256, 256),   # D = 3 -> 32x32 meeting grid
        cond_dim=3,
        rngs=nnx.Rngs(0),
        patch_size=2,                          # -> 16x16 = 256 tokens
    )
    assert params.hidden_size == 768           # default axes [16,24,24] * 12 heads
    assert params.n_obs_tokens == 256

    model = FieldDiT(params)
    n_params = sum(
        leaf.size for leaf in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))
    )
    print(f"\n[smoke] tokens={params.n_obs_tokens} params={n_params/1e6:.1f}M")

    obs = jnp.zeros((1, 256, 256, 1), dtype=jnp.bfloat16)
    cond = jnp.zeros((1, 3, 1), dtype=jnp.bfloat16)
    v = model(jnp.ones((1,)), obs, cond)
    assert v.shape == (1, 256, 256, 1)
    assert jnp.all(jnp.isfinite(v))
