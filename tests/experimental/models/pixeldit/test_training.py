"""Learning gates: prove PixelDiT can train (gate 1 — gradient aliveness).

Gradient aliveness via parameter snapshot comparison.

WHY 6 STEPS — the ignition cascade with c2i zero-init (``pit_post_modulation=False``):

  Step 1: Only ``final_layer`` has nonzero grads.  The zero-init kernel of
          ``final_layer.linear`` gates the output to exactly zero; its gradient
          is the only non-trivial one.  The optimizer moves ``final_layer``
          away from zero, so the model output is no longer identically zero.

  Step 2: The output is now nonzero, so gradients reach the pixel pathway:
          ``pixel_embedder``, and all of ``pixel_blocks`` (PiT compress /
          expand / attn / mlp and the PiT adaLN projections).

  Step 3: ``s_cond`` (= the patch-level output ``s``) now flows through the
          non-trivial PiT adaLN, so its gradient propagates back through the
          patch-level pathway: ``patch_blocks`` adaLN/gates open, and the
          ``s_embedder`` and ``t_conditioner`` get non-trivial gradients.

  Step 4: With the patch-block adaLN gates now open, cond tokens influence
          ``s_N`` through the gated joint attention in MMDiTBlock.  ``cond_embedder``
          parameters therefore receive gradient for the first time.

  Steps 5-6: Margin (numerical noise from momentum, discrete gates not
             perfectly on at exactly step 4).

The ``pit_post_modulation=True`` variant has *no* gated residuals in PiTBlock,
so the pixel pathway ignites at step 1 (no gates close the residuals).  The
patch pathway follows at step 2 and cond at step 3.  Six steps gives
comfortable margin in both variants.

We train for 6 steps and then compare per-subtree parameter snapshots (before
vs. after).  The snapshot comparison is more robust than checking gradients at a
particular step: it survives floating-point cascades where a gradient is
theoretically nonzero but rounds to zero in a single step.
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import copy
import tempfile

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from gensbi.core import FlowMatchingMethod
from gensbi.experimental.models.pixeldit.model import PixelDiT, PixelDiTParams
from gensbi.experimental.recipes import FieldConditionalPipeline

H = W = 16
COND_DIM = 2
HIDDEN_SIZE = 64
NUM_HEADS = 4
PATCH_DEPTH = 2
PIXEL_DEPTH = 2
PATCH_SIZE = 4
PIXEL_HIDDEN_SIZE = 8

# Index of the last patch block (0-based).  The last block's y-stream output-side
# parameters are provably dead: the reference forward pass (pixeldit_t2i.py) discards
# the final ``y`` output after the patch-block loop — only ``s`` (= ``x``) is carried
# forward into the PiT pathway.  This is reference-faithful behaviour, not a bug.
# The exempt leaves are exactly those parameters whose only effect on the computation
# graph is via the discarded y output: they cannot receive a gradient and will never
# move.
_LAST_PATCH_IDX = PATCH_DEPTH - 1

# Exact set of dead leaves (measured empirically; must match precisely — the test
# asserts equality so a new freeze or an unexpected revival both fail).
_DEAD_LEAVES: frozenset = frozenset({
    # y-stream output projection — writes only to y.
    ("patch_blocks", _LAST_PATCH_IDX, "proj_y", "kernel"),
    ("patch_blocks", _LAST_PATCH_IDX, "proj_y", "bias"),
    # y-stream post-attention norm — only conditions the y MLP input.
    ("patch_blocks", _LAST_PATCH_IDX, "norm_y2", "scale"),
    # y-stream MLP — operates on the post-norm y and writes only to y.
    ("patch_blocks", _LAST_PATCH_IDX, "mlp_y", "w1", "kernel"),
    ("patch_blocks", _LAST_PATCH_IDX, "mlp_y", "w2", "kernel"),
    ("patch_blocks", _LAST_PATCH_IDX, "mlp_y", "w3", "kernel"),
    # y-stream QUERY norm — the y query affects only the y output slice of
    # joint attention; the y KEY norm affects x outputs so it is alive.
    ("patch_blocks", _LAST_PATCH_IDX, "qk_norm_y", "query_norm", "scale"),
})


def _params(pit_post_modulation: bool, seed: int = 0) -> PixelDiTParams:
    return PixelDiTParams(
        in_channels=1,
        field_shape=(H, W),
        cond_dim=COND_DIM,
        rngs=nnx.Rngs(seed),
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        patch_depth=PATCH_DEPTH,
        pixel_depth=PIXEL_DEPTH,
        patch_size=PATCH_SIZE,
        pixel_hidden_size=PIXEL_HIDDEN_SIZE,
        pit_post_modulation=pit_post_modulation,
        param_dtype=jnp.float32,
    )


def _snapshot(model) -> nnx.State:
    """Return a deep copy of every nnx.Param leaf as an nnx.State."""
    state = nnx.state(model, nnx.Param)
    # jax arrays are immutable; copy.deepcopy of a pytree of jax arrays is safe.
    return copy.deepcopy(state)


def _path_tuple(path) -> tuple:
    """Convert a jax tree path (sequence of keys) to a plain hashable tuple."""
    parts = []
    for key in path:
        # DictKey wraps dict keys; SequenceKey wraps list indices.
        k = key.key if hasattr(key, "key") else key
        # SequenceKey stores the index as .idx
        if hasattr(k, "idx"):
            parts.append(k.idx)
        else:
            parts.append(k)
    return tuple(parts)


def _frozen_leaves(before: nnx.State, after: nnx.State) -> set:
    """Return paths (as plain tuples) of every Param leaf that did not change."""
    frozen = set()
    before_paths = jax.tree_util.tree_leaves_with_path(before)
    after_paths = jax.tree_util.tree_leaves_with_path(after)
    for (path_b, leaf_b), (_, leaf_a) in zip(before_paths, after_paths):
        if jnp.array_equal(leaf_b, leaf_a):
            frozen.add(_path_tuple(path_b))
    return frozen


def _subtree_changed(before: nnx.State, after: nnx.State, key: str) -> tuple[bool, list]:
    """Return (all_changed, frozen_paths) for every Param leaf in ``key``.

    ``all_changed`` is True iff every leaf in the subtree changed.
    ``frozen_paths`` lists the path tuples of leaves that did not change,
    *excluding* the known dead leaves in ``_DEAD_LEAVES``.
    """
    before_paths = jax.tree_util.tree_leaves_with_path(before[key])
    after_paths = jax.tree_util.tree_leaves_with_path(after[key])
    frozen = []
    for (path_b, leaf_b), (_, leaf_a) in zip(before_paths, after_paths):
        if jnp.array_equal(leaf_b, leaf_a):
            # Reconstruct the full path relative to the model root for the
            # exemption check: the subtree was keyed by ``key`` in the state.
            full_path = (key,) + _path_tuple(path_b)
            if full_path not in _DEAD_LEAVES:
                frozen.append(full_path)
    return len(frozen) == 0, frozen


@pytest.mark.parametrize("pit_post_modulation", [False, True])
def test_every_subtree_trains_within_ignition_cascade(pit_post_modulation):
    """All subtrees move within 6 optimizer steps (ignition cascade, see module docstring).

    Parametrized over pit_post_modulation=False (pre-modulation, 6-chunk
    gated DiT adaLN, reference c2i default) and pit_post_modulation=True
    (post-modulation, 4-chunk gate-free variant from ref line 168).

    The post-modulation variant has *no* gated residuals in PiTBlock, so the
    pixel pathway ignites at step 1 (immediately after final_layer moves).
    The patch pathway follows at step 2 and cond at step 3.  Six steps gives
    comfortable margin in both variants.

    Every nnx.Param leaf in every subtree must change, *except* for the
    reference-faithful dead leaves in ``patch_blocks[last]`` (see ``_DEAD_LEAVES``).
    The test also asserts that the dead-leaf set is *exactly* ``_DEAD_LEAVES`` —
    it fails if a previously-dead leaf comes alive or a new one freezes.
    """
    model = PixelDiT(_params(pit_post_modulation=pit_post_modulation))
    optimizer = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)

    B = 4
    obs = jax.random.normal(jax.random.PRNGKey(1), (B, H, W, 1))
    cond = jax.random.normal(jax.random.PRNGKey(2), (B, COND_DIM, 1))
    t = jnp.full((B,), 0.5)
    target = jax.random.normal(jax.random.PRNGKey(3), (B, H, W, 1))

    # Snapshot parameters BEFORE any training.
    before = _snapshot(model)

    def loss_fn(m):
        return jnp.mean((m(t, obs, cond) - target) ** 2)

    # Train for 6 steps — covers the full ignition cascade (see module docstring).
    losses = []
    for _ in range(6):
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        losses.append(float(loss))
        optimizer.update(model, grads)

    # Snapshot parameters AFTER training.
    after = _snapshot(model)

    # Every loss must be finite.
    for i, l in enumerate(losses):
        assert jnp.isfinite(l), f"loss at step {i} is not finite: {l}"

    # Every named subtree must have EVERY Param leaf change (barring the exempt
    # dead leaves in _DEAD_LEAVES).
    subtrees = [
        "s_embedder",
        "cond_embedder",
        "t_conditioner",
        "patch_blocks",
        "pixel_blocks",
        "final_layer",
        "pixel_embedder",
    ]
    unexpectedly_frozen = []
    for k in subtrees:
        all_changed, frozen_paths = _subtree_changed(before, after, k)
        if not all_changed:
            unexpectedly_frozen.extend(frozen_paths)

    assert not unexpectedly_frozen, (
        f"pit_post_modulation={pit_post_modulation}: "
        f"param leaf/leaves never moved after 6 training steps (outside the known dead set): "
        f"{unexpectedly_frozen}. "
        "This indicates a detached buffer or broken gradient path — "
        "do NOT raise the step count without understanding why."
    )

    # Assert the dead-leaf set is EXACTLY _DEAD_LEAVES (no regressions, no
    # surprise revivals).  This detects both a previously-dead leaf coming
    # alive (would disappear from measured_dead) and a new freeze appearing
    # (would appear in measured_dead but not in _DEAD_LEAVES).
    measured_dead = _frozen_leaves(before, after)
    # Normalise _DEAD_LEAVES paths to match the measured format.
    expected_dead = frozenset(
        _DEAD_LEAVES  # already plain tuples
    )
    assert measured_dead == expected_dead, (
        f"pit_post_modulation={pit_post_modulation}: "
        f"dead-leaf set mismatch.\n"
        f"  measured_dead - expected_dead (new freezes): {measured_dead - expected_dead}\n"
        f"  expected_dead - measured_dead (revivals):    {expected_dead - measured_dead}"
    )


# ---------------------------------------------------------------------------
# Gate 2 — tiny overfit + cond sensitivity
# ---------------------------------------------------------------------------


class _Loop:
    """Iterable dataset yielding the same (obs, cond) batch forever."""

    def __init__(self, batch):
        self.batch = batch

    def __iter__(self):
        while True:
            yield self.batch


def _checkerboard(H, W):
    """Return a (H, W, 1) float32 checkerboard pattern (+1/-1)."""
    ii = jnp.arange(H)[:, None]
    jj = jnp.arange(W)[None, :]
    return (((ii + jj) % 2) * 2 - 1).astype(jnp.float32)[:, :, None]


def _gradient_field(H, W):
    """Return a (H, W, 1) float32 horizontal gradient from -1 to +1."""
    row = jnp.linspace(-1.0, 1.0, W, dtype=jnp.float32)   # (W,)
    field = jnp.broadcast_to(row[None, :], (H, W))          # (H, W)
    return field[:, :, None]                                 # (H, W, 1)


def _make_gate2_dataset():
    """4 (obs, cond) pairs where obs is a deterministic function of cond.

    cond[i] has shape (2, 1): token-0 weight and token-1 weight.
    obs = checkerboard * cond[0] + gradient * cond[1]

    Cond values are chosen to be structurally distinct so that
    RMSNorm inside the cond embedder does not annihilate the differences.
    """
    H = W = 16
    checker = _checkerboard(H, W)    # (H, W, 1)
    grad = _gradient_field(H, W)     # (H, W, 1)

    # 4 distinct (w0, w1) weight pairs
    weights = jnp.array([
        [2.0, 0.0],
        [0.0, 2.0],
        [2.0, 0.0],
        [0.0, 2.0],
    ], dtype=jnp.float32)           # (4, 2)

    cond = weights[:, :, None]      # (4, 2, 1) — each row is one cond token pair

    w0 = weights[:, 0, None, None, None]   # (4, 1, 1, 1)
    w1 = weights[:, 1, None, None, None]
    obs = w0 * checker[None] + w1 * grad[None]   # (4, H, W, 1)

    return obs, cond


def _make_gate2_pipeline(model_dir, seed=0):
    _H = _W = 16
    _COND_DIM = 2

    obs, cond = _make_gate2_dataset()
    batch = (obs, cond)

    training_config = FieldConditionalPipeline.get_default_training_config()
    training_config["checkpoint_dir"] = model_dir

    model = PixelDiT(PixelDiTParams(
        in_channels=1,
        field_shape=(_H, _W),
        cond_dim=_COND_DIM,
        rngs=nnx.Rngs(seed),
        hidden_size=64,
        num_heads=4,
        patch_depth=2,
        pixel_depth=1,
        patch_size=4,
        pixel_hidden_size=8,
        param_dtype=jnp.float32,
    ))

    return FieldConditionalPipeline(
        model=model,
        train_dataset=_Loop(batch),
        val_dataset=_Loop(batch),
        field_shape=(_H, _W),
        dim_cond=_COND_DIM,
        method=FlowMatchingMethod(),
        ch_obs=1,
        training_config=training_config,
    )


def test_tiny_overfit_and_cond_sensitivity():
    """Gate 2: overfit a cond-dependent field; loss must drop >=100x and
    structurally different conds must produce different outputs.

    Dataset: 4 (obs, cond) pairs where obs = checkerboard * cond[0] +
    gradient * cond[1].  Trained via the flow-matching pipeline loss with
    a warmup + cosine schedule for 3000 steps (~44 s on CPU).

    Loss check: the CFM objective is stochastic (random t and noise each step),
    so the final loss is evaluated as the mean over 64 fixed random keys to
    get a stable estimate.  The 100x threshold is consistent with the model
    overfitting the 4-sample batch to the point where the velocity field is
    well-approximated; 3000 steps with a cosine schedule reliably reaches it.

    Cond sensitivity uses structurally different tokens cond_A=[[1],[0]] vs
    cond_B=[[0],[1]] — RMSNorm cannot annihilate these (uniform shifts would
    be annihilated; see FieldDiT gate-2 lesson in the module docstring).
    """
    _NSTEPS = 3000

    with tempfile.TemporaryDirectory() as model_dir:
        pipeline = _make_gate2_pipeline(model_dir, seed=2)
        pipeline._wrap_model()

        loss_fn = pipeline.get_loss_fn()

        @nnx.jit
        def eval_loss(model, batch, key):
            """Average CFM loss over 64 keys for a stable estimate."""
            keys = jax.random.split(key, 64)
            losses = jax.vmap(lambda k: loss_fn(model, batch, k))(keys)
            return jnp.mean(losses)

        @nnx.jit
        def train_step(model, optimizer, batch, key):
            loss, grads = nnx.value_and_grad(loss_fn)(model, batch, key)
            optimizer.update(model, grads)
            return loss

        obs, cond = _make_gate2_dataset()
        batch = (obs, cond)

        # Warmup + cosine schedule: peak 5e-3 at step 50, decay to 1e-5.
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=5e-3,
            warmup_steps=50,
            decay_steps=_NSTEPS,
            end_value=1e-5,
        )
        optimizer = nnx.Optimizer(
            pipeline.model, optax.adam(schedule), wrt=nnx.Param
        )

        eval_key = jax.random.PRNGKey(99)
        loss_init = eval_loss(pipeline.model, batch, eval_key)

        key = jax.random.PRNGKey(42)
        for _ in range(_NSTEPS):
            key, subkey = jax.random.split(key)
            train_step(pipeline.model, optimizer, batch, subkey)

        loss_final = eval_loss(pipeline.model, batch, eval_key)

        assert loss_final < loss_init / 100.0, (
            f"loss did not drop 100x after {_NSTEPS} steps: "
            f"{float(loss_init):.4f} -> {float(loss_final):.4f} "
            f"(ratio {float(loss_init) / float(loss_final):.1f}x); "
            "this may indicate a model bug — do NOT weaken the threshold without investigating"
        )

        # Cond sensitivity: structurally different tokens produce different outputs.
        # cond_A puts all weight on token-0 (checkerboard-like target),
        # cond_B puts all weight on token-1 (gradient-like target).
        t_probe = jnp.full((1,), 0.5)
        obs_probe = obs[:1]  # use first sample's obs as probe input

        cond_A = jnp.array([[[1.0], [0.0]]])   # (1, 2, 1)
        cond_B = jnp.array([[[0.0], [1.0]]])   # (1, 2, 1)

        v_A = pipeline.model(t_probe, obs_probe, cond_A)
        v_B = pipeline.model(t_probe, obs_probe, cond_B)

        gap = float(jnp.mean(jnp.square(v_A - v_B)))
        assert gap > 0.01, (
            f"cond barely changes the output (mean squared diff = {gap:.6f}); "
            "cond path may be dead — do NOT weaken this threshold without investigating"
        )


# ---------------------------------------------------------------------------
# Gate 3 — opt-in realistic-config smoke
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("GENSBI_RUN_BIG_SMOKE"),
    reason="opt-in: set GENSBI_RUN_BIG_SMOKE=1 (slow, ~GBs of RAM)",
)
def test_realistic_64_config_smoke():
    """64^2 field, paper-B-ish scale: instantiate, run one forward + one grad step.

    Config mirrors a paper-B DiT scaled to a field workload:
      field_shape=(64,64), patch_size=8 -> 8x8 = 64 patch tokens
      hidden_size=768, num_heads=12, patch_depth=12
      pixel_depth=2, pixel_hidden_size=16 (pixel MLP; attn uses hidden_size)
      cond_dim=2

    Prints param count, peak output shape, and walltime with a [smoke] prefix
    (visible only with -s / --capture=no).
    """
    import time

    _H = _W = 64
    _COND_DIM = 2

    params = PixelDiTParams(
        in_channels=1,
        field_shape=(_H, _W),
        cond_dim=_COND_DIM,
        rngs=nnx.Rngs(0),
        patch_size=8,
        hidden_size=768,
        num_heads=12,
        patch_depth=12,
        pixel_depth=2,
        pixel_hidden_size=16,
        param_dtype=jnp.float32,
    )

    model = PixelDiT(params)
    n_params = sum(
        leaf.size for leaf in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))
    )
    print(f"\n[smoke] tokens={params.n_obs_tokens} params={n_params / 1e6:.1f}M")

    obs = jnp.zeros((1, _H, _W, 1), dtype=jnp.float32)
    cond = jnp.zeros((1, _COND_DIM, 1), dtype=jnp.float32)
    t = jnp.ones((1,), dtype=jnp.float32)
    target = jnp.ones((1, _H, _W, 1), dtype=jnp.float32)

    # One forward pass
    t0 = time.perf_counter()
    v = model(t, obs, cond)
    fwd_time = time.perf_counter() - t0
    print(f"[smoke] forward shape={v.shape} walltime={fwd_time:.2f}s")

    assert v.shape == (1, _H, _W, 1)
    assert jnp.all(jnp.isfinite(v))

    # One loss + grad step
    def loss_fn(m):
        return jnp.mean((m(t, obs, cond) - target) ** 2)

    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    t1 = time.perf_counter()
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    grad_time = time.perf_counter() - t1
    print(f"[smoke] loss={float(loss):.4f} grad_step_walltime={grad_time:.2f}s")

    assert jnp.isfinite(loss), f"loss is not finite: {loss}"
