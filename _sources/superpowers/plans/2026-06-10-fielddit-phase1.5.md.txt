# FieldDiT Phase 1.5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the well-formed Phase-1 FieldDiT into a trainable, verified-alive model: harden the module (review findings), wire it into a field-shaped conditional pipeline, and add learning-gate tests.

**Architecture:** Three workstreams executed in order: W1 model hardening (tasks 1–8, all inside `src/gensbi/experimental/models/fielddit/`), W3a test backfill that locks W1 in (tasks 9–10), W2 field pipeline (tasks 11–13, `src/gensbi/core/prior.py` + `src/gensbi/experimental/recipes/`), W3b learning gates (tasks 14–16). Spec: `docs/superpowers/specs/2026-06-10-fielddit-phase1.5-design.md`.

**Tech Stack:** JAX, flax.nnx, optax, numpyro distributions, pytest. All tests run with `JAX_PLATFORMS=cpu`.

**Conventions for every task:**
- Run tests from the repo root: `JAX_PLATFORMS=cpu uv run pytest <path> -x -q` (or `mamba activate gensbi` instead of `uv run`).
- Existing baseline: `tests/experimental/models/fielddit/` = 30 passed; `tests/experimental/models/ + tests/recipes/` = 216 passed. No regressions allowed.
- Branch: `FieldDiT`. Commit after every task with the message given in the task.
- Single dtype policy (spec §6): `param_dtype` is both storage and compute dtype, bf16 default. Do NOT add a separate compute-`dtype` argument anywhere.

**Model assignment (for the SDD orchestrator):**
- Default: a fast implementer model (e.g. Sonnet-class) per task — every step carries the literal code, paths, commands, and expected output; the work is transcription + verification.
- **Tasks 7 and 13 need a capable implementer model (Opus-class or better).** They are the two integration-risky tasks: Task 7 has a real-debugging contingency (hunting residual non-hashable static leaves through nnx internals if the graphdef hash test still fails after the known fixes), and Task 13 is first contact between FieldDiT and the training-loop machinery, where unforeseen API mismatches surface. Task 14's `cond_embedder` contingency is bounded (the resolution is written out); Sonnet-class is fine there.
- **HARD RULE for all implementers and reviewers:** the "if this fails, investigate — do NOT weaken the test" notes in Tasks 9, 14, and 15 are binding. A failing gate assertion is the bug these tasks exist to catch. Loosening a threshold, broadening an `allclose` tolerance, or skipping a subtree check to go green is a plan violation — report BLOCKED instead.

---

## Task 1: Loud failures in `FieldDiT.__call__` — `conditioned` and obs-shape guards

**Files:**
- Modify: `src/gensbi/experimental/models/fielddit/model.py` (the `__call__` method, lines ~141–175)
- Test: `tests/experimental/models/fielddit/test_model.py`

**Why:** `conditioned=False` is silently ignored today — an unconditional CFG pass would quietly return the conditional output. A wrong obs spatial size fails deep inside attention with a cryptic broadcast error.

- [ ] **Step 1: Write the failing tests** (append to `tests/experimental/models/fielddit/test_model.py`):

```python
def test_fielddit_conditioned_false_raises():
    """No unconditional path exists yet (CFG deferred): must fail loudly."""
    model = _small_model()
    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 32, 32, 1))
    cond = jax.random.normal(jax.random.PRNGKey(2), (2, 3, 1))
    t = jnp.ones((2,))
    with pytest.raises(NotImplementedError, match="unconditional"):
        model(t, obs, cond, conditioned=False)


def test_fielddit_rejects_wrong_spatial_shape():
    """obs spatial dims must match field_shape; fail at the door, not in attention."""
    model = _small_model()  # field_shape (32, 32)
    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 16, 16, 1))
    cond = jax.random.normal(jax.random.PRNGKey(2), (2, 3, 1))
    t = jnp.ones((2,))
    with pytest.raises(ValueError, match="field_shape"):
        model(t, obs, cond)


def test_fielddit_rejects_wrong_channel_count():
    model = _small_model()  # in_channels 1
    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 32, 32, 3))
    cond = jax.random.normal(jax.random.PRNGKey(2), (2, 3, 1))
    t = jnp.ones((2,))
    with pytest.raises(ValueError, match="in_channels"):
        model(t, obs, cond)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/experimental/models/fielddit/test_model.py -x -q -k "conditioned_false or wrong_spatial or wrong_channel"`
Expected: 3 FAILED (no exception raised / wrong exception).

- [ ] **Step 3: Implement the guards.** In `model.py` `__call__`, change the signature comment for `conditioned` and insert the guards. The top of `__call__` becomes:

```python
    def __call__(
        self,
        t,
        obs,
        cond,
        obs_ids=None,      # accepted & ignored (ids built internally)
        cond_ids=None,     # accepted & ignored
        conditioned=True,  # only True supported (CFG/null-cond is deferred)
        guidance=None,
    ):
        if conditioned is not True:
            raise NotImplementedError(
                "FieldDiT has no unconditional path yet (CFG / null-conditioning "
                f"is deferred work); got conditioned={conditioned!r}"
            )

        p = self.params
        obs = jnp.asarray(obs, dtype=p.param_dtype)
        cond = jnp.asarray(cond, dtype=p.param_dtype)
        t = jnp.asarray(t, dtype=p.param_dtype)

        if obs.shape[1:3] != tuple(p.field_shape):
            raise ValueError(
                f"obs spatial shape {obs.shape[1:3]} does not match "
                f"field_shape {tuple(p.field_shape)}"
            )
        if obs.shape[-1] != p.in_channels:
            raise ValueError(
                f"obs has {obs.shape[-1]} channels, expected in_channels={p.in_channels}"
            )
```

(The rest of `__call__` is unchanged in this task.)

- [ ] **Step 4: Run the module test suite**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/experimental/models/fielddit/ -q`
Expected: 33 passed (30 + 3 new).

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/experimental/models/fielddit/model.py tests/experimental/models/fielddit/test_model.py
git commit -m "feat(fielddit): raise on conditioned=False; guard obs shape/channels"
```

---

## Task 2: `ScalarCondEmbedder` 2D-input guard

**Files:**
- Modify: `src/gensbi/experimental/models/fielddit/cond.py`
- Test: `tests/experimental/models/fielddit/test_cond.py`

**Why:** The `(B, k)` shorthand is only valid when `cond_in_channels == 1`; today a multi-channel embedder fed 2D input errors cryptically downstream (documented-only caveat).

- [ ] **Step 1: Write the failing test** (append to `tests/experimental/models/fielddit/test_cond.py`):

```python
def test_scalar_cond_embedder_rejects_2d_when_multichannel():
    """(B, k) shorthand is only valid for in_channels == 1."""
    emb = ScalarCondEmbedder(in_channels=2, hidden_size=16, rngs=nnx.Rngs(0), param_dtype=jnp.float32)
    cond_2d = jnp.ones((2, 3))
    with pytest.raises(ValueError, match="cond_in_channels"):
        emb(cond_2d)


def test_scalar_cond_embedder_accepts_2d_when_single_channel():
    emb = ScalarCondEmbedder(in_channels=1, hidden_size=16, rngs=nnx.Rngs(0), param_dtype=jnp.float32)
    tokens, summary = emb(jnp.ones((2, 3)))
    assert tokens.shape == (2, 3, 16)
    assert summary.shape == (2, 16)
```

(If `test_cond.py` does not already import `pytest`, add `import pytest` at the top.)

- [ ] **Step 2: Run tests to verify the first fails**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/experimental/models/fielddit/test_cond.py -x -q`
Expected: FAIL — `rejects_2d_when_multichannel` (no ValueError raised; a flax shape error or silent success instead).

- [ ] **Step 3: Implement.** In `cond.py`, store `in_channels` and guard:

```python
    def __init__(self, in_channels: int, hidden_size: int, rngs: nnx.Rngs, param_dtype: DTypeLike = jnp.bfloat16):
        self.in_channels = in_channels
        self.token_proj = nnx.Linear(
            in_features=in_channels, out_features=hidden_size, use_bias=True,
            rngs=rngs, param_dtype=param_dtype,
        )
        self.summary_proj = nnx.Linear(
            in_features=hidden_size, out_features=hidden_size, use_bias=True,
            rngs=rngs, param_dtype=param_dtype,
        )

    def __call__(self, cond):
        if cond.ndim == 2:
            if self.in_channels != 1:
                raise ValueError(
                    f"2D cond (B, k) is only valid when cond_in_channels == 1; "
                    f"this embedder has cond_in_channels={self.in_channels} — "
                    f"pass cond as (B, k, {self.in_channels})"
                )
            cond = cond[..., None]
        tokens = self.token_proj(cond)
        summary = self.summary_proj(jnp.mean(tokens, axis=1))
        return tokens, summary
```

Also update the class docstring: replace the "the 2D form is only valid when `in_channels == 1`" caveat with "the 2D form raises unless `in_channels == 1`".

- [ ] **Step 4: Run tests**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/experimental/models/fielddit/ -q`
Expected: all pass (35).

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/experimental/models/fielddit/cond.py tests/experimental/models/fielddit/test_cond.py
git commit -m "feat(fielddit): ScalarCondEmbedder rejects 2D cond when multichannel"
```

---

## Task 3: Compute the timestep embedding in float32

**Files:**
- Modify: `src/gensbi/experimental/models/fielddit/model.py` (`__call__`)
- Test: `tests/experimental/models/fielddit/test_model.py`

**Why:** Today `t` is cast to bfloat16 *before* `timestep_embedding`, quantizing `t * 1000` to ~2.0 resolution near t=0.5 (t-resolution ≈ 0.002–0.004). The sinusoid must be computed in f32; only the resulting embedding is cast to the model dtype.

- [ ] **Step 1: Write the failing test** (append to `test_model.py`). It spies on the `timestep_embedding` call to assert the dtype actually received:

```python
def test_timestep_embedding_receives_f32(monkeypatch):
    """The sinusoidal embedding must be computed in f32 even for a bf16 model
    (bf16 t quantizes ~0.0005 differences in t away before the sinusoid)."""
    import gensbi.experimental.models.fielddit.model as fielddit_model

    seen = {}
    orig = fielddit_model.timestep_embedding

    def spy(t, dim, **kwargs):
        seen["dtype"] = t.dtype
        return orig(t, dim, **kwargs)

    monkeypatch.setattr(fielddit_model, "timestep_embedding", spy)

    model = FieldDiT(_params(rngs=nnx.Rngs(0), param_dtype=jnp.bfloat16))
    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 32, 32, 1))
    cond = jax.random.normal(jax.random.PRNGKey(2), (2, 3, 1))
    v = model(jnp.ones((2,)), obs, cond)
    assert seen["dtype"] == jnp.float32
    assert v.dtype == jnp.bfloat16  # model dtype unchanged downstream
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/experimental/models/fielddit/test_model.py::test_timestep_embedding_receives_f32 -x -q`
Expected: FAIL — `seen["dtype"] == bfloat16`.

- [ ] **Step 3: Implement.** In `__call__`, replace

```python
        t = jnp.asarray(t, dtype=p.param_dtype)

        time_vec = self.time_in(timestep_embedding(t, 256))  # (B, hidden)
```

with

```python
        # timestep sinusoid in f32 (bf16 t quantizes t*1000 to ~2.0 ulp);
        # cast only the finished embedding to the model dtype
        t = jnp.asarray(t, dtype=jnp.float32)
        time_vec = self.time_in(
            timestep_embedding(t, 256).astype(p.param_dtype)
        )  # (B, hidden)
```

- [ ] **Step 4: Run tests**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/experimental/models/fielddit/ -q`
Expected: all pass (36).

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/experimental/models/fielddit/model.py tests/experimental/models/fielddit/test_model.py
git commit -m "fix(fielddit): compute timestep embedding in f32 before casting"
```

---

## Task 4: Boundary LayerNorm in `Untokenizer`

**Files:**
- Modify: `src/gensbi/experimental/models/fielddit/codec.py` (`Untokenizer`)
- Test: `tests/experimental/models/fielddit/test_codec.py`

**Why:** Flux1 leaves the residual stream through `LastLayer` (norm + projection); FieldDiT projects the raw stream — whose magnitude grows with transformer depth — straight into the conv decoder. A pre-projection LayerNorm bounds it. Zero-at-init is unaffected (guaranteed by the decoder's zero-init `conv_out`).

- [ ] **Step 1: Write the failing test** (append to `test_codec.py`). LayerNorm is exactly scale-invariant at init, which discriminates norm-then-proj from plain proj:

```python
def test_untokenizer_normalizes_residual_stream():
    """A 100x token-magnitude blowup must not reach the conv decoder:
    LayerNorm before the projection makes the output scale-invariant at init."""
    untok = Untokenizer(
        out_channels=4, patch_size=2, hidden_size=16, rngs=nnx.Rngs(0), param_dtype=jnp.float32
    )
    tokens = jax.random.normal(jax.random.PRNGKey(0), (2, 16, 16))
    grid = (4, 4)
    out_small = untok(tokens, grid)
    out_big = untok(tokens * 100.0, grid)
    assert jnp.allclose(out_small, out_big, atol=1e-4)
```

Note: `Untokenizer`'s first arg is currently named `out_channels` already — check the constructor in `codec.py:219` and use the existing keyword.

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/experimental/models/fielddit/test_codec.py::test_untokenizer_normalizes_residual_stream -x -q`
Expected: FAIL — outputs differ by ~100x.

- [ ] **Step 3: Implement.** In `Untokenizer.__init__` add a LayerNorm before the projection, and apply it in `__call__`:

```python
    def __init__(self, out_channels: int, patch_size: int, hidden_size: int, rngs: nnx.Rngs, param_dtype: DTypeLike = jnp.bfloat16):
        self.patch_size = patch_size
        self.out_channels = out_channels
        # bound the transformer residual stream before re-entering conv space
        # (Flux1 exits through LastLayer's norm for the same reason)
        self.norm = nnx.LayerNorm(
            num_features=hidden_size,
            epsilon=1e-6,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.proj = nnx.Linear(
            in_features=hidden_size,
            out_features=out_channels * patch_size * patch_size,
            use_bias=True,
            rngs=rngs,
            param_dtype=param_dtype,
        )

    def __call__(self, tokens, grid):
        x = self.proj(self.norm(tokens))  # (B, N, C * p * p)
        return depatchify_2d(x, size=self.patch_size, grid=tuple(grid))
```

(If the constructor's first parameter is currently named differently, e.g. `out_channels` was already the name — keep it; only the `norm` addition is new.)

- [ ] **Step 4: Run tests**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/experimental/models/fielddit/ -q`
Expected: all pass (37). The model-level zero-at-init test must still pass.

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/experimental/models/fielddit/codec.py tests/experimental/models/fielddit/test_codec.py
git commit -m "feat(fielddit): LayerNorm before Untokenizer projection (boundary norm)"
```

---

## Task 5: Rename `ObsDecoder`'s `in_channels` → `out_channels`

**Files:**
- Modify: `src/gensbi/experimental/models/fielddit/codec.py` (`ObsDecoder.__init__`, docstring)
- Modify: `src/gensbi/experimental/models/fielddit/model.py:121-124` (instantiation)
- Modify: `tests/experimental/models/fielddit/test_codec.py:65` (keyword usage)

**Why:** The parameter is the *output* channel count of `conv_out`; the current name invites a wiring mistake.

- [ ] **Step 1: Rename.** In `codec.py` `ObsDecoder.__init__`, change the parameter `in_channels: int` to `out_channels: int` and the `conv_out` construction to `out_features=out_channels`. Update the class docstring's mention accordingly ("`out_channels` is the channel count of the produced velocity field").

- [ ] **Step 2: Update call sites.** In `model.py` the decoder is constructed positionally (`ObsDecoder(p.in_channels, p.encoder_widths, ...)`) — no change needed, but verify. In `test_codec.py:65` the test passes `in_channels=1` as a keyword — change to `out_channels=1`.

- [ ] **Step 3: Verify no other call sites**

Run: `grep -rn "ObsDecoder(" src tests`
Expected: only `codec.py` (definition), `model.py` (positional), `test_codec.py` (now `out_channels=`).

- [ ] **Step 4: Run tests**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/experimental/models/fielddit/ -q`
Expected: all pass (37).

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/experimental/models/fielddit/codec.py src/gensbi/experimental/models/fielddit/model.py tests/experimental/models/fielddit/test_codec.py
git commit -m "refactor(fielddit): rename ObsDecoder in_channels -> out_channels"
```

---

## Task 6: `theta` default rule (10× tokens, capped at 10k)

**Files:**
- Modify: `src/gensbi/experimental/models/fielddit/model.py` (`FieldDiTParams`)
- Test: `tests/experimental/models/fielddit/test_model.py`

**Why:** Fixed `theta=10000` leaves most rope channels near-constant on small meeting grids. Rule of thumb (user-provided): `theta = 10 × token count`, capped at 10 000. An explicit `theta` always wins.

- [ ] **Step 1: Write the failing test** (append to `test_model.py`):

```python
def test_theta_default_derives_from_token_count():
    # default test config: 16 obs tokens + 3 cond tokens -> theta = 190
    p = _params()
    assert p.theta == 10 * (p.n_obs_tokens + p.cond_dim) == 190


def test_theta_explicit_override_wins():
    p = _params(theta=777)
    assert p.theta == 777


def test_theta_default_capped_at_10k():
    # 64x64 meeting grid (field 256, D=1, p=2 -> 128x128 feat -> 64x64 grid = 4096 tokens)
    p = _params(field_shape=(256, 256), encoder_widths=(8, 16))
    assert p.theta == 10_000
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/experimental/models/fielddit/test_model.py -x -q -k theta`
Expected: FAIL — `theta == 10000` for the first test (current fixed default).

- [ ] **Step 3: Implement.** In `FieldDiTParams`, change the field declaration

```python
    theta: int = 10000
```

to

```python
    theta: Optional[int] = None  # None -> min(10 * (n_obs_tokens + cond_dim), 10_000)
```

and at the END of `__post_init__` (after `self.n_obs_tokens` is computed) add:

```python
        if self.theta is None:
            # rule of thumb: 10x the joint token count, capped at the
            # literature default; rope frequency coverage then matches the
            # actual grid instead of assuming ~10k positions
            self.theta = min(10 * (self.n_obs_tokens + self.cond_dim), 10_000)
```

- [ ] **Step 4: Run tests**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/experimental/models/fielddit/ -q`
Expected: all pass (40).

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/experimental/models/fielddit/model.py tests/experimental/models/fielddit/test_model.py
git commit -m "feat(fielddit): derive theta default from token count (10x, cap 10k)"
```

---

## Task 7: GraphDef hygiene — no dataclass on the module, ids as `RopeIds` Variables

**Files:**
- Modify: `src/gensbi/experimental/models/fielddit/model.py` (most of the file)
- Modify: `src/gensbi/experimental/models/fielddit/core.py:45` (one line: tuple for `EmbedND`)
- Modify: `src/gensbi/experimental/models/fielddit/__init__.py` (export `RopeIds`)
- Test: `tests/experimental/models/fielddit/test_model.py`

**Why (verified by review):** storing `FieldDiTParams` on the module makes the GraphDef unhashable and never-equal across instances (`nnx.jit` retraces per instance — bites the pipeline's EMA pattern). Raw `jnp` int arrays as module attributes become anonymous state leaves that a blanket float-cast over the state would silently corrupt.

- [ ] **Step 1: Write the failing tests** (append to `test_model.py`):

```python
def test_graphdef_hashable_and_equal_across_instances():
    """Two identically-configured models must share a hashable, equal GraphDef
    (otherwise nnx.jit retraces per instance — EMA/eval patterns pay twice)."""
    m1 = FieldDiT(_params(rngs=nnx.Rngs(0)))
    m2 = FieldDiT(_params(rngs=nnx.Rngs(1)))
    g1, _ = nnx.split(m1)
    g2, _ = nnx.split(m2)
    hash(g1)  # must not raise
    assert g1 == g2


def test_rope_ids_are_filterable_variables():
    """obs/cond ids live in a dedicated Variable type: excluded from Param
    state and immune to blanket float casts over Params."""
    from gensbi.experimental.models.fielddit import RopeIds

    model = _small_model()
    ids_state = nnx.state(model, RopeIds)
    leaves = jax.tree_util.tree_leaves(ids_state)
    assert len(leaves) == 2  # obs_ids, cond_ids
    assert all(l.dtype == jnp.int32 for l in leaves)
    # and they are NOT in the Param state
    param_leaves = jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))
    assert all(jnp.issubdtype(l.dtype, jnp.floating) for l in param_leaves)


def test_model_does_not_store_params_dataclass():
    model = _small_model()
    assert not hasattr(model, "params")
```

- [ ] **Step 2: Update the two existing tests that touch ids on params.** In `test_params_derive_hidden_and_grid`, REMOVE the two lines

```python
    assert p.obs_ids.shape == (1, 16, 3)
    assert p.cond_ids.shape == (1, 3, 1)
```

and add a new model-level test right after it:

```python
def test_model_builds_rope_ids():
    model = _small_model()
    assert model.obs_ids[...].shape == (1, 16, 3)
    assert model.cond_ids[...].shape == (1, 3, 1)
```

- [ ] **Step 3: Run tests to verify the new ones fail**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/experimental/models/fielddit/test_model.py -x -q -k "graphdef or rope_ids or dataclass or builds_rope"`
Expected: FAIL — `hash(g1)` raises `TypeError: unhashable type: 'FieldDiTParams'`; `RopeIds` import fails.

- [ ] **Step 4: Implement in `model.py`.**

(a) Add the Variable subclass right after the imports:

```python
class RopeIds(nnx.Variable):
    """Integer rope/positional id buffers.

    A dedicated Variable type so the ids are (a) filterable with
    ``nnx.state(model, RopeIds)``, (b) excluded from ``nnx.Param`` state, and
    (c) safe from blanket float-dtype casts applied to the parameter state.
    """
```

(b) In `FieldDiTParams.__post_init__`, DELETE the two id-construction lines:

```python
        self.obs_ids, _ = init_ids_2d((self.feat_h, self.feat_w), semantic_id=0, size=p)
        self.cond_ids, _ = init_ids_1d(self.cond_dim, semantic_id=None)
```

(c) Resolve the stale comment. Replace

```python
        if self.axes_dim is None:
            self.axes_dim = [16, 24, 24] # Double check this, I think we use the semantic id on the last dimension, so we might want to use less features for that
```

with

```python
        if self.axes_dim is None:
            # (semantic, h, w) split of the rope head dims. In Phase 1 the
            # semantic axis is identical (unrotated) for every token: obs uses
            # semantic_id=0 and cond tokens use learned absolute embeddings
            # with zero rope ids — benign, the same situation as Flux1
            # txt/img. The semantic-dim budget and the 1D/2D id axis-order
            # unification are revisited wholesale in the Phase-2
            # co-tokenization design (see the 2026-06-10 phase-1.5 spec §2.6).
            self.axes_dim = [16, 24, 24]
```

and after the per-entry evenness assert, normalize: `self.axes_dim = tuple(self.axes_dim)`.

(d) Document the `rngs` semantics in the `FieldDiTParams` docstring — append:

```
    Note: ``rngs`` is a live ``nnx.Rngs`` stream (mirrors ``Flux1Params``).
    Constructing two models from the *same* params object yields *different*
    weights, because the stream advances; build a fresh ``FieldDiTParams``
    (or a fresh ``nnx.Rngs(seed)``) per model for reproducibility.
```

(e) In `FieldDiT.__init__`, replace `self.params = params` and the tail of the constructor. The constructor keeps using the local `p = params` for construction, and ends with:

```python
        # static primitives needed at call time (the dataclass itself is NOT
        # stored: it holds Rngs/derived arrays and would poison the GraphDef)
        self.field_shape = tuple(p.field_shape)
        self.in_channels = p.in_channels
        self.cond_dim = p.cond_dim
        self.use_cond_summary_in_vec = p.use_cond_summary_in_vec
        self.guidance_embed = p.guidance_embed
        self.param_dtype = p.param_dtype
        self.token_grid = tuple(p.token_grid)

        # rope id buffers (int32) — built here, kept out of Param state
        obs_ids, _ = init_ids_2d((p.feat_h, p.feat_w), semantic_id=0, size=p.patch_size)
        cond_ids, _ = init_ids_1d(p.cond_dim, semantic_id=None)
        self.obs_ids = RopeIds(obs_ids)
        self.cond_ids = RopeIds(cond_ids)
```

Remove the old `self.params = params` (top) and the old precomputed-ids block (bottom). Keep `p = params` as a local at the top of `__init__`.

(f) Rewrite `__call__` to use the module attributes (full method, incorporating tasks 1 and 3):

```python
    def __call__(
        self,
        t,
        obs,
        cond,
        obs_ids=None,      # accepted & ignored (ids built internally)
        cond_ids=None,     # accepted & ignored
        conditioned=True,  # only True supported (CFG/null-cond is deferred)
        guidance=None,
    ):
        if conditioned is not True:
            raise NotImplementedError(
                "FieldDiT has no unconditional path yet (CFG / null-conditioning "
                f"is deferred work); got conditioned={conditioned!r}"
            )

        obs = jnp.asarray(obs, dtype=self.param_dtype)
        cond = jnp.asarray(cond, dtype=self.param_dtype)

        if obs.shape[1:3] != self.field_shape:
            raise ValueError(
                f"obs spatial shape {obs.shape[1:3]} does not match "
                f"field_shape {self.field_shape}"
            )
        if obs.shape[-1] != self.in_channels:
            raise ValueError(
                f"obs has {obs.shape[-1]} channels, expected in_channels={self.in_channels}"
            )

        # timestep sinusoid in f32 (bf16 t quantizes t*1000 to ~2.0 ulp);
        # cast only the finished embedding to the model dtype
        t = jnp.asarray(t, dtype=jnp.float32)
        time_vec = self.time_in(
            timestep_embedding(t, 256).astype(self.param_dtype)
        )  # (B, hidden)

        cond_tokens, summary = self.cond_embedder(cond)  # (B, k, hidden), (B, hidden)
        assert cond_tokens.shape[1] == self.cond_dim, (
            f"cond has {cond_tokens.shape[1]} tokens but cond_dim={self.cond_dim}"
        )

        vec = time_vec
        if self.use_cond_summary_in_vec:
            vec = vec + summary
        if self.guidance_embed:
            if guidance is None:
                raise ValueError("guidance required when guidance_embed=True")
            vec = vec + self.guidance_in(guidance)

        feat, pos_skips, neg_skips = self.encoder(obs, time_vec)   # time-only modulation
        obs_tokens = self.tokenizer(feat)
        obs_tokens = self.core(
            obs_tokens, cond_tokens, vec, self.obs_ids[...], self.cond_ids[...]
        )
        feat = self.untokenizer(obs_tokens, self.token_grid)
        v = self.decoder(feat, vec, pos_skips, neg_skips)          # time + cond modulation
        return v
```

(g) In `core.py:45`, change `axes_dim=list(axes_dim)` to `axes_dim=tuple(axes_dim)` in the `EmbedND` construction (a list attribute on a submodule would re-poison the GraphDef hash).

(h) In `src/gensbi/experimental/models/fielddit/__init__.py`, add `RopeIds` to the re-exports (alongside `FieldDiT`, `FieldDiTParams`).

- [ ] **Step 5: Run the full module suite**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/experimental/models/fielddit/ -q`
Expected: all pass (44: 40 + 4 new, with `test_params_derive_hidden_and_grid` trimmed).

**If `hash(g1)` still fails:** print `repr(g1)` and look for remaining non-hashable static leaves (lists, dataclasses, arrays stored as plain attributes) on submodules; convert them to tuples/primitives the same way. `EmbedND.axes_dim` (step g) is the known one.

- [ ] **Step 6: Run the broader regression** (graphdef change touches everything):

Run: `JAX_PLATFORMS=cpu uv run pytest tests/experimental/models/ tests/recipes/ -q`
Expected: 216+ passed, no failures.

- [ ] **Step 7: Commit**

```bash
git add src/gensbi/experimental/models/fielddit/ tests/experimental/models/fielddit/test_model.py
git commit -m "refactor(fielddit): hashable GraphDef — drop params from module, RopeIds variables"
```

---

## Task 8: `cond_modulates_encoder` flag

**Files:**
- Modify: `src/gensbi/experimental/models/fielddit/model.py` (`FieldDiTParams` + `__call__`)
- Test: `tests/experimental/models/fielddit/test_model.py`

**Why (spec §2.5):** default OFF keeps the encoder condition-free (CFG branches can share the encoder pass; encoder stays a pure noisy-field feature extractor for Phase-2). ON makes encoder/decoder modulation fully symmetric — the escape hatch that turns handoff risk B4 into a GRF ablation.

- [ ] **Step 1: Write the failing tests** (append to `test_model.py`):

```python
def _open_gates(model):
    """Surgery so cond can reach the output at all: open the zero-init output
    conv and one encoder-stage modulation (everything is gated shut at init)."""
    k = model.decoder.conv_out.kernel
    k[...] = jnp.ones_like(k[...])
    mod = model.encoder.down.layers[0].block.layers[0].mod.lin
    mod.kernel[...] = 0.01 * jnp.ones_like(mod.kernel[...])


def test_cond_modulates_encoder_routes_cond_through_encoder():
    model = FieldDiT(_params(rngs=nnx.Rngs(0), cond_modulates_encoder=True))
    _open_gates(model)
    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 32, 32, 1))
    t = jnp.ones((2,))
    cond_a = jnp.zeros((2, 3, 1))
    cond_b = jnp.ones((2, 3, 1))
    v_a = model(t, obs, cond_a)
    v_b = model(t, obs, cond_b)
    # encoder modulation sees vec (incl. cond summary) -> output must differ
    assert not jnp.allclose(v_a, v_b)


def test_encoder_is_cond_free_by_default():
    model = FieldDiT(_params(rngs=nnx.Rngs(0)))  # flag off
    _open_gates(model)
    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 32, 32, 1))
    t = jnp.ones((2,))
    v_a = model(t, obs, jnp.zeros((2, 3, 1)))
    v_b = model(t, obs, jnp.ones((2, 3, 1)))
    # only the encoder path is opened; with a cond-free encoder (and all other
    # gates still zero-init) the cond cannot reach the output
    assert jnp.allclose(v_a, v_b)


def test_cond_modulates_encoder_preserves_zero_at_init():
    model = FieldDiT(_params(rngs=nnx.Rngs(0), cond_modulates_encoder=True))
    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 32, 32, 1))
    cond = jax.random.normal(jax.random.PRNGKey(2), (2, 3, 1))
    v = model(jnp.ones((2,)), obs, cond)
    assert jnp.allclose(v, 0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/experimental/models/fielddit/test_model.py -x -q -k cond_modulates`
Expected: FAIL — `FieldDiTParams` has no field `cond_modulates_encoder`.

- [ ] **Step 3: Implement.**

(a) `FieldDiTParams`: add the field (next to `use_cond_summary_in_vec`):

```python
    cond_modulates_encoder: bool = False  # ON: encoder gets the full vec (symmetric with decoder); OFF: time-only encoder, shareable across CFG branches
```

(b) `FieldDiT.__init__`: copy it with the other primitives:

```python
        self.cond_modulates_encoder = p.cond_modulates_encoder
```

(c) `FieldDiT.__call__`: replace the encoder call line with:

```python
        enc_vec = vec if self.cond_modulates_encoder else time_vec
        feat, pos_skips, neg_skips = self.encoder(obs, enc_vec)
```

(d) Update the `FieldDiT` class docstring: "The conv encoder is modulated by time only (or by the full ``vec`` when ``cond_modulates_encoder=True``); ..."

- [ ] **Step 4: Run tests**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/experimental/models/fielddit/ -q`
Expected: all pass (47).

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/experimental/models/fielddit/model.py tests/experimental/models/fielddit/test_model.py
git commit -m "feat(fielddit): cond_modulates_encoder flag (default off, B4 ablation hatch)"
```

---

## Task 9: Core de-identity test (cond-sensitivity + rope-activity)

**Files:**
- Test: `tests/experimental/models/fielddit/test_core.py`

**Why (verified by review):** all Flux1 blocks are AdaLN-zero gated, so `MMDiTCore(obs, …) == obs` bit-exactly at init — every existing core test exercises an identity map. With randomized params the core is cond- and position-sensitive, but no test pins it.

- [ ] **Step 1: Write the test** (append to `test_core.py`; reuse the file's existing core-construction helper if one exists, otherwise construct as below):

```python
from gensbi.recipes.utils import init_ids_1d, init_ids_2d


def _randomized_core(seed=0):
    """MMDiTCore with all float params replaced by small random values, so the
    AdaLN-zero gates are open and attention/rope/cond paths actually execute."""
    core = MMDiTCore(
        hidden_size=16, num_heads=2, mlp_ratio=2.0, depth=1, depth_single_blocks=1,
        axes_dim=[2, 2, 4], theta=100, n_cond_tokens=3, qkv_bias=False,
        rngs=nnx.Rngs(seed), param_dtype=jnp.float32,
    )
    graphdef, state = nnx.split(core)
    counter = iter(range(100_000))

    def _rand(x):
        if jnp.issubdtype(x.dtype, jnp.floating):
            k = jax.random.fold_in(jax.random.PRNGKey(42), next(counter))
            return 0.05 * jax.random.normal(k, x.shape, x.dtype)
        return x

    state = jax.tree_util.tree_map(_rand, state)
    return nnx.merge(graphdef, state)


def test_core_with_open_gates_is_cond_sensitive_and_rope_active():
    core = _randomized_core()
    B, hid = 2, 16
    obs_ids, n_obs = init_ids_2d((8, 8), semantic_id=0, size=2)  # 16 tokens
    cond_ids, _ = init_ids_1d(3, semantic_id=None)
    obs_tokens = jax.random.normal(jax.random.PRNGKey(1), (B, n_obs, hid))
    vec = jax.random.normal(jax.random.PRNGKey(2), (B, hid))
    cond_a = jax.random.normal(jax.random.PRNGKey(3), (B, 3, hid))
    cond_b = cond_a + 1.0

    out_a = core(obs_tokens, cond_a, vec, obs_ids, cond_ids)
    out_b = core(obs_tokens, cond_b, vec, obs_ids, cond_ids)
    assert jnp.all(jnp.isfinite(out_a))
    # the cond value path must reach the obs stream
    assert not jnp.allclose(out_a, out_b)

    # rope must break obs-token permutation equivariance: permuting the input
    # tokens (with FIXED position ids) must NOT just permute the output
    perm = jax.random.permutation(jax.random.PRNGKey(4), n_obs)
    out_perm = core(obs_tokens[:, perm, :], cond_a, vec, obs_ids, cond_ids)
    assert not jnp.allclose(out_perm, out_a[:, perm, :], atol=1e-5)
```

- [ ] **Step 2: Run the test**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/experimental/models/fielddit/test_core.py -x -q`
Expected: PASS (this pins existing-but-untested behavior; if it FAILS, that is a real wiring bug — stop and investigate before continuing).

- [ ] **Step 3: Commit**

```bash
git add tests/experimental/models/fielddit/test_core.py
git commit -m "test(fielddit): core de-identity — cond sensitivity and rope activity with open gates"
```

---

## Task 10: Model test backfill — non-square, patch_size=1, depth-1, split/merge + jit

**Files:**
- Test: `tests/experimental/models/fielddit/test_model.py`

**Why:** All four configurations work today (verified) but nothing pins them; non-square fields exercise the `depatchify_2d` grid argument and the `(h, w)` axis-order footguns.

- [ ] **Step 1: Write the tests** (append to `test_model.py`):

```python
def test_fielddit_non_square_field():
    model = FieldDiT(_params(rngs=nnx.Rngs(0), field_shape=(16, 32)))
    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 16, 32, 1))
    cond = jax.random.normal(jax.random.PRNGKey(2), (2, 3, 1))
    v = model(jnp.ones((2,)), obs, cond)
    assert v.shape == (2, 16, 32, 1)
    assert jnp.allclose(v, 0.0)  # zero-at-init must survive non-square grids


def test_fielddit_patch_size_one():
    model = FieldDiT(_params(rngs=nnx.Rngs(0), patch_size=1))
    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 32, 32, 1))
    cond = jax.random.normal(jax.random.PRNGKey(2), (2, 3, 1))
    v = model(jnp.ones((2,)), obs, cond)
    assert v.shape == obs.shape


def test_fielddit_single_level_encoder():
    model = FieldDiT(_params(rngs=nnx.Rngs(0), encoder_widths=(8, 16)))  # D = 1
    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 32, 32, 1))
    cond = jax.random.normal(jax.random.PRNGKey(2), (2, 3, 1))
    v = model(jnp.ones((2,)), obs, cond)
    assert v.shape == obs.shape


def test_fielddit_split_merge_jit_roundtrip():
    """The bare-nnx.Module-container idiom in the codec must survive
    split/merge and nnx.jit (checkpointing + compiled training rely on it)."""
    model = _small_model()
    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 32, 32, 1))
    cond = jax.random.normal(jax.random.PRNGKey(2), (2, 3, 1))
    t = jnp.ones((2,))

    v_eager = model(t, obs, cond)

    graphdef, state = nnx.split(model)
    model2 = nnx.merge(graphdef, state)
    v_merged = model2(t, obs, cond)
    assert jnp.array_equal(v_eager, v_merged)

    @nnx.jit
    def fwd(m, t, obs, cond):
        return m(t, obs, cond)

    v_jit = fwd(model, t, obs, cond)
    assert v_jit.shape == v_eager.shape
    assert jnp.allclose(v_jit, v_eager, atol=1e-5)
```

- [ ] **Step 2: Run the tests**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/experimental/models/fielddit/test_model.py -q`
Expected: all PASS (pinning existing behavior; a failure is a real bug — stop and investigate).

- [ ] **Step 3: Commit**

```bash
git add tests/experimental/models/fielddit/test_model.py
git commit -m "test(fielddit): backfill non-square, patch1, depth-1, split/merge+jit"
```

---

## Task 11: Shape-generic Gaussian prior

**Files:**
- Modify: `src/gensbi/core/prior.py`
- Modify: `src/gensbi/core/flow_matching.py:88`, `src/gensbi/core/score_matching.py:73,75`, `src/gensbi/core/diffusion_edm.py:68` (call sites)
- Test: `tests/test_prior.py` (extend — it already exists, already imports `pytest`, and already passes `mu`/`sigma` as keywords, so it is compatible with the keyword-only change)

**Why:** `make_gaussian_prior` hard-codes rank-2 event shapes, and `make_gaussian_prior(H, W, C)` today silently reads `C` as the prior **mean**. Fields need `event_shape=(H, W, C)`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_prior.py`):

```python
def test_legacy_two_int_form():
    prior = make_gaussian_prior(5, 1)
    assert prior.event_shape == (5, 1)
    assert is_gaussian_prior(prior)


def test_event_shape_tuple_form():
    prior = make_gaussian_prior((8, 8, 2))
    assert prior.event_shape == (8, 8, 2)
    assert is_gaussian_prior(prior)
    s = prior.sample(jax.random.PRNGKey(0), (4,))
    assert s.shape == (4, 8, 8, 2)


def test_three_positional_ints_raise():
    """make_gaussian_prior(H, W, C) used to silently read C as the MEAN."""
    with pytest.raises(TypeError):
        make_gaussian_prior(8, 8, 2)


def test_mu_sigma_keywords():
    prior = make_gaussian_prior((4, 4, 1), mu=2.0, sigma=3.0)
    assert jnp.allclose(prior.base_dist.loc, 2.0)
    assert jnp.allclose(prior.base_dist.scale, 3.0)
```

- [ ] **Step 2: Run tests to verify failures**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_prior.py -x -q`
Expected: `test_event_shape_tuple_form` FAILS (tuple `dim` breaks `jnp.full((dim, ch), …)`), `test_three_positional_ints_raise` FAILS (no TypeError — mu silently set).

- [ ] **Step 3: Implement** in `src/gensbi/core/prior.py`:

```python
def make_gaussian_prior(dim, ch=None, *, mu=0.0, sigma=1.0):
    """Create a Gaussian prior as a numpyro distribution.

    Two call forms:

    - ``make_gaussian_prior(dim, ch)`` — legacy rank-2 form,
      ``event_shape=(dim, ch)``.
    - ``make_gaussian_prior(event_shape)`` — a single tuple of any rank,
      e.g. ``(H, W, C)`` for pixel-space fields.

    ``mu`` and ``sigma`` are keyword-only: ``make_gaussian_prior(H, W, C)``
    raises ``TypeError`` instead of silently reading ``C`` as the mean.

    Returns
    -------
    dist.Independent
        ``Independent(Normal(loc, scale), len(event_shape))``.
    """
    if ch is None:
        if not isinstance(dim, (tuple, list)):
            raise TypeError(
                "pass (dim, ch) as two ints or a single event_shape tuple, "
                f"got dim={dim!r} with ch=None"
            )
        event_shape = tuple(dim)
    else:
        event_shape = (dim, ch)
    loc = jnp.full(event_shape, mu)
    scale = jnp.full(event_shape, sigma)
    return dist.Independent(dist.Normal(loc, scale), len(event_shape))
```

- [ ] **Step 4: Update the three method call sites** so they pass the shape as a tuple (works for any rank):

In `flow_matching.py:88`: `self.prior = make_gaussian_prior(tuple(event_shape))`
In `score_matching.py:73`: `self.prior = make_gaussian_prior(tuple(event_shape))`
In `score_matching.py:75`: `self.prior = make_gaussian_prior(tuple(event_shape), sigma=path.scheduler.sigma_max)`
In `diffusion_edm.py:68`: `self.prior = make_gaussian_prior(tuple(event_shape))`

(Each currently reads `make_gaussian_prior(*event_shape, …)` — the star-unpack would now raise on rank-3 shapes.)

- [ ] **Step 5: Run prior tests + the full recipes regression** (the methods are shared infrastructure):

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_prior.py tests/core/ tests/recipes/ -q`
Expected: all pass, no regressions (the four pre-existing prior tests keep passing untouched).

- [ ] **Step 6: Commit**

```bash
git add src/gensbi/core/prior.py src/gensbi/core/flow_matching.py src/gensbi/core/score_matching.py src/gensbi/core/diffusion_edm.py tests/test_prior.py
git commit -m "feat(core): shape-generic make_gaussian_prior; close positional-mu footgun"
```

---

## Task 12: `FieldConditionalWrapper` + `FieldConditionalPipeline`

**Files:**
- Create: `src/gensbi/experimental/recipes/field_pipeline.py`
- Modify: `src/gensbi/experimental/recipes/__init__.py` (exports)
- Test: `tests/experimental/recipes/test_field_pipeline.py` (created in Task 13; this task adds only the wrapper unit tests)

**Why (spec §3.2):** `ConditionalPipeline` flattens `dim_obs` to a token count, resolves ids the model doesn't want, and `_expand_dims`'s `ndim < 3` heuristic misreads unbatched fields. Note one deliberate deviation from the spec's deferral list: the **batch-1 cond → batch-N broadcast lives in the wrapper**, because sampling N samples for one `x_o` immediately produces batch-N obs against batch-1 cond (the model itself intentionally does not broadcast).

- [ ] **Step 1: Write the wrapper unit tests** (create `tests/experimental/recipes/test_field_pipeline.py`):

```python
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx
import pytest

from gensbi.experimental.recipes import FieldConditionalWrapper


class _EchoModel(nnx.Module):
    """Records the shapes it was called with and returns obs unchanged."""

    def __init__(self):
        self.seen = {}

    def __call__(self, *, t, obs, cond, obs_ids=None, cond_ids=None,
                 conditioned=True, guidance=None):
        self.seen = dict(t=t.shape, obs=obs.shape, cond=cond.shape)
        return obs


def test_wrapper_batches_unbatched_field_and_cond():
    m = _EchoModel()
    w = FieldConditionalWrapper(m)
    out = w(t=jnp.array(0.5), obs=jnp.ones((8, 8, 1)), cond=jnp.ones((3,)))
    assert m.seen["obs"] == (1, 8, 8, 1)   # (H,W,C) -> (1,H,W,C)
    assert m.seen["cond"] == (1, 3)        # (k,) -> (1,k)
    assert out.shape == (1, 8, 8, 1)


def test_wrapper_passes_batched_inputs_through():
    m = _EchoModel()
    w = FieldConditionalWrapper(m)
    w(t=jnp.ones((4,)), obs=jnp.ones((4, 8, 8, 1)), cond=jnp.ones((4, 3, 1)))
    assert m.seen["obs"] == (4, 8, 8, 1)
    assert m.seen["cond"] == (4, 3, 1)


def test_wrapper_broadcasts_batch1_cond_to_obs_batch():
    """Sampling N draws for one x_o: obs arrives batch-N, cond batch-1."""
    m = _EchoModel()
    w = FieldConditionalWrapper(m)
    w(t=jnp.ones((4,)), obs=jnp.ones((4, 8, 8, 1)), cond=jnp.ones((1, 3, 1)))
    assert m.seen["cond"] == (4, 3, 1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/experimental/recipes/test_field_pipeline.py -x -q`
Expected: FAIL — `FieldConditionalWrapper` does not exist.

- [ ] **Step 3: Implement** `src/gensbi/experimental/recipes/field_pipeline.py`:

```python
"""Field-shaped conditional pipeline (experimental).

``ConditionalPipeline`` assumes token-shaped observations ``(B, dim, ch)``:
it flattens ``dim_obs``, resolves embedding ids, and builds a rank-2 prior.
Pixel-space field models (FieldDiT) need ``(B, H, W, C)`` observations, a
rank-3 ``event_shape``, no external ids, and rank-aware input expansion.
"""

import jax.numpy as jnp

from gensbi.core.generative_method import GenerativeMethod
from gensbi.recipes.conditional_pipeline import ConditionalPipeline
from gensbi.recipes.pipeline import AbstractPipeline
from gensbi.utils.model_wrapping import ModelWrapper, _expand_time


class FieldConditionalWrapper(ModelWrapper):
    """Wrapper for field-shaped conditional models.

    Expansion is event-rank-aware (a field event is rank 3, ``(H, W, C)``):

    - ``obs`` with ``ndim == 3`` is treated as unbatched -> ``(1, H, W, C)``.
    - ``cond`` with ``ndim == 1`` (``(k,)``) -> ``(1, k)``; 2D+ cond is
      assumed batched (``(B, k)`` / ``(B, k, c)``).
    - batch-1 ``cond`` is broadcast to the obs batch (sampling N draws for a
      single ``x_o``; the model itself deliberately does not broadcast).
    - ids are passed through untouched (FieldDiT builds rope ids internally).
    """

    def __init__(self, model):
        super().__init__(model)

    def __call__(
        self,
        t,
        obs,
        cond,
        obs_ids=None,
        cond_ids=None,
        conditioned=True,
        guidance=None,
        **kwargs,
    ):
        if obs.ndim == 3:
            obs = obs[None, ...]
        if cond.ndim == 1:
            cond = cond[None, ...]
        if cond.shape[0] == 1 and obs.shape[0] > 1:
            cond = jnp.repeat(cond, obs.shape[0], axis=0)
        t = _expand_time(t)

        return self.model(
            obs=obs,
            t=t,
            cond=cond,
            obs_ids=obs_ids,
            cond_ids=cond_ids,
            conditioned=conditioned,
            guidance=guidance,
            **kwargs,
        )


class FieldConditionalPipeline(ConditionalPipeline):
    """Conditional pipeline for pixel-space fields ``(B, H, W, C)``.

    Differences from :class:`ConditionalPipeline`:

    - ``event_shape = (*field_shape, ch_obs)`` — prior, path, and sampling are
      field-shaped (``sample`` returns ``(nsamples, H, W, C)``).
    - no obs/cond id resolution: ``obs_ids``/``cond_ids`` are ``None`` in all
      extras; the model builds its rope ids internally and ignores them.
    - :class:`FieldConditionalWrapper` for event-rank-aware expansion.

    Datasets must yield ``(obs, cond)`` batches with ``obs`` of shape
    ``(B, H, W, C)`` and ``cond`` of shape ``(B, k)`` or ``(B, k, c)``.
    """

    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        field_shape,
        dim_cond,
        method: GenerativeMethod,
        ch_obs=1,
        ch_cond=1,
        params=None,
        training_config=None,
    ):
        self.method = method

        if training_config is None:
            training_config = self.get_default_training_config()
        extra = method.get_extra_training_config()
        for k, v in extra.items():
            training_config.setdefault(k, v)

        # bypass ConditionalPipeline.__init__ (id resolution + rank-2 path):
        # AbstractPipeline handles datasets/EMA/optimizer/training config
        AbstractPipeline.__init__(
            self,
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            dim_obs=tuple(field_shape),
            dim_cond=dim_cond,
            ch_obs=ch_obs,
            ch_cond=ch_cond,
            params=params,
            training_config=training_config,
        )

        self.field_shape = tuple(field_shape)
        self.event_shape = (*self.field_shape, ch_obs)
        self.obs_ids = None
        self.cond_ids = None

        self.path = method.build_path(self.training_config, event_shape=self.event_shape)
        self.loss_obj = method.build_loss(self.path)

    def _wrap_model(self):
        self.model_wrapped = FieldConditionalWrapper(self.model)
        self.ema_model_wrapped = FieldConditionalWrapper(self.ema_model)
```

The inherited `get_loss_fn`, `get_sampler`, `sample`, `get_log_prob_fn`, and `log_prob` from `ConditionalPipeline` are reused as-is: they put `cond`/`obs_ids=None`/`cond_ids=None` in the extras (the wrapper and model tolerate `None` ids), and `_expand_dims(x_o)` maps `(k,) -> (1, k, 1)` and `(B, k) -> (B, k, 1)`, both valid embedder inputs.

- [ ] **Step 4: Export.** In `src/gensbi/experimental/recipes/__init__.py`, add:

```python
from gensbi.experimental.recipes.field_pipeline import (
    FieldConditionalPipeline,
    FieldConditionalWrapper,
)
```

(and extend `__all__` if the file defines one).

- [ ] **Step 5: Run the wrapper tests**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/experimental/recipes/test_field_pipeline.py -q`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/gensbi/experimental/recipes/field_pipeline.py src/gensbi/experimental/recipes/__init__.py tests/experimental/recipes/test_field_pipeline.py
git commit -m "feat(recipes): FieldConditionalPipeline + rank-aware wrapper for pixel-space fields"
```

---

## Task 13: Field pipeline integration tests — construct, train, sample

**Files:**
- Test: `tests/experimental/recipes/test_field_pipeline.py` (extend)

- [ ] **Step 1: Write the integration tests** (append). A fixed repeating batch is the simplest dataset that satisfies the pipeline's `iter()` protocol:

```python
import tempfile

from gensbi.core import FlowMatchingMethod
from gensbi.experimental.models import FieldDiT, FieldDiTParams
from gensbi.experimental.recipes import FieldConditionalPipeline

H = W = 16
COND_DIM = 3


class _Loop:
    """Iterable dataset yielding the same (obs, cond) batch forever."""

    def __init__(self, batch):
        self.batch = batch

    def __iter__(self):
        while True:
            yield self.batch


def _tiny_fielddit(seed=0):
    return FieldDiT(FieldDiTParams(
        in_channels=1,
        field_shape=(H, W),
        encoder_widths=(4, 8),       # D = 1
        cond_dim=COND_DIM,
        rngs=nnx.Rngs(seed),
        res_blocks_down=1,
        res_blocks_up=1,
        patch_size=2,
        num_heads=2,
        axes_dim=[2, 2, 4],          # hidden 16
        depth=1,
        depth_single_blocks=1,
        param_dtype=jnp.float32,
    ))


def _make_pipeline(model_dir, seed=0):
    key = jax.random.PRNGKey(seed)
    obs = jax.random.normal(key, (32, H, W, 1))
    cond = jax.random.normal(jax.random.fold_in(key, 1), (32, COND_DIM, 1))
    batch = (obs, cond)

    training_config = FieldConditionalPipeline.get_default_training_config()
    training_config["checkpoint_dir"] = model_dir

    return FieldConditionalPipeline(
        model=_tiny_fielddit(seed),
        train_dataset=_Loop(batch),
        val_dataset=_Loop(batch),
        field_shape=(H, W),
        dim_cond=COND_DIM,
        method=FlowMatchingMethod(),
        ch_obs=1,
        training_config=training_config,
    )


def test_pipeline_constructs_with_field_event_shape():
    with tempfile.TemporaryDirectory() as model_dir:
        p = _make_pipeline(model_dir)
        assert p.event_shape == (H, W, 1)
        assert p.method.prior.event_shape == (H, W, 1)
        assert p.obs_ids is None and p.cond_ids is None


def test_pipeline_trains_two_steps():
    with tempfile.TemporaryDirectory() as model_dir:
        p = _make_pipeline(model_dir)
        before = jax.tree_util.tree_leaves(nnx.state(p.model, nnx.Param))
        before = [leaf.copy() for leaf in before]
        p.train(nnx.Rngs(0), nsteps=2, save_model=False)
        after = jax.tree_util.tree_leaves(nnx.state(p.model, nnx.Param))
        changed = any(
            not jnp.array_equal(b, a) for b, a in zip(before, after)
        )
        assert changed


def test_pipeline_samples_field_shaped_output():
    with tempfile.TemporaryDirectory() as model_dir:
        p = _make_pipeline(model_dir)
        p.train(nnx.Rngs(0), nsteps=2, save_model=False)
        p._wrap_model()
        x_o = jnp.ones((1, COND_DIM, 1))
        samples = p.sample(jax.random.PRNGKey(0), x_o, nsamples=4, step_size=0.5)
        assert samples.shape == (4, H, W, 1)
        assert jnp.all(jnp.isfinite(samples))
```

- [ ] **Step 2: Run the tests**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/experimental/recipes/test_field_pipeline.py -x -q`
Expected: 6 passed (3 wrapper + 3 integration). Likely first-run friction points, in order: (a) the `train()` signature — mirror `tests/recipes/test_conditional_pipeline.py:85` exactly; (b) sampler kwargs — `step_size` is the documented kwarg of `FlowMatchingMethod.build_sampler_fn` (`src/gensbi/core/flow_matching.py:197`).

- [ ] **Step 3: Run the full regression**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/experimental/ tests/recipes/ tests/core/ -q`
Expected: no regressions.

- [ ] **Step 4: Commit**

```bash
git add tests/experimental/recipes/test_field_pipeline.py
git commit -m "test(recipes): field pipeline construct/train/sample integration"
```

---

## Task 14: Learning gate 1 — one-step aliveness

**Files:**
- Test: `tests/experimental/models/fielddit/test_training.py` (create)

**Why (handoff B1):** at init, 149/151 parameter gradients are identically zero by design (zero-init gates). No test at init can distinguish a live conditioning path from a dead one. After ONE optimizer step the output conv is nonzero and gradient must flow everywhere.

- [ ] **Step 1: Write the test** (create `test_training.py`):

```python
"""Learning gates: prove FieldDiT can actually train (handoff B1).

Phase-1 tests prove the model is well-formed; these prove it is ALIVE:
gradients reach every subtree after one step, and conditioning genuinely
shapes the output after a tiny overfit.
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
    model = _tiny_model()
    optimizer = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)

    obs = jax.random.normal(jax.random.PRNGKey(1), (4, H, W, 1))
    cond = jax.random.normal(jax.random.PRNGKey(2), (4, COND_DIM, 1))
    t = jnp.full((4,), 0.5)
    target = jax.random.normal(jax.random.PRNGKey(3), (4, H, W, 1))

    def loss_fn(m):
        return jnp.mean((m(t, obs, cond) - target) ** 2)

    # step 0: only the zero-init output conv has nonzero grads (by design)
    loss0, grads0 = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads0)

    # after one step the output is no longer identically zero ...
    v = model(t, obs, cond)
    assert not jnp.allclose(v, 0.0)

    # ... and gradients reach every major subtree
    grads1 = nnx.grad(loss_fn)(model)
    gstate = nnx.state(grads1, nnx.Param)
    for subtree in ("encoder", "tokenizer", "core", "untokenizer", "decoder",
                    "cond_embedder", "time_in"):
        assert _subtree_grad_nonzero(gstate, subtree), (
            f"no gradient reaches '{subtree}' after one optimizer step — "
            "a dead path the zero-init design would otherwise mask"
        )
```

- [ ] **Step 2: Run the test**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/experimental/models/fielddit/test_training.py -x -q`
Expected: PASS. If a subtree assert fails, that is exactly the dead-path bug B1 exists to catch — debug the wiring (do NOT weaken the test). One foreseeable nuance: `cond_embedder` grads require the cond path to be live, which at step 1 flows through the core's modulation linears and the vec-summary; if specifically `cond_embedder` is still zero after step 1, run a second optimizer step before the subtree asserts and document why in the test.

- [ ] **Step 3: Commit**

```bash
git add tests/experimental/models/fielddit/test_training.py
git commit -m "test(fielddit): gate 1 — one-step gradient aliveness for every subtree"
```

---

## Task 15: Learning gate 2 — tiny overfit + conditioning sensitivity

**Files:**
- Test: `tests/experimental/models/fielddit/test_training.py` (extend)

**Why (handoff B1):** prove the loss can be driven down and that the condition genuinely shapes the output — the failure mode the zero-init design masks at init.

- [ ] **Step 1: Write the test** (append to `test_training.py`):

```python
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
```

- [ ] **Step 2: Run the test** (expect a few seconds on CPU)

Run: `JAX_PLATFORMS=cpu uv run pytest tests/experimental/models/fielddit/test_training.py -x -q`
Expected: 2 passed. If the 10x bound is not reached in 300 steps, raise the step count to 600 and/or learning rate to 1e-2 before weakening any threshold; if it still fails, the conditioning path is too weak — investigate, don't tune the test.

- [ ] **Step 3: Commit**

```bash
git add tests/experimental/models/fielddit/test_training.py
git commit -m "test(fielddit): gate 2 — tiny overfit proves live conditioning"
```

---

## Task 16: Opt-in realistic-size smoke test

**Files:**
- Test: `tests/experimental/models/fielddit/test_training.py` (extend)

**Why (handoff B5):** all tests use hidden=16 configs; nobody has instantiated the production-shape model. Opt-in (env flag), not CI.

- [ ] **Step 1: Write the test** (append):

```python
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
```

- [ ] **Step 2: Run it once, opt-in, and record the numbers** (token count, param count, wall time, peak RSS if available):

Run: `GENSBI_RUN_BIG_SMOKE=1 JAX_PLATFORMS=cpu uv run pytest tests/experimental/models/fielddit/test_training.py::test_realistic_256_config_smoke -x -q -s`
Expected: PASS, with the `[smoke]` line printed. Paste the printed numbers into the final commit message body or the wrap-up notes.

- [ ] **Step 3: Verify it is skipped by default**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/experimental/models/fielddit/test_training.py -q`
Expected: 2 passed, 1 skipped.

- [ ] **Step 4: Commit**

```bash
git add tests/experimental/models/fielddit/test_training.py
git commit -m "test(fielddit): opt-in 256^2 realistic-config smoke"
```

---

## Task 17: Final regression + wrap-up

**Files:**
- Possibly modify: anything surfaced by the regression run.

- [ ] **Step 1: Full regression**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/ -q`
Expected: everything green (baseline 216 in the affected dirs, plus ~25 new tests). Fix anything that broke; commit fixes individually with `fix(...)` messages.

- [ ] **Step 2: Verify the working tree is clean and the branch history is coherent**

Run: `git status && git log --oneline main..HEAD | head -30`
Expected: clean tree; one commit per task above on `FieldDiT`.

- [ ] **Step 3: Done.** Report: test counts before/after, the smoke-test numbers from Task 16, and any deviations from this plan (there is one planned deviation already: the batch-1 cond broadcast landed in `FieldConditionalWrapper`, see Task 12).
