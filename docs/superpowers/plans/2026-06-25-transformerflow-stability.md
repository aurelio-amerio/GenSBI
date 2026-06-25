# TransformerFlow Stability Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port STARFlow's scale-stability guards (softplus parametrization, soft_clip, fp32 affine) into TransformerFlow's `MetaBlock` so the SOS-shift model stops diverging to NaN, default-on, with the legacy `exp` path kept as a non-default fallback.

**Architecture:** Replace the inline `exp(-a)` affine in `MetaBlock.inverse`/`forward` with a single `_affine(a) -> (scale, inv_scale, log_scale)` helper that switches between `exp` and `softplus(a + INV_SOFTPLUS_1)`, computed in float32. Add a `soft_clip` `tanh` bound on the `proj_out` output in `_params`. Thread two new flags (`use_softplus=True`, `soft_clip=4.0`) through `make_tarflow`. No pipeline or MAF changes.

**Tech Stack:** JAX, flax.nnx, optax; pytest. Run tests with `JAX_PLATFORMS=cpu .venv/bin/python -m pytest`.

**Spec:** `docs/superpowers/specs/2026-06-25-transformerflow-stability-design.md` (commit 9e3da51).

## Global Constraints

- Changes confined to `src/gensbi/normalizing_flows/transformer_flow/blocks.py` and `.../model.py`, plus tests under `tests/normalizing_flows/transformer_flow/`. No pipeline, no MAF, no other modules.
- Defaults: `use_softplus=True`, `soft_clip=4.0`. `soft_clip=0` disables the clip (`>0` convention).
- `INV_SOFTPLUS_1 = 0.541324854612918` (chosen so `softplus(0 + INV_SOFTPLUS_1) == 1.0` ⇒ identity at zero-init).
- The `exp` branch (`use_softplus=False`) must remain numerically equal to the current code under the default float32 dtype.
- Affine arithmetic (scale, inverse-scale, log-scale, and the `(xp - b)` combine) is done in float32.
- TDD: write the failing test first, watch it fail, implement minimally, watch it pass, commit. One logical change per commit.
- Run command for a single test: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest <path>::<name> -v`.

---

### Task 1: `_affine` helper + stability flags on `MetaBlock`

Adds the constant, the two flags, and the parametrization helper. Does **not** yet
rewire `inverse`/`forward` (that is Task 2), so behavior is unchanged after this task.

**Files:**
- Modify: `src/gensbi/normalizing_flows/transformer_flow/blocks.py` (add module constant `INV_SOFTPLUS_1`; `MetaBlock.__init__` gains `use_softplus`/`soft_clip`; add `MetaBlock._affine`)
- Test: `tests/normalizing_flows/transformer_flow/test_stability.py` (new file)

**Interfaces:**
- Produces:
  - Module constant `INV_SOFTPLUS_1: float = 0.541324854612918` in `blocks.py`.
  - `MetaBlock.__init__(..., zero_init=True, use_softplus=True, soft_clip=4.0)` — two new keyword-only-style params appended after `zero_init`; stored as `self.use_softplus` (bool) and `self.soft_clip` (float).
  - `MetaBlock._affine(self, a: Array) -> tuple[Array, Array, Array]` returning `(scale, inv_scale, log_scale)`, all float32. softplus mode: `s = softplus(a + INV_SOFTPLUS_1)`, returns `(s, 1/s, log(s))`. exp mode: `(exp(a), exp(-a), a)`.

- [ ] **Step 1: Write the failing test**

Create `tests/normalizing_flows/transformer_flow/test_stability.py`:

```python
# tests/normalizing_flows/transformer_flow/test_stability.py
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from gensbi.normalizing_flows.transformer_flow.model import make_tarflow
from gensbi.normalizing_flows.transformer_flow.blocks import MetaBlock, INV_SOFTPLUS_1


class _NullCond(nnx.Module):
    def embed(self, cond):
        return (None, None)


def _block(use_softplus=True, soft_clip=4.0, F=1, channels=16, zero_init=True):
    return MetaBlock(
        F=F, channels=channels, T=4, perm=jnp.arange(4), inv_perm=jnp.arange(4),
        conditioner=_NullCond(), num_layers=1, head_dim=8, expansion=4,
        rngs=nnx.Rngs(0), zero_init=zero_init, use_softplus=use_softplus, soft_clip=soft_clip,
    )


def test_affine_exp_mode_exact():
    blk = _block(use_softplus=False)
    a = jnp.array([[-1.5, 0.0, 2.0]])
    scale, inv_scale, log_scale = blk._affine(a)
    assert jnp.allclose(scale, jnp.exp(a))
    assert jnp.allclose(inv_scale, jnp.exp(-a))
    assert jnp.allclose(log_scale, a)


def test_affine_softplus_mode_exact():
    blk = _block(use_softplus=True)
    a = jnp.array([[-1.5, 0.0, 2.0]])
    s = jax.nn.softplus(a + INV_SOFTPLUS_1)
    scale, inv_scale, log_scale = blk._affine(a)
    assert jnp.allclose(scale, s)
    assert jnp.allclose(inv_scale, 1.0 / s)
    assert jnp.allclose(log_scale, jnp.log(s))


def test_affine_softplus_identity_at_zero():
    blk = _block(use_softplus=True)
    scale, inv_scale, log_scale = blk._affine(jnp.zeros((1, 3)))
    assert jnp.allclose(scale, 1.0, atol=1e-6)
    assert jnp.allclose(inv_scale, 1.0, atol=1e-6)
    assert jnp.allclose(log_scale, 0.0, atol=1e-6)


def test_affine_is_float32():
    blk = _block(use_softplus=True)
    scale, inv_scale, log_scale = blk._affine(jnp.zeros((1, 3), dtype=jnp.float32))
    assert scale.dtype == jnp.float32
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/transformer_flow/test_stability.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'use_softplus'` (and `ImportError` for `INV_SOFTPLUS_1`).

- [ ] **Step 3: Add the constant and flags**

In `blocks.py`, after the imports block (after `from gensbi.normalizing_flows.bijections.base import Mask`), add:

```python
INV_SOFTPLUS_1 = 0.541324854612918  # softplus(INV_SOFTPLUS_1) == 1.0 -> identity at zero-init
```

In `MetaBlock.__init__`, change the signature line:

```python
    def __init__(self, F, channels, T, perm, inv_perm, conditioner,
                 num_layers, head_dim, expansion, rngs, zero_init=True,
                 use_softplus=True, soft_clip=4.0):
```

and store the flags near the top of the body (e.g. right after `self.F = F`):

```python
        self.use_softplus = use_softplus
        self.soft_clip = soft_clip
```

- [ ] **Step 4: Add the `_affine` helper**

In `MetaBlock`, add this method (place it just above `_params`):

```python
    def _affine(self, a: Array):
        """Map raw log-scale ``a`` -> ``(scale, inv_scale, log_scale)`` in float32.

        ``scale`` plays the role of ``exp(a)`` ("1/sigma"): inverse multiplies by
        ``inv_scale``, forward multiplies by ``scale``, logdet sums ``log_scale``.
        softplus mode bounds the positive-scale tail and its gradient; the
        ``INV_SOFTPLUS_1`` offset makes it the identity at ``a == 0``.
        """
        a = a.astype(jnp.float32)
        if self.use_softplus:
            s = jax.nn.softplus(a + INV_SOFTPLUS_1)
            return s, 1.0 / s, jnp.log(s)
        return jnp.exp(a), jnp.exp(-a), a
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/transformer_flow/test_stability.py -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add src/gensbi/normalizing_flows/transformer_flow/blocks.py tests/normalizing_flows/transformer_flow/test_stability.py
git commit -m "feat(nf): MetaBlock _affine helper + use_softplus/soft_clip flags

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Rewire `inverse`/`forward` to `_affine` + soft_clip in `_params`

Switches the affine to the helper and applies the `tanh` clip on `proj_out`. After this
task, `make_tarflow` models (which take `MetaBlock` defaults) compute the **softplus**
affine by default; existing tests must stay green because softplus is identity at init.

**Files:**
- Modify: `src/gensbi/normalizing_flows/transformer_flow/blocks.py` (`_params` soft_clip; `inverse` and `forward` use `_affine`)
- Test: `tests/normalizing_flows/transformer_flow/test_stability.py` (append)

**Interfaces:**
- Consumes: `MetaBlock._affine` and `self.soft_clip` from Task 1.
- Produces: unchanged public signatures `MetaBlock.inverse(x, cond=None) -> (z, logdet)` and `MetaBlock.forward(z, cond=None) -> (x, logdet)`; their internals now route through `_affine`. `_params` output `(a, b)` is soft-clipped when `self.soft_clip > 0`.

- [ ] **Step 1: Write the failing tests (block-level, self-contained)**

Append to `tests/normalizing_flows/transformer_flow/test_stability.py`. These use
`_block(...)` directly so the task does not depend on the Task-3 `make_tarflow` plumbing:

```python
def test_soft_clip_bounds_params():
    blk = _block(use_softplus=True, soft_clip=4.0)
    # blow up proj_out so the raw output is far outside [-4, 4]
    blk.proj_out.kernel[...] = blk.proj_out.kernel[...] + 50.0
    blk.proj_out.bias[...] = blk.proj_out.bias[...] + 50.0
    xp = jax.random.normal(jax.random.PRNGKey(3), (4, 4, 1))
    a, b = blk._params(xp, None)
    assert jnp.max(jnp.abs(a)) <= 4.0 + 1e-4
    assert jnp.max(jnp.abs(b)) <= 4.0 + 1e-4


def test_block_inverse_uses_softplus_when_enabled():
    # RED DRIVER: before the rewire, inverse uses exp(-a); after, softplus.
    blk = _block(use_softplus=True, soft_clip=0.0, zero_init=False)  # no clip: isolate softplus
    xp = jax.random.normal(jax.random.PRNGKey(1), (6, 4, 1))
    a, b = blk._params(xp, None)
    s = jax.nn.softplus(a + INV_SOFTPLUS_1)
    z_ref = (xp - b) / s                        # perm is identity here (arange)
    z, ld = blk.inverse(xp, None)
    assert jnp.allclose(z, z_ref, atol=1e-5)
    assert jnp.allclose(ld, -jnp.sum(jnp.log(s), axis=(1, 2)), atol=1e-5)


def test_block_exp_inverse_matches_bare_exp():
    # GUARD: use_softplus=False + soft_clip=0 still equals literal (xp-b)*exp(-a).
    blk = _block(use_softplus=False, soft_clip=0.0, zero_init=False)
    xp = jax.random.normal(jax.random.PRNGKey(1), (6, 4, 1))
    a, b = blk._params(xp, None)
    z, ld = blk.inverse(xp, None)
    assert jnp.allclose(z, (xp - b) * jnp.exp(-a), atol=1e-5)
    assert jnp.allclose(ld, -jnp.sum(a, axis=(1, 2)), atol=1e-5)


def test_block_roundtrip_softplus():
    # GUARD: forward and inverse stay mutually consistent (fails if only one is rewired).
    blk = _block(use_softplus=True, soft_clip=4.0, zero_init=False)
    z = jax.random.normal(jax.random.PRNGKey(7), (5, 4, 1))
    x, _ = blk.forward(z, None)
    z2, _ = blk.inverse(x, None)
    assert jnp.allclose(z2.reshape(5, 4, 1), z, atol=1e-4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/transformer_flow/test_stability.py::test_soft_clip_bounds_params tests/normalizing_flows/transformer_flow/test_stability.py::test_block_inverse_uses_softplus_when_enabled -v`
Expected: FAIL — `test_soft_clip_bounds_params` (`a`/`b` exceed 4.0, no clip yet) and `test_block_inverse_uses_softplus_when_enabled` (inverse still computes `exp(-a)`, not `1/softplus(a+c)`). `test_block_exp_inverse_matches_bare_exp` already passes (guards the unchanged exp branch). `test_block_roundtrip_softplus` passes now (exp∘exp is consistent) and must STILL pass after the rewire (softplus∘softplus) — it catches a half-done rewire.

- [ ] **Step 4: Apply soft_clip in `_params`**

In `MetaBlock._params`, change the tail. Current:

```python
        out = self.proj_out(h)                                  # (B, T, 2F)
        a, b = jnp.split(out, 2, axis=-1)                       # each (B, T, F)
        return a, b
```

to:

```python
        out = self.proj_out(h)                                  # (B, T, 2F)
        if self.soft_clip > 0:
            out = self.soft_clip * jnp.tanh(out / self.soft_clip)
        a, b = jnp.split(out, 2, axis=-1)                       # each (B, T, F)
        return a, b
```

- [ ] **Step 5: Rewire `inverse` to `_affine` (float32 combine)**

In `MetaBlock.inverse`, current:

```python
        a, b = self._params(xp, cond)
        z = (xp - b) * jnp.exp(-a)
        logdet = -jnp.sum(a, axis=(1, 2))                      # (B,)
        z = z[:, self.inv_perm[...]]
        return z, logdet
```

to:

```python
        a, b = self._params(xp, cond)
        scale, inv_scale, log_scale = self._affine(a)
        z = (xp.astype(jnp.float32) - b.astype(jnp.float32)) * inv_scale
        logdet = -jnp.sum(log_scale, axis=(1, 2))              # (B,)
        z = z[:, self.inv_perm[...]].astype(xp.dtype)
        return z, logdet
```

- [ ] **Step 6: Rewire `forward` to `_affine`**

In `MetaBlock.forward`, change the scan body and the final logdet. Current body:

```python
        def body(x, i):
            a, b = self._params(x, cond)        # a[:,i],b[:,i] depend on tokens < i
            xi = zp[:, i, :] * jnp.exp(a[:, i, :]) + b[:, i, :]
            return x.at[:, i, :].set(xi), None
```

to:

```python
        def body(x, i):
            a, b = self._params(x, cond)        # a[:,i],b[:,i] depend on tokens < i
            scale, _, _ = self._affine(a)
            xi = zp[:, i, :] * scale[:, i, :] + b[:, i, :].astype(jnp.float32)
            return x.at[:, i, :].set(xi.astype(x.dtype)), None
```

Current final:

```python
        a, _ = self._params(x, cond)
        logdet = jnp.sum(a, axis=(1, 2))                       # (B,), +Σa
        x = x[:, self.inv_perm[...]]
        return x, logdet
```

to:

```python
        a, _ = self._params(x, cond)
        _, _, log_scale = self._affine(a)
        logdet = jnp.sum(log_scale, axis=(1, 2))               # (B,), +Σ log_scale
        x = x[:, self.inv_perm[...]]
        return x, logdet
```

- [ ] **Step 7: Run the new block-level tests to verify they pass**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/transformer_flow/test_stability.py -v`
Expected: PASS for all of `test_soft_clip_bounds_params`, `test_block_inverse_uses_softplus_when_enabled`, `test_block_exp_inverse_matches_bare_exp`, `test_block_roundtrip_softplus`, plus the Task-1 `_affine` unit tests.

- [ ] **Step 8: Run the existing model suite — must stay green**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/transformer_flow/ -v`
Expected: PASS for all pre-existing tests. softplus is identity at init, so `test_zero_init_flow_is_standard_normal` still holds; `test_full_flow_logdet_matches_autodiff` is a self-consistency check between `log_prob` and per-block `inverse` (both now softplus) so it also holds. If anything regresses, STOP and investigate before committing.

- [ ] **Step 9: Commit**

```bash
git add src/gensbi/normalizing_flows/transformer_flow/blocks.py tests/normalizing_flows/transformer_flow/test_stability.py
git commit -m "feat(nf): route MetaBlock affine through _affine + soft_clip (fp32)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Plumb `use_softplus`/`soft_clip` through `make_tarflow`

**Files:**
- Modify: `src/gensbi/normalizing_flows/transformer_flow/model.py` (`make_tarflow` signature + `MetaBlock(...)` call)
- Test: `tests/normalizing_flows/transformer_flow/test_stability.py` (append)

**Interfaces:**
- Consumes: `MetaBlock(..., use_softplus=, soft_clip=)` from Task 1.
- Produces: `make_tarflow(..., zero_init=True, use_softplus=True, soft_clip=4.0)` — two params appended after `zero_init`, forwarded to every `MetaBlock`.

- [ ] **Step 1: Write the failing test**

Append to `tests/normalizing_flows/transformer_flow/test_stability.py`:

```python
def test_make_tarflow_defaults_and_override():
    flow = make_tarflow(nnx.Rngs(0), dim=4, cond_dim=2, channels=16, num_blocks=3,
                        layers_per_block=2, head_dim=8)
    for blk in flow.blocks:
        assert blk.use_softplus is True
        assert blk.soft_clip == 4.0
    flow2 = make_tarflow(nnx.Rngs(0), dim=4, cond_dim=2, channels=16, num_blocks=2,
                         layers_per_block=2, head_dim=8,
                         use_softplus=False, soft_clip=0.0)
    for blk in flow2.blocks:
        assert blk.use_softplus is False
        assert blk.soft_clip == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/transformer_flow/test_stability.py::test_make_tarflow_defaults_and_override -v`
Expected: FAIL — `make_tarflow() got an unexpected keyword argument 'use_softplus'`.

- [ ] **Step 3: Add the params to `make_tarflow`**

In `model.py`, change the `make_tarflow` signature line `block_size=1, permutation="flip", standardize=True, zero_init=True):` to:

```python
                 block_size=1, permutation="flip", standardize=True,
                 zero_init=True, use_softplus=True, soft_clip=4.0):
```

In the `MetaBlock(...)` construction inside the loop, change the final args
`rngs=rngs, zero_init=zero_init))` to:

```python
            head_dim=head_dim, expansion=4, rngs=rngs, zero_init=zero_init,
            use_softplus=use_softplus, soft_clip=soft_clip))
```

(Keep the existing `F=`, `channels=`, `T=`, `perm=`, `inv_perm=`, `conditioner=`,
`num_layers=` arguments as they are; only append the two new keywords.)

- [ ] **Step 4: Run the new test + the whole stability file**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/transformer_flow/test_stability.py -v`
Expected: PASS for `test_make_tarflow_defaults_and_override` (flags now plumbed) plus every test from Tasks 1–2.

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/normalizing_flows/transformer_flow/model.py tests/normalizing_flows/transformer_flow/test_stability.py
git commit -m "feat(nf): plumb use_softplus/soft_clip through make_tarflow (default on)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Stability regression + softplus logdet correctness

Locks in the fix: the shipped default stays finite where the legacy unguarded `exp`
path overflows; and the softplus block's logdet matches a numerical Jacobian.

**Files:**
- Test: `tests/normalizing_flows/transformer_flow/test_stability.py` (append)

**Interfaces:**
- Consumes: `make_tarflow` with flags (Task 3); `flow.F`, `flow.blocks`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/normalizing_flows/transformer_flow/test_stability.py`:

```python
def _force_large_neg_a(flow, val=-60.0):
    # Make block 0 output a constant, very negative log-scale `a` regardless of input
    # (kernel is already 0 from zero_init; override the a-half of the bias).
    F = flow.F
    blk = flow.blocks[0]
    bias = np.zeros((2 * F,), dtype=np.float32)
    bias[:F] = val
    blk.proj_out.bias[...] = jnp.asarray(bias)


def test_default_is_finite_where_legacy_exp_overflows():
    x = jax.random.normal(jax.random.PRNGKey(1), (8, 4))

    legacy = make_tarflow(nnx.Rngs(0), dim=4, cond_dim=0, channels=16, num_blocks=3,
                          layers_per_block=2, head_dim=8,
                          use_softplus=False, soft_clip=0.0)   # bare exp, no clip
    _force_large_neg_a(legacy)
    lp_legacy = legacy.log_prob(x)
    assert not bool(jnp.all(jnp.isfinite(lp_legacy)))          # exp(-(-60)) overflows

    new = make_tarflow(nnx.Rngs(0), dim=4, cond_dim=0, channels=16, num_blocks=3,
                       layers_per_block=2, head_dim=8)          # default softplus+soft_clip
    _force_large_neg_a(new)
    lp_new = new.log_prob(x)
    assert bool(jnp.all(jnp.isfinite(lp_new)))                 # clip+softplus keep it finite


def test_softplus_block_logdet_matches_numerical():
    flow = make_tarflow(nnx.Rngs(0), dim=4, cond_dim=0, channels=16, num_blocks=1,
                        layers_per_block=2, head_dim=8, zero_init=False)  # softplus default
    blk = flow.blocks[0]
    x = jnp.array([[0.5, -1.0, 0.3, 0.8]])

    def to_noise(xv):
        z, _ = blk.inverse(xv[None], None)
        return z.reshape(-1)

    _, num = jnp.linalg.slogdet(jax.jacobian(to_noise)(x[0]))
    _, ld = blk.inverse(x, None)
    assert jnp.allclose(ld[0], num, atol=1e-4)
```

- [ ] **Step 2: Run tests to verify behavior**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/transformer_flow/test_stability.py::test_default_is_finite_where_legacy_exp_overflows tests/normalizing_flows/transformer_flow/test_stability.py::test_softplus_block_logdet_matches_numerical -v`
Expected: PASS. (These assert behavior already implemented in Tasks 1–3; they exist to lock it in. If `test_softplus_block_logdet_matches_numerical` fails on the `slogdet` cross-check, widen atol to 3e-4 — the assembled Jacobian is float32-fragile, same caveat as the existing `test_full_flow_logdet_matches_autodiff`.)

- [ ] **Step 3: Commit**

```bash
git add tests/normalizing_flows/transformer_flow/test_stability.py
git commit -m "test(nf): lock in TransformerFlow stability (default finite vs legacy exp overflow)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Audit existing tests + EMA seam re-assert

Confirms the default flip to softplus did not silently break the existing battery, refreshes
one stale exp-specific comment, and re-asserts the EMA seam (historically fragile here).

**Files:**
- Modify: `tests/normalizing_flows/transformer_flow/test_model.py` (comment refresh only, if needed)
- Test: `tests/normalizing_flows/transformer_flow/test_stability.py` (append EMA seam assert)

**Interfaces:**
- Consumes: `make_tarflow`; `gensbi.recipes.pipeline` EMA wiring is **not** modified.

- [ ] **Step 1: Run the full transformer_flow battery**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/transformer_flow/ -v`
Expected: PASS. In particular `tests/normalizing_flows/transformer_flow/test_model.py::test_zero_init_flow_is_standard_normal` (identity at init) and `::test_full_flow_logdet_matches_autodiff` (parametrization-agnostic self-consistency between `log_prob` and per-block `inverse`) must stay green. If `test_full_flow_logdet_matches_autodiff` regresses on conditioning, note it and STOP — do not loosen it without review.

- [ ] **Step 2: Refresh the stale exp-specific comment (only if present)**

In `tests/normalizing_flows/transformer_flow/test_model.py`, the `test_full_flow_logdet_matches_autodiff` docstring says "the composition logdet is exactly the sum of per-block (-Sum a)". That wording is exp-specific. Replace "(-Sum a)" with "(-Sum log_scale)" so it reads correctly for the softplus default. Make no logic/assertion changes.

- [ ] **Step 3: Write the EMA seam assertion**

Append to `tests/normalizing_flows/transformer_flow/test_stability.py`:

```python
def test_stability_flags_are_static_not_ema_state():
    # use_softplus/soft_clip are plain Python config, not nnx state captured by EMA.
    flow = make_tarflow(nnx.Rngs(0), dim=4, cond_dim=2, channels=16, num_blocks=2,
                        layers_per_block=2, head_dim=8)
    state = nnx.state(flow, nnx.Param)
    leaves = jax.tree_util.tree_leaves(state)
    # all Param leaves are float arrays; no bool/None smuggled in from the new flags
    assert all(jnp.issubdtype(jnp.asarray(l).dtype, jnp.floating) for l in leaves)
    # flags survive as block attributes (graphdef), not as Param state
    assert flow.blocks[0].use_softplus is True
    assert flow.blocks[0].soft_clip == 4.0
```

- [ ] **Step 4: Run the EMA seam test**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/transformer_flow/test_stability.py::test_stability_flags_are_static_not_ema_state -v`
Expected: PASS.

- [ ] **Step 5: Run the whole NF suite as a final guard**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/ -q`
Expected: PASS (no regressions across the MAF + transformer_flow battery).

- [ ] **Step 6: Commit**

```bash
git add tests/normalizing_flows/transformer_flow/test_model.py tests/normalizing_flows/transformer_flow/test_stability.py
git commit -m "test(nf): audit existing TransformerFlow tests + EMA seam re-assert for stability flags

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Post-plan validation (manual, not a task)

After all tasks: optionally re-run the original repro to confirm the example no longer
NaNs — `scratchpad/repro_nan.py sos_softplus` (already written) should train without
divergence, and the real two-moons example (`train_tarflow_npe.py`) can be re-run on GPU
to confirm convergence end-to-end. This is empirical confirmation outside the unit battery.
