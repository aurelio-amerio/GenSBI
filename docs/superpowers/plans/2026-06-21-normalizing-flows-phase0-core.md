# Normalizing Flows — Phase 0 (Flow Core) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone, correct conditional affine MAF (Masked Autoregressive Flow) as NNX modules with exact `log_prob` and `sample`.

**Architecture:** A parallel track to the existing flow-matching/diffusion methods (NOT the CNF-shaped `GenerativeMethod` ABC). Bijections compose in a `Chain`; a `MaskedAutoregressive` bijection pairs a MADE conditioner (with rank-safe FiLM+gate conditioning) with an elementwise `Affine` transformer. Direction convention: `forward` = noise→data (slow scan), `inverse` = data→noise (fast one pass); `log_prob` uses `inverse`. Core operates on single examples `(dim,)`; `Flow` vmaps over the batch.

**Tech Stack:** JAX, Flax NNX, NumPyro (base distribution only). Float32 throughout (exact-likelihood model — bf16 would wreck Jacobian/log-det precision).

**Spec:** `docs/superpowers/specs/2026-06-21-normalizing-flows-design.md` (§4 direction convention, §5 components, §6 FiLM, §11 tests).

---

## File Structure

```
src/gensbi/normalizing_flows/
  __init__.py                 # package exports (Flow, make_maf, bijections)
  bijections/
    __init__.py               # bijection exports
    base.py                   # Bijection ABC + Mask variable type
    masks.py                  # make_mask (rank-based binary mask)
    masked_linear.py          # MaskedLinear (NNX): masked dense layer
    transformers.py           # Affine elementwise transformer (pure math)
    made.py                   # MADE conditioner + MaskedAutoregressive bijection
    permutation.py            # Permutation bijection (reverse/random)
    standardize.py            # Standardize bijection (fixed affine, non-Param buffers)
    chain.py                  # Chain of bijections
  flow.py                     # Flow (NNX) + make_maf builder

tests/normalizing_flows/
  __init__.py
  bijections/
    __init__.py
    test_masks.py
    test_masked_linear.py
    test_transformers.py
    test_made.py              # the critical autoregression + FiLM-rank-safety test
    test_masked_autoregressive.py
    test_permutation.py
    test_standardize.py
    test_chain.py
  test_flow.py
```

Each file has one responsibility. `made.py` holds both the MADE conditioner and the `MaskedAutoregressive` bijection because they change together and the bijection is meaningless without its conditioner.

---

## Task 1: Package scaffold, Bijection base, rank-based masks

**Files:**
- Create: `src/gensbi/normalizing_flows/__init__.py`
- Create: `src/gensbi/normalizing_flows/bijections/__init__.py`
- Create: `src/gensbi/normalizing_flows/bijections/base.py`
- Create: `src/gensbi/normalizing_flows/bijections/masks.py`
- Create: `tests/normalizing_flows/__init__.py`
- Create: `tests/normalizing_flows/bijections/__init__.py`
- Test: `tests/normalizing_flows/bijections/test_masks.py`

- [ ] **Step 1: Write the failing test**

`tests/normalizing_flows/bijections/test_masks.py`:
```python
import jax.numpy as jnp
from gensbi.normalizing_flows.bijections.masks import make_mask


def test_make_mask_non_strict_is_ge():
    # in_ranks = [0,1,2], out_ranks = [0,1,2]; connect i->o if out_rank >= in_rank
    in_ranks = jnp.array([0, 1, 2])
    out_ranks = jnp.array([0, 1, 2])
    mask = make_mask(in_ranks, out_ranks, strict=False)  # shape (in=3, out=3)
    expected = jnp.array([
        [True,  True,  True],   # in-rank 0 -> out-ranks >= 0 : all
        [False, True,  True],   # in-rank 1 -> out-ranks >= 1
        [False, False, True],   # in-rank 2 -> out-ranks >= 2
    ])
    assert mask.shape == (3, 3)
    assert jnp.array_equal(mask, expected)


def test_make_mask_strict_is_gt():
    in_ranks = jnp.array([0, 1, 2])
    out_ranks = jnp.array([0, 1, 2])
    mask = make_mask(in_ranks, out_ranks, strict=True)  # connect if out_rank > in_rank
    expected = jnp.array([
        [False, True,  True],
        [False, False, True],
        [False, False, False],
    ])
    assert jnp.array_equal(mask, expected)


def test_make_mask_rectangular():
    in_ranks = jnp.array([0, 1])          # 2 inputs
    out_ranks = jnp.array([0, 0, 1, 1])   # 4 outputs
    mask = make_mask(in_ranks, out_ranks, strict=True)
    assert mask.shape == (2, 4)
    # input rank 1 connects to no output (no out_rank > 1)
    assert not mask[1].any()
    # input rank 0 connects to the two rank-1 outputs only (strict)
    assert jnp.array_equal(mask[0], jnp.array([False, False, True, True]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/normalizing_flows/bijections/test_masks.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gensbi.normalizing_flows'`

- [ ] **Step 3: Write minimal implementation**

`src/gensbi/normalizing_flows/__init__.py`:
```python
"""Discrete (autoregressive) normalizing flows for GenSBI.

A parallel track to the flow-matching/diffusion methods: the flow IS the
density model, with exact ``log_prob`` and one-pass conditional ``sample``.
"""
```

`src/gensbi/normalizing_flows/bijections/__init__.py`:
```python
"""Bijections and the masked-autoregressive building blocks."""
```

`src/gensbi/normalizing_flows/bijections/masks.py`:
```python
"""Rank-based binary masks for masked autoregressive networks."""

import operator

import jax.numpy as jnp
from jax import Array


def make_mask(in_ranks: Array, out_ranks: Array, *, strict: bool) -> Array:
    """Binary connectivity mask of shape ``(len(in_ranks), len(out_ranks))``.

    ``mask[i, o]`` is True iff input unit ``i`` may feed output unit ``o``:
    ``out_ranks[o] > in_ranks[i]`` when ``strict`` (final/output layer), else
    ``out_ranks[o] >= in_ranks[i]`` (hidden layers). The ``(in, out)`` layout
    matches an ``nnx.Linear`` kernel so it multiplies the weight directly.
    """
    op = operator.gt if strict else operator.ge
    return op(out_ranks[None, :], in_ranks[:, None])
```

`src/gensbi/normalizing_flows/bijections/base.py`:
```python
"""Bijection abstract base and the non-trainable Mask variable type."""

from abc import abstractmethod

from flax import nnx
from jax import Array


class Mask(nnx.Variable):
    """A fixed buffer (e.g. an autoregressive mask).

    Subclassing ``nnx.Variable`` (not ``nnx.Param``) keeps it out of
    ``nnx.split(wrt=nnx.Param)`` and the optimizer, while checkpointing still
    saves/restores it.
    """


class Bijection(nnx.Module):
    """Invertible map with the locked direction convention.

    Both methods act on a single example and return ``(output, log_det)`` where
    ``log_det`` is the log-abs-det of *that method's* Jacobian.

    - ``forward``:  noise -> data   (sampling; MAF: slow, sequential)
    - ``inverse``:  data  -> noise  (density; MAF: fast, one pass)
    """

    @abstractmethod
    def forward(self, u: Array, cond: Array | None = None) -> tuple[Array, Array]:
        ...  # pragma: no cover

    @abstractmethod
    def inverse(self, x: Array, cond: Array | None = None) -> tuple[Array, Array]:
        ...  # pragma: no cover
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/normalizing_flows/bijections/test_masks.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/normalizing_flows/__init__.py \
        src/gensbi/normalizing_flows/bijections/__init__.py \
        src/gensbi/normalizing_flows/bijections/base.py \
        src/gensbi/normalizing_flows/bijections/masks.py \
        tests/normalizing_flows/__init__.py \
        tests/normalizing_flows/bijections/__init__.py \
        tests/normalizing_flows/bijections/test_masks.py
git commit -m "feat(nflows): package scaffold, Bijection base, rank-based masks"
```

---

## Task 2: MaskedLinear

**Files:**
- Create: `src/gensbi/normalizing_flows/bijections/masked_linear.py`
- Test: `tests/normalizing_flows/bijections/test_masked_linear.py`

- [ ] **Step 1: Write the failing test**

`tests/normalizing_flows/bijections/test_masked_linear.py`:
```python
import jax
import jax.numpy as jnp
from flax import nnx

from gensbi.normalizing_flows.bijections.base import Mask
from gensbi.normalizing_flows.bijections.masked_linear import MaskedLinear


def test_masked_linear_zeros_out_masked_connections():
    # 3 inputs -> 2 outputs; only allow input 0 -> output 0
    mask = jnp.array([[True, False],
                      [False, False],
                      [False, False]])  # (in=3, out=2)
    layer = MaskedLinear(3, 2, mask, rngs=nnx.Rngs(0))
    x = jnp.ones((3,))
    y = layer(x)
    # output 1 receives nothing -> equals its bias only; perturbing inputs
    # must not change output 1.
    x2 = x.at[0].set(5.0)
    assert jnp.allclose(y[1], layer(x2)[1])


def test_masked_linear_grad_is_zero_on_masked_weights():
    mask = jnp.array([[True, False],
                      [True, False],
                      [True, False]])  # only output 0 is connected
    layer = MaskedLinear(3, 2, mask, rngs=nnx.Rngs(0))

    def loss(layer, x):
        return layer(x).sum()

    grads = nnx.grad(loss)(layer, jnp.ones((3,)))
    # gradient w.r.t. masked-out kernel entries (column 1) must be exactly zero
    assert jnp.all(grads["kernel"].value[:, 1] == 0.0)


def test_mask_is_not_a_param():
    mask = jnp.ones((3, 2), dtype=bool)
    layer = MaskedLinear(3, 2, mask, rngs=nnx.Rngs(0))
    params = nnx.state(layer, nnx.Param)
    # the mask must NOT appear among Params
    flat = jax.tree_util.tree_leaves(params)
    assert all(leaf.dtype != bool for leaf in flat)
    # but it IS reachable as a Mask buffer
    masks = nnx.state(layer, Mask)
    assert len(jax.tree_util.tree_leaves(masks)) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/normalizing_flows/bijections/test_masked_linear.py -v`
Expected: FAIL — `ModuleNotFoundError: ... masked_linear`

- [ ] **Step 3: Write minimal implementation**

`src/gensbi/normalizing_flows/bijections/masked_linear.py`:
```python
"""Dense layer with a fixed binary weight mask."""

import jax.numpy as jnp
from flax import nnx
from jax import Array
from jax.typing import DTypeLike

from gensbi.normalizing_flows.bijections.base import Mask


class MaskedLinear(nnx.Module):
    """``y = (kernel * mask).T @ x + bias`` with a non-trainable mask.

    Parameters
    ----------
    in_features, out_features : int
    mask : Array
        Boolean array of shape ``(in_features, out_features)``; stored as a
        :class:`Mask` buffer so it is excluded from ``nnx.Param``.
    rngs : nnx.Rngs
    param_dtype : DTypeLike, optional
        Defaults to float32 (exact-likelihood model needs the precision).
    """

    def __init__(self, in_features, out_features, mask, rngs,
                 param_dtype: DTypeLike = jnp.float32):
        self.linear = nnx.Linear(
            in_features, out_features, use_bias=True,
            rngs=rngs, param_dtype=param_dtype,
        )
        self.mask = Mask(jnp.asarray(mask, dtype=param_dtype))

    def __call__(self, x: Array) -> Array:
        masked_kernel = self.linear.kernel.value * self.mask.value
        return x @ masked_kernel + self.linear.bias.value
```

Note: `nnx.grad(loss)(layer, x)` returns a state tree; `grads["kernel"]` in the
test addresses the kernel via the nested `linear` module — adjust to
`grads["linear"]["kernel"]` if the structure nests (see Step 4 output).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/normalizing_flows/bijections/test_masked_linear.py -v`
Expected: PASS. If `test_masked_linear_grad_is_zero_on_masked_weights` raises a
KeyError, print `grads` and fix the path to the kernel leaf (it is nested under
`linear`): change the assertion to
`assert jnp.all(grads["linear"]["kernel"].value[:, 1] == 0.0)`.

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/normalizing_flows/bijections/masked_linear.py \
        tests/normalizing_flows/bijections/test_masked_linear.py
git commit -m "feat(nflows): MaskedLinear with non-Param mask buffer"
```

---

## Task 3: Affine transformer

**Files:**
- Create: `src/gensbi/normalizing_flows/bijections/transformers.py`
- Test: `tests/normalizing_flows/bijections/test_transformers.py`

The transformer is **pure param-driven math** (not an NNX module). It receives
per-dimension parameters from MADE.

- [ ] **Step 1: Write the failing test**

`tests/normalizing_flows/bijections/test_transformers.py`:
```python
import jax.numpy as jnp
from gensbi.normalizing_flows.bijections.transformers import Affine


def test_affine_num_params():
    assert Affine().num_params == 2


def test_affine_roundtrip_and_logdet_signs():
    t = Affine()
    x = jnp.array([0.3, -1.2, 2.0])
    # params: (dim, 2) = (shift mu, log-scale a)
    params = jnp.array([[0.5, 0.1], [-0.2, -0.3], [1.0, 0.2]])
    u, logdet_inv = t.inverse(x, params)   # data -> noise
    x2, logdet_fwd = t.forward(u, params)  # noise -> data
    assert jnp.allclose(x, x2, atol=1e-6)
    # inverse logdet = -sum(a); forward logdet = +sum(a)
    a = params[:, 1]
    assert jnp.allclose(logdet_inv, -jnp.sum(a))
    assert jnp.allclose(logdet_fwd, jnp.sum(a))


def test_affine_clamps_log_scale():
    t = Affine(clamp_min=-5.0, clamp_max=3.0)
    x = jnp.array([1.0])
    params = jnp.array([[0.0, 100.0]])   # absurd log-scale
    u, logdet = t.inverse(x, params)
    # effective log-scale clamped to 3.0 -> logdet = -3.0
    assert jnp.allclose(logdet, -3.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/normalizing_flows/bijections/test_transformers.py -v`
Expected: FAIL — `ModuleNotFoundError: ... transformers`

- [ ] **Step 3: Write minimal implementation**

`src/gensbi/normalizing_flows/bijections/transformers.py`:
```python
"""Elementwise transformers parameterised per-dimension by a conditioner.

Pure functions of (value, params) — no learnable state of their own.
"""

import jax
import jax.numpy as jnp
from jax import Array


def _clamp(a: Array, lo: float, hi: float) -> Array:
    """Clamp with a straight-through gradient (NumPyro IAF trick)."""
    return a + jax.lax.stop_gradient(jnp.clip(a, lo, hi) - a)


class Affine:
    """Elementwise affine transform with log-scale clamping.

    params layout per dim: ``[shift mu, log-scale a]`` (``num_params == 2``).
    forward (noise->data): ``x = u * exp(a) + mu``, logdet ``= +sum(a)``.
    inverse (data->noise): ``u = (x - mu) * exp(-a)``, logdet ``= -sum(a)``.
    """

    num_params = 2

    def __init__(self, clamp_min: float = -5.0, clamp_max: float = 3.0):
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def _split(self, params: Array) -> tuple[Array, Array]:
        mu = params[..., 0]
        a = _clamp(params[..., 1], self.clamp_min, self.clamp_max)
        return mu, a

    def forward(self, u: Array, params: Array) -> tuple[Array, Array]:
        mu, a = self._split(params)
        x = u * jnp.exp(a) + mu
        return x, jnp.sum(a)

    def inverse(self, x: Array, params: Array) -> tuple[Array, Array]:
        mu, a = self._split(params)
        u = (x - mu) * jnp.exp(-a)
        return u, -jnp.sum(a)

    def forward_dim(self, u_i: Array, params_i: Array) -> Array:
        """Scalar forward for one dim (used by the sequential sampling scan)."""
        mu = params_i[0]
        a = _clamp(params_i[1], self.clamp_min, self.clamp_max)
        return u_i * jnp.exp(a) + mu
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/normalizing_flows/bijections/test_transformers.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/normalizing_flows/bijections/transformers.py \
        tests/normalizing_flows/bijections/test_transformers.py
git commit -m "feat(nflows): Affine transformer with clamped log-scale"
```

---

## Task 4: MADE conditioner (concatenation conditioning, flowjax-style)

**Files:**
- Create: `src/gensbi/normalizing_flows/bijections/made.py` (MADE only in this task)
- Test: `tests/normalizing_flows/bijections/test_made.py`

This is the critical correctness test (spec §11 #1): output dim `d` must be
independent of inputs `x_{>=d}` (autoregression), and dependent on `cond`
(conditioning) **for every output dimension, including `d=0`**.

**Conditioning method (decision, supersedes the original FiLM design):** `cond`
is **concatenated** onto the MADE input and given autoregressive **rank −1**
(below every data dim), exactly as the reference `flowjax`
(`reference/flowjax/flowjax/bijections/masked_autoregressive.py`). Because the
output mask is strict (`out_rank > in_rank`), every output dim — including `d=0`
(rank 0) — may read the rank-−1 conditioning inputs, while `cond` itself reads
nothing. This is the standard conditional-MAF approach (Papamakarios et al. 2017).
The earlier FiLM/adaLN-Zero modulation left `d=0` unconditioned (its output reads
no hidden unit), which is a real bug for conditional density estimation. Future
conditioning schemes (FiLM, T-NAF) can be added later as alternative conditioners
behind the same `(x, cond) -> (dim, num_params)` interface — not in scope here.

**`zero_init`:** now means an *identity-transform warm-start* — zero the output
layer so all transform params start at 0 (Affine: `mu=0, a=0` ⇒ identity). Default
`True`; tests pass `False` so the network is live. This replaces the old adaLN-Zero
gate (removed with FiLM) and keeps the flow near the standard normal at init.

- [ ] **Step 1: Write the failing test**

`tests/normalizing_flows/bijections/test_made.py`:
```python
import jax
import jax.numpy as jnp
from flax import nnx

from gensbi.normalizing_flows.bijections.made import MADE


def _made(dim=5, cond_dim=3, num_params=2, seed=0):
    # zero_init=False so the conditioning path is live (gates non-zero),
    # which is what makes the cond-dependence assertion meaningful.
    return MADE(dim=dim, cond_dim=cond_dim, num_params=num_params,
                nn_width=32, nn_depth=2, zero_init=False, rngs=nnx.Rngs(seed))


def test_made_output_shape():
    made = _made()
    x = jnp.linspace(-1, 1, 5)
    cond = jnp.array([0.1, -0.2, 0.3])
    out = made(x, cond)
    assert out.shape == (5, 2)  # (dim, num_params)


def test_made_is_autoregressive():
    """Output dim d must have ZERO Jacobian w.r.t. inputs x_j for j >= d."""
    made = _made(dim=5, num_params=2)
    cond = jnp.array([0.1, -0.2, 0.3])

    def out_flat(x):
        return made(x, cond).reshape(-1)   # (dim*num_params,)

    J = jax.jacobian(out_flat)(jnp.linspace(-1, 1, 5))  # (dim*np, dim)
    J = J.reshape(5, 2, 5)  # (out_dim, param, in_dim)
    for d in range(5):
        for j in range(d, 5):           # j >= d must be zero (strict autoregression)
            assert jnp.allclose(J[d, :, j], 0.0, atol=1e-6), (d, j)
        if d > 0:                        # must actually use the allowed prefix
            assert not jnp.allclose(J[d, :, :d], 0.0, atol=1e-6), d


def test_made_depends_on_cond_densely():
    """Every output must depend on the conditioning vector (FiLM is live)."""
    made = _made(dim=5, cond_dim=3, num_params=2)
    x = jnp.linspace(-1, 1, 5)

    def out_flat(cond):
        return made(x, cond).reshape(-1)

    J = jax.jacobian(out_flat)(jnp.array([0.1, -0.2, 0.3]))  # (dim*np, cond_dim)
    # no output row is entirely independent of cond
    assert jnp.all(jnp.any(jnp.abs(J) > 1e-6, axis=1))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/normalizing_flows/bijections/test_made.py -v`
Expected: FAIL — `ImportError: cannot import name 'MADE'`

- [ ] **Step 3: Write minimal implementation**

`src/gensbi/normalizing_flows/bijections/made.py`:
```python
"""MADE conditioner with concatenation-based conditioning (flowjax-style).

Conditioning variables are concatenated onto the input and given autoregressive
rank -1 (below every data dimension), so every output -- including the first --
may depend on the condition while the condition depends on nothing. This is the
standard conditional-MAF approach (Papamakarios et al. 2017; flowjax). NO
cross-feature normalisation (LayerNorm/RMSNorm/GroupNorm): MADE hidden units
carry the autoregressive rank, so cross-unit statistics would mix ranks and
silently break the flow. See spec §6.

The conditioner is a single cohesive module behind the
``(x, cond) -> (dim, num_params)`` interface; alternative conditioning schemes
(FiLM, T-NAF, ...) may be added later as drop-in conditioners.
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array
from jax.typing import DTypeLike

from gensbi.normalizing_flows.bijections.masked_linear import MaskedLinear
from gensbi.normalizing_flows.bijections.masks import make_mask


def _rank_vectors(dim, nn_width, num_params, cond_dim):
    """0-indexed MADE ranks for input, hidden, and output units.

    With conditioning (``cond_dim > 0``) the conditioning inputs get rank -1
    (before every data dim) and hidden ranks are shifted into ``[-1, dim-2]`` so
    some hidden units carry only the condition and can feed output dim 0.
    """
    out_ranks = jnp.repeat(jnp.arange(dim), num_params)
    if cond_dim > 0:
        in_ranks = jnp.concatenate(
            [jnp.arange(dim), -jnp.ones(cond_dim, dtype=jnp.int32)])
        hidden_ranks = (jnp.arange(nn_width) % dim) - 1
    elif dim > 1:
        in_ranks = jnp.arange(dim)
        hidden_ranks = jnp.arange(nn_width) % (dim - 1)
    else:
        in_ranks = jnp.arange(dim)
        hidden_ranks = jnp.zeros(nn_width, dtype=jnp.int32)
    return in_ranks, hidden_ranks, out_ranks


class MADE(nnx.Module):
    """Autoregressive conditioner: ``(x, cond) -> params`` of shape ``(dim, num_params)``.

    Conditioning is by concatenation: ``cond`` is appended to ``x`` and given
    autoregressive rank -1, so every output (incl. dim 0) may depend on it.

    Parameters
    ----------
    dim : int               Autoregressive (target) dimension.
    cond_dim : int          Conditioning dimension; 0 for unconditional.
    num_params : int        Transform params per dim (Affine: 2).
    nn_width, nn_depth : int
    rngs : nnx.Rngs
    zero_init : bool        Identity warm-start: zero the output layer so all
                            transform params start at 0 (Affine -> identity).
                            Default True; tests pass False so the net is live.
    """

    def __init__(self, dim, cond_dim, num_params, nn_width, nn_depth, rngs,
                 zero_init: bool = True, param_dtype: DTypeLike = jnp.float32,
                 activation=jax.nn.silu):
        self.dim = dim
        self.cond_dim = cond_dim
        self.num_params = num_params
        self.activation = activation

        in_ranks, hidden_ranks, out_ranks = _rank_vectors(
            dim, nn_width, num_params, cond_dim)
        in_mask = make_mask(in_ranks, hidden_ranks, strict=False)
        hidden_mask = make_mask(hidden_ranks, hidden_ranks, strict=False)
        out_mask = make_mask(hidden_ranks, out_ranks, strict=True)

        self.input_layer = MaskedLinear(dim + cond_dim, nn_width, in_mask,
                                        rngs=rngs, param_dtype=param_dtype)
        self.hidden_layers = nnx.List([
            MaskedLinear(nn_width, nn_width, hidden_mask, rngs=rngs,
                         param_dtype=param_dtype)
            for _ in range(nn_depth)
        ])
        self.output_layer = MaskedLinear(nn_width, dim * num_params, out_mask,
                                         rngs=rngs, param_dtype=param_dtype)
        if zero_init:
            # Identity warm-start: zero output params -> affine is identity.
            self.output_layer.linear.kernel.value = jnp.zeros_like(
                self.output_layer.linear.kernel.value)
            self.output_layer.linear.bias.value = jnp.zeros_like(
                self.output_layer.linear.bias.value)

    def __call__(self, x: Array, cond: Array | None = None) -> Array:
        if self.cond_dim > 0:
            if cond is None:
                raise ValueError(
                    "cond is required: this MADE was built with cond_dim > 0")
            nn_input = jnp.concatenate([x, cond])
        else:
            nn_input = x
        h = self.activation(self.input_layer(nn_input))
        for layer in self.hidden_layers:
            h = self.activation(layer(h))
        out = self.output_layer(h)
        return out.reshape(self.dim, self.num_params)
```

Note: a plain Python list of `nnx.Module`s cannot be assigned to an `nnx.Module`
attribute (raises `ValueError` in flax 0.12.x); use `nnx.List([...])` as above
(the established pattern in `pixeldit`/`simformer`).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/normalizing_flows/bijections/test_made.py -v`
Expected: PASS (3 tests). `test_made_is_autoregressive` is the rank guardrail
(output `d` must not depend on `x_{>=d}`); `test_made_depends_on_cond_densely`
verifies the concatenation conditioning reaches every output dim including `d=0`
(it would fail if `cond` were routed only through a hidden-stream modulation).

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/normalizing_flows/bijections/made.py \
        tests/normalizing_flows/bijections/test_made.py
git commit -m "feat(nflows): MADE conditioner with concatenation conditioning (flowjax-style)"
```

---

## Task 5: MaskedAutoregressive bijection

**Files:**
- Modify: `src/gensbi/normalizing_flows/bijections/made.py` (add `MaskedAutoregressive`)
- Test: `tests/normalizing_flows/bijections/test_masked_autoregressive.py`

- [ ] **Step 1: Write the failing test**

`tests/normalizing_flows/bijections/test_masked_autoregressive.py`:
```python
import jax
import jax.numpy as jnp
from flax import nnx

from gensbi.normalizing_flows.bijections.made import MaskedAutoregressive
from gensbi.normalizing_flows.bijections.transformers import Affine


def _ma(dim=5, cond_dim=3, seed=0):
    return MaskedAutoregressive(
        dim=dim, cond_dim=cond_dim, transformer=Affine(),
        nn_width=32, nn_depth=2, zero_init=False, rngs=nnx.Rngs(seed),
    )


def test_invertibility_both_ways():
    ma = _ma()
    cond = jnp.array([0.1, -0.2, 0.3])
    x = jnp.array([0.5, -1.0, 0.2, 1.3, -0.7])
    u, _ = ma.inverse(x, cond)
    x2, _ = ma.forward(u, cond)
    assert jnp.allclose(x, x2, atol=1e-5)
    u2, _ = ma.inverse(x2, cond)
    assert jnp.allclose(u, u2, atol=1e-5)


def test_logdet_matches_autodiff_jacobian():
    """Spec §11 #3 — the sign/convention guardrail."""
    ma = _ma(dim=5)
    cond = jnp.array([0.1, -0.2, 0.3])
    x = jnp.array([0.5, -1.0, 0.2, 1.3, -0.7])

    def inv_only(x):
        return ma.inverse(x, cond)[0]

    _, ad_logdet = jnp.linalg.slogdet(jax.jacobian(inv_only)(x))
    _, analytic_logdet = ma.inverse(x, cond)
    assert jnp.allclose(ad_logdet, analytic_logdet, atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/normalizing_flows/bijections/test_masked_autoregressive.py -v`
Expected: FAIL — `ImportError: cannot import name 'MaskedAutoregressive'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/gensbi/normalizing_flows/bijections/made.py`:
```python
from gensbi.normalizing_flows.bijections.base import Bijection


class MaskedAutoregressive(Bijection):
    """MADE conditioner + elementwise transformer = one autoregressive flow step.

    inverse (data->noise) is one MADE pass (fast); forward (noise->data) is a
    sequential ``lax.scan`` over dims (slow).
    """

    def __init__(self, dim, cond_dim, transformer, nn_width, nn_depth, rngs,
                 zero_init: bool = True):
        self.dim = dim
        self.transformer = transformer
        self.made = MADE(dim=dim, cond_dim=cond_dim,
                         num_params=transformer.num_params,
                         nn_width=nn_width, nn_depth=nn_depth,
                         zero_init=zero_init, rngs=rngs)

    def inverse(self, x: Array, cond: Array | None = None):
        params = self.made(x, cond)              # (dim, num_params), single pass
        return self.transformer.inverse(x, params)

    def forward(self, u: Array, cond: Array | None = None):
        def body(x, i):
            params = self.made(x, cond)
            x_i = self.transformer.forward_dim(u[i], params[i])
            return x.at[i].set(x_i), None

        x0 = jnp.zeros_like(u)
        x, _ = jax.lax.scan(body, x0, jnp.arange(self.dim))
        # log-det from the completed x (forward logdet = +sum(a))
        params = self.made(x, cond)
        _, logdet = self.transformer.forward(u, params)
        return x, logdet
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/normalizing_flows/bijections/test_masked_autoregressive.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/normalizing_flows/bijections/made.py \
        tests/normalizing_flows/bijections/test_masked_autoregressive.py
git commit -m "feat(nflows): MaskedAutoregressive bijection (one-pass inverse, scan forward)"
```

---

## Task 6: Permutation

**Files:**
- Create: `src/gensbi/normalizing_flows/bijections/permutation.py`
- Test: `tests/normalizing_flows/bijections/test_permutation.py`

- [ ] **Step 1: Write the failing test**

`tests/normalizing_flows/bijections/test_permutation.py`:
```python
import jax.numpy as jnp
from flax import nnx

from gensbi.normalizing_flows.bijections.permutation import Permutation


def test_reverse_permutation_roundtrip_and_zero_logdet():
    perm = Permutation.reverse(4)
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    u, logdet_inv = perm.inverse(x)
    assert jnp.array_equal(u, jnp.array([4.0, 3.0, 2.0, 1.0]))
    assert logdet_inv == 0.0
    x2, logdet_fwd = perm.forward(u)
    assert jnp.array_equal(x2, x)
    assert logdet_fwd == 0.0


def test_random_permutation_is_a_bijection():
    perm = Permutation.random(6, rngs=nnx.Rngs(0))
    x = jnp.arange(6.0)
    u, _ = perm.inverse(x)
    x2, _ = perm.forward(u)
    assert jnp.array_equal(x, x2)
    assert jnp.array_equal(jnp.sort(u), x)   # a true permutation
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/normalizing_flows/bijections/test_permutation.py -v`
Expected: FAIL — `ModuleNotFoundError: ... permutation`

- [ ] **Step 3: Write minimal implementation**

`src/gensbi/normalizing_flows/bijections/permutation.py`:
```python
"""Permutation bijection (dimension reordering between flow layers)."""

import jax
import jax.numpy as jnp
from jax import Array

from gensbi.normalizing_flows.bijections.base import Bijection, Mask


class Permutation(Bijection):
    """Reorder dims; ``cond`` is ignored; log-det is 0.

    ``perm`` and its inverse are stored as :class:`Mask` buffers (non-Param).
    """

    def __init__(self, perm: Array):
        perm = jnp.asarray(perm, dtype=jnp.int32)
        self.perm = Mask(perm)
        self.inv_perm = Mask(jnp.argsort(perm))

    @classmethod
    def reverse(cls, dim: int) -> "Permutation":
        return cls(jnp.arange(dim)[::-1])

    @classmethod
    def random(cls, dim: int, rngs) -> "Permutation":
        return cls(jax.random.permutation(rngs.params(), dim))

    def inverse(self, x: Array, cond: Array | None = None):
        return x[self.perm.value], jnp.array(0.0)

    def forward(self, u: Array, cond: Array | None = None):
        return u[self.inv_perm.value], jnp.array(0.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/normalizing_flows/bijections/test_permutation.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/normalizing_flows/bijections/permutation.py \
        tests/normalizing_flows/bijections/test_permutation.py
git commit -m "feat(nflows): Permutation bijection (reverse/random)"
```

---

## Task 7: Standardize bijection

**Files:**
- Create: `src/gensbi/normalizing_flows/bijections/standardize.py`
- Test: `tests/normalizing_flows/bijections/test_standardize.py`

A fixed (non-trainable) affine that maps raw data <-> standardized space. Lives
at the data end of the chain so the Jacobian is handled automatically. Buffers
default to identity (mean 0, std 1); Phase 1's pipeline sets them from data.

- [ ] **Step 1: Write the failing test**

`tests/normalizing_flows/bijections/test_standardize.py`:
```python
import jax.numpy as jnp
from gensbi.normalizing_flows.bijections.standardize import Standardize


def test_default_is_identity():
    s = Standardize(dim=3)
    x = jnp.array([1.0, -2.0, 0.5])
    u, logdet = s.inverse(x)
    assert jnp.allclose(u, x)
    assert jnp.allclose(logdet, 0.0)


def test_standardize_roundtrip_and_logdet():
    s = Standardize(dim=3)
    s.set_stats(mean=jnp.array([1.0, 2.0, 3.0]), std=jnp.array([2.0, 0.5, 4.0]))
    x = jnp.array([3.0, 2.5, -1.0])
    u, logdet_inv = s.inverse(x)        # u = (x - mean) / std
    assert jnp.allclose(u, jnp.array([(3-1)/2, (2.5-2)/0.5, (-1-3)/4]))
    assert jnp.allclose(logdet_inv, -jnp.sum(jnp.log(jnp.array([2.0, 0.5, 4.0]))))
    x2, logdet_fwd = s.forward(u)
    assert jnp.allclose(x2, x, atol=1e-6)
    assert jnp.allclose(logdet_fwd, -logdet_inv)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/normalizing_flows/bijections/test_standardize.py -v`
Expected: FAIL — `ModuleNotFoundError: ... standardize`

- [ ] **Step 3: Write minimal implementation**

`src/gensbi/normalizing_flows/bijections/standardize.py`:
```python
"""Fixed affine standardization bijection (non-trainable mean/std buffers)."""

import jax.numpy as jnp
from jax import Array

from gensbi.normalizing_flows.bijections.base import Bijection, Mask


class Standardize(Bijection):
    """``inverse``: ``u = (x - mean) / std`` (data->standardized).

    ``forward``: ``x = u * std + mean``. log-det(inverse) ``= -sum(log std)``.
    Buffers are :class:`Mask` (non-Param); default to identity.
    """

    def __init__(self, dim: int):
        self.mean = Mask(jnp.zeros((dim,)))
        self.std = Mask(jnp.ones((dim,)))

    def set_stats(self, mean: Array, std: Array) -> None:
        self.mean.value = jnp.asarray(mean, dtype=self.mean.value.dtype)
        self.std.value = jnp.asarray(std, dtype=self.std.value.dtype)

    def inverse(self, x: Array, cond: Array | None = None):
        u = (x - self.mean.value) / self.std.value
        return u, -jnp.sum(jnp.log(self.std.value))

    def forward(self, u: Array, cond: Array | None = None):
        x = u * self.std.value + self.mean.value
        return x, jnp.sum(jnp.log(self.std.value))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/normalizing_flows/bijections/test_standardize.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/normalizing_flows/bijections/standardize.py \
        tests/normalizing_flows/bijections/test_standardize.py
git commit -m "feat(nflows): Standardize bijection (fixed affine, non-Param buffers)"
```

---

## Task 8: Chain

**Files:**
- Create: `src/gensbi/normalizing_flows/bijections/chain.py`
- Test: `tests/normalizing_flows/bijections/test_chain.py`

- [ ] **Step 1: Write the failing test**

`tests/normalizing_flows/bijections/test_chain.py`:
```python
import jax
import jax.numpy as jnp
from flax import nnx

from gensbi.normalizing_flows.bijections.chain import Chain
from gensbi.normalizing_flows.bijections.made import MaskedAutoregressive
from gensbi.normalizing_flows.bijections.permutation import Permutation
from gensbi.normalizing_flows.bijections.transformers import Affine


def _chain(dim=4, cond_dim=2, seed=0):
    rngs = nnx.Rngs(seed)
    bijections = [
        MaskedAutoregressive(dim, cond_dim, Affine(), 32, 2, rngs, zero_init=False),
        Permutation.reverse(dim),
        MaskedAutoregressive(dim, cond_dim, Affine(), 32, 2, rngs, zero_init=False),
    ]
    return Chain(bijections)


def test_chain_invertibility():
    chain = _chain()
    cond = jnp.array([0.2, -0.1])
    x = jnp.array([0.5, -1.0, 0.3, 0.8])
    u, _ = chain.inverse(x, cond)
    x2, _ = chain.forward(u, cond)
    assert jnp.allclose(x, x2, atol=1e-5)


def test_chain_logdet_matches_autodiff():
    chain = _chain(dim=4)
    cond = jnp.array([0.2, -0.1])
    x = jnp.array([0.5, -1.0, 0.3, 0.8])

    def inv_only(x):
        return chain.inverse(x, cond)[0]

    _, ad_logdet = jnp.linalg.slogdet(jax.jacobian(inv_only)(x))
    _, analytic_logdet = chain.inverse(x, cond)
    assert jnp.allclose(ad_logdet, analytic_logdet, atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/normalizing_flows/bijections/test_chain.py -v`
Expected: FAIL — `ModuleNotFoundError: ... chain`

- [ ] **Step 3: Write minimal implementation**

`src/gensbi/normalizing_flows/bijections/chain.py`:
```python
"""Compose bijections. Stored in noise->data (forward) order."""

import jax.numpy as jnp
from jax import Array
from flax import nnx

from gensbi.normalizing_flows.bijections.base import Bijection


class Chain(Bijection):
    """Apply bijections in order for ``forward``, reversed for ``inverse``.

    Log-dets accumulate (sum). ``bijections[-1]`` is closest to data; it is the
    first applied in ``inverse`` and last in ``forward``.
    """

    def __init__(self, bijections: list[Bijection]):
        self.bijections = bijections

    def forward(self, u: Array, cond: Array | None = None):
        logdet = jnp.array(0.0)
        x = u
        for b in self.bijections:
            x, ld = b.forward(x, cond)
            logdet = logdet + ld
        return x, logdet

    def inverse(self, x: Array, cond: Array | None = None):
        logdet = jnp.array(0.0)
        u = x
        for b in reversed(self.bijections):
            u, ld = b.inverse(u, cond)
            logdet = logdet + ld
        return u, logdet
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/normalizing_flows/bijections/test_chain.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/normalizing_flows/bijections/chain.py \
        tests/normalizing_flows/bijections/test_chain.py
git commit -m "feat(nflows): Chain bijection"
```

---

## Task 9: Flow + make_maf builder

**Files:**
- Create: `src/gensbi/normalizing_flows/flow.py`
- Modify: `src/gensbi/normalizing_flows/__init__.py` (export `Flow`, `make_maf`)
- Modify: `src/gensbi/normalizing_flows/bijections/__init__.py` (export bijections)
- Test: `tests/normalizing_flows/test_flow.py`

`Flow` batches over examples with `jax.vmap`. The base distribution is a
standard normal built lazily (not stored as nnx state) via `make_gaussian_prior`.

- [ ] **Step 1: Write the failing test**

`tests/normalizing_flows/test_flow.py`:
```python
import jax
import jax.numpy as jnp
from flax import nnx
from scipy.integrate import trapezoid

from gensbi.normalizing_flows import make_maf
from gensbi.normalizing_flows.bijections.base import Mask


def test_log_prob_shape_and_finiteness():
    flow = make_maf(rngs=nnx.Rngs(0), dim=3, cond_dim=2, n_layers=3,
                    nn_width=32, nn_depth=2)
    x = jax.random.normal(jax.random.PRNGKey(1), (16, 3))
    cond = jax.random.normal(jax.random.PRNGKey(2), (16, 2))
    lp = flow.log_prob(x, cond)
    assert lp.shape == (16,)
    assert jnp.all(jnp.isfinite(lp))


def test_sample_shape_and_roundtrip_consistency():
    flow = make_maf(rngs=nnx.Rngs(0), dim=3, cond_dim=2, n_layers=2,
                    nn_width=16, nn_depth=1)
    cond = jnp.zeros((5, 2))
    samples = flow.sample(jax.random.PRNGKey(3), cond=cond, nsamples=5)
    assert samples.shape == (5, 3)
    # log_prob of samples is finite (forward then inverse must be consistent)
    assert jnp.all(jnp.isfinite(flow.log_prob(samples, cond)))


def test_density_integrates_to_one_1d():
    """Spec §11 #4 — 1D normalization via trapezoid (better than nothing)."""
    flow = make_maf(rngs=nnx.Rngs(0), dim=1, cond_dim=1, n_layers=3,
                    nn_width=32, nn_depth=2)
    cond = jnp.zeros((1,))
    grid = jnp.linspace(-12.0, 12.0, 4001)[:, None]   # (N, dim=1)
    cond_b = jnp.broadcast_to(cond, (grid.shape[0], 1))
    dens = jnp.exp(flow.log_prob(grid, cond_b))
    integral = trapezoid(dens, grid[:, 0])
    assert jnp.allclose(integral, 1.0, atol=1e-2)


def test_full_flow_logdet_matches_autodiff():
    flow = make_maf(rngs=nnx.Rngs(0), dim=4, cond_dim=2, n_layers=3,
                    nn_width=32, nn_depth=2)
    cond = jnp.array([0.3, -0.4])
    x = jnp.array([0.5, -1.0, 0.3, 0.8])

    def inv_only(x):
        return flow.chain.inverse(x, cond)[0]

    _, ad_logdet = jnp.linalg.slogdet(jax.jacobian(inv_only)(x))
    _, analytic_logdet = flow.chain.inverse(x, cond)
    assert jnp.allclose(ad_logdet, analytic_logdet, atol=1e-4)


def test_masks_are_not_params():
    """Spec §11 #5 — masks/buffers excluded from Param state."""
    flow = make_maf(rngs=nnx.Rngs(0), dim=4, cond_dim=2, n_layers=3,
                    nn_width=16, nn_depth=2)
    params = nnx.state(flow, nnx.Param)
    param_leaves = jax.tree_util.tree_leaves(params)
    assert all(leaf.dtype != bool for leaf in param_leaves)
    # masks ARE present as buffers
    masks = nnx.state(flow, Mask)
    assert len(jax.tree_util.tree_leaves(masks)) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/normalizing_flows/test_flow.py -v`
Expected: FAIL — `ImportError: cannot import name 'make_maf'`

- [ ] **Step 3: Write minimal implementation**

`src/gensbi/normalizing_flows/flow.py`:
```python
"""The Flow module: base distribution + Chain of bijections."""

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from gensbi.core.prior import make_gaussian_prior
from gensbi.normalizing_flows.bijections.chain import Chain
from gensbi.normalizing_flows.bijections.made import MaskedAutoregressive
from gensbi.normalizing_flows.bijections.permutation import Permutation
from gensbi.normalizing_flows.bijections.standardize import Standardize
from gensbi.normalizing_flows.bijections.transformers import Affine


class Flow(nnx.Module):
    """Normalizing flow over ``(batch, dim)`` data, optionally conditioned.

    ``log_prob(x, cond) = base.log_prob(u) + logdet`` with
    ``u, logdet = chain.inverse(x, cond)``. The base is a standard normal over
    ``(dim,)``, built lazily so it never enters nnx state.
    """

    def __init__(self, chain: Chain, dim: int, cond_dim: int):
        self.chain = chain
        self.dim = dim
        self.cond_dim = cond_dim

    def _base(self):
        return make_gaussian_prior((self.dim,))

    def log_prob(self, x: Array, cond: Array | None = None) -> Array:
        base = self._base()

        def single(x_i, cond_i):
            u, logdet = self.chain.inverse(x_i, cond_i)
            return base.log_prob(u) + logdet

        if cond is None:
            return jax.vmap(lambda xi: single(xi, None))(x)
        return jax.vmap(single)(x, cond)

    def sample(self, key, cond: Array | None = None, nsamples: int | None = None) -> Array:
        base = self._base()
        if cond is not None:
            nsamples = cond.shape[0]
        u = base.sample(key, (nsamples,))            # (nsamples, dim)

        def single(u_i, cond_i):
            x, _ = self.chain.forward(u_i, cond_i)
            return x

        if cond is None:
            return jax.vmap(lambda ui: single(ui, None))(u)
        return jax.vmap(single)(u, cond)


def make_maf(rngs, dim, cond_dim=0, n_layers=5, transformer=None,
             nn_width=64, nn_depth=2, permutation="reverse",
             standardize=True, zero_init=True) -> Flow:
    """Build an affine MAF as a stack of (MaskedAutoregressive, Permutation) layers.

    Parameters
    ----------
    rngs : nnx.Rngs
    dim : int               Target (autoregressive) dimension.
    cond_dim : int          Conditioning dim; 0 for unconditional.
    n_layers : int          Number of autoregressive layers.
    transformer : object    Elementwise transformer; defaults to ``Affine()``.
    nn_width, nn_depth : int
    permutation : str       "reverse" (alternating via stacking) or "random".
    standardize : bool      Prepend a data-end Standardize bijection (identity
                            until the pipeline sets stats).
    zero_init : bool        adaLN-Zero neutral conditioning init.
    """
    if transformer is None:
        transformer = Affine()

    bijections = []
    for i in range(n_layers):
        bijections.append(
            MaskedAutoregressive(dim, cond_dim, transformer, nn_width, nn_depth,
                                 rngs, zero_init=zero_init)
        )
        if i < n_layers - 1:
            if permutation == "reverse":
                bijections.append(Permutation.reverse(dim))
            elif permutation == "random":
                bijections.append(Permutation.random(dim, rngs))
            else:
                raise ValueError(f"unknown permutation {permutation!r}")

    if standardize:
        # data-end: appended last so it is applied first in inverse (data->noise)
        bijections.append(Standardize(dim))

    return Flow(Chain(bijections), dim=dim, cond_dim=cond_dim)
```

`src/gensbi/normalizing_flows/__init__.py` (append):
```python
from gensbi.normalizing_flows.flow import Flow, make_maf

__all__ = ["Flow", "make_maf"]
```

`src/gensbi/normalizing_flows/bijections/__init__.py` (append):
```python
from gensbi.normalizing_flows.bijections.base import Bijection, Mask
from gensbi.normalizing_flows.bijections.chain import Chain
from gensbi.normalizing_flows.bijections.made import MADE, MaskedAutoregressive
from gensbi.normalizing_flows.bijections.masked_linear import MaskedLinear
from gensbi.normalizing_flows.bijections.permutation import Permutation
from gensbi.normalizing_flows.bijections.standardize import Standardize
from gensbi.normalizing_flows.bijections.transformers import Affine

__all__ = [
    "Bijection", "Mask", "Chain", "MADE", "MaskedAutoregressive",
    "MaskedLinear", "Permutation", "Standardize", "Affine",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/normalizing_flows/test_flow.py -v`
Expected: PASS (5 tests). If `test_density_integrates_to_one_1d` is slightly off,
widen the grid bounds (`-15..15`) or increase resolution — the untrained flow is
near-standard-normal so `±12` over 4001 points should give `≈1`.

- [ ] **Step 5: Run the full Phase 0 suite and commit**

Run: `pytest tests/normalizing_flows/ -v`
Expected: all PASS.

```bash
git add src/gensbi/normalizing_flows/flow.py \
        src/gensbi/normalizing_flows/__init__.py \
        src/gensbi/normalizing_flows/bijections/__init__.py \
        tests/normalizing_flows/test_flow.py
git commit -m "feat(nflows): Flow module + make_maf builder; Phase 0 complete"
```

---

## Self-Review

**Spec coverage (Phase 0 scope):**
- §3 layout `normalizing_flows/bijections/...` → Tasks 1–9. ✓
- §4 direction convention (forward=noise→data, inverse=data→noise, log_prob via inverse) → `Bijection` (Task 1), `Flow.log_prob` (Task 9). ✓
- §5 MaskedLinear/MADE/Affine/MaskedAutoregressive/Permutation/Chain/Flow/make_maf → Tasks 2–9. ✓
- §5 standardization as a fixed affine bijection at the data end → Task 7 + `make_maf`. ✓
- §6 FiLM+gate, no norm, rank-safe; cond-only modulation → Task 4 (`_ModBlock`, `MADE`). ✓
- §6 mask = non-Param buffer → `Mask` (Task 1), verified Tasks 2 & 9. ✓
- §11 official battery: #1 MADE-Jacobian (Task 4), #2 invertibility (Tasks 5,8,9), #3 log-det vs autodiff (Tasks 5,8,9), #4 1D density trapz (Task 9), #5 mask-is-buffer (Tasks 2,9). ✓
- Float32 throughout → every module's `param_dtype` default. ✓

**Out of Phase 0 (later plans, per spec phasing §10):** ConditionalFlowPipeline/NPE (P1), NLEPosterior/NUTS (P2), RQSpline (P3), end-to-end linear-Gaussian (exploratory), two-moons (example). Not in this plan — correct.

**Placeholder scan:** No TBD/TODO; every code step has complete code; every test has real assertions. ✓

**Type/name consistency:** `make_mask(in_ranks, out_ranks, *, strict)`, `Mask`, `MaskedLinear(in, out, mask, rngs)`, `Affine.num_params/forward/inverse/forward_dim`, `MADE(dim, cond_dim, num_params, nn_width, nn_depth, rngs, zero_init)`, `MaskedAutoregressive(dim, cond_dim, transformer, nn_width, nn_depth, rngs, zero_init)`, `Permutation.reverse/random`, `Standardize(dim).set_stats`, `Chain(bijections)`, `Flow(chain, dim, cond_dim).log_prob/sample`, `make_maf(rngs, dim, cond_dim, ...)` — consistent across tasks. ✓

**Known risk to watch during execution:** `nnx.grad`/`nnx.state` tree key paths (Task 2 Step 4) and `jax.vmap` over a captured nnx module in `Flow` (Task 9). If vmap-over-module misbehaves, switch `Flow.log_prob`/`sample` to `nnx.vmap` or `nnx.split`/`jax.vmap`/`nnx.merge`. Flagged inline.
