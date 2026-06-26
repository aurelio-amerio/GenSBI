# NF Model Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recast the MAF and TarFlow normalizing flows as uniform `Model(Params(...))` classes under `gensbi.models`, leaving `gensbi.normalizing_flows` as a pure bijection-abstraction library.

**Architecture:** Three tiers — pure NF abstractions stay in `normalizing_flows/bijections/`; shared primitives (`patching`, `tokenizers`) move to `models/core/`; each model's machinery co-locates with its model class in `models/{maf,tarflow}/`. The `make_maf`/`make_tarflow` factories are dropped; construction is `MAFlow(MAFlowParams(...))` / `TarFlow(TarFlowParams(...))`. Moving `patchify` out of `recipes` severs the `normalizing_flows → recipes` edge so `models → normalizing_flows` is acyclic.

**Tech Stack:** Python, JAX, Flax NNX, einops, pytest.

## Global Constraints

- **General pipelines must stay green:** Flux1, Simformer, Flux1Joint, and the unified pipelines (`tests/recipes/`, `tests/flow_matching/`) are the only user-facing surface; their behaviour must not change.
- **Behaviour-identical refactor** except the one deliberate change: TarFlow head parameterization `(channels, head_dim)` → `(head_dim, num_heads)` with `channels = head_dim * num_heads` derived. Default `(head_dim=16, num_heads=4)` reproduces `channels=64`.
- **No import cycles:** `models/core` and `normalizing_flows/bijections` are leaves; `normalizing_flows` never imports from `models` except the `models.core` leaf (transiently, while `transformer_flow` still lives under `normalizing_flows` — Tasks 1–4 — and fully removed in Task 5). MAF migrates additively (Tasks 3→4) because its machinery never imports `models.core`; **TarFlow must migrate atomically (Task 5)** because its `conditioners` import `models.core` at module top, which would close a cycle in any additive intermediate. Verified by a fresh-interpreter smoke test of **both** `gensbi.models` and `gensbi.normalizing_flows` after every task.
- **`rngs` is a field on every `*Params` dataclass** (matches `Flux1Params`).
- **No deprecated shims:** `make_maf`/`make_tarflow`, `Flow`, `TransformerFlow` are removed, not aliased (un-merged branch, major release).
- **Spec:** `docs/superpowers/specs/2026-06-26-nf-model-restructure-design.md`.

---

## File Structure

**Created:**
- `src/gensbi/models/core/__init__.py` — re-exports core primitives
- `src/gensbi/models/core/patching.py` — `patchify_2d`, `depatchify_2d`
- `src/gensbi/models/core/tokenizers.py` — `VectorTokenizer`, `ImageTokenizer`
- `src/gensbi/models/maf/__init__.py` — re-exports `MAFlowParams`, `MAFlow`
- `src/gensbi/models/maf/model.py` — `MAFlowParams`, `MAFlow`
- `src/gensbi/models/maf/{made,masked_linear,masks}.py` — moved from `bijections/`
- `src/gensbi/models/tarflow/__init__.py` — re-exports `TarFlowParams`, `TarFlow`
- `src/gensbi/models/tarflow/model.py` — `TarFlowParams`, `TarFlow`
- `src/gensbi/models/tarflow/{blocks,conditioners}.py` — moved from `transformer_flow/`
- `tests/models/core/`, `tests/models/maf/`, `tests/models/tarflow/` — relocated + new tests
- `tests/test_import_smoke.py` — fresh-interpreter cycle guard

**Modified:**
- `src/gensbi/recipes/utils.py` — remove `patchify_2d`/`depatchify_2d`
- `src/gensbi/models/__init__.py` — export the four NF symbols
- `src/gensbi/normalizing_flows/__init__.py` + `bijections/__init__.py` — trim to Tier-1
- `src/gensbi/experimental/models/fielddit/codec.py`, `experimental/models/glue/embedder.py` — repoint patchify import
- `src/gensbi/models/flux1/model.py` — docstring patchify path
- `src/gensbi/recipes/flow_pipeline.py` — docstring/message wording

**Deleted:**
- `src/gensbi/normalizing_flows/flow.py`
- `src/gensbi/normalizing_flows/transformer_flow/` (entire subpackage)

---

### Task 0: Baseline full-suite snapshot (before any changes)

Establish the "no unexpected regression" reference: which tests pass/fail on the current branch HEAD, in this CPU environment, *before* the refactor. Some tests may already fail or skip on CPU (e.g. GPU-oriented experimental tests); recording that now prevents misattributing them to the refactor in Task 7.

**Files:** none (read-only).

- [ ] **Step 1: Confirm clean starting state**

Run: `git status --short`
Expected: only untracked `reference/` directories — no modified tracked files.

- [ ] **Step 2: Run the entire suite and capture the result (~20 min, background)**

This exceeds the 10-minute foreground command cap, so run it in the **background** and poll for completion. Tee the output to a scratch file:
Run: `pytest tests 2>&1 | tee "$SCRATCH/baseline_pytest.txt"` (background; `$SCRATCH` = the session scratchpad dir)
Expected: a final summary line, e.g. `N passed, K skipped, M failed`.

- [ ] **Step 3: Record the baseline failing set**

From `baseline_pytest.txt`, note the IDs of any tests that **fail** at baseline (the `FAILED ...` lines). This list is the regression reference for Task 7: if the baseline is fully green, Task 7 must be fully green; otherwise Task 7 must introduce **no new** failures beyond this list. (No commit — this is a read-only snapshot.)

---

### Task 1: Move `patchify` to `models/core/`

Severs the `normalizing_flows → recipes` edge. No model code depends on `models/` yet, so this is a safe foundational move.

**Files:**
- Create: `src/gensbi/models/core/__init__.py`, `src/gensbi/models/core/patching.py`
- Modify: `src/gensbi/recipes/utils.py` (remove the two functions), `src/gensbi/normalizing_flows/transformer_flow/tokenizers.py:11`, `src/gensbi/normalizing_flows/transformer_flow/conditioners.py:18`, `src/gensbi/experimental/models/fielddit/codec.py:14`, `src/gensbi/experimental/models/glue/embedder.py:6`, `src/gensbi/models/flux1/model.py:68` (docstring), `tests/recipes/test_pipeline_utils.py:98-99` (test-side patchify import), `tests/normalizing_flows/transformer_flow/test_tokenizers.py:34` (test-side patchify import)
- Create: `tests/models/core/__init__.py`, `tests/models/core/test_patching.py`, `tests/test_import_smoke.py`

**Interfaces:**
- Produces: `gensbi.models.core.patching.patchify_2d(x, size=2)`, `depatchify_2d(x, size=2, grid=None)` (signatures unchanged from `recipes.utils`).

- [ ] **Step 1: Find every patchify import from `recipes.utils`**

Run: `grep -rn "recipes.utils import.*patchify\|recipes\.utils\.patchify\|patchify_2d\|depatchify_2d" tests/ src/ | grep -v __pycache__`
Expected: the `src/` hits listed in **Files**, plus the two test files (`test_pipeline_utils.py`, `test_tokenizers.py`). If any other consumer appears, add it to this task's edit list. (Note: `test_pipeline_utils.py`'s import is *multi-line*, so a single-line `import.*patchify` grep would miss it — this broader grep catches it.)

- [ ] **Step 2: Create `models/core/patching.py` (verbatim move)**

```python
"""Invertible 2D patchify/depatchify — pure einops reshapes (no learned state).

Moved out of recipes.utils so model/flow code can depend on it without pulling
in the recipes package (which imports gensbi.models, creating a cycle).
"""

import jax
from jax import Array
from einops import rearrange


@jax.jit(static_argnames=["size"])
def patchify_2d(x: Array, size=2):
    return rearrange(x, "b (h ph) (w pw) c -> b (h w) (c ph pw)", ph=size, pw=size)


@jax.jit(static_argnames=["size", "grid"])
def depatchify_2d(x: Array, size=2, grid=None):
    """Inverse of :func:`patchify_2d`.

    Parameters
    ----------
    x : Array
        Patchified tensor of shape ``(B, h*w, C*size*size)``.
    size : int
        Patch edge length used by :func:`patchify_2d`.
    grid : tuple of int, optional
        The ``(h, w)`` patch grid. The grid cannot be inferred from the token
        count alone, so it is required for non-square grids. If ``None``, a
        square grid (``h == w``) is assumed.
    """
    if grid is None:
        n = x.shape[1]
        side = int(round(n ** 0.5))
        if side * side != n:
            raise ValueError(
                f"Cannot infer a square grid from {n} tokens; pass grid=(h, w)."
            )
        h = w = side
    else:
        h, w = grid
    return rearrange(
        x, "b (h w) (c ph pw) -> b (h ph) (w pw) c", h=h, w=w, ph=size, pw=size
    )
```

- [ ] **Step 3: Create `models/core/__init__.py`**

```python
"""Shared, model-agnostic primitives (the reuse home across architectures)."""

from gensbi.models.core.patching import patchify_2d, depatchify_2d

__all__ = ["patchify_2d", "depatchify_2d"]
```

- [ ] **Step 4: Remove the two functions from `recipes/utils.py`**

Delete the `@jax.jit(...) def patchify_2d` and `@jax.jit(...) def depatchify_2d` definitions (the block at `recipes/utils.py:99-132`). Leave `init_ids_2d` and the rest untouched (its docstring mentions `patchify_2d` in prose only — no import).

- [ ] **Step 5: Repoint the importers (source + tests)**

Change the import in each file to point at the new module:
- `normalizing_flows/transformer_flow/tokenizers.py:11`: `from gensbi.models.core.patching import patchify_2d, depatchify_2d`
- `normalizing_flows/transformer_flow/conditioners.py:18`: `from gensbi.models.core.patching import patchify_2d`
- `experimental/models/fielddit/codec.py:14`: `from gensbi.models.core.patching import patchify_2d, depatchify_2d`
- `experimental/models/glue/embedder.py:6`: `from gensbi.models.core.patching import patchify_2d`
- `models/flux1/model.py:68` (docstring): change `gensbi.recipes.utils.patchify_2d` to `gensbi.models.core.patching.patchify_2d`
- `tests/normalizing_flows/transformer_flow/test_tokenizers.py:34`: `from gensbi.models.core.patching import patchify_2d` (this test is run in Step 8 before Task 2 moves it, so it must be repointed now)
- `tests/recipes/test_pipeline_utils.py`: this test imports `patchify_2d, depatchify_2d` inside a `from gensbi.recipes.utils import (...)` block (lines 98-99). Remove those two names from that block and add a separate `from gensbi.models.core.patching import patchify_2d, depatchify_2d`. Its 5 patchify tests then validate the new location unchanged.

- [ ] **Step 6: Write the patching round-trip test**

`tests/models/core/__init__.py`: empty file.
`tests/models/core/test_patching.py`:

```python
import jax.numpy as jnp
from gensbi.models.core.patching import patchify_2d, depatchify_2d


def test_patchify_shape_and_roundtrip():
    x = jnp.arange(1 * 4 * 4 * 2).reshape(1, 4, 4, 2).astype(jnp.float32)
    p = patchify_2d(x, size=2)
    assert p.shape == (1, 4, 2 * 2 * 2)          # (B, (h w), C*ph*pw)
    xr = depatchify_2d(p, size=2)
    assert jnp.allclose(xr, x)


def test_depatchify_nonsquare_requires_grid():
    p = jnp.zeros((1, 6, 8))                       # 6 = 3*2 patches, not square
    xr = depatchify_2d(p, size=2, grid=(2, 3))
    assert xr.shape == (1, 4, 6, 2)
```

- [ ] **Step 7: Write the fresh-interpreter cycle guard**

`tests/test_import_smoke.py`:

```python
import subprocess
import sys


def _fresh_import(module: str):
    r = subprocess.run([sys.executable, "-c", f"import {module}"],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"import {module} failed:\n{r.stderr}"


def test_import_models_clean():
    _fresh_import("gensbi.models")


def test_import_normalizing_flows_clean():
    _fresh_import("gensbi.normalizing_flows")
```

- [ ] **Step 8: Run tests**

Run: `pytest tests/models/core/test_patching.py tests/test_import_smoke.py tests/recipes/test_pipeline_utils.py tests/normalizing_flows/transformer_flow/test_tokenizers.py tests/normalizing_flows/transformer_flow/test_conditioners.py tests/experimental/models/fielddit/test_codec.py tests/experimental/models/glue/test_embedder.py -v`
Expected: PASS (patchify works at the new path; the repointed `test_pipeline_utils.py`, tokenizer/conditioner, and experimental fielddit/glue tests still pass; no cycle).

- [ ] **Step 9: Commit**

```bash
git add src/gensbi/models/core tests/models/core tests/test_import_smoke.py \
  src/gensbi/recipes/utils.py src/gensbi/normalizing_flows/transformer_flow/tokenizers.py \
  src/gensbi/normalizing_flows/transformer_flow/conditioners.py \
  src/gensbi/experimental/models/fielddit/codec.py \
  src/gensbi/experimental/models/glue/embedder.py src/gensbi/models/flux1/model.py \
  tests/recipes/test_pipeline_utils.py \
  tests/normalizing_flows/transformer_flow/test_tokenizers.py
git commit -m "refactor(nf): move patchify_2d/depatchify_2d to models/core (sever NF->recipes edge)"
```

---

### Task 2: Move tokenizers to `models/core/`

**Files:**
- Create: `src/gensbi/models/core/tokenizers.py`
- Modify: `src/gensbi/models/core/__init__.py`, `src/gensbi/normalizing_flows/transformer_flow/model.py` (its `tokenizers` import)
- Delete: `src/gensbi/normalizing_flows/transformer_flow/tokenizers.py`
- Move test: `tests/normalizing_flows/transformer_flow/test_tokenizers.py` → `tests/models/core/test_tokenizers.py`

**Interfaces:**
- Produces: `gensbi.models.core.tokenizers.VectorTokenizer(dim, block_size=1)`, `ImageTokenizer(height, width, channels, patch_size)`, each with `.T`, `.F`, `.example_shape`, `.tokenize(x)`, `.detokenize(tokens)`.

- [ ] **Step 1: Create `models/core/tokenizers.py` (verbatim move, repointed patchify import)**

```python
"""Invertible tokenizers — the modeled-variable reshape seam.

Adapted from apple/ml-tarflow (TarFlow); see models/tarflow/LICENSE.apple.

A tokenizer maps the modeled variable to a token sequence ``(B, T, F)`` and back.
It MUST be volume-preserving (a fixed invertible reshape, log-det 0) — never a
learned lossy encoder — so the change-of-variables stays exact. Pure reshape, no
parameters, which is why it is a shared core primitive.
"""

from jax import Array
from gensbi.models.core.patching import patchify_2d, depatchify_2d


class VectorTokenizer:
    """Reshape a vector ``(B, dim)`` into ``T = dim // block_size`` tokens of
    ``F = block_size`` features. Pure reshape: invertible, log-det 0."""

    def __init__(self, dim: int, block_size: int = 1):
        if dim % block_size != 0:
            raise ValueError(
                f"block_size ({block_size}) must divide dim ({dim})")
        self.dim = dim
        self.F = block_size
        self.T = dim // block_size
        self.example_shape = (dim,)

    def tokenize(self, x: Array) -> Array:
        return x.reshape(x.shape[0], self.T, self.F)

    def detokenize(self, tokens: Array) -> Array:
        return tokens.reshape(tokens.shape[0], self.dim)


class ImageTokenizer:
    """Patchify an image ``(B, H, W, C)`` into ``T = (H/p)(W/p)`` tokens of
    ``F = C*p*p`` features via :func:`patchify_2d`. Pure reshape: invertible,
    log-det 0. Raster causal order (fixed by ``patchify_2d``)."""

    def __init__(self, height: int, width: int, channels: int, patch_size: int):
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(
                f"patch_size ({patch_size}) must divide height ({height}) and "
                f"width ({width})")
        self.height = height
        self.width = width
        self.channels = channels
        self.patch_size = patch_size
        self.grid = (height // patch_size, width // patch_size)
        self.T = self.grid[0] * self.grid[1]
        self.F = channels * patch_size * patch_size
        self.example_shape = (height, width, channels)

    def tokenize(self, x: Array) -> Array:
        return patchify_2d(x, size=self.patch_size)

    def detokenize(self, tokens: Array) -> Array:
        return depatchify_2d(tokens, size=self.patch_size, grid=self.grid)
```

- [ ] **Step 2: Re-export from `models/core/__init__.py`**

```python
"""Shared, model-agnostic primitives (the reuse home across architectures)."""

from gensbi.models.core.patching import patchify_2d, depatchify_2d
from gensbi.models.core.tokenizers import VectorTokenizer, ImageTokenizer

__all__ = ["patchify_2d", "depatchify_2d", "VectorTokenizer", "ImageTokenizer"]
```

- [ ] **Step 3: Delete old tokenizers + repoint the TransformerFlow import**

Delete `src/gensbi/normalizing_flows/transformer_flow/tokenizers.py`.
In `src/gensbi/normalizing_flows/transformer_flow/model.py:21-23`, change:
```python
from gensbi.normalizing_flows.transformer_flow.tokenizers import (
    VectorTokenizer, ImageTokenizer,
)
```
to:
```python
from gensbi.models.core.tokenizers import VectorTokenizer, ImageTokenizer
```

- [ ] **Step 4: Move the tokenizer test**

```bash
git mv tests/normalizing_flows/transformer_flow/test_tokenizers.py tests/models/core/test_tokenizers.py
```
In the moved file, change the two `from gensbi.normalizing_flows.transformer_flow.tokenizers import ...` lines (1, 33) to `from gensbi.models.core.tokenizers import ...`. (Its `patchify_2d` import at line 34 already points at `gensbi.models.core.patching` from Task 1.)

- [ ] **Step 5: Run tests**

Run: `pytest tests/models/core/ tests/test_import_smoke.py tests/normalizing_flows/transformer_flow/test_model.py -v`
Expected: PASS (tokenizers work at new path; old TarFlow path still constructs via repointed import; no cycle).

- [ ] **Step 6: Commit**

```bash
git add -A src/gensbi/models/core tests/models/core \
  src/gensbi/normalizing_flows/transformer_flow/model.py
git rm src/gensbi/normalizing_flows/transformer_flow/tokenizers.py
git commit -m "refactor(nf): move VectorTokenizer/ImageTokenizer to models/core"
```

---

### Task 3: Add `MAFlowParams` + `MAFlow` (additive; old `make_maf`/`Flow` still live)

Create the new MAF model class alongside the old factory, verified by a parity test against `make_maf`. `MADE` stays in `bijections/` for now (relocated in Task 4); no `normalizing_flows → models` edge is introduced (MAF machinery does not import `models.core`, so the additive intermediate is cycle-free — confirmed by the smoke test).

**Files:**
- Create: `src/gensbi/models/maf/__init__.py`, `src/gensbi/models/maf/model.py`
- Modify: `src/gensbi/models/__init__.py`
- Create: `tests/models/maf/__init__.py`, `tests/models/maf/test_maflow.py`

**Interfaces:**
- Consumes: `gensbi.normalizing_flows.bijections.{made.MaskedAutoregressive, chain.Chain, permutation.Permutation, standardize.Standardize, transformers.Affine, base.Bijection}`; `gensbi.core.prior.make_gaussian_prior`.
- Produces: `MAFlowParams(rngs, dim, cond_dim=0, n_layers=5, transformer=None, nn_width=64, nn_depth=2, permutation="reverse", standardize=True, zero_init=True)`; `MAFlow(params)` with `.log_prob(x, cond=None)`, `.sample(key, cond=None, nsamples=None)`, `.set_standardization(mean, std)`, `.dim`, `.cond_dim`.

- [ ] **Step 1: Write the failing test**

`tests/models/maf/__init__.py`: empty file.
`tests/models/maf/test_maflow.py`:

```python
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from gensbi.models import MAFlow, MAFlowParams
from gensbi.normalizing_flows import make_maf          # still present in Task 3
from gensbi.normalizing_flows.bijections.transformers import Affine


def test_params_defaults_and_validation():
    p = MAFlowParams(rngs=nnx.Rngs(0), dim=3)
    assert isinstance(p.transformer, Affine)            # None -> Affine()
    assert p.cond_dim == 0 and p.n_layers == 5
    with pytest.raises(ValueError):
        MAFlowParams(rngs=nnx.Rngs(0), dim=3, permutation="nope")


def test_maflow_matches_make_maf():
    cfg = dict(dim=3, cond_dim=2, n_layers=3, nn_width=16, nn_depth=2)
    old = make_maf(nnx.Rngs(0), **cfg)
    new = MAFlow(MAFlowParams(rngs=nnx.Rngs(0), **cfg))
    x = jax.random.normal(jax.random.key(1), (5, 3))
    c = jax.random.normal(jax.random.key(2), (5, 2))
    assert jnp.allclose(old.log_prob(x, c), new.log_prob(x, c), atol=1e-5)


def test_maflow_sample_shape_and_standardize():
    flow = MAFlow(MAFlowParams(rngs=nnx.Rngs(0), dim=4, cond_dim=2, n_layers=3))
    c = jax.random.normal(jax.random.key(0), (7, 2))
    s = flow.sample(jax.random.key(1), cond=c)
    assert s.shape == (7, 4)
    flow.set_standardization(jnp.ones(4), 2.0 * jnp.ones(4))   # no raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/models/maf/test_maflow.py -v`
Expected: FAIL with `ImportError: cannot import name 'MAFlow' from 'gensbi.models'`.

- [ ] **Step 3: Create `models/maf/model.py`**

```python
"""MAF: affine/spline masked-autoregressive normalizing flow.

Self-contained density model (absorbs the former ``Flow`` container and the
``make_maf`` factory). Builds a Chain of (MaskedAutoregressive, Permutation)
layers + an optional data-end Standardize, over a standard-normal base.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from gensbi.core.prior import make_gaussian_prior
from gensbi.normalizing_flows.bijections.base import Bijection
from gensbi.normalizing_flows.bijections.made import MaskedAutoregressive
from gensbi.normalizing_flows.bijections.chain import Chain
from gensbi.normalizing_flows.bijections.permutation import Permutation
from gensbi.normalizing_flows.bijections.standardize import Standardize
from gensbi.normalizing_flows.bijections.transformers import Affine


@dataclass
class MAFlowParams:
    """Architecture parameters for :class:`MAFlow`.

    Only ``rngs`` and ``dim`` are required. ``transformer`` defaults to
    ``Affine()`` (pass ``RQSpline()`` for a spline flow).
    """

    rngs: nnx.Rngs
    dim: int
    cond_dim: int = 0
    n_layers: int = 5
    transformer: Bijection | None = None
    nn_width: int = 64
    nn_depth: int = 2
    permutation: str = "reverse"
    standardize: bool = True
    zero_init: bool = True

    def __post_init__(self):
        if self.transformer is None:
            self.transformer = Affine()
        if self.permutation not in ("reverse", "random"):
            raise ValueError(f"unknown permutation {self.permutation!r}")


class MAFlow(nnx.Module):
    """Affine/spline MAF over ``(batch, dim)`` data, optionally conditioned.

    ``log_prob(x, cond) = base.log_prob(u) + logdet`` with
    ``u, logdet = chain.inverse(x, cond)``; the base is a standard normal over
    ``(dim,)`` built lazily so it never enters nnx state.
    """

    def __init__(self, params: MAFlowParams):
        rngs = params.rngs
        dim = params.dim
        bijections = []
        for i in range(params.n_layers):
            bijections.append(
                MaskedAutoregressive(dim, params.cond_dim, params.transformer,
                                     params.nn_width, params.nn_depth, rngs,
                                     zero_init=params.zero_init))
            if i < params.n_layers - 1:
                if params.permutation == "reverse":
                    bijections.append(Permutation.reverse(dim))
                else:
                    bijections.append(Permutation.random(dim, rngs))
        if params.standardize:
            bijections.append(Standardize(dim))
        self.chain = Chain(bijections)
        self.dim = dim
        self.cond_dim = params.cond_dim

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
        u = base.sample(key, (nsamples,))

        def single(u_i, cond_i):
            x, _ = self.chain.forward(u_i, cond_i)
            return x

        if cond is None:
            return jax.vmap(lambda ui: single(ui, None))(u)
        return jax.vmap(single)(u, cond)

    def set_standardization(self, mean, std) -> None:
        """Set the data-end Standardize bijection's mean/std buffers in place.

        Raises ValueError if built with ``standardize=False``.
        """
        mean = jnp.asarray(mean)
        std = jnp.asarray(std)
        for b in self.chain.bijections:
            if isinstance(b, Standardize):
                b.set_stats(mean, std)
                return
        raise ValueError(
            "MAFlow has no Standardize bijection (built with standardize=False).")
```

- [ ] **Step 4: Create `models/maf/__init__.py`**

```python
from gensbi.models.maf.model import MAFlowParams, MAFlow

__all__ = ["MAFlowParams", "MAFlow"]
```

- [ ] **Step 5: Export from `models/__init__.py`**

Add to `src/gensbi/models/__init__.py`:
```python
from .maf import MAFlowParams, MAFlow
```
and add `"MAFlowParams", "MAFlow"` to `__all__`.

- [ ] **Step 6: Run tests**

Run: `pytest tests/models/maf/test_maflow.py tests/test_import_smoke.py -v`
Expected: PASS (parity with `make_maf`, sample shape, no cycle — both smoke imports clean).

- [ ] **Step 7: Commit**

```bash
git add src/gensbi/models/maf src/gensbi/models/__init__.py tests/models/maf
git commit -m "feat(nf): add MAFlowParams + MAFlow (uniform Model(Params) constructor)"
```

---

### Task 4: Cut over to `MAFlow`; remove `Flow`/`make_maf`; relocate MADE family

Switch all MAF call sites to the new API, delete the old container/factory, then move the MAF-specific bijection machinery (`MADE`, `MaskedLinear`, `masks`) into `models/maf/`. Order matters: `flow.py` is deleted **before** MADE moves, so `normalizing_flows` never imports from `models`.

**Files:**
- Modify (call sites): `tests/normalizing_flows/test_flow_pipeline.py`, `test_flow_pipeline_e2e.py`, `test_nle.py`, `scripts/maf_nle_recovery.py`
- Move + modify (call sites): `tests/normalizing_flows/test_flow.py` → `tests/models/maf/test_maflow_density.py`; `tests/normalizing_flows/test_flow_spline_battery.py` → `tests/models/maf/test_maflow_spline.py`
- Delete: `src/gensbi/normalizing_flows/flow.py`
- Modify: `src/gensbi/normalizing_flows/__init__.py`, `src/gensbi/recipes/flow_pipeline.py` (docstring + `NotImplementedError` wording), `tests/models/maf/test_maflow.py` (drop the `make_maf` parity test)
- Move: `bijections/{made,masked_linear,masks}.py` → `models/maf/`; tests `tests/normalizing_flows/bijections/{test_made,test_masked_autoregressive,test_masked_linear,test_masks}.py` → `tests/models/maf/`
- Modify: `src/gensbi/normalizing_flows/bijections/__init__.py`, `src/gensbi/models/maf/model.py` (repoint `MaskedAutoregressive` import)

**Interfaces:**
- Consumes: `MAFlow`, `MAFlowParams` from Task 3.
- Produces: `gensbi.models.maf.made.{MADE, MaskedAutoregressive}`, `gensbi.models.maf.masked_linear.MaskedLinear`, `gensbi.models.maf.masks.make_mask`. `gensbi.normalizing_flows` no longer exports `Flow` or `make_maf`.

**Call-site rewrite rule (apply uniformly):**
`make_maf(RNGS, dim=D, cond_dim=C, **kw)` → `MAFlow(MAFlowParams(rngs=RNGS, dim=D, cond_dim=C, **kw))`; all keyword args (`n_layers`, `nn_width`, `nn_depth`, `permutation`, `standardize`, `zero_init`, `transformer`) carry over unchanged. Replace `from gensbi.normalizing_flows import make_maf` with `from gensbi.models import MAFlow, MAFlowParams`.

- [ ] **Step 1: Rewrite the MAF pipeline/script call sites**

In `tests/normalizing_flows/test_flow_pipeline.py` (lines 11, 38, 160), `test_flow_pipeline_e2e.py` (11, 48), `test_nle.py` (10, 23, 53), and `scripts/maf_nle_recovery.py` (51, 112): apply the rewrite rule above. Example (`test_nle.py:23`):
```python
# before
flow = make_maf(nnx.Rngs(0), dim=dim, cond_dim=dim, n_layers=4, nn_width=32)
# after
flow = MAFlow(MAFlowParams(rngs=nnx.Rngs(0), dim=dim, cond_dim=dim, n_layers=4, nn_width=32))
```

- [ ] **Step 2: Move + rewrite the MAF density/spline tests**

```bash
git mv tests/normalizing_flows/test_flow.py tests/models/maf/test_maflow_density.py
git mv tests/normalizing_flows/test_flow_spline_battery.py tests/models/maf/test_maflow_spline.py
```
Apply the rewrite rule to every `make_maf(...)` call in both moved files (`test_flow.py` lines 11/21/32/43/58/74/89/103; `test_flow_spline_battery.py` lines 14/43), and swap the `make_maf` import to `from gensbi.models import MAFlow, MAFlowParams`. The spline file keeps its existing `from gensbi.normalizing_flows.bijections.transformers import RQSpline` import (Tier-1, unchanged) and passes `transformer=RQSpline(num_bins=8, range_bound=6.0)`.

- [ ] **Step 3: Drop the parity test (its dependency is being removed)**

In `tests/models/maf/test_maflow.py`, delete `test_maflow_matches_make_maf` and the `from gensbi.normalizing_flows import make_maf` import. The density/spline suites now cover behaviour.

- [ ] **Step 4: Delete `flow.py` and trim `normalizing_flows/__init__.py`**

```bash
git rm src/gensbi/normalizing_flows/flow.py
```
Edit `src/gensbi/normalizing_flows/__init__.py` — remove the `from gensbi.normalizing_flows.flow import Flow, make_maf` line and drop `"Flow"`, `"make_maf"` from `__all__`. (TransformerFlow exports stay until Task 5.)

- [ ] **Step 5: Update `flow_pipeline.py` wording**

In `src/gensbi/recipes/flow_pipeline.py`: docstring line 50 `make_maf(rngs, dim=dim_obs, cond_dim=dim_cond)` → `MAFlow(MAFlowParams(rngs=rngs, dim=dim_obs, cond_dim=dim_cond))`. The `NotImplementedError` at line ~80 references `make_maf` ("build it with make_maf and pass it as model=") → "build a `MAFlow` and pass it as `model=`." (The line ~84 message references "Flow", not `make_maf` — update that prose to "a `MAFlow`" too for consistency, optional.)

- [ ] **Step 6: Run tests (verify the cut-over before relocating machinery)**

Run: `pytest tests/models/maf tests/normalizing_flows/test_flow_pipeline.py tests/normalizing_flows/test_nle.py tests/test_import_smoke.py -v`
Expected: PASS. Confirms no remaining `make_maf`/`Flow` references.

- [ ] **Step 7: Relocate the MADE family**

```bash
git mv src/gensbi/normalizing_flows/bijections/made.py src/gensbi/models/maf/made.py
git mv src/gensbi/normalizing_flows/bijections/masked_linear.py src/gensbi/models/maf/masked_linear.py
git mv src/gensbi/normalizing_flows/bijections/masks.py src/gensbi/models/maf/masks.py
```
In `models/maf/made.py`, repoint its siblings:
```python
from gensbi.models.maf.masked_linear import MaskedLinear
from gensbi.models.maf.masks import make_mask
```
(its `from gensbi.normalizing_flows.bijections.base import Bijection` stays — `base` is Tier-1). `masked_linear.py` keeps `from gensbi.normalizing_flows.bijections.base import Mask`.
In `models/maf/model.py`, change `from gensbi.normalizing_flows.bijections.made import MaskedAutoregressive` → `from gensbi.models.maf.made import MaskedAutoregressive`.

- [ ] **Step 8: Trim `bijections/__init__.py`**

Edit `src/gensbi/normalizing_flows/bijections/__init__.py` to drop the MADE-family imports/exports. Final content:
```python
"""Pure normalizing-flow bijection abstractions (Tier 1)."""

from gensbi.normalizing_flows.bijections.base import Bijection, Mask
from gensbi.normalizing_flows.bijections.chain import Chain
from gensbi.normalizing_flows.bijections.permutation import Permutation
from gensbi.normalizing_flows.bijections.standardize import Standardize
from gensbi.normalizing_flows.bijections.transformers import Affine, RQSpline

__all__ = [
    "Bijection", "Mask", "Chain", "Permutation", "Standardize",
    "Affine", "RQSpline",
]
```

- [ ] **Step 9: Move the MADE-family tests**

```bash
git mv tests/normalizing_flows/bijections/test_made.py tests/models/maf/test_made.py
git mv tests/normalizing_flows/bijections/test_masked_autoregressive.py tests/models/maf/test_masked_autoregressive.py
git mv tests/normalizing_flows/bijections/test_masked_linear.py tests/models/maf/test_masked_linear.py
git mv tests/normalizing_flows/bijections/test_masks.py tests/models/maf/test_masks.py
```
In each moved file, change imports from `gensbi.normalizing_flows.bijections.{made,masked_linear,masks}` to `gensbi.models.maf.{made,masked_linear,masks}`.

- [ ] **Step 10: Run tests**

Run: `pytest tests/models/maf tests/normalizing_flows/bijections tests/normalizing_flows/test_nle.py tests/normalizing_flows/test_flow_pipeline.py tests/test_import_smoke.py -v`
Expected: PASS. `bijections/` now holds Tier-1 only; MAF is self-contained; no cycle.

- [ ] **Step 11: Commit**

```bash
git add -A src/gensbi/models src/gensbi/normalizing_flows src/gensbi/recipes/flow_pipeline.py tests
git commit -m "refactor(nf): cut MAF over to MAFlow, drop Flow/make_maf, relocate MADE family to models/maf"
```

---

### Task 5: Migrate TarFlow atomically — relocate machinery, add `TarFlow`, drop `TransformerFlow`/`make_tarflow`

**Why this is one task, not two.** Unlike MAF, TarFlow's `conditioners.py` imports `models.core.patching` at module top. An additive intermediate (adding `TarFlow` to `gensbi.models` while `normalizing_flows/__init__` still re-exports `transformer_flow`) closes a real import cycle: `import gensbi.normalizing_flows` → `transformer_flow` → `conditioners` → `models.core` → triggers `import gensbi.models` → `.tarflow` → re-imports the half-initialized `conditioners` → `ImportError`. So TarFlow must migrate in a single commit: relocate `blocks`/`conditioners` into `models/tarflow/`, build `TarFlow` against the local copies, and remove `transformer_flow` from the `normalizing_flows` import graph together. (There is also no valid intermediate where `transformer_flow/model.py` coexists with the moved `blocks`/`conditioners` — its imports would dangle — which is the same reason the deletion happens right after the relocation.)

**Files:**
- Move: `transformer_flow/blocks.py` → `models/tarflow/blocks.py`; `transformer_flow/conditioners.py` → `models/tarflow/conditioners.py`; `transformer_flow/LICENSE.apple` + `LICENSE.starflow` → `models/tarflow/`
- Modify: `src/gensbi/models/tarflow/blocks.py` (`head_dim` → `num_heads`)
- Create: `src/gensbi/models/tarflow/model.py`, `src/gensbi/models/tarflow/__init__.py`
- Modify: `src/gensbi/models/__init__.py`, `src/gensbi/normalizing_flows/__init__.py`
- Delete: `src/gensbi/normalizing_flows/transformer_flow/model.py`, `src/gensbi/normalizing_flows/transformer_flow/__init__.py` (the whole subpackage)
- Move + modify (tests): `tests/normalizing_flows/transformer_flow/{test_model,test_stability,test_structured_integration,test_structured_boundary,test_blocks_attention,test_blocks_meta,test_conditioners,test_pipeline_integration}.py` → `tests/models/tarflow/`; `tests/normalizing_flows/transformer_flow/test_exports.py` → `tests/models/test_nf_exports.py`
- Delete: `tests/normalizing_flows/transformer_flow/__init__.py`
- Create: `tests/models/tarflow/__init__.py`, `tests/models/tarflow/test_tarflow.py`
- Modify (scripts): `scripts/tarflow_nle_recovery.py`, `tarflow_field_nle_recovery.py`, `tarflow_image_npe_recovery.py`

**Interfaces:**
- Consumes: `gensbi.models.core.tokenizers.{VectorTokenizer, ImageTokenizer}`; `gensbi.models.tarflow.blocks.{AttentionBlock, MetaBlock, INV_SOFTPLUS_1}`; `gensbi.models.tarflow.conditioners.{VectorConditioner, VectorPrefixConditioner, ImagePrefixConditioner}`; `gensbi.normalizing_flows.bijections.base.Mask`.
- Produces: `TarFlowParams(rngs, dim=None, cond_dim=0, modeled="vector", img_size=None, patch_size=None, img_channels=1, cond="add", cond_img_size=None, cond_patch_size=None, cond_channels=1, prefix_tokens=1, head_dim=16, num_heads=4, num_blocks=8, layers_per_block=2, block_size=1, permutation="flip", standardize=True, zero_init=True, use_softplus=True, soft_clip=4.0)` with derived `.channels`; `TarFlow(params)` with `.log_prob`, `.sample`, `.set_standardization`, `.T`, `.F`, `.example_shape`. `gensbi.normalizing_flows` no longer exports `TransformerFlow`/`make_tarflow`; `AttentionBlock`/`MetaBlock` now take `num_heads`.

**Call-site rewrite rule (apply uniformly):**
`make_tarflow(RNGS, ..., channels=CH, head_dim=HD, **kw)` → `TarFlow(TarFlowParams(rngs=RNGS, ..., head_dim=HD, num_heads=CH // HD, **kw))`. `head_dim` defaults to 16 when omitted, so `num_heads = CH // 16`. Drop the `channels=` kwarg. All other kwargs (`dim`, `cond_dim`, `modeled`, `img_size`, `patch_size`, `cond`, `cond_img_size`, `cond_patch_size`, `num_blocks`, `layers_per_block`, `permutation`, `use_softplus`, `soft_clip`, etc.) carry over unchanged. When `channels`/`head_dim` are runtime values (e.g. `args.channels`), emit `num_heads=args.channels // args.head_dim` as an expression, not a precomputed literal. Replace `from gensbi.normalizing_flows import make_tarflow` (and `make_tarflow, TransformerFlow`) with `from gensbi.models import TarFlow, TarFlowParams`.

- [ ] **Step 1: Relocate blocks + conditioners + LICENSEs**

```bash
git mv src/gensbi/normalizing_flows/transformer_flow/blocks.py src/gensbi/models/tarflow/blocks.py
git mv src/gensbi/normalizing_flows/transformer_flow/conditioners.py src/gensbi/models/tarflow/conditioners.py
git mv src/gensbi/normalizing_flows/transformer_flow/LICENSE.apple src/gensbi/models/tarflow/LICENSE.apple
git mv src/gensbi/normalizing_flows/transformer_flow/LICENSE.starflow src/gensbi/models/tarflow/LICENSE.starflow
```
`blocks.py` keeps `from gensbi.normalizing_flows.bijections.base import Mask` (Tier-1). `conditioners.py` keeps `from gensbi.models.core.patching import patchify_2d` (set in Task 1). No other import edits in these two files yet.

- [ ] **Step 2: Delete the `transformer_flow` package + trim `normalizing_flows/__init__.py`**

```bash
git rm src/gensbi/normalizing_flows/transformer_flow/model.py
git rm src/gensbi/normalizing_flows/transformer_flow/__init__.py
```
(The directory is now empty — `blocks`/`conditioners`/`tokenizers`/LICENSEs all moved out. Remove the dir if anything remains.)
Edit `src/gensbi/normalizing_flows/__init__.py` to its final Tier-1 form:
```python
"""Pure normalizing-flow abstractions (bijections + change-of-variables).

Concrete flow models live under ``gensbi.models`` (``MAFlow``, ``TarFlow``).
"""

from gensbi.normalizing_flows.bijections import (
    Bijection, Mask, Chain, Permutation, Standardize, Affine, RQSpline,
)

__all__ = [
    "Bijection", "Mask", "Chain", "Permutation", "Standardize",
    "Affine", "RQSpline",
]
```

- [ ] **Step 3: Normalize the head parameter in `models/tarflow/blocks.py` (`head_dim` → `num_heads`)**

- Remove the `# TODO` comment at the top of the file (formerly `blocks.py:17`).
- `AttentionBlock.__init__(self, channels, num_heads, expansion, rngs)`: replace the head logic with
  ```python
  if channels % num_heads != 0:
      raise ValueError(
          f"channels ({channels}) must be a multiple of num_heads ({num_heads})")
  self.num_heads = num_heads
  self.head_dim = channels // num_heads
  ```
- `MetaBlock.__init__(...)`: rename its `head_dim` parameter to `num_heads` and pass it through: `AttentionBlock(channels, num_heads, expansion, rngs)`.

- [ ] **Step 4: Create `models/tarflow/model.py`**

```python
"""TarFlow: transformer autoregressive normalizing flow.

Adapted from apple/ml-tarflow (TarFlow) and apple/ml-starflow (STARFlow);
see models/tarflow/LICENSE.apple and LICENSE.starflow.

Self-contained ``(B, T, F)`` density model (absorbs the former
``TransformerFlow`` container and the ``make_tarflow`` factory). Head sizing
follows the Flux1 convention: specify ``head_dim`` and ``num_heads``; total
width ``channels = head_dim * num_heads`` is derived.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from gensbi.models.core.tokenizers import VectorTokenizer, ImageTokenizer
from gensbi.models.tarflow.blocks import MetaBlock
from gensbi.models.tarflow.conditioners import (
    VectorConditioner, VectorPrefixConditioner, ImagePrefixConditioner,
)
from gensbi.normalizing_flows.bijections.base import Mask

_LOG2PI = jnp.log(2.0 * jnp.pi)


@dataclass
class TarFlowParams:
    """Architecture parameters for :class:`TarFlow`.

    ``modeled`` selects the tokenizer (vector/image); ``cond`` selects the
    conditioner (additive bias / vector-prefix / image-prefix). Head sizing is
    ``(head_dim, num_heads)`` with ``channels = head_dim * num_heads`` derived.
    """

    rngs: nnx.Rngs
    dim: int | None = None
    cond_dim: int = 0
    modeled: str = "vector"
    img_size: int | None = None
    patch_size: int | None = None
    img_channels: int = 1
    cond: str = "add"
    cond_img_size: int | None = None
    cond_patch_size: int | None = None
    cond_channels: int = 1
    prefix_tokens: int = 1
    head_dim: int = 16
    num_heads: int = 4
    num_blocks: int = 8
    layers_per_block: int = 2
    block_size: int = 1
    permutation: str = "flip"
    standardize: bool = True
    zero_init: bool = True
    use_softplus: bool = True
    soft_clip: float = 4.0

    def __post_init__(self):
        if self.modeled not in ("vector", "image"):
            raise ValueError(f"unknown modeled {self.modeled!r}")
        if self.modeled == "vector" and self.dim is None:
            raise ValueError("modeled='vector' requires dim")
        if self.modeled == "image" and (self.img_size is None or self.patch_size is None):
            raise ValueError("modeled='image' requires img_size and patch_size")
        if self.cond not in ("add", "vector_prefix", "image_prefix"):
            raise ValueError(f"unknown cond {self.cond!r}")
        if self.cond == "image_prefix" and (self.cond_img_size is None or self.cond_patch_size is None):
            raise ValueError("cond='image_prefix' requires cond_img_size and cond_patch_size")
        if self.permutation not in ("flip", "random"):
            raise ValueError(f"unknown permutation {self.permutation!r}")
        self.channels = self.head_dim * self.num_heads


class TarFlow(nnx.Module):
    """Stack of MetaBlocks + tokenizer + standardization + N(0, I) base."""

    def __init__(self, params: TarFlowParams):
        rngs = params.rngs
        channels = params.channels

        if params.modeled == "vector":
            tokenizer = VectorTokenizer(params.dim, params.block_size)
        else:
            tokenizer = ImageTokenizer(params.img_size, params.img_size,
                                       params.img_channels, params.patch_size)
        T, F = tokenizer.T, tokenizer.F

        def make_cond():
            if params.cond == "add":
                return VectorConditioner(params.cond_dim, channels, rngs=rngs)
            if params.cond == "vector_prefix":
                return VectorPrefixConditioner(params.cond_dim, channels,
                                               params.prefix_tokens, rngs=rngs)
            m = (params.cond_img_size // params.cond_patch_size) ** 2
            return ImagePrefixConditioner(params.cond_channels,
                                          params.cond_patch_size, channels, m,
                                          rngs=rngs)

        blocks = []
        for i in range(params.num_blocks):
            if params.permutation == "flip":
                perm = jnp.arange(T) if i % 2 == 0 else jnp.arange(T)[::-1]
            else:
                perm = jax.random.permutation(rngs.params(), T)
            blocks.append(MetaBlock(
                F=F, channels=channels, T=T, perm=perm, inv_perm=jnp.argsort(perm),
                conditioner=make_cond(), num_layers=params.layers_per_block,
                num_heads=params.num_heads, expansion=4, rngs=rngs,
                zero_init=params.zero_init, use_softplus=params.use_softplus,
                soft_clip=params.soft_clip))

        self.blocks = nnx.List(blocks)
        self.tokenizer = tokenizer
        self.dim = params.dim
        self.cond_dim = params.cond_dim
        self.T = T
        self.F = F
        self.example_shape = tokenizer.example_shape
        self._standardize = params.standardize
        self.mean = Mask(jnp.zeros(self.example_shape))
        self.std = Mask(jnp.ones(self.example_shape))

    def _base_log_prob(self, z: Array) -> Array:
        return -0.5 * jnp.sum(z ** 2, axis=(1, 2)) - 0.5 * self.T * self.F * _LOG2PI

    def _ensure_batched(self, x: Array) -> Array:
        x = jnp.asarray(x)
        if x.ndim == len(self.example_shape):
            x = x[None]
        return x

    def log_prob(self, x: Array, cond: Array | None = None) -> Array:
        x = self._ensure_batched(x)
        u = (x - self.mean[...]) / self.std[...]
        logdet = -jnp.sum(jnp.log(self.std[...]))
        z = self.tokenizer.tokenize(u)
        total = jnp.broadcast_to(logdet, (x.shape[0],))
        for blk in self.blocks:
            z, ld = blk.inverse(z, cond)
            total = total + ld
        return self._base_log_prob(z) + total

    def sample(self, key, cond: Array | None = None, nsamples: int | None = None):
        if cond is not None:
            nsamples = cond.shape[0]
        z = jax.random.normal(key, (nsamples, self.T, self.F))
        x = z
        for blk in reversed(self.blocks):
            x, _ = blk.forward(x, cond)
        x = self.tokenizer.detokenize(x)
        return x * self.std[...] + self.mean[...]

    def set_standardization(self, mean, std) -> None:
        if not self._standardize:
            raise ValueError("TarFlow built with standardize=False")
        self.mean[...] = jnp.asarray(mean, dtype=self.mean[...].dtype)
        self.std[...] = jnp.asarray(std, dtype=self.std[...].dtype)
```

- [ ] **Step 5: Create `models/tarflow/__init__.py` + export from `models/__init__.py`**

`src/gensbi/models/tarflow/__init__.py`:
```python
from gensbi.models.tarflow.model import TarFlowParams, TarFlow

__all__ = ["TarFlowParams", "TarFlow"]
```
In `src/gensbi/models/__init__.py`, add `from .tarflow import TarFlowParams, TarFlow` and add `"TarFlowParams", "TarFlow"` to `__all__`.

- [ ] **Step 6: Write the new standalone TarFlow tests**

`tests/models/tarflow/__init__.py`: empty file.
`tests/models/tarflow/test_tarflow.py`:

```python
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from gensbi.models import TarFlow, TarFlowParams


def test_params_channels_derived_and_validation():
    p = TarFlowParams(rngs=nnx.Rngs(0), dim=4, head_dim=16, num_heads=4)
    assert p.channels == 64                              # head_dim * num_heads
    with pytest.raises(ValueError):
        TarFlowParams(rngs=nnx.Rngs(0), modeled="image")  # img_size/patch_size missing
    with pytest.raises(ValueError):
        TarFlowParams(rngs=nnx.Rngs(0), dim=4, cond="image_prefix")  # cond img args missing


def test_default_head_gives_channels_64():
    p = TarFlowParams(rngs=nnx.Rngs(0), dim=4)
    assert (p.head_dim, p.num_heads, p.channels) == (16, 4, 64)


def test_tarflow_log_prob_and_sample_shapes():
    flow = TarFlow(TarFlowParams(rngs=nnx.Rngs(0), dim=4, cond_dim=2,
                                 head_dim=8, num_heads=2, num_blocks=2,
                                 layers_per_block=1))
    x = jax.random.normal(jax.random.key(1), (3, 4))
    c = jax.random.normal(jax.random.key(2), (3, 2))
    lp = flow.log_prob(x, c)
    assert lp.shape == (3,) and bool(jnp.all(jnp.isfinite(lp)))
    s = flow.sample(jax.random.key(3), cond=c)
    assert s.shape == (3, 4)
```

- [ ] **Step 7: Relocate + rewrite the existing TarFlow tests**

```bash
git mv tests/normalizing_flows/transformer_flow/test_model.py tests/models/tarflow/test_model.py
git mv tests/normalizing_flows/transformer_flow/test_stability.py tests/models/tarflow/test_stability.py
git mv tests/normalizing_flows/transformer_flow/test_structured_integration.py tests/models/tarflow/test_structured_integration.py
git mv tests/normalizing_flows/transformer_flow/test_structured_boundary.py tests/models/tarflow/test_structured_boundary.py
git mv tests/normalizing_flows/transformer_flow/test_blocks_attention.py tests/models/tarflow/test_blocks_attention.py
git mv tests/normalizing_flows/transformer_flow/test_blocks_meta.py tests/models/tarflow/test_blocks_meta.py
git mv tests/normalizing_flows/transformer_flow/test_conditioners.py tests/models/tarflow/test_conditioners.py
git mv tests/normalizing_flows/transformer_flow/test_pipeline_integration.py tests/models/tarflow/test_pipeline_integration.py
```
Apply, in the moved files:
- **`make_tarflow` call sites** (`test_model.py`, `test_stability.py`, `test_structured_integration.py`, `test_structured_boundary.py`, `test_pipeline_integration.py`): apply the call-site rewrite rule; swap imports to `from gensbi.models import TarFlow, TarFlowParams`. Rename `test_stability.py::test_make_tarflow_defaults_and_override` → `test_tarflow_defaults_and_override`. Example (`test_pipeline_integration.py`): `make_tarflow(nnx.Rngs(0), dim=M, cond_dim=D, channels=16, num_blocks=2)` → `TarFlow(TarFlowParams(rngs=nnx.Rngs(0), dim=M, cond_dim=D, head_dim=16, num_heads=1, num_blocks=2))` (channels 16, head_dim default 16 ⇒ num_heads 1).
- **Direct block imports** (`test_stability.py:8`, `test_blocks_attention.py`, `test_blocks_meta.py`): change `from gensbi.normalizing_flows.transformer_flow.blocks import ...` → `from gensbi.models.tarflow.blocks import ...` (this includes `MetaBlock, INV_SOFTPLUS_1`).
- **Direct conditioner imports** (`test_conditioners.py`): change `from gensbi.normalizing_flows.transformer_flow.conditioners import ...` → `from gensbi.models.tarflow.conditioners import ...`.
- **Direct block construction with `head_dim`** → convert to `num_heads`:
  - `test_stability.py` `_block` helper: `MetaBlock(..., head_dim=8, ...)` with `channels=16` → `MetaBlock(..., num_heads=2, ...)`.
  - `test_blocks_attention.py` / `test_blocks_meta.py`: any `AttentionBlock(channels, head_dim, ...)` / `MetaBlock(..., head_dim=HD, ...)` → pass `num_heads = channels // HD` (the block tests use `channels=8, head_dim=4` ⇒ `num_heads=2`).

- [ ] **Step 8: Rewrite the exports test**

Replace the body of `tests/normalizing_flows/transformer_flow/test_exports.py` with:
```python
def test_models_export_nf_classes():
    from gensbi.models import MAFlow, MAFlowParams, TarFlow, TarFlowParams
    from gensbi.models.maf import MAFlow as M2
    from gensbi.models.tarflow import TarFlow as T2
    assert MAFlow is M2 and TarFlow is T2
    assert MAFlowParams is not None and TarFlowParams is not None
```
Then move it: `git mv tests/normalizing_flows/transformer_flow/test_exports.py tests/models/test_nf_exports.py`. Remove the now-empty package init: `git rm tests/normalizing_flows/transformer_flow/__init__.py` (and the directory if empty).

- [ ] **Step 9: Rewrite the TarFlow recovery scripts**

In `scripts/tarflow_nle_recovery.py` (53, 115), `tarflow_field_nle_recovery.py` (39, 77), `tarflow_image_npe_recovery.py` (36, 72): apply the call-site rewrite rule, emitting `num_heads` as a runtime expression where `channels`/`head_dim` are runtime values (e.g. `num_heads=args.channels // args.head_dim`).

- [ ] **Step 10: Run focused TarFlow tests + both smoke imports**

Run: `pytest tests/models/tarflow tests/models/test_nf_exports.py tests/test_import_smoke.py -v`
Expected: PASS. `normalizing_flows/` is now bijections-only; `TarFlow` is self-contained; both `import gensbi.models` and `import gensbi.normalizing_flows` are clean; default `(head_dim=16, num_heads=4)` reproduces prior `channels=64` behaviour; `test_stability.py`'s EMA/buffer-seam assertion runs.

- [ ] **Step 11: Blast-radius check + commit**

Run: `pytest tests/models tests/normalizing_flows tests/recipes tests/flow_matching tests/core -q -m "not slow"`
Expected: PASS.
```bash
git add -A src/gensbi/models src/gensbi/normalizing_flows tests scripts
git commit -m "refactor(nf): migrate TarFlow to models/tarflow (atomic), drop TransformerFlow/make_tarflow, num_heads convention"
```

---

### Task 6: Static verification — structure + stale references + cycle guard

**Files:** none (verification; fix any gap found).

- [ ] **Step 1: Confirm no stale references remain**

Run: `grep -rn "make_maf\|make_tarflow\|TransformerFlow\|normalizing_flows.flow\|transformer_flow\|recipes.utils import.*patchify" src/ tests/ scripts/ | grep -v __pycache__`
Expected: no hits (other than unrelated substrings). Fix any that appear by applying the matching rewrite rule from Task 4 or 5.

- [ ] **Step 2: Confirm `normalizing_flows/` is bijections-only**

Run: `find src/gensbi/normalizing_flows -name '*.py' | grep -v __pycache__`
Expected: only `__init__.py` and `bijections/*.py` (`base`, `chain`, `permutation`, `standardize`, `transformers`, `__init__`). No `flow.py`, no `transformer_flow/`, no `made`/`masked_linear`/`masks`.

- [ ] **Step 3: Fresh-interpreter cycle guard**

Run: `pytest tests/test_import_smoke.py -v`
Expected: PASS (both `import gensbi.models` and `import gensbi.normalizing_flows` succeed in a clean process).

- [ ] **Step 4: Focused NF pass**

Run: `pytest tests/models tests/normalizing_flows -q`
Expected: PASS (all relocated + new NF tests green).

- [ ] **Step 5: Blast-radius check (fast — general pipelines + experimental, no slow)**

Run: `pytest tests/recipes tests/flow_matching tests/core tests/experimental -q -m "not slow"`
Expected: PASS. Confirms the general-pipeline contract and the experimental fielddit/glue (which import the relocated `patchify`) are unaffected. The exhaustive incl-slow run is the dedicated final gate in **Task 7**.

- [ ] **Step 6: Commit (only if Step 1 required fixes; otherwise skip)**

```bash
git add -A
git commit -m "test(nf): static verification — no stale refs, no import cycles, NF=bijections-only"
```

---

### Task 7: Final full-suite regression gate (the very last step)

The exhaustive safety net: run the **entire** GenSBI test suite from a clean, fully-committed tree and confirm zero regressions against the Task 0 baseline. Run this only after every other task is committed.

**Files:** none (read-only). **Precondition:** Tasks 1–6 complete and committed.

- [ ] **Step 1: Confirm clean, committed state**

Run: `git status --short`
Expected: only untracked `reference/` directories — no modified or staged tracked files. (If anything is uncommitted, the gate is invalid — commit or stash first.)

- [ ] **Step 2: Run the entire suite incl. slow (~20 min, background)**

This exceeds the 10-minute foreground command cap — run in the **background** and poll to completion. Tee to a scratch file:
Run: `pytest tests 2>&1 | tee "$SCRATCH/final_pytest.txt"` (background; includes the 4 `slow`-marked tests; uses the configured `-n 2`)
Expected: a summary line with **`0 failed`** and no errors.

- [ ] **Step 3: Compare against the Task 0 baseline**

Diff the `FAILED ...` lines in `final_pytest.txt` against the Task 0 baseline failing set.
- Fully-green baseline ⇒ `final_pytest.txt` must be fully green.
- Non-empty baseline ⇒ no failure may appear that was **not** already in the baseline (pre-existing CPU-only failures may remain; **new** ones are regressions).
The total pass *count* will legitimately differ from baseline — the refactor moved/renamed tests, removed the `make_maf` parity test, and rewrote `test_exports`. Count drift is expected; **new failures are not**.

- [ ] **Step 4: Triage any new failure (do not skip)**

If a test that was green at baseline now fails, STOP — do not declare done. Classify it:
- Missed import/call-site update → apply the matching rewrite rule from Task 4 (MAF) or Task 5 (TarFlow), re-commit, re-run from Step 2.
- Genuine behavioural regression → diagnose and fix (use superpowers:systematic-debugging), re-commit, re-run from Step 2.

- [ ] **Step 5: Declare complete**

Only when Step 2 reports zero failures/errors and Step 3 shows no new failures vs. baseline. The restructure is verified end-to-end with no unexpected regression.

---

## Self-Review

**Spec coverage:**
- Tier-1 pure abstractions stay in `normalizing_flows/bijections/` — Task 4 Step 8, Task 5 Step 2, verified Task 6 Step 2. ✓
- Tier-2 `models/core/` (`patching`, `tokenizers`) — Tasks 1–2. ✓
- Tier-3 machinery co-located (`MADE` family → `models/maf`; blocks/conditioners → `models/tarflow`) — Task 4 Step 7, Task 5 Step 1. ✓
- `MAFlowParams`/`MAFlow`, `TarFlowParams`/`TarFlow` with `rngs` field — Tasks 3, 5. ✓
- Drop `make_maf`/`make_tarflow`; API from `gensbi.models` — Tasks 4, 5; exports Tasks 3 Step 5, 5 Step 5. ✓
- Head convention `(head_dim, num_heads)`, channels derived, default preserves 64, `blocks.py` TODO resolved — Task 5 (Params + Step 3 block rename); default checked in `test_tarflow.py`. ✓
- Sever `normalizing_flows → recipes`; acyclic; smoke test — Task 1 + `tests/test_import_smoke.py`, run every task; TarFlow's cycle risk handled by the atomic Task 5. ✓
- LICENSE attribution travels to `models/tarflow/` — Task 5 Step 1. ✓
- General pipelines stay green + full blast radius (incl. `tests/experimental` fielddit/glue, `test_pipeline_utils` patchify) — Task 5 Step 11 + Task 6 Step 5 (fast) + Task 7 (entire suite incl. slow, vs. Task 0 baseline). ✓
- No unexpected regression — Task 0 captures a baseline; Task 7 gates on zero new failures across the whole suite. ✓
- EMA/buffer-seam survives the `Mask` move into `TarFlow` — `test_stability.py` (which asserts the seam) is relocated and import/`num_heads`-corrected in Task 5 Step 7 and run in Steps 10–11.

**Placeholder scan:** No `TBD`/`TODO`/"handle edge cases" in steps; all new modules and new tests have complete code; relocations give exact `git mv` + import edits; call-site rewrites give an exact rule + worked example + file/line list.

**Type consistency:** `MAFlow`/`MAFlowParams`, `TarFlow`/`TarFlowParams` names and signatures match across the Interfaces blocks and the `gensbi.models` exports. `AttentionBlock`/`MetaBlock` parameter renamed `head_dim` → `num_heads` consistently in Task 5 Step 3 (blocks), Step 4 (TarFlow's `MetaBlock(..., num_heads=...)` call), and Step 7 (every direct-construction test, incl. `test_stability.py`'s `_block`). `make_gaussian_prior`, `Chain`, `MaskedAutoregressive`, `Standardize`, `Mask` signatures match the absorbed `flow.py`/`model.py` source. Neither `MAFlow` nor `TarFlow` stores `self.params` (unused; keeps `rngs` out of the attribute tree).
