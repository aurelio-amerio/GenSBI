# First-Class healpix-rope Pipeline Ids + Repo Restructuring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make spherical HEALPix RoPE ids a first-class `ConditionalPipeline` strategy (`HealpixRope`), then restructure three repos so HEAL-SWIN-nnx becomes a standalone PyPI package that gensbi depends on and mirrors, with the spherical GRF example living in GenSBI-examples under the YAML-config convention.

**Architecture:** `id_embedding_strategy` tuple slots learn to accept `str | IdStrategy` objects via one dispatch arm in `_resolve_embedding_ids`; `HealpixRope(nside, base_pixels)` carries the geometry the string enum cannot. Restructuring is strictly ordered: GenSBI API → HEAL-SWIN-nnx slim+publishable (**HARD STOP** for the user's manual PyPI publish) → gensbi dependency + `gensbi.models.healswin` mirror → example migration.

**Tech Stack:** JAX/Flax NNX, healpy, uv (HEAL-SWIN-nnx), HTCondor, sbibm-jax, grain.

**Spec:** `docs/superpowers/specs/2026-07-19-healpix-rope-pipeline-and-restructure-design.md`

## Global Constraints

- Repos: GenSBI = `/lustre/ific.uv.es/ml/ific088/github/GenSBI`; HEAL-SWIN-nnx = `/lhome/ific/a/aamerio/data/github/HEAL-SWIN-nnx`; GenSBI-examples = `/lhome/ific/a/aamerio/data/github/GenSBI-examples`. All work on short-lived branches off `main`, merged when the task's tests are green.
- GenSBI tests run in the **mamba `gensbi` env** (NOT a `.venv`): activate it (or `mamba run -n gensbi`) before any `python -m pytest`. HEAL-SWIN-nnx uses `uv run pytest` (its pyproject sets `JAX_PLATFORMS=cpu` and `-n 2`).
- Existing string strategies (`"absolute"`, `"pos1d"`, `"rope1d"`, `"pos2d"`, `"rope2d"`) must behave **byte-identically** — no renames, no deprecations, no `init_ids_1d` axis-order changes.
- Version floors (copy verbatim): gensbi gains `heal-swin-nnx>=0.1.0`; GenSBI-examples gains `sbibm-jax[loader]>=0.1.3` and `jax>=0.10.2, <0.12.0`. HEAL-SWIN-nnx publishes as version `0.1.0`.
- **HARD STOP after Task 8:** the user manually publishes `heal-swin-nnx 0.1.0` to PyPI. Do not start Task 9 until the PyPI check in its Step 1 passes.
- Commit messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

## Phase 1 — GenSBI: the strategy-object API

Work in `/lustre/ific.uv.es/ml/ific088/github/GenSBI`, branch `healpix-rope-pipeline` off `main`:

```bash
cd /lustre/ific.uv.es/ml/ific088/github/GenSBI && git checkout -b healpix-rope-pipeline main
```

### Task 1: Harden `init_ids_healpix` base_pixels validation

**Files:**
- Modify: `src/gensbi/recipes/utils.py` (function `init_ids_healpix`, ~line 119; new helper above it)
- Test: `tests/recipes/test_healpix_ids.py`

**Interfaces:**
- Produces: `_validate_base_pixels(base_pixels) -> list[int]` in `gensbi/recipes/utils.py` — raises `ValueError` on empty, non-integer, out-of-range, or duplicate entries. Task 2 reuses it.

- [ ] **Step 1: Write the failing tests** (append to `tests/recipes/test_healpix_ids.py`)

```python
def test_init_ids_healpix_rejects_empty_base_pixels():
    # Previously leaked a raw numpy "need at least one array to concatenate".
    with pytest.raises(ValueError, match="non-empty"):
        init_ids_healpix(2, base_pixels=[])


def test_init_ids_healpix_rejects_non_integer_base_pixels():
    # Previously slipped validation and reached hp.pix2vec.
    with pytest.raises(ValueError, match="integer"):
        init_ids_healpix(2, base_pixels=[1.5])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/recipes/test_healpix_ids.py -q -k "empty_base or non_integer"`
Expected: 2 FAILED — empty raises `ValueError: need at least one array to concatenate` (message mismatch), non-integer raises a healpy `TypeError` (or passes through), not our messages.

- [ ] **Step 3: Implement.** In `src/gensbi/recipes/utils.py`, add above `init_ids_healpix`:

```python
def _validate_base_pixels(base_pixels):
    """Validate a base-pixel subset spec; return it as a list of ints."""
    base_pixels = list(base_pixels)
    if not base_pixels:
        raise ValueError(
            "base_pixels must be non-empty; omit it (or pass None) for full sky"
        )
    if any(not isinstance(b, (int, np.integer)) for b in base_pixels):
        raise ValueError(f"base_pixels entries must be integers, got {base_pixels}")
    if any(b < 0 or b > 11 for b in base_pixels) or len(set(base_pixels)) != len(
        base_pixels
    ):
        raise ValueError(
            f"base_pixels must be unique integers in [0, 11], got {base_pixels}"
        )
    return base_pixels
```

and replace the existing validation block inside `init_ids_healpix` (the lines from `if base_pixels is None:` through the `raise ValueError("base_pixels must be unique integers...")`) with:

```python
    if base_pixels is None:
        base_pixels = range(12)
    base_pixels = _validate_base_pixels(base_pixels)
```

- [ ] **Step 4: Run the whole healpix test file**

Run: `python -m pytest tests/recipes/test_healpix_ids.py -q`
Expected: all pass (the 9 existing tests still green — the old unique/range message is preserved by `_validate_base_pixels`).

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/recipes/utils.py tests/recipes/test_healpix_ids.py
git commit -m "fix: proper errors for empty/non-integer base_pixels in init_ids_healpix"
```

### Task 2: `HealpixRope` strategy object

**Files:**
- Create: `src/gensbi/recipes/id_strategies.py`
- Modify: `src/gensbi/recipes/__init__.py`
- Test: `tests/recipes/test_id_strategies.py`

**Interfaces:**
- Consumes: `init_ids_healpix`, `healpix_rope_theta`, `_validate_base_pixels` from `gensbi.recipes.utils` (Task 1).
- Produces: `gensbi.recipes.HealpixRope` — frozen dataclass `HealpixRope(nside: int, base_pixels: tuple | None = None)` with `name: ClassVar[str] = "healpix-rope"`, `num_tokens: int` property, `theta: int` property, `build(dim) -> (ids, resolved_dim)`. Also `gensbi.recipes.IdStrategy` (documentation Protocol). Tasks 3 and 11 use `HealpixRope`.

- [ ] **Step 1: Write the failing tests** (new file `tests/recipes/test_id_strategies.py`)

```python
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import pytest

from gensbi.recipes import HealpixRope, IdStrategy
from gensbi.recipes.utils import healpix_rope_theta, init_ids_healpix


def test_healpix_rope_build_matches_init_ids_healpix():
    ids, n = HealpixRope(nside=2).build(48)
    ref_ids, ref_n = init_ids_healpix(2)
    assert n == ref_n == 48
    np.testing.assert_array_equal(np.asarray(ids), np.asarray(ref_ids))


def test_healpix_rope_subset_build_matches_init_ids_healpix():
    ids, n = HealpixRope(nside=2, base_pixels=(3, 7)).build(8)
    ref_ids, ref_n = init_ids_healpix(2, base_pixels=[3, 7])
    assert n == ref_n == 8
    np.testing.assert_array_equal(np.asarray(ids), np.asarray(ref_ids))


def test_healpix_rope_dim_mismatch_names_both_numbers():
    # The error must name the given dim AND the expected token count.
    with pytest.raises(ValueError, match=r"(?s)47.*48|48.*47"):
        HealpixRope(nside=2).build(47)


def test_healpix_rope_theta_property():
    assert HealpixRope(nside=2).theta == healpix_rope_theta(2) == 480


def test_healpix_rope_num_tokens():
    assert HealpixRope(nside=2).num_tokens == 48
    assert HealpixRope(nside=2, base_pixels=(0, 1, 2)).num_tokens == 12


def test_healpix_rope_validates_at_construction():
    with pytest.raises(ValueError, match="power of 2"):
        HealpixRope(nside=3)
    with pytest.raises(ValueError, match="non-empty"):
        HealpixRope(nside=2, base_pixels=())
    with pytest.raises(ValueError, match="integer"):
        HealpixRope(nside=2, base_pixels=(1.5,))


def test_healpix_rope_satisfies_id_strategy_protocol():
    strat = HealpixRope(nside=2)
    assert isinstance(strat, IdStrategy)
    assert strat.name == "healpix-rope"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/recipes/test_id_strategies.py -q`
Expected: collection error — `ImportError: cannot import name 'HealpixRope' from 'gensbi.recipes'`.

- [ ] **Step 3: Implement.** Create `src/gensbi/recipes/id_strategies.py`:

```python
"""Structured id-builder strategies for the recipe pipelines.

Two DIFFERENT vocabularies share the word "rope" — they collided once
(2026-07-19 handoff) and are deliberately kept distinct:

- **Model-side** strategy strings (e.g. ``Flux1Params.id_embedding_strategy``):
  ``"rope"`` means *apply RoPE to whatever ids arrive at the forward pass*.
- **Pipeline-side** builder strategies (``ConditionalPipeline``'s
  ``id_embedding_strategy``): strings like ``"rope1d"``/``"rope2d"`` — and the
  objects in this module — *build* the id arrays themselves.

A :class:`HealpixRope` pipeline strategy pairs with model-side
``("absolute", "rope")`` plus a 3-entry even ``axes_dim`` summing to the
per-head dim (e.g. ``(22, 22, 20)`` for 64).
"""

from dataclasses import dataclass
from typing import ClassVar, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable

from gensbi.recipes.utils import (
    _validate_base_pixels,
    healpix_rope_theta,
    init_ids_healpix,
)


@runtime_checkable
class IdStrategy(Protocol):
    """Structural interface for pipeline id-builder strategy objects.

    Any object with a ``name`` and a ``build(dim) -> (ids, resolved_dim)``
    method can be passed in an ``id_embedding_strategy`` tuple slot; the
    pipeline calls ``build`` with the corresponding ``dim_obs``/``dim_cond``.
    Strategies own their full geometry — unlike the string strategies they
    receive no ``semantic_id``/``size``.
    """

    name: str

    def build(self, dim):
        """Return ``(ids, resolved_dim)`` for a stream of ``dim`` tokens."""
        ...


@dataclass(frozen=True)
class HealpixRope:
    """Spherical RoPE ids for tokens on a HEALPix grid (name: "healpix-rope").

    Wraps :func:`gensbi.recipes.utils.init_ids_healpix` (see there for the
    method and its rationale) with the geometry needed to build the ids —
    which the string-enum API cannot carry, since the pipeline only passes a
    token count.

    Parameters
    ----------
    nside : int
        HEALPix resolution of the *token* grid (power of 2).
    base_pixels : sequence of int, optional
        Base pixels (0..11) covered by the grid; ``None`` = full sky.
    """

    nside: int
    base_pixels: Optional[Union[Tuple[int, ...], Sequence[int]]] = None

    name: ClassVar[str] = "healpix-rope"

    def __post_init__(self):
        if self.nside < 1 or (self.nside & (self.nside - 1)) != 0:
            raise ValueError(f"nside must be a power of 2, got {self.nside}")
        if self.base_pixels is not None:
            object.__setattr__(
                self, "base_pixels", tuple(_validate_base_pixels(self.base_pixels))
            )

    @property
    def num_tokens(self) -> int:
        """Token count of the grid: ``n_faces * nside**2``."""
        n_faces = 12 if self.base_pixels is None else len(self.base_pixels)
        return n_faces * self.nside**2

    @property
    def theta(self) -> int:
        """Suggested model-side RoPE theta: :func:`healpix_rope_theta`."""
        return healpix_rope_theta(self.nside)

    def build(self, dim):
        """Return ``(ids, num_tokens)``; ``dim`` must match the grid."""
        if dim != self.num_tokens:
            sky = (
                "full sky"
                if self.base_pixels is None
                else f"{len(self.base_pixels)} base pixels"
            )
            raise ValueError(
                f"{self.name}: the pipeline stream has dim={dim} tokens, but "
                f"nside={self.nside} ({sky}) implies {self.num_tokens} tokens"
            )
        return init_ids_healpix(self.nside, self.base_pixels)
```

In `src/gensbi/recipes/__init__.py`, add after the existing pipeline imports:

```python
from .id_strategies import IdStrategy, HealpixRope
```

and add `"IdStrategy",` and `"HealpixRope",` to `__all__`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/recipes/test_id_strategies.py -q`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/recipes/id_strategies.py src/gensbi/recipes/__init__.py tests/recipes/test_id_strategies.py
git commit -m "feat: HealpixRope IdStrategy object carrying HEALPix geometry"
```

### Task 3: Dispatch arm in `_resolve_embedding_ids` + namespace docs + pipeline integration

**Files:**
- Modify: `src/gensbi/recipes/utils.py` (`_resolve_embedding_ids`, ~line 267)
- Modify: `src/gensbi/recipes/conditional_pipeline.py` (class docstring, ~lines 83-91)
- Test: `tests/recipes/test_id_strategies.py` (append)

**Interfaces:**
- Consumes: `HealpixRope` (Task 2); `MockConditionalModel` from `tests/recipes/mock_models.py` (existing; `__call__(t, obs, obs_ids, cond, cond_ids, conditioned=True, guidance=None)`).
- Produces: `_resolve_embedding_ids(dim, strategy, semantic_id, size=2)` now accepts any object with `.build`; `ConditionalPipeline(..., id_embedding_strategy=("absolute", HealpixRope(nside=2)))` works. Task 11 relies on this.

- [ ] **Step 1: Write the failing tests** (append to `tests/recipes/test_id_strategies.py`)

```python
import grain
import jax
from flax import nnx  # noqa: F401  (parity with sibling test imports)

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from mock_models import MockConditionalModel

from gensbi.core import FlowMatchingMethod
from gensbi.recipes import ConditionalPipeline
from gensbi.recipes.utils import _resolve_embedding_ids, init_ids_1d


def _tiny_datasets(dim_obs, dim_cond, n=64, batch=8):
    key = jax.random.PRNGKey(0)
    theta = jax.random.normal(key, (n, dim_obs, 1))
    x = jax.random.normal(key, (n, dim_cond, 1))

    def make(sl):
        data = np.concatenate([np.asarray(theta[sl]), np.asarray(x[sl])], axis=1)
        return (
            grain.MapDataset.source(data)
            .repeat()
            .to_iter_dataset()
            .batch(batch)
            .map(lambda d: (d[:, :dim_obs], d[:, dim_obs:]))
        )

    return make(slice(0, n // 2)), make(slice(n // 2, n))


def test_conditional_pipeline_accepts_healpix_rope_strategy():
    train_ds, val_ds = _tiny_datasets(3, 48)
    pipeline = ConditionalPipeline(
        model=MockConditionalModel(),
        train_dataset=train_ds,
        val_dataset=val_ds,
        dim_obs=3,
        dim_cond=48,
        method=FlowMatchingMethod(),
        id_embedding_strategy=("absolute", HealpixRope(nside=2)),
    )
    ref_ids, _ = init_ids_healpix(2)
    np.testing.assert_array_equal(np.asarray(pipeline.cond_ids), np.asarray(ref_ids))
    assert pipeline.dim_cond == 48
    # obs stream untouched: 1D absolute ids as before
    obs_ref, _ = init_ids_1d(3, semantic_id=0)
    np.testing.assert_array_equal(np.asarray(pipeline.obs_ids), np.asarray(obs_ref))


def test_conditional_pipeline_healpix_rope_dim_mismatch_raises():
    train_ds, val_ds = _tiny_datasets(3, 47)
    with pytest.raises(ValueError, match="healpix-rope"):
        ConditionalPipeline(
            model=MockConditionalModel(),
            train_dataset=train_ds,
            val_dataset=val_ds,
            dim_obs=3,
            dim_cond=47,
            method=FlowMatchingMethod(),
            id_embedding_strategy=("absolute", HealpixRope(nside=2)),
        )


def test_resolve_embedding_ids_dispatches_to_strategy_objects():
    ids, n = _resolve_embedding_ids(48, HealpixRope(nside=2), semantic_id=1)
    ref_ids, _ = init_ids_healpix(2)
    np.testing.assert_array_equal(np.asarray(ids), np.asarray(ref_ids))
    assert n == 48


def test_resolve_embedding_ids_unknown_string_mentions_objects():
    # Prefix preserved for backward compat; message now teaches the object API.
    with pytest.raises(ValueError, match="Unknown id embedding strategy"):
        _resolve_embedding_ids(10, "rope3d", semantic_id=1)
    with pytest.raises(ValueError, match="IdStrategy"):
        _resolve_embedding_ids(10, "rope3d", semantic_id=1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/recipes/test_id_strategies.py -q`
Expected: the two pipeline tests and the dispatch test FAIL with `ValueError: Unknown id embedding strategy: HealpixRope(...)`; the message test FAILS on the `IdStrategy` match.

- [ ] **Step 3: Implement.** In `src/gensbi/recipes/utils.py`, `_resolve_embedding_ids`: insert the object arm first and extend the error. Replace the function body's dispatch with:

```python
    if hasattr(strategy, "build"):
        return strategy.build(dim)
    if strategy in _EMBEDDINGS_1D:
        return init_ids_1d(dim, semantic_id=semantic_id)
    elif strategy in _EMBEDDINGS_2D:
        return init_ids_2d(dim, semantic_id=semantic_id, size=size)
    else:
        raise ValueError(
            f"Unknown id embedding strategy: {strategy!r}. Expected one of "
            f"{sorted(_EMBEDDINGS_1D | _EMBEDDINGS_2D)} or an IdStrategy object "
            "with a build(dim) method (e.g. gensbi.recipes.HealpixRope)."
        )
```

Also update the function docstring `strategy` parameter description to:

```
    strategy : str or IdStrategy
        Embedding strategy name (e.g., "absolute", "pos1d", "rope1d",
        "pos2d", "rope2d") or a strategy object with ``build(dim)`` (e.g.
        :class:`gensbi.recipes.HealpixRope`). NOTE these pipeline-side
        builder names are a different vocabulary from the model-side
        ``id_embedding_strategy`` strings (where "rope" means "apply RoPE
        to the ids the pipeline provides") — see
        :mod:`gensbi.recipes.id_strategies`.
```

In `src/gensbi/recipes/conditional_pipeline.py`, replace the `id_embedding_strategy` parameter docstring (lines 83-85) with:

```
    id_embedding_strategy : tuple of (str or IdStrategy), optional
        Per-stream (obs, cond) id-builder strategy. Strings pick a built-in
        1D/2D grid builder; an :class:`~gensbi.recipes.IdStrategy` object
        (e.g. :class:`~gensbi.recipes.HealpixRope`) builds ids from its own
        geometry. Default is ``("absolute", "absolute")``. NOTE this
        pipeline-side vocabulary is distinct from the model-side
        ``id_embedding_strategy`` (e.g. ``Flux1Params``), where "rope"
        means "apply RoPE to the provided ids"; a ``HealpixRope`` pipeline
        strategy pairs with model-side ``("absolute", "rope")`` and a
        3-entry ``axes_dim``.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/recipes/test_id_strategies.py tests/recipes/test_healpix_ids.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/recipes/utils.py src/gensbi/recipes/conditional_pipeline.py tests/recipes/test_id_strategies.py
git commit -m "feat: ConditionalPipeline accepts IdStrategy objects (healpix-rope)"
```

### Task 4: Full fast suite + merge Phase 1

- [ ] **Step 1: Run the fast test suite**

Run: `python -m pytest tests/ -q -n auto -m "not slow and not experimental and not extraslow"`
Expected: all pass, zero regressions (string strategies byte-identical).

- [ ] **Step 2: Merge to main**

```bash
git checkout main && git merge --ff-only healpix-rope-pipeline && git branch -d healpix-rope-pipeline
```

Expected: fast-forward; `git log --oneline -3` shows the three Phase-1 commits on main.

---

## Phase 2 — HEAL-SWIN-nnx: slim down, make publishable

Work in `/lhome/ific/a/aamerio/data/github/HEAL-SWIN-nnx`. The repo is currently checked out on `healpix-rope` with an untracked `.github/` directory.

### Task 5: Fold the `healpix-rope` branch into main

- [ ] **Step 1: Fast-forward merge and delete the branch**

```bash
cd /lhome/ific/a/aamerio/data/github/HEAL-SWIN-nnx
git checkout main && git merge --ff-only healpix-rope && git branch -d healpix-rope
```

Expected: main now at `0748eda` ("feat: spherical HEALPix RoPE cond ids in GRF flow-matching example"). This puts the port source for Task 11 permanently in main's history.

### Task 6: Removal commit (example + gensbi coupling)

**Files:**
- Delete: `examples/spherical_grf_flowmatch.py`, `examples/sub/spherical_grf_flowmatch.sub`, `examples/sub/run_spherical_grf_flowmatch.sh`, `examples/spherical_grf_fm_results.txt`, `examples/spherical_grf_fm_quick_results.txt`, `examples/imgs/spherical_grf_fm*.png` (16 files)
- Modify: `pyproject.toml`, `README.md`

- [ ] **Step 1: Branch and delete the files**

```bash
git checkout -b slim-for-pypi
git rm examples/spherical_grf_flowmatch.py \
       examples/sub/spherical_grf_flowmatch.sub \
       examples/sub/run_spherical_grf_flowmatch.sh \
       examples/spherical_grf_fm_results.txt \
       examples/spherical_grf_fm_quick_results.txt
git rm examples/imgs/spherical_grf_fm*.png
```

Expected: `git status` shows 21 deletions, nothing else touched.

- [ ] **Step 2: Strip pyproject coupling.** In `pyproject.toml`, delete the `gensbi = [...]` block from `[project.optional-dependencies]` AND the entire `[tool.uv.sources]` table (both the `sbibm-jax` and `gensbi` local-path entries). The `examples`, `cuda12`, `cuda13` extras and `[dependency-groups]` stay.

- [ ] **Step 3: README pointer.** In `README.md`, where examples are described (or at the end of the examples section), add:

```markdown
The spherical GRF simulation-based-inference example (HealSwin encoder +
gensbi Flux1 flow matching) lives in
[GenSBI-examples](https://github.com/aurelio-amerio/GenSBI-examples) under
`examples/sbi-benchmarks/spherical_grf/`.
```

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml README.md
git commit -m "refactor!: move spherical GRF SBI example to GenSBI-examples, drop gensbi coupling

Removes the gensbi optional extra and all local [tool.uv.sources] path
overrides so the package has no absolute-path or unpublished deps and can
ship to PyPI.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 7: Fix and commit the GitHub workflows + package metadata

**Files:**
- Modify (currently untracked): `.github/workflows/python-publish.yml`, `.github/workflows/python-app.yml`
- Modify: `pyproject.toml`

Both workflows were copied from GenSBI and NOT adapted: the publish URL points at `pypi.org/p/gensbi`, and the CI references dependency groups (`lint`, `test`, `docs`), pytest markers, junitparser/genbadge tooling, and a docs build that do not exist in this repo.

- [ ] **Step 1: Rewrite `.github/workflows/python-publish.yml`** (only the environment URL changes):

```yaml
name: Publish Python 🐍 distribution 📦 to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    name: Build and publish 📦 to PyPI
    runs-on: ubuntu-latest
    environment:
      name: pypi
      url: https://pypi.org/p/heal-swin-nnx
    permissions:
      id-token: write
      contents: read
    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        uses: astral-sh/setup-uv@v7
      - name: Set up Python
        run: uv python install 3.13
      - name: Build
        run: uv build
      - name: Publish distribution 📦 to PyPI
        run: uv publish
```

- [ ] **Step 2: Rewrite `.github/workflows/python-app.yml`** to match what this repo actually has (`dev` dependency group with pytest/pytest-env/pytest-xdist; pyproject pytest config already sets `-n 2` and `JAX_PLATFORMS=cpu`; no lint group, no docs build, no badges):

```yaml
name: Tests

on:
  push:
    branches: ["main"]
  pull_request:
    branches: ["main"]

permissions:
  contents: read

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        uses: astral-sh/setup-uv@v7
      - name: Set up Python 3.12
        run: uv python install 3.12
      - name: Install dependencies
        run: uv sync
      - name: Run tests
        run: uv run pytest
```

- [ ] **Step 3: Add `[project.urls]` to `pyproject.toml`** (after the `[project]` table's `dependencies`):

```toml
[project.urls]
Homepage = "https://github.com/aurelio-amerio/HEAL-SWIN-nnx"
Issues = "https://github.com/aurelio-amerio/HEAL-SWIN-nnx/issues"
```

- [ ] **Step 4: Commit**

```bash
git add .github/ pyproject.toml
git commit -m "ci: adapt copied workflows to heal-swin-nnx, add project urls

python-publish.yml pointed at pypi.org/p/gensbi; python-app.yml referenced
dependency groups, markers, and badge tooling this repo does not have.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 8: Build verification, merge — then HARD STOP

- [ ] **Step 1: Re-sync the env** (also restores the CUDA jaxlib packages the old gensbi source-override uninstalled — closes handoff pending-gate 2):

Run: `uv sync --extra cuda12 --extra examples`
Expected: resolves cleanly; installs `jax[cuda12]` packages; no `sbibm-jax`/`gensbi` from local paths.

- [ ] **Step 2: Full test suite**

Run: `uv run pytest`
Expected: all pass (the deleted example is not under `tests/`).

- [ ] **Step 3: Build and wheel-install test**

```bash
uv build
python -m venv /tmp/aamerio/healswin-wheel-test
/tmp/aamerio/healswin-wheel-test/bin/pip install dist/heal_swin_nnx-0.1.0-py3-none-any.whl pytest pytest-env pytest-xdist
JAX_PLATFORMS=cpu /tmp/aamerio/healswin-wheel-test/bin/python -c "import heal_swin_nnx; print(sorted(heal_swin_nnx.__all__))"
JAX_PLATFORMS=cpu /tmp/aamerio/healswin-wheel-test/bin/python -m pytest tests/ -q
```

Expected: import prints the 13-name `__all__` (Buffer … SwinUnet); test suite green against the **installed wheel** (catches packaging omissions).

- [ ] **Step 4: Merge and stop**

```bash
git checkout main && git merge --ff-only slim-for-pypi && git branch -d slim-for-pypi
```

**HARD STOP.** Report to the user: HEAL-SWIN-nnx main is publish-ready. The user pushes main and publishes `heal-swin-nnx 0.1.0` manually (either `uv publish` after `uv build`, or by cutting a GitHub release so `python-publish.yml` trusted-publishing runs). **Do not proceed to Task 9 until the user confirms.**

---

## Phase 3 — GenSBI: dependency + mirror (after PyPI publish)

Work in `/lustre/ific.uv.es/ml/ific088/github/GenSBI`, branch `healswin-mirror` off `main`.

### Task 9: `heal-swin-nnx` dependency + `gensbi.models.healswin` mirror

**Files:**
- Modify: `pyproject.toml`, `src/gensbi/models/__init__.py`
- Create: `src/gensbi/models/healswin.py`
- Test: `tests/models/test_healswin_mirror.py`

**Interfaces:**
- Consumes: published `heal-swin-nnx` (13 public names in `heal_swin_nnx.__all__`: `Buffer`, `HealConv`, `HealConvDecoder`, `HealConvEncoder`, `HealConvParams`, `HealSwin`, `HealSwinDecoder`, `HealSwinEncoder`, `HealSwinParams`, `SwinDecoder`, `SwinEncoder`, `SwinParams`, `SwinUnet`).
- Produces: `from gensbi.models import HealSwinEncoder, HealSwinParams` (Task 11 imports exactly this); `gensbi.models.healswin` mirrors the full surface.

- [ ] **Step 1: Verify the publish landed** (gate for this whole phase)

Run: `curl -s https://pypi.org/pypi/heal-swin-nnx/json | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['info']['version'])"`
Expected: `0.1.0`. If not, STOP — the user has not published yet.

- [ ] **Step 2: Add the dependency and install it**

```bash
git checkout -b healswin-mirror main
```

In `pyproject.toml` `[project] dependencies`, after `"healpy>=1.19.0",` add:

```toml
    "heal-swin-nnx>=0.1.0",
```

Then install into the mamba env: `pip install "heal-swin-nnx>=0.1.0"`
Expected: installs from PyPI without pulling new transitive deps (jax/flax/einops/numpy/healpy already present).

- [ ] **Step 3: Write the failing test** (new file `tests/models/test_healswin_mirror.py`)

```python
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
from flax import nnx


def test_mirror_reexports_are_the_same_objects():
    import heal_swin_nnx
    from gensbi.models import HealSwinEncoder, HealSwinParams
    from gensbi.models import healswin

    assert HealSwinEncoder is heal_swin_nnx.HealSwinEncoder
    assert HealSwinParams is heal_swin_nnx.HealSwinParams
    assert set(healswin.__all__) == set(heal_swin_nnx.__all__)


def test_healswin_encoder_tiny_forward_via_gensbi():
    from gensbi.models import HealSwinEncoder, HealSwinParams

    # Known-good tiny config from HEAL-SWIN-nnx's own test suite
    # (tests/test_model.py::tiny_params): 8 faces at nside 16, 2 stages.
    p = HealSwinParams(
        nside=16, in_channels=3, out_channels=5, base_pixels=tuple(range(8)),
        embed_dim=16, depths=(2, 2), num_heads=(2, 4), drop_path_rate=0.0,
    )
    enc = HealSwinEncoder(p, rngs=nnx.Rngs(0))
    enc.eval()
    tokens, skips = enc(jnp.ones((1, p.npix, 3)))
    # N/(patch * 4^(L-1)) tokens, embed_dim * 2^(L-1) features
    assert tokens.shape == (1, p.npix // 4 // 4, 32)
```

- [ ] **Step 4: Run test to verify it fails**

Run: `python -m pytest tests/models/test_healswin_mirror.py -q`
Expected: FAIL — `ImportError: cannot import name 'HealSwinEncoder' from 'gensbi.models'`.

- [ ] **Step 5: Implement.** Create `src/gensbi/models/healswin.py`:

```python
"""Mirror of the standalone ``heal_swin_nnx`` package.

GenSBI depends on `heal-swin-nnx <https://pypi.org/p/heal-swin-nnx>`_ (the
HEALPix-native spherical Swin V2 U-Net in Flax NNX) and re-exports its public
API here, so spherical encoders are importable alongside the other gensbi
models. The SBI-relevant names — :class:`HealSwinEncoder` and
:class:`HealSwinParams` — are also exported from :mod:`gensbi.models`
directly; everything else (full U-Nets, decoders, HealConv, planar Swin) is
available from this module or from ``heal_swin_nnx`` itself.
"""

from heal_swin_nnx import (
    Buffer,
    HealConv,
    HealConvDecoder,
    HealConvEncoder,
    HealConvParams,
    HealSwin,
    HealSwinDecoder,
    HealSwinEncoder,
    HealSwinParams,
    SwinDecoder,
    SwinEncoder,
    SwinParams,
    SwinUnet,
)

__all__ = [
    "Buffer",
    "HealConv",
    "HealConvDecoder",
    "HealConvEncoder",
    "HealConvParams",
    "HealSwin",
    "HealSwinDecoder",
    "HealSwinEncoder",
    "HealSwinParams",
    "SwinDecoder",
    "SwinEncoder",
    "SwinParams",
    "SwinUnet",
]
```

In `src/gensbi/models/__init__.py`, after the `.tarflow` import add:

```python
from .healswin import HealSwinEncoder, HealSwinParams
```

and add `"HealSwinEncoder",` and `"HealSwinParams",` to `__all__`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/models/test_healswin_mirror.py -q`
Expected: 2 passed.

- [ ] **Step 7: Fast suite + merge**

```bash
python -m pytest tests/ -q -n auto -m "not slow and not experimental and not extraslow"
git add pyproject.toml src/gensbi/models/healswin.py src/gensbi/models/__init__.py tests/models/test_healswin_mirror.py
git commit -m "feat: depend on heal-swin-nnx and mirror it via gensbi.models.healswin"
git checkout main && git merge --ff-only healswin-mirror && git branch -d healswin-mirror
```

---

## Phase 4 — GenSBI-examples: the spherical_grf example

Work in `/lhome/ific/a/aamerio/data/github/GenSBI-examples`, branch `spherical-grf-example` off `main`:

```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI-examples && git checkout -b spherical-grf-example main
```

### Task 10: pyproject dependency fixes

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Verify sbibm-jax 0.1.3 is on PyPI** (the example needs its `spherical_grf` task + `TaskDataset`)

Run: `curl -s https://pypi.org/pypi/sbibm-jax/json | python3 -c "import json,sys; d=json.load(sys.stdin); print(sorted(d['releases']))"`
Expected: includes `0.1.3`. If not, STOP and tell the user (they were uploading it on 2026-07-19).

- [ ] **Step 2: Update `[project] dependencies`** to (changes: add `gensbi`, fix stale `jax` pin to gensbi's floor, add the sbibm-jax version floor):

```toml
dependencies = [
    "gensbi",
    "jax>=0.10.2, <0.12.0",
    "datasets",
    "huggingface_hub",
    "matplotlib>=3.10",
    "numpy>=2.3.5",
    "flax>=0.12.4",
    "grain>=0.2.15",
    "imageio>=2.37.2",
    "sbibm-jax[loader]>=0.1.3",
]
```

(Once a gensbi release containing `HealpixRope` + the healswin mirror ships, bump `"gensbi"` to that version floor — noted as a release-time follow-up, not done here. Local runs meanwhile use the mamba env's local gensbi tree; `heal-swin-nnx` arrives transitively through it.)

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build(deps): gensbi + sbibm-jax>=0.1.3, fix stale jax pin"
```

### Task 11: Port the example with YAML configs

**Files:**
- Create: `examples/sbi-benchmarks/spherical_grf/train-spherical-grf.py`
- Create: `examples/sbi-benchmarks/spherical_grf/config/config_healpix.yaml`
- Create: `examples/sbi-benchmarks/spherical_grf/config/config_pos1d.yaml`

**Interfaces:**
- Consumes: `gensbi.models.{HealSwinEncoder, HealSwinParams, Flux1, Flux1Params}` (Task 9), `gensbi.recipes.{ConditionalPipeline, HealpixRope}` (Tasks 2-3), `gensbi.recipes.utils.{init_ids_1d, parse_training_config}`, `gensbi.recipes.flux1.parse_flux1_params` (returns a dict with keys `in_channels, vec_in_dim, context_in_dim, mlp_ratio, num_heads, depth, depth_single_blocks, axes_dim, val_emb_dim, id_emb_dim, id_merge_mode, qkv_bias, theta, id_embedding_strategy, param_dtype`), `sbibm_jax.data.TaskDataset`, `sbibm_jax.tasks.get_task`.
- Produces: a headless script `train-spherical-grf.py --config config/config_{healpix,pos1d}.yaml` with `SMOKE=1`/`QUICK=1` debug modes; Task 12's sub files invoke it.

- [ ] **Step 1: Create the directory and extract the port source** (the final example state is in HEAL-SWIN-nnx main history after Task 5):

```bash
mkdir -p examples/sbi-benchmarks/spherical_grf/config
git -C /lhome/ific/a/aamerio/data/github/HEAL-SWIN-nnx show 0748eda:examples/spherical_grf_flowmatch.py \
  > examples/sbi-benchmarks/spherical_grf/train-spherical-grf.py
```

- [ ] **Step 2: Write `config/config_healpix.yaml`:**

```yaml
task_name: spherical_grf

strategy:
  method: "flow"
  model: "flux"

# Pipeline-side cond-id builder: "healpix-rope" (spherical RoPE on HEALPix
# pixel-center 3D coords, gensbi.recipes.HealpixRope) or "pos1d" (1D
# sinusoidal ids over NEST order — the A/B baseline, see config_pos1d.yaml).
ids:
  cond_kind: "healpix-rope"

data:
  nside: 64
  seed: 0

# HealSwin encoder: patch embed nside 64->32, then 4 mergings 32->...->2.
encoder:
  embed_dim: 32
  depths: [2, 2, 6, 2, 2]
  num_heads: [4, 8, 16, 16, 16]
  window_size: 16

model:
  in_channels: 1
  vec_in_dim: null
  context_in_dim: 512      # = encoder embed_dim * 2^(len(depths)-1); asserted
  mlp_ratio: 4.0
  num_heads: 6
  depth: 4
  depth_single_blocks: 4
  axes_dim: [22, 22, 20]   # (x, y, z); sum = 64 = hidden/heads
  theta: null              # null -> derived from HealpixRope(nside_bottleneck).theta
  qkv_bias: true
  param_dtype: "float32"
  id_embedding_strategy: ["absolute", "rope"]   # model-side: apply RoPE to cond ids

optimizer:
  warmup_steps: 500
  decay_transition: 0.85
  max_lr: 1.0e-3
  min_lr: 0.0

training:
  batch_size: 128
  val_batch_size: 256
  nsteps: 20000
  ema_decay: 0.999
  multistep: 1
  early_stopping: true
  val_every: 100
  experiment_id: "spherical_grf_fm_healpix"
  train_model: true
  restore_model: false

evaluation:
  observations: [1, 2, 3]
  num_posterior_samples: 10000
  sample_step_size: 0.01
  tarp_pairs: 200
  tarp_posterior_samples: 1000
```

- [ ] **Step 3: Write `config/config_pos1d.yaml`** — identical except the four A/B lines and the experiment id:

```yaml
task_name: spherical_grf

strategy:
  method: "flow"
  model: "flux"

# 1D sinusoidal cond ids over NEST order — baseline arm of the A/B against
# config_healpix.yaml (spherical HEALPix RoPE).
ids:
  cond_kind: "pos1d"

data:
  nside: 64
  seed: 0

encoder:
  embed_dim: 32
  depths: [2, 2, 6, 2, 2]
  num_heads: [4, 8, 16, 16, 16]
  window_size: 16

model:
  in_channels: 1
  vec_in_dim: null
  context_in_dim: 512
  mlp_ratio: 4.0
  num_heads: 6
  depth: 4
  depth_single_blocks: 4
  axes_dim: [64]           # hidden_size = sum(axes_dim) * heads = 384
  theta: null              # null -> Flux1Params default (10 * (dim_obs + dim_cond) = 510)
  qkv_bias: true
  param_dtype: "float32"
  id_embedding_strategy: ["absolute", "pos1d"]

optimizer:
  warmup_steps: 500
  decay_transition: 0.85
  max_lr: 1.0e-3
  min_lr: 0.0

training:
  batch_size: 128
  val_batch_size: 256
  nsteps: 20000
  ema_decay: 0.999
  multistep: 1
  early_stopping: true
  val_every: 100
  experiment_id: "spherical_grf_fm_pos1d"
  train_model: true
  restore_model: false

evaluation:
  observations: [1, 2, 3]
  num_posterior_samples: 10000
  sample_step_size: 0.01
  tarp_pairs: 200
  tarp_posterior_samples: 1000
```

- [ ] **Step 4: Rewrite the script's header + imports.** In `train-spherical-grf.py`, update the module docstring's run commands to:

```
    python train-spherical-grf.py --config config/config_healpix.yaml

Or submit to a GPU node: ``condor_submit sub/spherical_grf.sub``
(edit `config = ...` in the sub file to pick the A/B arm).

Debug modes (both CPU-safe, both accept --config):

    SMOKE=1 JAX_PLATFORMS=cpu python train-spherical-grf.py
        forward-shape check, no data, no training
    QUICK=1 JAX_PLATFORMS=cpu python train-spherical-grf.py
        tiny end-to-end run (few steps, few samples)
```

and replace the import lines

```python
from heal_swin_nnx import HealSwinEncoder, HealSwinParams

from gensbi.core import FlowMatchingMethod
from gensbi.models import Flux1, Flux1Params
from gensbi.recipes import ConditionalPipeline
from gensbi.recipes.utils import init_ids_1d, init_ids_healpix, healpix_rope_theta
```

with

```python
import argparse

import yaml

from gensbi.core import FlowMatchingMethod
from gensbi.models import Flux1, Flux1Params, HealSwinEncoder, HealSwinParams
from gensbi.recipes import ConditionalPipeline, HealpixRope
from gensbi.recipes.flux1 import parse_flux1_params
from gensbi.recipes.utils import init_ids_1d, parse_training_config
```

- [ ] **Step 5: Replace the whole `# --- config (tune here) ---` block** (from `QUICK = ...` through the `# ---...---` closing line, i.e. everything between the flags parsing and `def make_encoder_params`) with:

```python
QUICK = os.environ.get("QUICK") == "1"

# --- configuration (edit the YAML files, not this block) ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_parser = argparse.ArgumentParser(description="Spherical GRF flow-matching NPE")
_parser.add_argument(
    "--config", default=os.path.join("config", "config_healpix.yaml"),
    help="YAML run config: config/config_healpix.yaml or config/config_pos1d.yaml",
)
_args, _ = _parser.parse_known_args()
CONFIG_PATH = (_args.config if os.path.isabs(_args.config)
               else os.path.join(BASE_DIR, _args.config))

with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

# task / data
NSIDE = CONFIG["data"]["nside"]
NPIX = 12 * NSIDE ** 2
SEED = CONFIG["data"]["seed"]
DIM_THETA = 3
THETA_LABELS = (r"$\log A$", r"$n$", r"$\alpha$")

# HealSwin encoder geometry (bottleneck derived, never written twice)
_ENC = CONFIG["encoder"]
EMBED_DIM = _ENC["embed_dim"]
DEPTHS = tuple(_ENC["depths"])
ENC_NUM_HEADS = tuple(_ENC["num_heads"])
WINDOW_SIZE = _ENC["window_size"]
NSIDE_BOTTLENECK = NSIDE // (2 * 2 ** (len(DEPTHS) - 1))  # patch /2, then mergings
COND_TOKENS = 12 * NSIDE_BOTTLENECK ** 2
COND_FEATURES = EMBED_DIM * 2 ** (len(DEPTHS) - 1)

# Flux1 model hyperparameters (model-side id_embedding_strategy lives here)
FLUX_PARAMS_DICT = parse_flux1_params(CONFIG_PATH)
assert FLUX_PARAMS_DICT["context_in_dim"] == COND_FEATURES, (
    f"model.context_in_dim={FLUX_PARAMS_DICT['context_in_dim']} must equal "
    f"encoder features {COND_FEATURES}")

# Pipeline-side cond-id builder: HealpixRope object or "pos1d" string.
COND_ID_KIND = CONFIG["ids"]["cond_kind"]
if COND_ID_KIND == "healpix-rope":
    COND_STRATEGY = HealpixRope(nside=NSIDE_BOTTLENECK)
    if FLUX_PARAMS_DICT["theta"] is None:
        FLUX_PARAMS_DICT["theta"] = COND_STRATEGY.theta
elif COND_ID_KIND == "pos1d":
    COND_STRATEGY = "pos1d"
else:
    raise ValueError(f"unknown ids.cond_kind: {COND_ID_KIND!r}")

# training / data loading
_TRAIN = CONFIG["training"]
BATCH_SIZE = 8 if QUICK else _TRAIN["batch_size"]
VAL_BATCH_SIZE = 8 if QUICK else _TRAIN["val_batch_size"]
NSTEPS = 5 if QUICK else _TRAIN["nsteps"]
NUM_WORKERS = 0 if QUICK else min(8, max(1, (os.cpu_count() or 2) - 2))
TRAIN_MODEL = _TRAIN["train_model"]
RESTORE_MODEL = _TRAIN["restore_model"]

# evaluation
_EVAL = CONFIG["evaluation"]
EVAL_OBSERVATIONS = (1,) if QUICK else tuple(_EVAL["observations"])
NUM_POSTERIOR_SAMPLES = 64 if QUICK else _EVAL["num_posterior_samples"]
SAMPLE_STEP_SIZE = 0.25 if QUICK else _EVAL["sample_step_size"]
TARP_PAIRS = 2 if QUICK else _EVAL["tarp_pairs"]
TARP_POSTERIOR_SAMPLES = 8 if QUICK else _EVAL["tarp_posterior_samples"]

EXPERIMENT_ID = _TRAIN["experiment_id"] + ("_quick" if QUICK else "")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints", EXPERIMENT_ID)
IMGS_DIR = os.path.join(BASE_DIR, "imgs")
RESULTS_FILE = os.path.join(BASE_DIR, f"{EXPERIMENT_ID}_results.txt")
# ------------------------------------------------------------------------
```

- [ ] **Step 6: Replace `make_flux_params` and `make_training_config` and `make_pipeline`** with:

```python
def make_flux_params(rngs: nnx.Rngs) -> Flux1Params:
    return Flux1Params(
        rngs=rngs, dim_obs=DIM_THETA, dim_cond=COND_TOKENS, **FLUX_PARAMS_DICT)
```

```python
def make_training_config():
    cfg = ConditionalPipeline.get_default_training_config()
    cfg.update(parse_training_config(CONFIG_PATH))
    cfg["checkpoint_dir"] = CHECKPOINT_DIR
    if QUICK:
        cfg["nsteps"] = NSTEPS
        cfg["warmup_steps"] = 2
        cfg["val_every"] = 2
        cfg["decay_transition"] = 0
    return cfg
```

```python
def make_pipeline(model, train_loader, val_loader):
    # Cond ids are first-class now: pass "pos1d" or a HealpixRope IdStrategy
    # straight to the pipeline. The model-side vocabulary (Flux1Params
    # id_embedding_strategy, e.g. ("absolute", "rope")) is configured
    # independently in the YAML `model:` section.
    return ConditionalPipeline(
        model, train_loader, val_loader,
        dim_obs=DIM_THETA, dim_cond=COND_TOKENS,
        method=FlowMatchingMethod(),
        ch_obs=1, ch_cond=COND_FEATURES,
        id_embedding_strategy=("absolute", COND_STRATEGY),
        training_config=make_training_config(),
    )
```

(`make_encoder_params`, `SphericalGRFModel`, `make_datasets`, `prep_x`, `evaluate`, `tarp_diagnostic`, and `main` port **verbatim** — every module-level name they reference keeps its old meaning. Only `make_flux_params`'s old inline `theta=`/`id_embedding_strategy=` logic is gone, absorbed by the config block. In `main()`'s first `log(...)` call, change `ids={ID_EMBEDDING}` to `ids={FLUX_PARAMS_DICT['id_embedding_strategy']}`, and drop the now-stale `RESTORE_MODEL` comment about "gensbi 0.4.0" if present — the `pipeline._wrap_model()` call itself stays.)

- [ ] **Step 7: Rewrite the SMOKE block** at the bottom of the file:

```python
if __name__ == "__main__" and os.environ.get("SMOKE") == "1":
    # Forward-shape smoke check: no data, no training; runs on CPU.
    model = SphericalGRFModel(rngs=nnx.Rngs(0))
    model.eval()
    B = 2
    obs_ids, _ = init_ids_1d(DIM_THETA, 0)  # (1, 3, 2) — broadcast over batch
    if COND_ID_KIND == "healpix-rope":
        cond_ids, _ = COND_STRATEGY.build(COND_TOKENS)  # (1, 48, 3) float32
    else:
        cond_ids, _ = init_ids_1d(COND_TOKENS, 1)  # (1, 48, 2)
    v = model(
        t=jnp.full((B,), 0.5),
        obs=jnp.zeros((B, DIM_THETA, 1)),
        obs_ids=obs_ids,
        cond=jnp.zeros((B, NPIX, 1)),
        cond_ids=cond_ids,
    )
    print("vector field shape:", v.shape)
    assert v.shape == (B, DIM_THETA, 1)
    print("forward smoke check OK")
```

- [ ] **Step 8: SMOKE both configs** (mamba `gensbi` env — it has local gensbi with Phases 1+3 plus heal-swin-nnx):

```bash
cd examples/sbi-benchmarks/spherical_grf
SMOKE=1 JAX_PLATFORMS=cpu python train-spherical-grf.py --config config/config_healpix.yaml
SMOKE=1 JAX_PLATFORMS=cpu python train-spherical-grf.py --config config/config_pos1d.yaml
```

Expected: both print `vector field shape: (2, 3, 1)` then `forward smoke check OK`.

- [ ] **Step 9: Commit**

```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI-examples
git add examples/sbi-benchmarks/spherical_grf/train-spherical-grf.py \
        examples/sbi-benchmarks/spherical_grf/config/
git commit -m "feat: spherical GRF flow-matching example (HealSwin + healpix-rope ids)

Ported from HEAL-SWIN-nnx 0748eda; YAML-config layout per the lensing
convention; the healpix-vs-pos1d A/B is now two config files. Uses the
first-class HealpixRope pipeline strategy — the fake-strategy +
cond_ids-override workaround is gone.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 12: HTCondor files, QUICK end-to-end smokes, merge

**Files:**
- Create: `examples/sbi-benchmarks/spherical_grf/sub/spherical_grf.sub`
- Create: `examples/sbi-benchmarks/spherical_grf/sub/run_spherical_grf.sh`
- Create: `examples/sbi-benchmarks/spherical_grf/sub/condor_logs/.gitignore`

**Interfaces:**
- Consumes: `train-spherical-grf.py --config config/<file>.yaml` (Task 11).

- [ ] **Step 1: Write `sub/spherical_grf.sub`** (adapted from the HEAL-SWIN-nnx original; the lustre GenSBI-examples checkout exists at `/lustre/ific.uv.es/ml/ific088/github/GenSBI-examples`):

```
# HTCondor submit file for the spherical GRF flow-matching NPE example.
#
#     condor_submit sub/spherical_grf.sub
#
# Pick the A/B arm by editing `config` below (config_healpix.yaml or
# config_pos1d.yaml). `initialdir` makes the condor_logs/ paths resolve
# relative to this file's own directory; the executable is given absolute, so
# this can be submitted from anywhere. Edit `repo_root` if your checkout
# lives elsewhere.
repo_root   = /lustre/ific.uv.es/ml/ific088/github/GenSBI-examples
example_dir = $(repo_root)/examples/sbi-benchmarks/spherical_grf
config      = config_healpix.yaml

universe   = vanilla
initialdir = $(example_dir)/sub
executable = $(example_dir)/sub/run_spherical_grf.sh
arguments  = "$(example_dir) $(config)"
getenv     = True

request_cpus   = 8
request_memory = 32 GB
request_gpus   = 1

+UseNvidiaA100 = True
# cuDNN dropped Volta support; require Ampere or newer.
requirements = (TARGET.Gpus_Capability >= 8.0)

log    = condor_logs/spherical_grf_$Fn(config)_a100.log
output = condor_logs/spherical_grf_$Fn(config)_a100.out
error  = condor_logs/spherical_grf_$Fn(config)_a100.err

queue
```

- [ ] **Step 2: Write `sub/run_spherical_grf.sh`** and make it executable:

```bash
#!/bin/bash
# HTCondor executable wrapper for the spherical GRF flow-matching example.
#
#   $1 = example directory, $2 = config file name (inside $1/config/)
set -euo pipefail

cd "$1"

# The script picks the device itself: JAX_PLATFORMS=cuda for the main
# process, cpu for spawned grain workers. Unset any inherited value so a
# stray JAX_PLATFORMS=cpu from the submit environment can't force CPU-only
# training. Uses the python on PATH (getenv=True passes the submitter's
# environment) — activate an env with gensbi + heal-swin-nnx + sbibm-jax
# before condor_submit.
unset JAX_PLATFORMS
exec python train-spherical-grf.py --config "config/$2"
```

Run: `chmod +x examples/sbi-benchmarks/spherical_grf/sub/run_spherical_grf.sh`

- [ ] **Step 3: Write `sub/condor_logs/.gitignore`:**

```
*
!.gitignore
```

- [ ] **Step 4: QUICK end-to-end smokes, both configs** (CPU-safe; streams from the HF cache — the ~24 GB `spherical_grf` dataset was already downloaded by the earlier HEAL-SWIN-nnx runs, so this reuses it; if `HF_HOME` differs from that machine's, expect a download):

```bash
cd examples/sbi-benchmarks/spherical_grf
QUICK=1 JAX_PLATFORMS=cpu python train-spherical-grf.py --config config/config_healpix.yaml
QUICK=1 JAX_PLATFORMS=cpu python train-spherical-grf.py --config config/config_pos1d.yaml
```

Expected: each completes end-to-end (5 train steps, 1 eval observation, tiny TARP) and writes `spherical_grf_fm_healpix_quick_results.txt` / `spherical_grf_fm_pos1d_quick_results.txt` with finite losses (reference values from the pre-port A/B: train ≈ 4.37, val ≈ 5.2). The two arms' losses must differ from each other (proves the id path actually switches).

- [ ] **Step 5: Commit the sub files only** (QUICK artifacts — `*_quick_results.txt`, `imgs/`, `checkpoints/` — stay untracked):

```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI-examples
git add examples/sbi-benchmarks/spherical_grf/sub/
git commit -m "feat: HTCondor submit files for the spherical GRF example (A/B via config arg)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 6: Merge**

```bash
git checkout main && git merge --ff-only spherical-grf-example && git branch -d spherical-grf-example
```

Then report to the user: the GPU A/B (promotion gate) is ready — `condor_submit sub/spherical_grf.sub` twice, once per `config` value, from the lustre checkout after `git pull`.

---

## Post-plan loose ends (user actions, recorded for visibility)

1. Push all three repos' main branches.
2. Cut a gensbi release including Phases 1+3; then bump GenSBI-examples' `gensbi` floor to it.
3. Run the GPU A/B (TARP + marginals) — the promotion gate for healpix RoPE.
