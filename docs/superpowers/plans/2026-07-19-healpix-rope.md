# HEALPix RoPE for Flux1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Spherical rotary position embedding for HEALPix-grid tokens feeding Flux1's conditioning stream: standard N-d RoPE applied to 3D Cartesian pixel-center coordinates, per the approved spec `docs/superpowers/specs/2026-07-19-healpix-rope-design.md`.

**Architecture:** A new ids-builder `init_ids_healpix` + theta helper `healpix_rope_theta` in `src/gensbi/recipes/utils.py` (siblings of `init_ids_1d`). Zero changes to Flux1: `rope()` already accepts float positions, `EmbedND` handles N axes, and the mixed `("absolute", "rope")` strategy already gives obs tokens dummy zero (= origin) rope ids (`model.py:427`). First consumer: the HEAL-SWIN-nnx GRF example switches its cond ids from `pos1d` to spherical.

**Tech Stack:** JAX/flax-nnx, healpy (host-side precompute only, lazy import), pytest.

## Global Constraints

- Run GenSBI tests with the mamba env: `mamba run -n gensbi python -m pytest ...` (NOT `.venv` — per project memory it hides real failures).
- Test files set `os.environ["JAX_PLATFORMS"] = "cpu"` before importing jax (convention: `tests/models/flux1/test_model_flux.py:4`).
- Do NOT modify `src/gensbi/models/flux1/` (model.py, math.py, layers.py) — the design requires zero attention/model changes.
- `healpy>=1.19.0` becomes a regular dependency in `pyproject.toml`, but is imported lazily *inside* `init_ids_healpix` (healpy pulls matplotlib; keep `import gensbi` light).
- ids are float32, shape `(1, N, 3)`. Assumption (spec): token grid = full HEALPix grid at a power-of-2 `nside` (power-of-4 pixels-per-token upstream); non-conforming patchings are out of scope.
- GenSBI commits go on branch `healpix-rope` (already checked out). HEAL-SWIN-nnx commits (Task 5) go on a new branch `healpix-rope` in that repo.
- End commit messages with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: `healpix_rope_theta` + healpy dependency

**Files:**
- Modify: `pyproject.toml` (dependencies list, after `"scipy>=1.18.0",` at line 25)
- Modify: `src/gensbi/recipes/utils.py` (add function after `init_ids_1d`, which ends at line 53)
- Test: `tests/recipes/test_healpix_ids.py` (create)

**Interfaces:**
- Produces: `healpix_rope_theta(nside: int) -> int` in `gensbi.recipes.utils` — returns `10 * 12 * nside**2`. Tasks 4 and 5 import it.

- [ ] **Step 1: Write the failing test**

Create `tests/recipes/test_healpix_ids.py`:

```python
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import jax.numpy as jnp
import pytest

from gensbi.recipes.utils import healpix_rope_theta


def test_healpix_rope_theta_follows_project_convention():
    # Project convention: theta = 10 * token count (Flux1Params defaults to
    # 10 * (dim_obs + dim_cond) at model.py:184). Full sky has 12*nside^2 tokens.
    assert healpix_rope_theta(2) == 480
    assert healpix_rope_theta(4) == 1920
    assert healpix_rope_theta(4) > healpix_rope_theta(2)
    assert isinstance(healpix_rope_theta(2), int)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /lustre/ific.uv.es/ml/ific088/github/GenSBI && mamba run -n gensbi python -m pytest tests/recipes/test_healpix_ids.py -v`
Expected: FAIL with `ImportError: cannot import name 'healpix_rope_theta'`

- [ ] **Step 3: Implement helper and add dependency**

In `src/gensbi/recipes/utils.py`, insert after the `init_ids_1d` function (after line 53):

```python
def healpix_rope_theta(nside: int) -> int:
    """Suggested RoPE ``theta`` for a full-sky HEALPix token grid at ``nside``.

    Follows the project convention ``theta = 10 * token count`` (the same rule
    :class:`~gensbi.models.flux1.model.Flux1Params` applies by default via
    ``10 * (dim_obs + dim_cond)``): a full-sky grid has ``12 * nside**2``
    tokens. Exposed so spherical models can derive theta from the intuitive
    knob (``nside``, always known after the encoder) instead of setting it by
    hand.
    """
    return 10 * 12 * nside**2
```

In `pyproject.toml`, after `    "scipy>=1.18.0",` (line 25) add:

```toml
    "healpy>=1.19.0",
```

Install into the test env:

Run: `mamba run -n gensbi pip install "healpy>=1.19.0"`
Expected: `Successfully installed healpy-...` (or already satisfied)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /lustre/ific.uv.es/ml/ific088/github/GenSBI && mamba run -n gensbi python -m pytest tests/recipes/test_healpix_ids.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
cd /lustre/ific.uv.es/ml/ific088/github/GenSBI
git add pyproject.toml src/gensbi/recipes/utils.py tests/recipes/test_healpix_ids.py
git commit -m "feat: add healpix_rope_theta helper and healpy dependency

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: `init_ids_healpix` — core builder

**Files:**
- Modify: `src/gensbi/recipes/utils.py` (add function after `healpix_rope_theta` from Task 1)
- Test: `tests/recipes/test_healpix_ids.py` (extend)

**Interfaces:**
- Consumes: nothing from other tasks (healpy from Task 1's dependency).
- Produces: `init_ids_healpix(nside: int, base_pixels: "Sequence[int] | None" = None) -> tuple[jax.Array, int]` in `gensbi.recipes.utils`. Returns `(ids, num_tokens)` mirroring `init_ids_1d`'s `(ids, dim)` convention; `ids` is float32 `(1, num_tokens, 3)` in NEST order, scaled to pixel units. Tasks 3–5 use it.

- [ ] **Step 1: Write the failing tests**

Append to `tests/recipes/test_healpix_ids.py`:

```python
from gensbi.recipes.utils import init_ids_healpix

# 1 / (pixel angular size) = nside * sqrt(3/pi); adjacent tokens ~1 apart.
def _pixel_unit_radius(nside):
    return nside * np.sqrt(3.0 / np.pi)


def test_init_ids_healpix_shape_dtype_scale():
    ids, n = init_ids_healpix(2)
    assert n == 48  # 12 * nside^2
    assert ids.shape == (1, 48, 3)
    assert ids.dtype == jnp.float32
    # every token direction lies on the sphere of radius r(nside)
    norms = np.linalg.norm(np.asarray(ids[0]), axis=-1)
    np.testing.assert_allclose(norms, _pixel_unit_radius(2), rtol=1e-5)


def test_init_ids_healpix_nest_order_roundtrip():
    # Token i must be the center of NEST pixel i: healpy round-trip catches
    # any ordering or indexing bug in the builder.
    import healpy as hp

    nside = 4
    ids, n = init_ids_healpix(nside)
    vecs = np.asarray(ids[0]) / _pixel_unit_radius(nside)
    pix = hp.vec2pix(nside, vecs[:, 0], vecs[:, 1], vecs[:, 2], nest=True)
    np.testing.assert_array_equal(pix, np.arange(n))


def test_init_ids_healpix_validates_nside():
    with pytest.raises(ValueError, match="power of 2"):
        init_ids_healpix(3)
    with pytest.raises(ValueError, match="power of 2"):
        init_ids_healpix(0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /lustre/ific.uv.es/ml/ific088/github/GenSBI && mamba run -n gensbi python -m pytest tests/recipes/test_healpix_ids.py -v`
Expected: 1 PASS (theta test), 3 FAIL with `ImportError: cannot import name 'init_ids_healpix'`

- [ ] **Step 3: Implement the builder**

In `src/gensbi/recipes/utils.py`, insert after `healpix_rope_theta`:

```python
def init_ids_healpix(nside: int, base_pixels=None):
    """Build spherical RoPE ids for tokens on a HEALPix grid, returning
    ``(ids, num_tokens)``.

    Method: standard N-dimensional RoPE (RoFormer, arXiv:2104.09864 — the
    mechanism implemented by Flux1's ``EmbedND``) applied uniformly, on all
    three axes and all frequency bands, to the 3D Cartesian coordinates of
    HEALPix pixel centers on the unit sphere. Each token maps to its
    pixel-center unit vector (``healpy.pix2vec``, NEST ordering), scaled to
    pixel units so adjacent tokens differ by ~1 in coordinate (radius
    ``nside * sqrt(3/pi)`` = 1/pixel angular size), which keeps ``theta``'s
    semantics identical to 2D-image usage (see :func:`healpix_rope_theta`).

    Attention scores then depend on positions only through the chord vector
    ``n_q - n_k``, whose norm ``2 sin(gamma/2)`` is strictly monotone in
    great-circle distance ``gamma`` — geodesic geometry with no projection
    step, hence no face-seam or polar artifacts, and any ``base_pixels``
    subset works by construction. Caveat: ``d(chord)/d(gamma) -> 0`` at
    antipodes, so resolution among near-antipodal separations is mildly
    compressed (benign for near/far attention). This is NOT an adaptation of
    SpheRoPE (arXiv:2606.32033 — closest prior work; ERP grid, pretrained
    constraints); see also StereoRoPE (arXiv:2606.31248, documents the
    failure of index-based RoPE on HEALPix) and Unlu (arXiv:2310.04454, an
    SO(3) feature-rotation alternative not adopted). Full rationale:
    ``docs/superpowers/specs/2026-07-19-healpix-rope-design.md``.

    Use with Flux1 via ``id_embedding_strategy=("absolute", "rope")`` and a
    3-entry ``axes_dim`` (each even, summing to the per-head dim, e.g.
    ``(22, 22, 20)`` for 64). Obs/theta-stream tokens automatically get
    origin (0, 0, 0) rope ids — the identity rotation, i.e. an exactly
    isotropic positional readout of the conditioning tokens.

    Parameters
    ----------
    nside : int
        HEALPix resolution of the *token* grid (power of 2). With HEAL-SWIN
        style encoders this is the bottleneck nside; tokens must correspond
        to single HEALPix pixels (power-of-4 pixels-per-token upstream).
    base_pixels : sequence of int, optional
        Base pixels (0..11) covered by the token grid, for partial-sky
        models. ``None`` (default) means full sky. Tokens are ordered by
        base pixel as given, NEST within each.

    Returns
    -------
    ids : jax.Array
        ``(1, num_tokens, 3)`` float32 scaled pixel-center coordinates.
    num_tokens : int
        ``len(base_pixels) * nside**2``.
    """
    import healpy as hp  # lazy: healpy pulls matplotlib, keep import light

    if nside < 1 or (nside & (nside - 1)) != 0:
        raise ValueError(f"nside must be a power of 2, got {nside}")
    if base_pixels is None:
        base_pixels = range(12)
    base_pixels = list(base_pixels)
    if any(b < 0 or b > 11 for b in base_pixels) or len(set(base_pixels)) != len(
        base_pixels
    ):
        raise ValueError(
            f"base_pixels must be unique integers in [0, 11], got {base_pixels}"
        )

    face_len = nside**2
    pix = np.concatenate(
        [b * face_len + np.arange(face_len) for b in base_pixels]
    )
    x, y, z = hp.pix2vec(nside, pix, nest=True)  # float64 host-side
    radius = nside * np.sqrt(3.0 / np.pi)  # 1 / pixel angular size
    ids = radius * np.stack([x, y, z], axis=-1)[None, ...]
    return jnp.asarray(ids, dtype=jnp.float32), ids.shape[1]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /lustre/ific.uv.es/ml/ific088/github/GenSBI && mamba run -n gensbi python -m pytest tests/recipes/test_healpix_ids.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
cd /lustre/ific.uv.es/ml/ific088/github/GenSBI
git add src/gensbi/recipes/utils.py tests/recipes/test_healpix_ids.py
git commit -m "feat: add init_ids_healpix spherical RoPE ids builder

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: subset semantics + seam-freedom property tests

**Files:**
- Test: `tests/recipes/test_healpix_ids.py` (extend; implementation changes only if a test exposes a bug)

**Interfaces:**
- Consumes: `init_ids_healpix(nside, base_pixels=None) -> (ids, num_tokens)` from Task 2.
- Produces: nothing new — locks in behavior later tasks rely on.

- [ ] **Step 1: Write the tests**

Append to `tests/recipes/test_healpix_ids.py`:

```python
def test_init_ids_healpix_base_pixel_subset_matches_full_sky():
    # Subset ids must be exactly the corresponding rows of the full-sky ids:
    # the encoding depends only on token directions, never on token count.
    nside = 2
    full, _ = init_ids_healpix(nside)
    subset, n_sub = init_ids_healpix(nside, base_pixels=[3, 7])
    assert n_sub == 2 * nside**2
    face_len = nside**2
    expected = jnp.concatenate(
        [
            full[:, 3 * face_len : 4 * face_len],
            full[:, 7 * face_len : 8 * face_len],
        ],
        axis=1,
    )
    np.testing.assert_array_equal(np.asarray(subset), np.asarray(expected))


def test_init_ids_healpix_rejects_bad_base_pixels():
    with pytest.raises(ValueError, match="base_pixels"):
        init_ids_healpix(2, base_pixels=[0, 12])
    with pytest.raises(ValueError, match="base_pixels"):
        init_ids_healpix(2, base_pixels=[1, 1])


def test_no_face_seam_discontinuity():
    # The failure documented for index-based RoPE on HEALPix (StereoRoPE,
    # arXiv:2606.31248) is a discontinuity across base-face boundaries. In
    # chord coordinates, grid-neighbor distances must be uniform across the
    # whole sphere — face boundaries and poles included. An index-seam bug
    # would make some neighbor pairs ~nside times farther than others.
    import healpy as hp

    nside = 4
    ids, n = init_ids_healpix(nside)
    coords = np.asarray(ids[0])
    neigh = hp.get_all_neighbours(nside, np.arange(n), nest=True)  # (8, n)
    dists = []
    for p in range(n):
        for q in neigh[:, p]:
            if q >= 0:
                dists.append(np.linalg.norm(coords[p] - coords[q]))
    dists = np.asarray(dists)
    # pixel units: neighbor spacing ~1 (sides) to ~sqrt(2) (diagonals), with
    # HEALPix shape distortion on top; no seam outliers anywhere.
    assert dists.max() / dists.min() < 4.0
    assert 0.5 < dists.mean() < 2.0
```

- [ ] **Step 2: Run tests**

Run: `cd /lustre/ific.uv.es/ml/ific088/github/GenSBI && mamba run -n gensbi python -m pytest tests/recipes/test_healpix_ids.py -v`
Expected: PASS (7 tests). If `test_no_face_seam_discontinuity` fails on the ratio bound, that indicates a real builder bug (e.g. RING/NEST mix-up) — debug the builder, do not loosen the bound past 4.0.

- [ ] **Step 3: Commit**

```bash
cd /lustre/ific.uv.es/ml/ific088/github/GenSBI
git add tests/recipes/test_healpix_ids.py
git commit -m "test: subset semantics and face-seam-freedom for healpix ids

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Flux1 end-to-end integration smoke

**Files:**
- Test: `tests/recipes/test_healpix_ids.py` (extend; NO source changes expected — if Flux1 rejects float 3-axis ids, stop and report, do not patch the model)

**Interfaces:**
- Consumes: `init_ids_healpix` (Task 2), `healpix_rope_theta` (Task 1), existing `Flux1`/`Flux1Params` and `init_ids_1d`.
- Produces: proof that the `("absolute", "rope")` + 3-axis float ids path works unmodified; Task 5 relies on it.

- [ ] **Step 1: Write the failing test**

Append to `tests/recipes/test_healpix_ids.py`:

```python
def test_embednd_consumes_healpix_ids():
    # 3-axis float ids through the existing EmbedND: correct freqs_cis shape,
    # all finite. (That scores depend only on the per-axis coordinate
    # differences is guaranteed by rope()'s construction; no test needed.)
    from gensbi.models.flux1.layers import EmbedND

    ids, n = init_ids_healpix(2)
    emb = EmbedND(dim=12, theta=healpix_rope_theta(2), axes_dim=[4, 4, 4])
    pe = emb(ids)
    # rope() -> (1, N, d/2, 2, 2) per axis, concat on axis -3, expand_dims(1)
    assert pe.shape == (1, 1, n, 6, 2, 2)
    assert bool(jnp.isfinite(pe).all())


def test_flux1_forward_with_healpix_rope():
    from flax import nnx

    from gensbi.models.flux1.model import Flux1, Flux1Params
    from gensbi.recipes.utils import init_ids_1d

    nside = 2
    cond_ids, n_cond = init_ids_healpix(nside)  # (1, 48, 3) float32
    dim_theta = 3
    params = Flux1Params(
        in_channels=1,
        vec_in_dim=None,
        context_in_dim=8,
        mlp_ratio=2.0,
        num_heads=4,
        depth=1,
        depth_single_blocks=1,
        qkv_bias=True,
        dim_obs=dim_theta,
        dim_cond=n_cond,
        axes_dim=[4, 4, 4],  # 3 axes for (x, y, z); sum = per-head dim 12
        theta=healpix_rope_theta(nside),
        id_embedding_strategy=("absolute", "rope"),
        rngs=nnx.Rngs(0),
        param_dtype=jnp.float32,
    )
    model = Flux1(params)
    batch = 2
    obs = jnp.zeros((batch, dim_theta, 1))
    cond = jnp.ones((batch, n_cond, 8))
    obs_ids, _ = init_ids_1d(dim_theta, 0)
    t = jnp.array([0.3, 0.7])
    out = model(t=t, obs=obs, obs_ids=obs_ids, cond=cond, cond_ids=cond_ids)
    assert out.shape == (batch, dim_theta, 1)
    assert bool(jnp.isfinite(out).all())
```

- [ ] **Step 2: Run test to verify current status**

Run: `cd /lustre/ific.uv.es/ml/ific088/github/GenSBI && mamba run -n gensbi python -m pytest tests/recipes/test_healpix_ids.py::test_flux1_forward_with_healpix_rope -v`
Expected: PASS on first run — the model path already exists (obs gets dummy origin rope ids at `model.py:427`; `rope()` accepts float positions). If it FAILS, the failure is a genuine integration finding: report it (with traceback) rather than modifying `src/gensbi/models/flux1/`.

- [ ] **Step 3: Run the full flux1 + recipes suites to check for regressions**

Run: `cd /lustre/ific.uv.es/ml/ific088/github/GenSBI && mamba run -n gensbi python -m pytest tests/recipes/test_healpix_ids.py tests/models/flux1/ -v`
Expected: all PASS

- [ ] **Step 4: Commit**

```bash
cd /lustre/ific.uv.es/ml/ific088/github/GenSBI
git add tests/recipes/test_healpix_ids.py
git commit -m "test: Flux1 e2e smoke with spherical healpix RoPE cond ids

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: switch the HEAL-SWIN-nnx GRF example to spherical RoPE

**Files (separate repo: `/lhome/ific/a/aamerio/data/github/HEAL-SWIN-nnx`):**
- Modify: `examples/spherical_grf_flowmatch.py` (config block lines 88–93; ids at lines 334–335; header comment line 5)

**Interfaces:**
- Consumes: `init_ids_healpix`, `healpix_rope_theta` from `gensbi.recipes.utils` (Tasks 1–2); the verified Flux1 path (Task 4).
- Produces: the runnable first consumer; `COND_ID_KIND` switch for the spherical-vs-pos1d A/B.

- [ ] **Step 1: Create a branch and install the updated gensbi into the example venv**

The venv has a non-editable `gensbi 0.4.0`; replace with an editable install of the working tree so example runs see the new builder:

```bash
cd /lhome/ific/a/aamerio/data/github/HEAL-SWIN-nnx
git checkout -b healpix-rope
.venv/bin/pip install --no-deps -e /lustre/ific.uv.es/ml/ific088/github/GenSBI
.venv/bin/python -c "from gensbi.recipes.utils import init_ids_healpix; print(init_ids_healpix(2)[0].shape)"
```

Expected final line: `(1, 48, 3)`

- [ ] **Step 2: Edit the config block**

In `examples/spherical_grf_flowmatch.py`, replace lines 88–93:

```python
# Flux1 posterior model
FLUX_DEPTH = 4                       # double-stream blocks
FLUX_DEPTH_SINGLE = 4                # single-stream blocks
FLUX_NUM_HEADS = 6
FLUX_AXES_DIM = (64,)                # hidden_size = sum(axes_dim) * heads = 384
ID_EMBEDDING = ("absolute", "pos1d") # learned theta-token ids, sinusoidal cond ids
```

with:

```python
# Flux1 posterior model
FLUX_DEPTH = 4                       # double-stream blocks
FLUX_DEPTH_SINGLE = 4                # single-stream blocks
FLUX_NUM_HEADS = 6
# Cond-stream positional ids: "healpix" = spherical RoPE on pixel-center
# 3D coordinates (see gensbi init_ids_healpix); "pos1d" = 1D sinusoidal
# ids over NEST order (baseline for A/B comparison).
COND_ID_KIND = "healpix"
NSIDE_BOTTLENECK = NSIDE // (2 * 2 ** (len(DEPTHS) - 1))  # patch /2, 4 mergings
if COND_ID_KIND == "healpix":
    ID_EMBEDDING = ("absolute", "rope")   # learned theta ids, spherical cond RoPE
    FLUX_AXES_DIM = (22, 22, 20)          # (x, y, z); sum = 64 = hidden/heads
else:
    ID_EMBEDDING = ("absolute", "pos1d")  # learned theta ids, sinusoidal cond ids
    FLUX_AXES_DIM = (64,)                 # hidden_size = sum(axes_dim) * heads = 384
assert COND_TOKENS == 12 * NSIDE_BOTTLENECK**2
```

- [ ] **Step 3: Pass theta and update ids construction**

In `make_flux_params` (currently lines 131–147), add one argument to `Flux1Params(...)` after `axes_dim=list(FLUX_AXES_DIM),`:

```python
        theta=(healpix_rope_theta(NSIDE_BOTTLENECK)
               if COND_ID_KIND == "healpix" else None),
```

Replace the ids lines (currently 334–335):

```python
    obs_ids, _ = init_ids_1d(DIM_THETA, 0)    # (1, 3, 2) — broadcast over batch
    cond_ids, _ = init_ids_1d(COND_TOKENS, 1)  # (1, 48, 2)
```

with:

```python
    obs_ids, _ = init_ids_1d(DIM_THETA, 0)  # (1, 3, 2) — broadcast over batch
    if COND_ID_KIND == "healpix":
        cond_ids, _ = init_ids_healpix(NSIDE_BOTTLENECK)  # (1, 48, 3) float32
    else:
        cond_ids, _ = init_ids_1d(COND_TOKENS, 1)  # (1, 48, 2)
```

Update the import (currently line 60):

```python
from gensbi.recipes.utils import init_ids_1d, init_ids_healpix, healpix_rope_theta
```

Update the header description (line 5) from `which condition a gensbi Flux1` context to mention spherical RoPE, e.g. change the line containing `bottleneck tokens (nside 2, 512 features), which condition a gensbi Flux1` to:

```python
bottleneck tokens (nside 2, 512 features), which condition a gensbi Flux1
posterior model via spherical HEALPix RoPE ids (see COND_ID_KIND).
```

(Keep surrounding docstring lines intact; only extend that sentence.)

Also update the run log line (currently line 299) `f"heads={FLUX_NUM_HEADS} ids={ID_EMBEDDING}")` to include the kind:

```python
        f"heads={FLUX_NUM_HEADS} ids={ID_EMBEDDING} cond_ids={COND_ID_KIND}")
```

- [ ] **Step 4: QUICK smoke run, both id kinds**

```bash
cd /lhome/ific/a/aamerio/data/github/HEAL-SWIN-nnx
QUICK=1 JAX_PLATFORMS=cpu .venv/bin/python examples/spherical_grf_flowmatch.py
```

Expected: runs to completion (5 steps, tiny batches), finite losses printed, no shape errors. Then temporarily set `COND_ID_KIND = "pos1d"`, rerun the same command, expect the baseline still works, and set it back to `"healpix"`.

- [ ] **Step 5: Commit (HEAL-SWIN-nnx repo)**

```bash
cd /lhome/ific/a/aamerio/data/github/HEAL-SWIN-nnx
git add examples/spherical_grf_flowmatch.py
git commit -m "feat: spherical HEALPix RoPE cond ids in GRF flow-matching example

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

Note: do NOT commit `examples/checkpoints/` or `examples/imgs/*.png` (pre-existing untracked artifacts).

---

## Acceptance (post-plan, user-run)

Full (non-QUICK) GRF example training on GPU, A/B `COND_ID_KIND = "healpix"` vs `"pos1d"`, comparing TARP calibration and posterior marginals — per the spec, this is the user's run and gates promotion beyond the example.
