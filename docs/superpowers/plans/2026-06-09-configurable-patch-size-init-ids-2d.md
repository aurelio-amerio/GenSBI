# Configurable Patch Size for 2D ID Embeddings — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the 2D positional-ID grid honor a configurable patch size (default 2) so it matches `patchify_2d(..., size=p)`, instead of hardcoding patch size 2.

**Architecture:** The patch size is threaded through three layers — the leaf `init_ids_2d`, the strategy resolver `_resolve_embedding_ids`, and the public `ConditionalPipeline.__init__`. The pipeline accepts an `int | tuple[int, int]`, normalized via a pure helper `_normalize_patch_size` to a `(obs, cond)` tuple. The 1D code path (`init_ids_1d`) never receives `size`, so any non-patchified input ignores it by construction. Defaults reproduce the old hardcoded `// 2` byte-for-byte, so existing models are unaffected.

**Tech Stack:** Python, JAX/NumPy, einops, Flax NNX, pytest, grain.

**Repos:** Library changes (Tasks 1–3) are in the **GenSBI** repo (`/lustre/ific.uv.es/ml/ific088/github/GenSBI`). The end-to-end acceptance (Task 4) touches the **GenSBI-examples** repo GRF script.

---

## File Structure

- `src/gensbi/recipes/utils.py` — add `_normalize_patch_size` helper; add `size` param to `init_ids_2d` and `_resolve_embedding_ids`.
- `src/gensbi/recipes/conditional_pipeline.py` — add `size` param to `ConditionalPipeline.__init__`; normalize and forward per-input.
- `tests/recipes/test_pipeline_utils.py` — unit tests for the helper, the leaf, and the resolver.
- `tests/recipes/test_conditional_pipeline.py` — wiring test: construct a pipeline with a 2D obs strategy + `size=8` and assert the resulting `obs_ids` token count.
- (GenSBI-examples) `examples/sbi-benchmarks/gaussian_random_field/grf.py` — pass `size=8` to the pipeline; run the existing smoke call.

### Design decisions already settled (do not reopen)

- Parameter name is **`size`** (consistent with `patchify_2d(x, size=2)` / `depatchify_2d`).
- `size=1` is a real, meaningful value meaning "no patchification" (one token per pixel) — not a sentinel. Document it as the value to use for non-patched inputs.
- **No `dim % size` divisibility guard.** Keep the existing silent floor-division behavior. (Decided during brainstorming.)
- Defaults stay `2` (leaf/resolver) and `2 → (2, 2)` (pipeline) for backward compatibility.

---

### Task 1: `_normalize_patch_size` helper + `size` param on `init_ids_2d`

**Files:**
- Modify: `src/gensbi/recipes/utils.py:38-47` (`init_ids_2d`); add new helper near it.
- Test: `tests/recipes/test_pipeline_utils.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/recipes/test_pipeline_utils.py` (the `init_ids_2d` import already exists at the top of the file; add `_normalize_patch_size` to that import line):

```python
from gensbi.recipes.utils import _normalize_patch_size


def test_normalize_patch_size():
    assert _normalize_patch_size(8) == (8, 8)
    assert _normalize_patch_size(2) == (2, 2)
    assert _normalize_patch_size((8, 1)) == (8, 1)
    assert _normalize_patch_size([4, 2]) == (4, 2)
    with pytest.raises(ValueError):
        _normalize_patch_size((1, 2, 3))


def test_init_ids_2d_with_patch_size():
    dim = (16, 16)
    # patch size 8 -> 2x2 = 4 tokens
    ids, dim_ = init_ids_2d(dim, size=8)
    assert ids.shape == (1, (16 // 8) * (16 // 8), 3)
    assert dim_ == 4
    # default size=2 is unchanged (backward compat)
    ids2, dim2 = init_ids_2d(dim)
    assert dim2 == (16 // 2) * (16 // 2)
    assert ids2.shape == (1, 64, 3)
    # size=1 means no patchification: one token per pixel
    _, dim1 = init_ids_2d(dim, size=1)
    assert dim1 == 16 * 16
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /lustre/ific.uv.es/ml/ific088/github/GenSBI && python -m pytest tests/recipes/test_pipeline_utils.py::test_normalize_patch_size tests/recipes/test_pipeline_utils.py::test_init_ids_2d_with_patch_size -v`
Expected: FAIL — `ImportError` / `cannot import name '_normalize_patch_size'`, and `init_ids_2d() got an unexpected keyword argument 'size'`.

- [ ] **Step 3: Add the helper and the `size` parameter**

In `src/gensbi/recipes/utils.py`, add the helper just above `init_ids_2d`:

```python
def _normalize_patch_size(size):
    """Normalize a patch-size spec into an ``(obs, cond)`` tuple.

    Parameters
    ----------
    size : int or tuple of int
        A single int is broadcast to both inputs (``8 -> (8, 8)``). A
        length-2 tuple is taken as ``(obs_size, cond_size)`` so the two
        inputs can use different patch sizes. Use ``1`` for an input that
        is not patchified.

    Returns
    -------
    tuple of int
        ``(obs_size, cond_size)``.
    """
    if isinstance(size, int):
        return (size, size)
    size = tuple(size)
    if len(size) != 2:
        raise ValueError(
            f"size must be an int or a length-2 (obs, cond) tuple, got {size!r}"
        )
    return size
```

Then replace `init_ids_2d` (currently `src/gensbi/recipes/utils.py:38-47`) with:

```python
def init_ids_2d(dim: Tuple[int, int], semantic_id: int = 0, size: int = 2):
    """Build 2D positional IDs for a patchified image grid.

    The grid has one entry per patch, i.e. ``(dim[0] // size, dim[1] // size)``,
    matching ``patchify_2d(x, size=size)``. ``size`` is the patch edge length;
    use ``size=1`` for no patchification (one token per pixel).
    """
    img_ids = np.zeros((dim[0] // size, dim[1] // size, 3), dtype=np.int32)
    img_ids[..., 0] = semantic_id
    img_ids[..., 1] = img_ids[..., 1] + np.arange(dim[0] // size)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + np.arange(dim[1] // size)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=1)

    dim = (dim[0] // size) * (dim[1] // size)

    return jnp.array(img_ids, dtype=jnp.int32), dim
```

- [ ] **Step 4: Run tests to verify they pass (and the existing leaf test still passes)**

Run: `cd /lustre/ific.uv.es/ml/ific088/github/GenSBI && python -m pytest tests/recipes/test_pipeline_utils.py::test_normalize_patch_size tests/recipes/test_pipeline_utils.py::test_init_ids_2d_with_patch_size tests/recipes/test_pipeline_utils.py::test_init_ids_2d -v`
Expected: PASS (3 passed). The existing `test_init_ids_2d` confirms the default `size=2` path is unchanged.

- [ ] **Step 5: Commit**

```bash
cd /lustre/ific.uv.es/ml/ific088/github/GenSBI
git add src/gensbi/recipes/utils.py tests/recipes/test_pipeline_utils.py
git commit -m "feat(recipes): add configurable patch size to init_ids_2d"
```

---

### Task 2: Thread `size` through `_resolve_embedding_ids`

**Files:**
- Modify: `src/gensbi/recipes/utils.py:85-115` (`_resolve_embedding_ids`)
- Test: `tests/recipes/test_pipeline_utils.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/recipes/test_pipeline_utils.py` (`_resolve_embedding_ids` is already imported in that file):

```python
def test_resolve_embedding_ids_patch_size():
    # 2D strategy uses size -> 16/8 * 16/8 = 4 tokens
    ids, dim_ = _resolve_embedding_ids((16, 16), "rope2d", semantic_id=0, size=8)
    assert dim_ == 4
    assert ids.shape == (1, 4, 3)
    # 1D strategy ignores size by construction (routes to init_ids_1d)
    ids1, dim1 = _resolve_embedding_ids(5, "absolute", semantic_id=0, size=8)
    assert dim1 == 5
    assert ids1.shape == (1, 5, 2)  # semantic_id given -> last dim is 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /lustre/ific.uv.es/ml/ific088/github/GenSBI && python -m pytest tests/recipes/test_pipeline_utils.py::test_resolve_embedding_ids_patch_size -v`
Expected: FAIL — `_resolve_embedding_ids() got an unexpected keyword argument 'size'`.

- [ ] **Step 3: Add the `size` parameter and forward it to the 2D branch only**

Replace the signature and body of `_resolve_embedding_ids` in `src/gensbi/recipes/utils.py`. Change the signature line:

```python
def _resolve_embedding_ids(dim, strategy: str, semantic_id: int, size: int = 2):
```

and update the dispatch so only the 2D branch receives `size` (the 1D branch must NOT pass it — `init_ids_1d` takes no `size`):

```python
    if strategy in _EMBEDDINGS_1D:
        return init_ids_1d(dim, semantic_id=semantic_id)
    elif strategy in _EMBEDDINGS_2D:
        return init_ids_2d(dim, semantic_id=semantic_id, size=size)
    else:
        raise ValueError(f"Unknown id embedding strategy: {strategy}")
```

Also add a one-line note to the `size` parameter in the docstring's Parameters section, e.g.:

```
    size : int, optional
        Patch edge length for 2D strategies (default 2). Ignored for 1D
        strategies. Use 1 for no patchification.
```

- [ ] **Step 4: Run test to verify it passes (and the existing resolver test still passes)**

Run: `cd /lustre/ific.uv.es/ml/ific088/github/GenSBI && python -m pytest tests/recipes/test_pipeline_utils.py::test_resolve_embedding_ids_patch_size tests/recipes/test_pipeline_utils.py::test_resolve_embedding_ids_unknown_strategy -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
cd /lustre/ific.uv.es/ml/ific088/github/GenSBI
git add src/gensbi/recipes/utils.py tests/recipes/test_pipeline_utils.py
git commit -m "feat(recipes): forward patch size through _resolve_embedding_ids"
```

---

### Task 3: Thread `size` through `ConditionalPipeline.__init__`

**Files:**
- Modify: `src/gensbi/recipes/conditional_pipeline.py:42` (import), `:100-141` (`__init__`)
- Test: `tests/recipes/test_conditional_pipeline.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/recipes/test_conditional_pipeline.py`. It reuses the module-level `train_dataset` / `val_dataset` and `MockConditionalModel` already defined in that file (datasets are only stored at construction, not iterated, so their shape need not match the 2D `dim_obs`):

```python
def test_conditional_pipeline_patch_size():
    """size threads to obs_ids for a 2D obs strategy; cond (1D) is unaffected."""
    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        training_config = ConditionalPipeline.get_default_training_config()
        training_config["checkpoint_dir"] = model_dir

        p = ConditionalPipeline(
            model=MockConditionalModel(),
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            dim_obs=(16, 16),
            dim_cond=3,
            method=FlowMatchingMethod(),
            id_embedding_strategy=("rope2d", "absolute"),
            size=8,
            training_config=training_config,
        )
        # obs: 16//8 * 16//8 = 4 patch tokens
        assert p.obs_ids.shape[1] == 4
        assert p.dim_obs == 4
        # cond is 1D -> size ignored, 3 tokens
        assert p.cond_ids.shape[1] == 3
        assert p.dim_cond == 3


def test_conditional_pipeline_patch_size_tuple():
    """A tuple lets obs and cond differ; cond 1D still ignores its entry."""
    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        training_config = ConditionalPipeline.get_default_training_config()
        training_config["checkpoint_dir"] = model_dir

        p = ConditionalPipeline(
            model=MockConditionalModel(),
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            dim_obs=(16, 16),
            dim_cond=3,
            method=FlowMatchingMethod(),
            id_embedding_strategy=("rope2d", "absolute"),
            size=(8, 1),
            training_config=training_config,
        )
        assert p.obs_ids.shape[1] == 4
        assert p.cond_ids.shape[1] == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /lustre/ific.uv.es/ml/ific088/github/GenSBI && python -m pytest tests/recipes/test_conditional_pipeline.py::test_conditional_pipeline_patch_size tests/recipes/test_conditional_pipeline.py::test_conditional_pipeline_patch_size_tuple -v`
Expected: FAIL — `ConditionalPipeline.__init__() got an unexpected keyword argument 'size'`.

- [ ] **Step 3: Add `size`, normalize it, and forward per-input**

In `src/gensbi/recipes/conditional_pipeline.py`, update the import at line 42 to also bring in the helper:

```python
from gensbi.recipes.utils import (
    _resolve_embedding_ids,
    _normalize_patch_size,
    build_edm_path,
    build_sm_path,
)
```

Add `size` to the `__init__` signature (after `id_embedding_strategy`):

```python
        id_embedding_strategy=("absolute", "absolute"),
        size=2,
        params=None,
        training_config=None,
```

Then replace the two `_resolve_embedding_ids` calls (currently `src/gensbi/recipes/conditional_pipeline.py:137-141`) with:

```python
        obs_size, cond_size = _normalize_patch_size(size)
        self.obs_ids, self.dim_obs = _resolve_embedding_ids(
            dim_obs, id_embedding_strategy[0], semantic_id=0, size=obs_size
        )
        self.cond_ids, self.dim_cond = _resolve_embedding_ids(
            dim_cond, id_embedding_strategy[1], semantic_id=1, size=cond_size
        )
```

Add a short note to the class/`__init__` docstring describing `size`: `int | tuple[int, int]`, default `2`; broadcast int applies the same patch size to obs and cond; tuple is `(obs, cond)`; `1` means no patchification; ignored for 1D strategies.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /lustre/ific.uv.es/ml/ific088/github/GenSBI && python -m pytest tests/recipes/test_conditional_pipeline.py -v`
Expected: PASS — the two new tests pass and the existing pipeline tests are unaffected.

- [ ] **Step 5: Run the full recipes test suite (regression guard)**

Run: `cd /lustre/ific.uv.es/ml/ific088/github/GenSBI && python -m pytest tests/recipes/ -q`
Expected: PASS — no existing test regresses (default `size=2` preserves prior behavior).

- [ ] **Step 6: Commit**

```bash
cd /lustre/ific.uv.es/ml/ific088/github/GenSBI
git add src/gensbi/recipes/conditional_pipeline.py tests/recipes/test_conditional_pipeline.py
git commit -m "feat(recipes): accept int|tuple patch size in ConditionalPipeline"
```

---

### Task 4: End-to-end acceptance on the GRF example (cross-repo)

This is the real definition of done: the GRF pipeline must build `obs_ids` whose token count matches `patchify_2d(obs, size=8)`, and the smoke forward pass must run clean.

**Files:**
- Modify: `examples/sbi-benchmarks/gaussian_random_field/grf.py` (in the **GenSBI-examples** repo) — the `ConditionalPipeline(...)` construction around lines 210–225.

- [ ] **Step 1: Pass `size=8` to the GRF pipeline**

In `examples/sbi-benchmarks/gaussian_random_field/grf.py`, add `size=8` to the `ConditionalPipeline(...)` call (alongside `id_embedding_strategy=("rope2d", "absolute")`):

```python
pipeline = ConditionalPipeline(
    model,
    train_dataset,
    val_dataset,
    dim_obs=(64, 64),
    dim_cond=2,
    ch_obs=1,
    ch_cond=64,
    method=FlowMatchingMethod(),
    id_embedding_strategy=("rope2d", "absolute"),
    size=8,
)
```

- [ ] **Step 2: Verify the ID grid now matches the patch count**

After construction, `pipeline.obs_ids.shape[1]` must equal `(64 // 8) * (64 // 8) == 64`, the same token count as `patchify_2d(data[1], size=8)` produces. Confirm by checking `pipeline.obs_ids.shape` is `(1, 64, 3)` (previously it was `(1, 1024, 3)`).

- [ ] **Step 3: Run the smoke forward pass**

The smoke call already in the script (`examples/sbi-benchmarks/gaussian_random_field/grf.py:233`):

```python
obs = patchify_2d(data[1], size=8)   # (B, 64, 64)
cond = data[0]
model(0.5, obs, pipeline.obs_ids, cond, pipeline.cond_ids)
```

Expected: runs without a shape-mismatch error between `obs` (64 tokens) and `pipeline.obs_ids` (64 entries).

- [ ] **Step 4: Caveat check (not part of this change) — model output patch size**

This change only fixes the ID grid. For GRF to *train* on correct shapes, the model's final layer must also reconstruct patch size 8: `out_features == patch_size**2 * out_channels == 8*8*1 == 64`, which is consistent with `in_channels=64`. Confirm the `Flux1` model is configured for patch size 8 (final-layer `patch_size`), not 2. If it is still 2, flag it to the user — it is their model config, separate from this library change, but it would otherwise silently produce broken training shapes.

- [ ] **Step 5: Commit the example (GenSBI-examples repo, on the `gaussian_random_field` branch)**

```bash
cd /lustre/ific.uv.es/ml/ific088/github/GenSBI-examples
git add examples/sbi-benchmarks/gaussian_random_field/grf.py
git commit -m "feat(grf): use patch size 8 for obs id embeddings"
```

---

## Self-Review

- **Spec coverage:** leaf `init_ids_2d` size param (Task 1) ✓; `_normalize_patch_size` int|tuple→tuple with `8 → (8,8)` broadcast (Task 1) ✓; resolver threading with 1D-ignores-by-construction (Task 2) ✓; pipeline `int|tuple` param + per-input forwarding (Task 3) ✓; backward-compat defaults verified by reusing existing tests (Tasks 1–3) ✓; end-to-end GRF acceptance + final-layer caveat (Task 4) ✓.
- **No divisibility guard:** intentionally omitted per the settled decision.
- **Type/name consistency:** `_normalize_patch_size`, `_resolve_embedding_ids(..., size=...)`, `init_ids_2d(..., size=...)`, and `ConditionalPipeline(..., size=...)` use the same name `size` throughout. The pipeline normalizes once and forwards plain ints (`obs_size`, `cond_size`) to the resolver, which forwards a plain int to the leaf — no tuple reaches `init_ids_2d`.
- **Test dims:** the new `init_ids_2d` / resolver / pipeline tests use `(16, 16)` (divisible by 8), avoiding the `6 // 8 = 0` trap in the existing `(6, 6)` test.
