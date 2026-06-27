# Safetensors Serialization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add portable safetensors save/load for any flax `nnx` model, plus thin pipeline convenience methods, so a trained model's weights can be exported to a single framework-neutral file and loaded back into a model rebuilt from its `Params`.

**Architecture:** Two standalone functions in `gensbi/utils/serialization.py` flatten `nnx.state(model).to_pure_dict()` to a flat `{str: array}` safetensors file (state-path tuples joined with `"."`, ints preserved) and load it back via `nnx.restore_int_paths`, validating against the rebuilt model and updating it in place with `nnx.update`. The pipeline gains `export_safetensors`/`import_safetensors` that only select the primary or EMA model and delegate.

**Tech Stack:** Python, flax nnx 0.12.7, `flax.traverse_util`, safetensors 0.8.0 (`safetensors.flax` + `safetensors.safe_open`), jax, pytest.

## Global Constraints

- **Dependencies already present:** `safetensors[jax]>=0.8.0`, `flax` (nnx) — do not add new deps.
- **Test environment:** run tests in the **`gensbi` mamba env** (not `.venv`). `pytest` auto-applies `JAX_PLATFORMS=cpu` and `-n 2` from `pyproject.toml` — no need to set them manually.
- **No `Variable.value` access** (deprecated in flax 0.12.7): go through `state.to_pure_dict()` / `state.replace_by_pure_dict()` exclusively.
- **Branch:** `maf` (un-merged, not pushed). Additive feature, no breaking changes. Commit after every task.
- **Key separator:** `"."`. Assumes no non-integer nnx path component contains a literal `.` — guarded at save time.
- **Spec:** `docs/superpowers/specs/2026-06-27-safetensors-serialization-design.md`.

---

## File Structure

- **Create** `src/gensbi/utils/serialization.py` — the two public functions (`save_safetensors`, `load_safetensors`) + private helpers (`_join_key`, `_flat_arrays`) + constants. Single responsibility: nnx ↔ safetensors translation.
- **Modify** `src/gensbi/utils/__init__.py` — re-export the two public functions (currently re-exports nothing).
- **Modify** `src/gensbi/recipes/pipeline.py` — add `export_safetensors`/`import_safetensors` to `AbstractPipeline` (after `restore_model`); add the import.
- **Create** `tests/utils/test_serialization.py` — all tests.

---

### Task 1: `save_safetensors` + helpers

**Files:**
- Create: `src/gensbi/utils/serialization.py`
- Test: `tests/utils/test_serialization.py`

**Interfaces:**
- Consumes: nothing (entry task).
- Produces:
  - `save_safetensors(model, path, *, metadata: Mapping|None = None, wrt=None) -> None`
  - `_join_key(path: tuple) -> str` — joins an nnx state-path tuple with `"."`, stringifying int indices; raises `ValueError` if a non-int component contains `"."`.
  - `_flat_arrays(model, wrt) -> dict[tuple, array]` — `flatten_dict(nnx.state(model[, wrt]).to_pure_dict())`.
  - Module constants `_SEP = "."`, `_DEFAULT_METADATA = {"format": "gensbi", "version": "1", "framework": "flax-nnx"}`.
  - On-disk metadata always includes `format`, `version`, `framework`, `model_class=type(model).__name__`; caller `metadata` is stringified and merged over these.

- [ ] **Step 1: Write the failing tests**

Create `tests/utils/test_serialization.py`:

```python
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
import flax.traverse_util as tu
from safetensors import safe_open
from safetensors.flax import save_file

from gensbi.utils.serialization import (
    save_safetensors,
    load_safetensors,
    _join_key,
)


# --- test fixtures: a generic nnx module with an nnx.List + BatchStat ---
class _TinyBlock(nnx.Module):
    def __init__(self, rngs):
        self.lin = nnx.Linear(3, 3, rngs=rngs)
        self.bn = nnx.BatchNorm(3, rngs=rngs)  # carries BatchStat (mean/var)


class _TinyNet(nnx.Module):
    def __init__(self, seed):
        rngs = nnx.Rngs(seed)
        self.blocks = nnx.List([_TinyBlock(rngs) for _ in range(2)])


def _make_maf(seed, *, dim=3):
    from gensbi.models import MAFlow, MAFlowParams

    params = MAFlowParams(rngs=nnx.Rngs(seed), dim=dim, zero_init=False)
    return MAFlow(params)


# --- Task 1 tests ---
def test_join_key_joins_ints_and_guards_separator():
    assert _join_key(("blocks", 0, "lin", "kernel")) == "blocks.0.lin.kernel"
    with pytest.raises(ValueError, match="separator"):
        _join_key(("bad.name", "kernel"))


def test_save_writes_dotjoined_keys_and_metadata(tmp_path):
    model = _make_maf(0)
    path = tmp_path / "m.safetensors"
    save_safetensors(model, path, metadata={"note": "hello"})

    with safe_open(str(path), framework="flax") as f:
        keys = list(f.keys())
        meta = f.metadata()

    assert keys, "no tensors written"
    assert all(isinstance(k, str) for k in keys)
    # MAFlow's Chain uses nnx.List -> integer index appears as a dot segment
    assert any("." in k for k in keys)
    assert meta["format"] == "gensbi"
    assert meta["version"] == "1"
    assert meta["framework"] == "flax-nnx"
    assert meta["model_class"] == "MAFlow"
    assert meta["note"] == "hello"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/utils/test_serialization.py -k "join_key or save_writes" -v`
Expected: FAIL — `ImportError`/`cannot import name 'save_safetensors'` (module does not exist yet).

- [ ] **Step 3: Create the module with helpers + `save_safetensors`**

Create `src/gensbi/utils/serialization.py`:

```python
"""Portable safetensors save/load for flax ``nnx`` models.

Exports the weights of any :class:`flax.nnx.Module` to a single,
framework-neutral ``.safetensors`` file, and loads them back into a model the
caller has already reconstructed from its ``Params``. The file stores a flat
``{str: array}`` table (nnx state paths joined with ``"."``) plus a small
provenance ``metadata`` blob; it does *not* carry enough information to rebuild
the model architecture (an explicit non-goal).

See ``docs/superpowers/specs/2026-06-27-safetensors-serialization-design.md``.
"""

from __future__ import annotations

import warnings
from typing import Any, Mapping, Optional

import numpy as np
from flax import nnx
import flax.traverse_util as tu
from safetensors import safe_open
from safetensors.flax import load_file, save_file

_SEP = "."
_DEFAULT_METADATA = {"format": "gensbi", "version": "1", "framework": "flax-nnx"}


def _join_key(path: tuple) -> str:
    """Join an nnx state-path tuple into a safetensors string key.

    Integer ``nnx.List`` indices are stringified; a non-integer component that
    contains the ``"."`` separator is unrepresentable and raises ``ValueError``.
    """
    parts = []
    for p in path:
        s = str(p)
        if not isinstance(p, int) and _SEP in s:
            raise ValueError(
                f"state path component {s!r} contains the key separator "
                f"{_SEP!r}; this model cannot be safetensors-serialized"
            )
        parts.append(s)
    return _SEP.join(parts)


def _flat_arrays(model, wrt) -> dict:
    """Flatten model state to ``{tuple_path: array}`` (ints preserved)."""
    state = nnx.state(model) if wrt is None else nnx.state(model, wrt)
    return tu.flatten_dict(state.to_pure_dict())


def save_safetensors(
    model,
    path,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
    wrt=None,
) -> None:
    """Save ``model``'s weights to a single ``.safetensors`` file.

    Parameters
    ----------
    model : nnx.Module
        Any flax nnx module.
    path : str | os.PathLike
        Destination ``.safetensors`` file.
    metadata : mapping, optional
        Extra provenance, stringified and merged over (overriding) the defaults
        ``format``/``version``/``framework``/``model_class``.
    wrt : nnx filter, optional
        Restrict the saved variable collections (e.g. ``nnx.Param``). Default
        saves the full state.
    """
    flat = _flat_arrays(model, wrt)
    tensors = {_join_key(k): np.asarray(v) for k, v in flat.items()}
    meta = dict(_DEFAULT_METADATA)
    meta["model_class"] = type(model).__name__
    if metadata:
        meta.update({str(k): str(v) for k, v in metadata.items()})
    save_file(tensors, str(path), metadata=meta)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/utils/test_serialization.py -k "join_key or save_writes" -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/utils/serialization.py tests/utils/test_serialization.py
git commit -m "feat(serialization): safetensors save_safetensors + key helpers"
```

---

### Task 2: `load_safetensors` + utils re-export

**Files:**
- Modify: `src/gensbi/utils/serialization.py` (append `load_safetensors`)
- Modify: `src/gensbi/utils/__init__.py`
- Test: `tests/utils/test_serialization.py` (append load tests)

**Interfaces:**
- Consumes: `save_safetensors`, `_join_key` (Task 1).
- Produces: `load_safetensors(model, path, *, strict: bool = True) -> model`. Updates `model` in place via `nnx.update` and returns it. `strict=True` requires the file's key set to equal the model's and every shared key to match shape (raises `ValueError` otherwise). `strict=False` loads the intersection. Loaded arrays are cast to the model leaf's dtype. A `model_class` metadata mismatch emits a `UserWarning` (not an error). Re-exported from `gensbi.utils`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/utils/test_serialization.py`:

```python
# --- Task 2 tests ---
def test_roundtrip_maf_fidelity_and_logprob(tmp_path):
    src = _make_maf(0)
    path = tmp_path / "m.safetensors"
    save_safetensors(src, path)

    dst = _make_maf(123)  # different init
    x = jax.random.normal(jax.random.PRNGKey(7), (16, 3))
    assert not bool(jnp.allclose(src.log_prob(x), dst.log_prob(x)))  # differ before load

    out = load_safetensors(dst, path)
    assert out is dst  # in-place, returns the model

    s_leaves = jax.tree.leaves(nnx.state(src))
    d_leaves = jax.tree.leaves(nnx.state(dst))
    assert all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(s_leaves, d_leaves)
    )
    assert bool(jnp.allclose(src.log_prob(x), dst.log_prob(x), atol=1e-6))


def test_roundtrip_generic_module_with_list_and_batchstat(tmp_path):
    src = _TinyNet(0)
    # perturb every leaf (incl. BatchStat mean/var) so nothing equals a fresh init
    st = nnx.state(src)
    flat = tu.flatten_dict(st.to_pure_dict())
    flat = {k: jnp.asarray(v) + 1.0 for k, v in flat.items()}
    st.replace_by_pure_dict(tu.unflatten_dict(flat))
    nnx.update(src, st)

    path = tmp_path / "n.safetensors"
    save_safetensors(src, path)
    with safe_open(str(path), framework="flax") as f:
        assert any(k.endswith("mean") for k in f.keys())  # BatchStat is included

    dst = _TinyNet(1)
    load_safetensors(dst, path)
    assert all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(jax.tree.leaves(nnx.state(src)), jax.tree.leaves(nnx.state(dst)))
    )


def test_strict_rejects_shape_mismatch(tmp_path):
    path = tmp_path / "m.safetensors"
    save_safetensors(_make_maf(0, dim=3), path)
    dst = _make_maf(0, dim=4)  # same structure, different array shapes
    with pytest.raises(ValueError):
        load_safetensors(dst, path)


def test_non_strict_loads_param_subset(tmp_path):
    src = _TinyNet(0)
    path = tmp_path / "p.safetensors"
    save_safetensors(src, path, wrt=nnx.Param)  # omits BatchStat keys

    dst = _TinyNet(1)
    with pytest.raises(ValueError):  # file is missing BatchStat keys
        load_safetensors(dst, path, strict=True)

    load_safetensors(dst, path, strict=False)  # loads the Param overlap
    sp = jax.tree.leaves(nnx.state(src, nnx.Param))
    dp = jax.tree.leaves(nnx.state(dst, nnx.Param))
    assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(sp, dp))


def test_load_casts_to_model_dtype(tmp_path):
    model = _TinyNet(0)
    ref = tu.flatten_dict(nnx.state(model).to_pure_dict())
    # write a file holding float16 versions of every key
    tensors = {".".join(map(str, k)): np.asarray(v).astype(np.float16) for k, v in ref.items()}
    path = tmp_path / "h.safetensors"
    save_file(tensors, str(path), metadata={"model_class": "_TinyNet"})

    load_safetensors(model, path)
    assert all(
        np.asarray(v).dtype == np.float32 for v in jax.tree.leaves(nnx.state(model))
    )


def test_model_class_mismatch_warns(tmp_path):
    path = tmp_path / "n.safetensors"
    save_safetensors(_TinyNet(0), path, metadata={"model_class": "OtherNet"})
    dst = _TinyNet(1)
    with pytest.warns(UserWarning, match="model_class"):
        load_safetensors(dst, path)


def test_load_safetensors_is_reexported_from_utils():
    from gensbi.utils import save_safetensors as s, load_safetensors as l

    assert callable(s) and callable(l)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/utils/test_serialization.py -k "roundtrip or strict or non_strict or casts or mismatch or reexported" -v`
Expected: FAIL — `cannot import name 'load_safetensors'` is satisfied at module level (it does not exist yet) → collection/usage error.

- [ ] **Step 3: Append `load_safetensors` to the module**

Add to the end of `src/gensbi/utils/serialization.py`:

```python
def load_safetensors(model, path, *, strict: bool = True):
    """Load weights from a ``.safetensors`` file into ``model`` in place.

    The caller must have reconstructed ``model`` from its ``Params`` first; that
    model is the structural schema. Returns ``model``.

    Parameters
    ----------
    model : nnx.Module
        Target model, rebuilt from its ``Params``.
    path : str | os.PathLike
        Source ``.safetensors`` file.
    strict : bool
        If True (default), the file's key set must equal the model's and every
        shared key must match shape (``ValueError`` otherwise). If False, only
        the intersection is loaded; model leaves absent from the file keep their
        current values and file keys absent from the model are ignored.
    """
    loaded = load_file(str(path))  # {str: jax.Array}

    with safe_open(str(path), framework="flax") as f:
        saved_meta = f.metadata() or {}
    saved_class = saved_meta.get("model_class")
    target_class = type(model).__name__
    if saved_class is not None and saved_class != target_class:
        warnings.warn(
            f"safetensors model_class={saved_class!r} does not match target "
            f"model {target_class!r}; loading anyway",
            stacklevel=2,
        )

    # Reconstruct the int-keyed pure dict from the flat file (official helper),
    # then re-flatten to tuple keys for comparison against the model schema.
    file_flat = tu.flatten_dict(
        nnx.restore_int_paths(tu.unflatten_dict(loaded, sep=_SEP))
    )

    state = nnx.state(model)
    ref = tu.flatten_dict(state.to_pure_dict())  # {tuple: array}

    missing = set(ref) - set(file_flat)
    extra = set(file_flat) - set(ref)
    if strict and (missing or extra):
        raise ValueError(
            "safetensors key mismatch:\n"
            f"  missing from file ({len(missing)}): "
            f"{sorted(_join_key(k) for k in missing)[:10]}\n"
            f"  unexpected in file ({len(extra)}): "
            f"{sorted(_join_key(k) for k in extra)[:10]}"
        )

    new = {}
    for k, want in ref.items():
        if k in file_flat:
            arr = file_flat[k]
            if arr.shape != want.shape:
                raise ValueError(
                    f"shape mismatch for {_join_key(k)!r}: "
                    f"file {tuple(arr.shape)} vs model {tuple(want.shape)}"
                )
            new[k] = arr.astype(want.dtype)
        else:
            new[k] = want  # strict=False: keep the model's current value

    state.replace_by_pure_dict(tu.unflatten_dict(new))
    nnx.update(model, state)
    return model
```

- [ ] **Step 4: Re-export from `gensbi.utils`**

Replace the contents of `src/gensbi/utils/__init__.py` with (keep the existing module docstring, then add the exports):

```python
"""
Utility functions for GenSBI.

This module provides general utility functions including mathematical operations,
model wrapping utilities, plotting functions, and model serialization.
"""

from .serialization import save_safetensors, load_safetensors

__all__ = ["save_safetensors", "load_safetensors"]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/utils/test_serialization.py -v`
Expected: PASS (all tests from Tasks 1 + 2 pass).

- [ ] **Step 6: Commit**

```bash
git add src/gensbi/utils/serialization.py src/gensbi/utils/__init__.py tests/utils/test_serialization.py
git commit -m "feat(serialization): load_safetensors + gensbi.utils re-export"
```

---

### Task 3: Pipeline convenience + integration check

**Files:**
- Modify: `src/gensbi/recipes/pipeline.py` (add import; add two methods to `AbstractPipeline` after `restore_model`, ~line 564, before the `_wrap_model` abstractmethod)
- Test: `tests/utils/test_serialization.py` (append pipeline test)

**Interfaces:**
- Consumes: `save_safetensors`, `load_safetensors` (Task 2); `AbstractPipeline` attributes `self.model`, `self.ema_model`.
- Produces:
  - `AbstractPipeline.export_safetensors(self, path, *, ema=True, metadata=None) -> None`
  - `AbstractPipeline.import_safetensors(self, path, *, ema=True, strict=True) -> None`

- [ ] **Step 1: Write the failing test**

Append to `tests/utils/test_serialization.py`:

```python
# --- Task 3 test (pipeline convenience via lightweight stub) ---
def test_pipeline_export_import_selects_ema(tmp_path):
    from gensbi.recipes.pipeline import AbstractPipeline

    class _Stub:  # stands in for a pipeline; only .model / .ema_model are used
        pass

    src = _Stub()
    src.model = _TinyNet(0)       # primary weights
    src.ema_model = _TinyNet(1)   # ema weights (distinct)

    # default ema=True exports the EMA model
    ema_path = tmp_path / "ema.safetensors"
    AbstractPipeline.export_safetensors(src, ema_path)
    tgt = _Stub()
    tgt.model = _TinyNet(2)
    tgt.ema_model = _TinyNet(3)
    AbstractPipeline.import_safetensors(tgt, ema_path)  # loads into tgt.ema_model
    assert all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(
            jax.tree.leaves(nnx.state(src.ema_model)),
            jax.tree.leaves(nnx.state(tgt.ema_model)),
        )
    )

    # ema=False selects the primary model on both export and import
    primary_path = tmp_path / "primary.safetensors"
    AbstractPipeline.export_safetensors(src, primary_path, ema=False)
    tgt2 = _Stub()
    tgt2.model = _TinyNet(4)
    tgt2.ema_model = _TinyNet(5)
    AbstractPipeline.import_safetensors(tgt2, primary_path, ema=False)
    assert all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(
            jax.tree.leaves(nnx.state(src.model)),
            jax.tree.leaves(nnx.state(tgt2.model)),
        )
    )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/utils/test_serialization.py::test_pipeline_export_import_selects_ema -v`
Expected: FAIL — `AttributeError: type object 'AbstractPipeline' has no attribute 'export_safetensors'`.

- [ ] **Step 3: Add the import to `pipeline.py`**

In `src/gensbi/recipes/pipeline.py`, near the other gensbi imports (the file already has `from gensbi.utils.misc import get_colored_value` around line 35), add:

```python
from gensbi.utils.serialization import save_safetensors, load_safetensors
```

- [ ] **Step 4: Add the two methods to `AbstractPipeline`**

In `src/gensbi/recipes/pipeline.py`, immediately after the `restore_model` method (which ends ~line 564, returning before the `@abc.abstractmethod def _wrap_model`), insert:

```python
    def export_safetensors(self, path, *, ema=True, metadata=None):
        """Export trained weights to a single ``.safetensors`` file.

        ``ema=True`` (default) exports the EMA model -- usually the weights you
        want for inference and for sharing. Pass ``ema=False`` for the primary
        model. This is a thin wrapper over
        :func:`gensbi.utils.serialization.save_safetensors`.
        """
        model = self.ema_model if ema else self.model
        save_safetensors(model, path, metadata=metadata)

    def import_safetensors(self, path, *, ema=True, strict=True):
        """Load weights from a ``.safetensors`` file into this pipeline in place.

        ``ema=True`` (default) loads into the EMA model. Thin wrapper over
        :func:`gensbi.utils.serialization.load_safetensors`.
        """
        model = self.ema_model if ema else self.model
        load_safetensors(model, path, strict=strict)
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `pytest tests/utils/test_serialization.py::test_pipeline_export_import_selects_ema -v`
Expected: PASS.

- [ ] **Step 6: Run the full serialization module + import smoke**

Run: `pytest tests/utils/test_serialization.py tests/test_import_smoke.py -v`
Expected: PASS (all serialization tests + the import smoke test).

- [ ] **Step 7: Commit**

```bash
git add src/gensbi/recipes/pipeline.py tests/utils/test_serialization.py
git commit -m "feat(recipes): pipeline export_/import_safetensors convenience"
```

---

## Self-Review

**Spec coverage:**
- Standalone `save_safetensors`/`load_safetensors` in `gensbi/utils/serialization.py` → Tasks 1, 2. ✔
- Re-export from `gensbi.utils` → Task 2, Step 4 + `test_load_safetensors_is_reexported_from_utils`. ✔
- `to_pure_dict` + `flatten_dict(sep=None)` + `_join_key` on save; `restore_int_paths` + validation + dtype cast + `replace_by_pure_dict`/`nnx.update` on load → Tasks 1, 2. ✔
- Full-state default + `wrt` filter → `save_safetensors(..., wrt=...)`, tested via `test_non_strict_loads_param_subset`. ✔
- Metadata (`format`/`version`/`model_class`/`framework`) → Task 1 + `test_save_writes_dotjoined_keys_and_metadata`. ✔
- Strict vs partial load + shape validation + dtype cast + `model_class` warning → Task 2 tests. ✔
- Separator guard at save time → `_join_key` + `test_join_key_joins_ints_and_guards_separator`. ✔
- In-place load returning the model → `test_roundtrip_maf_fidelity_and_logprob` (`out is dst`). ✔
- Thin pipeline `export_/import_safetensors` (EMA default) → Task 3. ✔
- Generic (non-flow) module incl. `nnx.List` + `BatchStat` → `test_roundtrip_generic_module_with_list_and_batchstat`. ✔
- Non-goals (auto-reconstruction, per-model methods, orbax replacement, sharding, cross-framework) → not implemented, by design. ✔

**Placeholder scan:** No TBD/TODO/"handle edge cases"; every code step shows complete code. ✔

**Type consistency:** `save_safetensors(model, path, *, metadata, wrt)`, `load_safetensors(model, path, *, strict)`, `_join_key(path)`, `_flat_arrays(model, wrt)`, `_SEP`, `_DEFAULT_METADATA` used identically across tasks; pipeline methods match their wrapped signatures. ✔
