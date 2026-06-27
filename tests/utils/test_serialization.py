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
    load_safetensors,
    save_safetensors,
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
    flat = tu.flatten_dict(nnx.to_pure_dict(st))
    flat = {k: jnp.asarray(v) + 1.0 for k, v in flat.items()}
    nnx.replace_by_pure_dict(st, tu.unflatten_dict(flat))
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
    ref = tu.flatten_dict(nnx.to_pure_dict(nnx.state(model)))
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
