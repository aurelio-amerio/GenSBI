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
