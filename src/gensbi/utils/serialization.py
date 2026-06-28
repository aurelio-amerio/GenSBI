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


def _flat_arrays(model, wrt) -> dict[tuple, Any]:
    """Flatten model state to ``{tuple_path: array}`` (ints preserved)."""
    state = nnx.state(model) if wrt is None else nnx.state(model, wrt)
    return tu.flatten_dict(nnx.to_pure_dict(state))


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


def load_safetensors(model, path, *, strict: bool = True):
    """Load weights from a ``.safetensors`` file into ``model`` in place.

    The caller must have reconstructed ``model`` from its ``Params`` first;
    that model is the structural schema.

    Parameters
    ----------
    model : nnx.Module
        Target model, rebuilt from its ``Params``.
    path : str | os.PathLike
        Source ``.safetensors`` file.
    strict : bool
        If True (default), the file's key set must equal the model's and
        every shared key must match shape (``ValueError`` otherwise). If
        False, only the intersection is loaded; model leaves absent from
        the file keep their current values and file keys absent from the
        model are ignored.

    Returns
    -------
    model : nnx.Module
        The same ``model`` object, updated in place with the loaded
        weights.
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
    ref = tu.flatten_dict(nnx.to_pure_dict(state))  # {tuple: array}

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

    nnx.replace_by_pure_dict(state, tu.unflatten_dict(new))
    nnx.update(model, state)
    return model
