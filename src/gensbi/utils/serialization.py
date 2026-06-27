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

from typing import Any, Mapping, Optional

import numpy as np
from flax import nnx
import flax.traverse_util as tu
from safetensors.flax import save_file

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
