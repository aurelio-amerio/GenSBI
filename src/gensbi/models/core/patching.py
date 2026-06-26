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
