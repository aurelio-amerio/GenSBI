"""Shared standardization-statistic helpers for flow models."""

import jax.numpy as jnp


def fit_stat(s, example_shape, dtype=None):
    """Broadcast a standardization statistic to ``example_shape``.

    Accepted shapes for ``s`` (with ``example_shape = (dim, C, ...)``):

    - ``(dim,)`` — reshaped to ``(dim, 1, ...)`` then broadcast
      (per-dimension stats, the tabular default);
    - ``(dim, C)`` / ``example_shape`` — used as-is;
    - ``(C,)`` — broadcast along the leading axes (per-channel stats);
    - scalar — broadcast everywhere.

    Ambiguous case: when ``s`` is 1-D and ``C == dim``, the ``(dim,)``
    per-dimension interpretation wins over the ``(C,)`` per-channel one,
    since the shape-match check below tests ``s.shape[0] == example_shape[0]``
    (i.e. against ``dim``) first.

    Parameters
    ----------
    s : array-like
        Statistic (mean or std) to fit.
    example_shape : tuple of int
        Target per-example shape, e.g. ``(dim, channels)``.
    dtype : jnp.dtype or None, optional
        If given, cast ``s`` before broadcasting (used when writing into an
        existing buffer). Default is ``None``.

    Returns
    -------
    Array
        ``s`` broadcast to ``example_shape``.
    """
    s = jnp.asarray(s, dtype=dtype)
    if s.ndim == 1 and s.shape[0] == example_shape[0]:
        s = s.reshape((example_shape[0],) + (1,) * (len(example_shape) - 1))
    return jnp.broadcast_to(s, example_shape)
