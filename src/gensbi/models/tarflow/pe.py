"""Rotary positional embedding for the transformer flow.

Adapted from apple/ml-starflow (STARFlow); see
models/tarflow/LICENSE.starflow. Faithful JAX port of the reference
``misc/pe.py``, restricted to the paths GenSBI uses.
Deliberately omitted relative to the reference: the ``is_1d`` branch (used
only for the pretrained-LM top block), ``freqs_for='pixel'/'constant'`` and
``custom_freqs``, the ``latent_len`` head-dim split (STARFlow's text-prefix
"3D" axis — GenSBI positions prefix tokens at the identity rotation
instead; see docs/superpowers/specs/2026-07-11-tarflow-rope-kvcache-design.md),
video ``duplicate`` handling, and the deprecated checkpoint-compat buffers.
"""

import math

import jax.numpy as jnp
from flax import nnx
from jax import Array

from gensbi.normalizing_flows.bijections.base import Mask


def rotate_half(x: Array) -> Array:
    """Rotate adjacent channel pairs: ``(x1, x2) -> (-x2, x1)``.

    Port of the reference ``rotate_half`` (einops ``(d r)`` with ``r=2``
    means adjacent pairs).

    Parameters
    ----------
    x : Array
        Input whose last dimension is even.

    Returns
    -------
    Array
        Same shape as ``x``.
    """
    xp = x.reshape(*x.shape[:-1], -1, 2)          # '... (d r) -> ... d r'
    x1, x2 = xp[..., 0], xp[..., 1]
    xp = jnp.stack((-x2, x1), axis=-1)
    return xp.reshape(*x.shape)                    # '... d r -> ... (d r)'


def apply_rope(t: Array, freqs: Array) -> Array:
    """Apply a rotary embedding: ``t*cos(freqs) + rotate_half(t)*sin(freqs)``.

    Parameters
    ----------
    t : Array
        Tensor to rotate; last dimension even.
    freqs : Array
        Rotation angles, broadcastable to ``t``'s shape.

    Returns
    -------
    Array
        Rotated tensor, same shape as ``t``.
    """
    return t * jnp.cos(freqs) + rotate_half(t) * jnp.sin(freqs)


def get_positions(h: int, w: int, pt_seq_len: int | None = None) -> Array:
    """2D patch positions in raster order (reference ``'2d'`` mode).

    Coordinates are normalized by ``sqrt(h*w)`` and rescaled to
    ``pt_seq_len`` (the reference's resolution-transfer schedule). For a
    square grid with ``pt_seq_len == h == w`` this reduces to plain integer
    coordinates.

    Parameters
    ----------
    h, w : int
        Patch-grid height and width.
    pt_seq_len : int or None, optional
        Pre-training sequence length; defaults to ``sqrt(h*w)``.

    Returns
    -------
    Array
        Positions of shape ``(h*w, 2)``.
    """
    mean_len = math.sqrt(h * w)
    pt_seq_len = pt_seq_len or mean_len
    px = jnp.arange(h) / mean_len * pt_seq_len
    py = jnp.arange(w) / mean_len * pt_seq_len
    px, py = [p.reshape(-1) for p in jnp.meshgrid(px, py, indexing="ij")]
    return jnp.stack([px, py], axis=-1)


class VisionRotaryEmbedding(nnx.Module):
    """2D vision RoPE with the ``'lang'`` frequency schedule.

    Port of the reference ``VisionRotaryEmbeddingFast`` without the
    ``latent_len`` split: each of the two position axes is rotated with the
    same ``1 / theta**(2i/dim)`` frequency table, and the concatenated
    per-axis angles are repeated pairwise to cover ``2*dim`` channels
    (= the full head dimension when ``dim = head_dim // 2``).

    Parameters
    ----------
    dim : int
        Half the attention head dimension. Must be even.
    pt_seq_len : int, optional
        Pre-training sequence length recorded for position building.
        Default is 16.
    theta : int, optional
        Frequency base. Default is 10000.
    """

    def __init__(self, dim: int, pt_seq_len: int = 16, theta: int = 10000):
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")
        self.pt_seq_len = pt_seq_len
        self.freqs = Mask(
            1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)))

    def __call__(self, pos: Array) -> Array:
        """Build rotation angles for 2D positions.

        Parameters
        ----------
        pos : Array
            Positions of shape ``(..., 2)``.

        Returns
        -------
        Array
            Angles of shape ``(..., 2*dim)``.
        """
        freqs = self.freqs[...]
        freqs_all = jnp.concatenate([
            jnp.einsum("...,f->...f", pos[..., 0], freqs),
            jnp.einsum("...,f->...f", pos[..., 1], freqs),
        ], axis=-1)
        return jnp.repeat(freqs_all, 2, axis=-1)   # '... n -> ... (n r)', r=2
