"""Transformer blocks for the transformer flow.

Adapted from apple/ml-tarflow (TarFlow); see models/tarflow/LICENSE.apple.
Prefix-concatenation conditioning and SOS shift adapted from apple/ml-starflow
(STARFlow); see models/tarflow/LICENSE.starflow.
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from gensbi.normalizing_flows.bijections.base import Mask

INV_SOFTPLUS_1 = 0.541324854612918  # softplus(INV_SOFTPLUS_1) == 1.0 -> identity at zero-init


class AttentionBlock(nnx.Module):
    """Pre-norm residual block combining causal self-attention and an MLP.

    LayerNorm is applied over the channel axis only (not across tokens),
    so no future-token information leaks into earlier positions.

    Parameters
    ----------
    channels : int
        Token embedding width. Must be divisible by ``num_heads``.
    num_heads : int
        Number of attention heads.
    expansion : int
        MLP hidden-size multiplier; the MLP has ``channels * expansion``
        neurons in its hidden layer.
    rngs : nnx.Rngs
        Flax RNG container used to initialize all sub-layers.
    """

    def __init__(self, channels: int, num_heads: int, expansion: int,
                 rngs: nnx.Rngs):
        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be a multiple of num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm1 = nnx.LayerNorm(channels, rngs=rngs)
        self.qkv = nnx.Linear(channels, 3 * channels, rngs=rngs)
        self.proj = nnx.Linear(channels, channels, rngs=rngs)
        self.norm2 = nnx.LayerNorm(channels, rngs=rngs)
        self.mlp_in = nnx.Linear(channels, channels * expansion, rngs=rngs)
        self.mlp_out = nnx.Linear(channels * expansion, channels, rngs=rngs)

    def __call__(self, x: Array, mask: Array | None = None) -> Array:
        """Apply pre-norm residual self-attention followed by a residual MLP.

        Parameters
        ----------
        x : Array
            Token sequence of shape ``(B, T, C)``.
        mask : Array or None, optional
            Attention mask of shape broadcastable to ``(1, 1, T, T)``, e.g. a
            prefix-causal mask. If ``None``, a standard causal mask is used.

        Returns
        -------
        Array
            Output token sequence of shape ``(B, T, C)``.
        """
        B, T, C = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]   # (B, T, nh, hd)
        if mask is None:
            attn = jax.nn.dot_product_attention(q, k, v, is_causal=True)
        else:
            attn = jax.nn.dot_product_attention(q, k, v, mask=mask[None, None])
        x = x + self.proj(attn.reshape(B, T, C))
        h = self.mlp_out(jax.nn.gelu(self.mlp_in(self.norm2(x))))
        return x + h


class MetaBlock(nnx.Module):
    """One exact autoregressive bijection over a token sequence.

    Implements the :class:`~gensbi.normalizing_flows.bijections.base.Bijection`
    direction contract: :meth:`inverse` maps data to noise (density-evaluation
    direction) via a single parallel affine pass with a triangular Jacobian;
    :meth:`forward` maps noise to data (sampling direction) via a sequential
    causal scan that re-runs the attention pass at each token position.

    The affine scale is computed via softplus by default (bounded tail,
    identity at zero-init) or via ``exp`` when ``use_softplus=False`` (legacy).

    Parameters
    ----------
    F : int
        Feature dimension per token (number of input channels per token).
    channels : int
        Internal embedding width for the transformer blocks.
    T : int
        Number of tokens in the sequence.
    perm : Array
        Token permutation applied before the affine transform.
    inv_perm : Array
        Inverse permutation of ``perm``; applied after the affine transform.
    conditioner : VectorConditioner or VectorPrefixConditioner or ImagePrefixConditioner
        Module that provides ``(bias, prefix)`` conditioning signals via its
        ``embed`` method.
    num_layers : int
        Number of :class:`AttentionBlock` layers stacked inside this block.
    num_heads : int
        Number of attention heads passed to each :class:`AttentionBlock`.
    expansion : int
        MLP expansion factor passed to each :class:`AttentionBlock`.
    rngs : nnx.Rngs
        Flax RNG container used to initialize all sub-layers.
    zero_init : bool, optional
        If ``True`` (default), initialize the output projection ``proj_out``
        to zero so the block starts as the identity map.
    use_softplus : bool, optional
        If ``True`` (default), use softplus to compute the affine scale
        (numerically stable, bounded tail). If ``False``, use ``exp``
        (legacy behavior).
    soft_clip : float, optional
        Soft-clip magnitude applied via ``tanh`` to raw network outputs
        before splitting into ``(a, b)``. Default is ``4.0``. Set to ``0``
        to disable clipping.
    """

    def __init__(self, F, channels, T, perm, inv_perm, conditioner,
                 num_layers, num_heads, expansion, rngs, zero_init=True,
                 use_softplus=True, soft_clip=4.0):
        self.F = F
        self.use_softplus = use_softplus
        self.soft_clip = soft_clip
        self.T = T
        self.perm = Mask(jnp.asarray(perm, dtype=jnp.int32))
        self.inv_perm = Mask(jnp.asarray(inv_perm, dtype=jnp.int32))
        self.conditioner = conditioner
        self.proj_in = nnx.Linear(F, channels, rngs=rngs)
        self.sos_embed = nnx.Param(
            jax.random.normal(rngs.params(), (1, 1, F)) * 1e-2)
        self.pos_embed = nnx.Param(
            jax.random.normal(rngs.params(), (T, channels)) * 1e-2)
        self.attn_blocks = nnx.List(
            [AttentionBlock(channels, num_heads, expansion, rngs)
             for _ in range(num_layers)])
        self.proj_out = nnx.Linear(channels, 2 * F, rngs=rngs)
        if zero_init:
            self.proj_out.kernel[...] = jnp.zeros_like(self.proj_out.kernel[...])
            self.proj_out.bias[...] = jnp.zeros_like(self.proj_out.bias[...])

    def _prefix_mask(self, M: int, T: int) -> Array:
        """Prefix-LM mask over [prefix(M); modeled(T)]: modeled is causal and
        sees all prefix; prefix is bidirectional among itself and never sees
        modeled (cond→x blocked)."""
        S = M + T
        idx = jnp.arange(S)
        is_modeled_q = idx[:, None] >= M
        is_prefix_k = idx[None, :] < M
        causal = idx[None, :] <= idx[:, None]
        return jnp.where(is_modeled_q, is_prefix_k | causal, is_prefix_k)

    def _affine(self, a: Array):
        """Map raw log-scale ``a`` -> ``(scale, inv_scale, log_scale)`` in float32.

        ``scale`` plays the role of ``exp(a)`` ("1/sigma"): inverse multiplies by
        ``inv_scale``, forward multiplies by ``scale``, logdet sums ``log_scale``.
        softplus mode bounds the positive-scale tail and its gradient; the
        ``INV_SOFTPLUS_1`` offset makes it the identity at ``a == 0``.
        """
        a = a.astype(jnp.float32)
        if self.use_softplus:
            s = jax.nn.softplus(a + INV_SOFTPLUS_1)
            return s, 1.0 / s, jnp.log(s)
        return jnp.exp(a), jnp.exp(-a), a

    def _embed_cond(self, cond: Array | None):
        """Condition-only signals ``(bias, prefix, mask)`` for :meth:`_params_core`.

        These depend on ``cond`` but not on the modeled tokens, so the sampling
        scan computes them once instead of re-running the (potentially expensive,
        e.g. image-patchify) conditioner and rebuilding the prefix mask at every
        token step.
        """
        bias, prefix = self.conditioner.embed(cond)
        mask = self._prefix_mask(prefix.shape[1], self.T) if prefix is not None else None
        return bias, prefix, mask

    def _params_core(self, x_perm: Array, bias, prefix, mask):
        """(a, b) for the permuted tokens given precomputed conditioning.

        SOS input-shift makes token i's params depend only on tokens < i (and the
        condition). ``bias``/``prefix``/``mask`` come from :meth:`_embed_cond`.
        """
        B = x_perm.shape[0]
        sos = jnp.broadcast_to(self.sos_embed[...], (B, 1, self.F))
        x_in = jnp.concatenate([sos, x_perm[:, :-1]], axis=1)   # (B, T, F)
        h = self.proj_in(x_in) + self.pos_embed[...][None]      # (B, T, C)
        if bias is not None:
            h = h + bias[:, None, :]
        if prefix is not None:
            M = prefix.shape[1]
            h = jnp.concatenate([prefix, h], axis=1)            # (B, M+T, C)
            for blk in self.attn_blocks:
                h = blk(h, mask)
            h = h[:, M:]                                        # (B, T, C) strip
        else:
            for blk in self.attn_blocks:
                h = blk(h)
        out = self.proj_out(h)                                  # (B, T, 2F)
        if self.soft_clip > 0:
            out = self.soft_clip * jnp.tanh(out / self.soft_clip)
        a, b = jnp.split(out, 2, axis=-1)                       # each (B, T, F)
        return a, b

    def _params(self, x_perm: Array, cond: Array | None):
        """(a, b) for the permuted tokens (single-shot; embeds the condition)."""
        bias, prefix, mask = self._embed_cond(cond)
        return self._params_core(x_perm, bias, prefix, mask)

    def inverse(self, x: Array, cond: Array | None = None):
        """Map data to noise (the density-evaluation direction).

        Applies a per-token parallel affine transform
        ``z = (x − b) · inv_scale`` after permuting tokens. Token ``i``'s
        parameters ``(a, b)`` depend only on tokens at positions ``< i``
        (causal attention on a shift-by-one input), giving a triangular
        Jacobian computable in a single forward pass.

        Parameters
        ----------
        x : Array
            Data-space token sequence of shape ``(B, T, F)`` or a flat array
            that will be reshaped to ``(B, T, F)``.
        cond : Array or None, optional
            Conditioning input, or ``None`` for an unconditional transform.

        Returns
        -------
        z : Array
            Noise-space output of shape ``(B, T, F)``.
        logabsdet : Array
            Log absolute determinant of the Jacobian of the inverse map,
            shape ``(B,)``. Equal to ``-Σ log_scale`` over token and feature
            dimensions.
        """
        x = x.reshape(x.shape[0], self.T, self.F)
        xp = x[:, self.perm[...]]
        a, b = self._params(xp, cond)
        scale, inv_scale, log_scale = self._affine(a)
        z = (xp.astype(jnp.float32) - b.astype(jnp.float32)) * inv_scale
        logdet = -jnp.sum(log_scale, axis=(1, 2))              # (B,)
        z = z[:, self.inv_perm[...]].astype(xp.dtype)
        return z, logdet

    def forward(self, z: Array, cond: Array | None = None):
        """Map noise to data (the sampling direction).

        Sequentially scans over token positions via ``jax.lax.scan``,
        re-running the causal attention pass at each step so that token
        ``i``'s parameters are conditioned on already-generated tokens
        ``0, …, i-1`` (mirrors ``MaskedAutoregressive.forward``).

        Parameters
        ----------
        z : Array
            Noise-space token sequence of shape ``(B, T, F)`` or a flat array
            that will be reshaped to ``(B, T, F)``.
        cond : Array or None, optional
            Conditioning input, or ``None`` for an unconditional transform.

        Returns
        -------
        x : Array
            Data-space output of shape ``(B, T, F)``.
        logabsdet : Array
            Log absolute determinant of the Jacobian of the forward map,
            shape ``(B,)``. Equal to ``+Σ log_scale`` over token and feature
            dimensions.
        """
        z = z.reshape(z.shape[0], self.T, self.F)
        zp = z[:, self.perm[...]]
        bias, prefix, mask = self._embed_cond(cond)            # constant over the scan

        def body(x, i):
            # a[:,i],b[:,i] depend only on tokens < i, so log_scale[:,i] is final at
            # step i: accumulate it here instead of a second full pass after the scan.
            a, b = self._params_core(x, bias, prefix, mask)
            scale, _, log_scale = self._affine(a)
            xi = zp[:, i, :] * scale[:, i, :] + b[:, i, :].astype(jnp.float32)
            x = x.at[:, i, :].set(xi.astype(x.dtype))
            return x, log_scale[:, i, :]                        # (B, F)

        x = jnp.zeros_like(zp)
        x, log_scale_steps = jax.lax.scan(body, x, jnp.arange(self.T))  # (T, B, F)
        logdet = jnp.sum(log_scale_steps, axis=(0, 2))         # (B,), +Σ log_scale
        x = x[:, self.inv_perm[...]]
        return x, logdet
