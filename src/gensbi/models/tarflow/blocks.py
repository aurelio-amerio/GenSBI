"""Transformer blocks for the transformer flow.

Adapted from apple/ml-tarflow (TarFlow); see models/tarflow/LICENSE.apple.
Prefix-concatenation conditioning and SOS shift adapted from apple/ml-starflow
(STARFlow); see models/tarflow/LICENSE.starflow.
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array
from jax.typing import DTypeLike

from gensbi.models.tarflow.pe import apply_rope, get_positions
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
    param_dtype : DTypeLike, optional
        Dtype for all stored (master) kernel/bias/scale parameters. Default
        is ``float32``.
    dtype : DTypeLike, optional
        Compute dtype forwarded to each ``Linear``. Default is ``float32``,
        matching ``param_dtype``, so with default arguments this is a
        bit-identical no-op cast. ``norm1``/``norm2`` are fp32 islands and
        always run at ``float32`` regardless of this knob; their output
        self-heals when it feeds the following compute-dtype ``Linear``
        (``promote_dtype`` downcasts it there).
    """

    def __init__(self, channels: int, num_heads: int, expansion: int,
                 rngs: nnx.Rngs, param_dtype: DTypeLike = jnp.float32,
                 dtype: DTypeLike = jnp.float32):
        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be a multiple of num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        # fp32 islands: norms stay at float32 regardless of the dtype knob;
        # their output self-heals via promote_dtype when feeding qkv/mlp_in.
        self.norm1 = nnx.LayerNorm(channels, rngs=rngs, param_dtype=param_dtype,
                                    dtype=jnp.float32)
        self.qkv = nnx.Linear(channels, 3 * channels, rngs=rngs,
                              param_dtype=param_dtype, dtype=dtype)
        self.proj = nnx.Linear(channels, channels, rngs=rngs,
                               param_dtype=param_dtype, dtype=dtype)
        self.norm2 = nnx.LayerNorm(channels, rngs=rngs, param_dtype=param_dtype,
                                    dtype=jnp.float32)
        self.mlp_in = nnx.Linear(channels, channels * expansion, rngs=rngs,
                                 param_dtype=param_dtype, dtype=dtype)
        self.mlp_out = nnx.Linear(channels * expansion, channels, rngs=rngs,
                                  param_dtype=param_dtype, dtype=dtype)

    def _qkv(self, x: Array):
        """Project to unrotated (q, k, v), each of shape ``(B, S, nh, hd)``."""
        B, S, C = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, S, 3, self.num_heads, self.head_dim)
        return qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

    def _finish(self, x: Array, attn: Array) -> Array:
        """Residual attention output + residual MLP (shared tail)."""
        B, S, C = x.shape
        x = x + self.proj(attn.reshape(B, S, C))
        h = self.mlp_out(jax.nn.gelu(self.mlp_in(self.norm2(x))))
        return x + h

    def __call__(self, x: Array, mask: Array | None = None,
                 freqs_cis: Array | None = None, return_kv: bool = False):
        """Apply pre-norm residual self-attention followed by a residual MLP.

        Parameters
        ----------
        x : Array
            Token sequence of shape ``(B, T, C)``.
        mask : Array or None, optional
            Attention mask broadcastable to ``(1, 1, T, T)``; ``None`` means
            standard causal.
        freqs_cis : Array or None, optional
            Rotary angles ``(T, head_dim)`` applied to q/k before attention.
        return_kv : bool, optional
            If ``True``, also return the **unrotated** ``(k, v)`` — the KV
            cache stores unrotated keys (the cached path re-rotates the full
            cache each step, as the reference does). Default ``False``.

        Returns
        -------
        Array or tuple
            ``out`` of shape ``(B, T, C)``, or ``(out, k, v)`` when
            ``return_kv=True``.
        """
        q, k, v = self._qkv(x)
        k_raw, v_raw = k, v
        if freqs_cis is not None:
            q = apply_rope(q, freqs_cis[None, :, None, :])
            k = apply_rope(k, freqs_cis[None, :, None, :])
        if mask is None:
            attn = jax.nn.dot_product_attention(q, k, v, is_causal=True)
        else:
            attn = jax.nn.dot_product_attention(q, k, v, mask=mask[None, None])
        out = self._finish(x, attn)
        if return_kv:
            return out, k_raw, v_raw
        return out

    def decode(self, x_new: Array, k_cache: Array, v_cache: Array,
               index, freqs_cis: Array | None = None):
        """Single-token decode step against a preallocated KV cache.

        The cache stores **unrotated** k (reference behavior: rope is applied
        after the cache read, re-rotating the whole prefix each step). Slots
        beyond ``index`` are zero-filled and masked out of the attention.

        Parameters
        ----------
        x_new : Array
            New token, shape ``(B, 1, C)``.
        k_cache, v_cache : Array
            Caches of shape ``(B, S, nh, hd)`` with ``S`` total slots.
        index : int or traced scalar
            Slot to write; attention sees slots ``<= index``.
        freqs_cis : Array or None, optional
            Rotary angles ``(S, head_dim)`` for **all** slots; the new
            token's q uses row ``index``.

        Returns
        -------
        tuple
            ``(out, k_cache, v_cache)`` with ``out`` of shape ``(B, 1, C)``.
        """
        q, k, v = self._qkv(x_new)                     # (B, 1, nh, hd)
        k_cache = jax.lax.dynamic_update_slice_in_dim(k_cache, k, index, axis=1)
        v_cache = jax.lax.dynamic_update_slice_in_dim(v_cache, v, index, axis=1)
        k_all = k_cache
        if freqs_cis is not None:
            fq = jax.lax.dynamic_slice_in_dim(freqs_cis, index, 1, axis=0)
            q = apply_rope(q, fq[None, :, None, :])
            k_all = apply_rope(k_cache, freqs_cis[None, :, None, :])
        S = k_cache.shape[1]
        mask = (jnp.arange(S) <= index)[None, None, None, :]   # (1,1,1,S)
        attn = jax.nn.dot_product_attention(q, k_all, v_cache, mask=mask)
        return self._finish(x_new, attn), k_cache, v_cache


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
        Token permutation applied before the affine transform. The inverse
        permutation is derived internally via ``argsort``.
    conditioner : AdditiveBiasConditioner or VectorConditioner or ImageConditioner
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
    rope : VisionRotaryEmbedding or None, optional
        If given, 2D rotary position embeddings replace the learned
        ``pos_embed`` (which is then set to ``None``). Positions are laid
        out with the ``M`` prefix (conditioning) slots at the identity
        rotation (zero angles) followed by the ``T`` image slots on the
        normalized ``grid`` (raster order, not permuted by ``perm``).
        Default is ``None`` (learned ``pos_embed``, unchanged behavior).
    grid : tuple of int or None, optional
        ``(h, w)`` patch-grid shape used to build rope positions when
        ``rope`` is given; required in that case. Default is ``None``.
    param_dtype : DTypeLike, optional
        Dtype for all stored (master) kernel/bias/embedding parameters.
        Default is ``float32``.
    dtype : DTypeLike, optional
        Compute dtype forwarded to ``proj_in``/``proj_out``/``AttentionBlock``
        layers. Default is ``float32``, matching ``param_dtype``, so with
        default arguments this is a bit-identical no-op cast. The
        softplus/soft_clip affine-scale computation in :meth:`_affine` stays
        unconditionally fp32 regardless of this knob.
    """

    def __init__(self, F, channels, T, perm, conditioner,
                 num_layers, num_heads, expansion, rngs, zero_init=True,
                 use_softplus=True, soft_clip=4.0, rope=None, grid=None,
                 param_dtype: DTypeLike = jnp.float32,
                 dtype: DTypeLike = jnp.float32):
        self.F = F
        self.use_softplus = use_softplus
        self.soft_clip = soft_clip
        self.T = T
        self.dtype = dtype
        perm = jnp.asarray(perm, dtype=jnp.int32)
        self.perm = Mask(perm)
        self.inv_perm = Mask(jnp.argsort(perm))
        self.conditioner = conditioner
        self.proj_in = nnx.Linear(F, channels, rngs=rngs,
                                  param_dtype=param_dtype, dtype=dtype)
        self.sos_embed = nnx.Param(
            (jax.random.normal(rngs.params(), (1, 1, F)) * 1e-2).astype(param_dtype))
        if rope is not None:
            if grid is None:
                raise ValueError("rope requires grid=(h, w)")
            if grid[0] * grid[1] != T:
                raise ValueError(
                    f"grid={grid} does not match T={T} "
                    f"(grid[0] * grid[1] must equal the token count T)")
            # Positions in raster sequence-slot order, NOT permuted with the
            # token flip (faithful to the reference); prefix slots at the
            # identity rotation (zeros), image slots on the normalized 2D grid.
            M = getattr(conditioner, "M", 0)
            pos_img = get_positions(grid[0], grid[1], rope.pt_seq_len)
            pos = jnp.concatenate(
                [jnp.zeros((M, 2), dtype=pos_img.dtype), pos_img], axis=0)
            self.freqs_cis = Mask(rope(pos))            # (M+T, head_dim)
            self.pos_embed = None
        else:
            self.freqs_cis = None
            self.pos_embed = nnx.Param(
                (jax.random.normal(rngs.params(), (T, channels)) * 1e-2).astype(param_dtype))
        self.attn_blocks = nnx.List(
            [AttentionBlock(channels, num_heads, expansion, rngs,
                            param_dtype=param_dtype, dtype=dtype)
             for _ in range(num_layers)])
        self.proj_out = nnx.Linear(channels, 2 * F, rngs=rngs,
                                   param_dtype=param_dtype, dtype=dtype)
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
        sos = jnp.broadcast_to(
            self.sos_embed[...].astype(x_perm.dtype), (B, 1, self.F))
        x_in = jnp.concatenate([sos, x_perm[:, :-1]], axis=1)   # (B, T, F)
        h = self.proj_in(x_in)                                  # (B, T, C)
        if self.pos_embed is not None:
            h = h + self.pos_embed[...].astype(h.dtype)[None]
        freqs = self.freqs_cis[...] if self.freqs_cis is not None else None
        if bias is not None:
            h = h + bias[:, None, :]
        if prefix is not None:
            M = prefix.shape[1]
            h = jnp.concatenate([prefix, h], axis=1)            # (B, M+T, C)
            for blk in self.attn_blocks:
                h = blk(h, mask, freqs)
            h = h[:, M:]                                        # (B, T, C) strip
        else:
            for blk in self.attn_blocks:
                h = blk(h, None, freqs)
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

    def _forward_reference(self, z: Array, cond: Array | None = None):
        """Map noise to data via full recompute (reference path for the KV cache).

        ``forward`` is the production path (KV-cached); this method is its
        correctness oracle, retained for the equivalence test suite — do not
        delete it.

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

    def forward(self, z: Array, cond: Array | None = None):
        """Map noise to data (the sampling direction), KV-cached.

        Prefills the per-layer caches with the condition prefix (one parallel
        pass under the bidirectional prefix mask, matching the training-path
        mask rows), then scans over token positions decoding a single token
        per step against the caches. Verified equivalent to
        :meth:`_forward_reference` (full recompute) by the test suite.

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
        B = zp.shape[0]
        # Unlike _params/_forward_reference, the cached path builds its own
        # bidirectional prefix mask for the prefill below, so it embeds the
        # condition directly instead of via _embed_cond (which would also
        # build the (M+T)^2 prefix-LM mask, only to discard it here).
        bias, prefix = self.conditioner.embed(cond)
        M = prefix.shape[1] if prefix is not None else 0
        S = M + self.T
        nh = self.attn_blocks[0].num_heads
        hd = self.attn_blocks[0].head_dim
        L = len(self.attn_blocks)
        freqs = self.freqs_cis[...] if self.freqs_cis is not None else None

        k_caches = jnp.zeros((L, B, S, nh, hd), dtype=zp.dtype)
        v_caches = jnp.zeros_like(k_caches)

        # Prefill: prefix rows attend bidirectionally among themselves,
        # exactly as in the training prefix-LM mask (rows < M).
        if prefix is not None:
            h = prefix
            prefix_mask = jnp.ones((M, M), dtype=bool)
            pf = freqs[:M] if freqs is not None else None
            for layer, blk in enumerate(self.attn_blocks):
                h, k, v = blk(h, prefix_mask, pf, return_kv=True)
                k_caches = k_caches.at[layer, :, :M].set(k)
                v_caches = v_caches.at[layer, :, :M].set(v)

        sos = jnp.broadcast_to(self.sos_embed[...].astype(zp.dtype), (B, 1, self.F))

        def body(carry, t):
            x, k_caches, v_caches = carry
            prev = jax.lax.dynamic_slice_in_dim(x, jnp.maximum(t - 1, 0), 1,
                                                axis=1)
            x_in = jnp.where(t == 0, sos, prev)                # (B, 1, F)
            h = self.proj_in(x_in)                             # (B, 1, C)
            if self.pos_embed is not None:
                pe = jax.lax.dynamic_slice_in_dim(self.pos_embed[...], t, 1,
                                                  axis=0)
                h = h + pe.astype(h.dtype)[None]
            if bias is not None:
                h = h + bias[:, None, :]
            slot = M + t
            for layer, blk in enumerate(self.attn_blocks):
                h, k_l, v_l = blk.decode(h, k_caches[layer], v_caches[layer],
                                         slot, freqs)
                k_caches = k_caches.at[layer].set(k_l)
                v_caches = v_caches.at[layer].set(v_l)
            out = self.proj_out(h)                             # (B, 1, 2F)
            if self.soft_clip > 0:
                out = self.soft_clip * jnp.tanh(out / self.soft_clip)
            a, b = jnp.split(out, 2, axis=-1)                  # (B, 1, F)
            scale, _, log_scale = self._affine(a[:, 0])        # (B, F) fp32
            z_t = jax.lax.dynamic_slice_in_dim(zp, t, 1, axis=1)[:, 0]
            x_t = z_t.astype(jnp.float32) * scale \
                + b[:, 0].astype(jnp.float32)
            x = jax.lax.dynamic_update_slice_in_dim(
                x, x_t.astype(x.dtype)[:, None], t, axis=1)
            return (x, k_caches, v_caches), log_scale          # (B, F)

        x0 = jnp.zeros_like(zp)
        (x, _, _), log_scale_steps = jax.lax.scan(
            body, (x0, k_caches, v_caches), jnp.arange(self.T))
        logdet = jnp.sum(log_scale_steps, axis=(0, 2))         # (B,)
        x = x[:, self.inv_perm[...]]
        return x, logdet
