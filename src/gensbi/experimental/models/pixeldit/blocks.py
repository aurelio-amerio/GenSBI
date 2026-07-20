"""Transformer blocks for the PixelDiT port: patch-level MMDiT and pixel-level PiT.

:class:`MMDiTBlock` is a faithful port of ``MMDiTBlockT2I`` +
``MMDiTJointAttention`` (``reference/PixelDiT/pixdit_core/pixeldit_t2i.py:19-132``)
collapsed into a single flax.nnx module. :class:`PiTBlock` ports the
pixel-level block (``pixeldit_c2i.py:114-187``); see its docstring.

Two streams ``x`` (obs patches, ``(B, Lx, D)``) and ``y`` (cond, ``(B, Ly, D)``)
each get their own qkv / proj / norms / mlp / adaLN, then share one joint
scaled-dot-product attention over the concatenated token axis (cond first,
matching the reference ``torch.cat([y, x])`` order).

Rope is applied *per stream before* the joint concat, because the two streams
carry different rope tables (a 2D grid for obs, a 1D table — or none — for
cond). The library's usual ``attention(pe=...)`` pattern applies one ``pe`` to
the whole concatenated sequence, which cannot express per-stream tables; so we
call :func:`flux1.math.apply_rope` ourselves on each stream and then invoke
``attention(..., pe=None)``, which reduces to plain sdpa plus the output
rearrange — math identical to the reference.

The adaLN projections are *plain* linears with **no internal silu**: ``c``
arrives pre-activated, so reusing flux1's ``Modulation`` (which applies silu
inside) would double-activate. ``zero_init`` zero-initialises the adaLN
kernel+bias so the gated residuals start as an exact identity (c2i recipe);
``zero_init=False`` uses default init (t2i recipe).

No attention-mask support: the cond length is fixed (YAGNI).
"""

import jax
import jax.numpy as jnp
from einops import rearrange
from flax import nnx
from jax.typing import DTypeLike

from gensbi.models.flux1.layers import QKNorm, SelfAttention
from gensbi.models.flux1.math import apply_rope, attention

from .modules import PixelMLP, SwiGLU


class MMDiTBlock(nnx.Module):
    """Joint-attention MMDiT block with per-stream rope (ref lines 19-132)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        *,
        zero_init: bool = True,
        rngs: nnx.Rngs,
        dtype: DTypeLike = jnp.bfloat16,
        param_dtype: DTypeLike = jnp.float32,
    ):
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        D = hidden_size

        # --- per-stream norms (fp32 island; each feeds a compute Linear next,
        # whose own promote_dtype performs the downcast) ---
        self.norm_x1 = nnx.RMSNorm(D, epsilon=1e-6, rngs=rngs, dtype=jnp.float32, param_dtype=param_dtype)
        self.norm_y1 = nnx.RMSNorm(D, epsilon=1e-6, rngs=rngs, dtype=jnp.float32, param_dtype=param_dtype)
        self.norm_x2 = nnx.RMSNorm(D, epsilon=1e-6, rngs=rngs, dtype=jnp.float32, param_dtype=param_dtype)
        self.norm_y2 = nnx.RMSNorm(D, epsilon=1e-6, rngs=rngs, dtype=jnp.float32, param_dtype=param_dtype)

        # --- per-stream qkv (no bias) ---
        self.qkv_x = nnx.Linear(D, 3 * D, use_bias=False, rngs=rngs, dtype=dtype, param_dtype=param_dtype)
        self.qkv_y = nnx.Linear(D, 3 * D, use_bias=False, rngs=rngs, dtype=dtype, param_dtype=param_dtype)

        # --- per-stream per-head q/k RMSNorm (over head_dim) ---
        self.qk_norm_x = QKNorm(self.head_dim, rngs=rngs, dtype=dtype, param_dtype=param_dtype)
        self.qk_norm_y = QKNorm(self.head_dim, rngs=rngs, dtype=dtype, param_dtype=param_dtype)

        # --- per-stream output projection (with bias) ---
        self.proj_x = nnx.Linear(D, D, use_bias=True, rngs=rngs, dtype=dtype, param_dtype=param_dtype)
        self.proj_y = nnx.Linear(D, D, use_bias=True, rngs=rngs, dtype=dtype, param_dtype=param_dtype)

        # --- per-stream SwiGLU MLPs ---
        self.mlp_x = SwiGLU(D, mlp_ratio, rngs=rngs, dtype=dtype, param_dtype=param_dtype)
        self.mlp_y = SwiGLU(D, mlp_ratio, rngs=rngs, dtype=dtype, param_dtype=param_dtype)

        # --- per-stream adaLN: plain Linear D -> 6D, NO internal silu ---
        # c arrives pre-activated; do NOT reuse flux1.Modulation (silu inside).
        if zero_init:
            adaln_kwargs = dict(
                kernel_init=jax.nn.initializers.zeros,
                bias_init=jax.nn.initializers.zeros,
            )
        else:
            adaln_kwargs = {}
        self.adaLN_x = nnx.Linear(
            D, 6 * D, use_bias=True, rngs=rngs, dtype=dtype, param_dtype=param_dtype, **adaln_kwargs
        )
        self.adaLN_y = nnx.Linear(
            D, 6 * D, use_bias=True, rngs=rngs, dtype=dtype, param_dtype=param_dtype, **adaln_kwargs
        )

    def _qkv(self, qkv_lin, qk_norm, h, pe):
        """Project, split heads, q/k-norm, and (optionally) apply per-stream rope."""
        qkv = qkv_lin(h)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = qk_norm(q, k, v)
        if pe is not None:
            q, k = apply_rope(q, k, pe)
        return q, k, v

    def __call__(self, x, y, c, pe_x, pe_y=None):
        # Each adaLN_*(c) -> 6 modulation pieces of shape (B, 1, D).
        shift_msa_x, scale_msa_x, gate_msa_x, shift_mlp_x, scale_mlp_x, gate_mlp_x = (
            jnp.split(self.adaLN_x(c), 6, axis=-1)
        )
        shift_msa_y, scale_msa_y, gate_msa_y, shift_mlp_y, scale_mlp_y, gate_mlp_y = (
            jnp.split(self.adaLN_y(c), 6, axis=-1)
        )

        # Modulated pre-attention norms.
        x_norm = self.norm_x1(x) * (1 + scale_msa_x) + shift_msa_x
        y_norm = self.norm_y1(y) * (1 + scale_msa_y) + shift_msa_y

        # Per-stream qkv + rope (rope applied here, before the joint concat).
        q_x, k_x, v_x = self._qkv(self.qkv_x, self.qk_norm_x, x_norm, pe_x)
        q_y, k_y, v_y = self._qkv(self.qkv_y, self.qk_norm_y, y_norm, pe_y)

        # Joint attention over [cond, obs] along the token axis (ref order).
        q = jnp.concatenate((q_y, q_x), axis=2)
        k = jnp.concatenate((k_y, k_x), axis=2)
        v = jnp.concatenate((v_y, v_x), axis=2)

        # pe=None: rope already applied per stream, so attention is pure sdpa
        # plus the "B L (H D)" output rearrange.
        attn = attention(q, k, v, pe=None)
        Ly = y.shape[1]
        attn_y, attn_x = attn[:, :Ly], attn[:, Ly:]

        # Gated attention residual.
        x = x + gate_msa_x * self.proj_x(attn_x)
        y = y + gate_msa_y * self.proj_y(attn_y)

        # Gated SwiGLU residual.
        x = x + gate_mlp_x * self.mlp_x(self.norm_x2(x) * (1 + scale_mlp_x) + shift_mlp_x)
        y = y + gate_mlp_y * self.mlp_y(self.norm_y2(y) * (1 + scale_mlp_y) + shift_mlp_y)

        return x, y


class PiTBlock(nnx.Module):
    """Pixel-level DiT block: pixel-wise AdaLN + token compaction (ref
    ``pixeldit_c2i.py:114-187``).

    Operates on per-patch pixel tokens ``x`` of shape ``(B*L, p^2, D_pix)``
    with per-patch conditioning ``s_cond`` of shape ``(B*L, D)``. The adaLN
    projection maps ``D -> n_mod * D_pix * p^2`` and is reshaped to
    ``(B*L, p^2, n_mod * D_pix)``, so every pixel in the patch receives its
    own modulation parameters ("pixel-wise AdaLN").

    Attention runs on *compacted* tokens: each patch's ``p^2 * D_pix`` pixels
    are compressed to a single ``attn_dim`` token, attended globally over the
    ``(B, L)`` patch grid with 2D rope, then expanded back. The attention
    itself is flux1's :class:`SelfAttention`, which matches the reference
    ``RotaryAttention`` (no-bias qkv, per-head RMSNorm on q/k, biased proj).

    ``post_modulation`` selects the gate-free post-modulation variant
    (``n_mod = 4``, chunk order ``scale1, shift1, scale2, shift2`` — scale
    before shift, ref line 168). The default pre-variant uses the standard
    six-chunk DiT order and is an exact identity at ``zero_init=True``; the
    post variant is *not* (no gates close the residuals).

    ``pe`` is the output of
    ``precompute_freqs_cis_2d(attn_dim // num_heads, grid_h, grid_w, ...)``
    over the patch grid, shape ``(1, 1, L, head_dim/2, 2, 2)``.
    """

    def __init__(
        self,
        pixel_dim: int,
        context_dim: int,
        patch_size: int,
        attn_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        post_modulation: bool = False,
        *,
        zero_init: bool = True,
        rngs: nnx.Rngs,
        dtype: DTypeLike = jnp.bfloat16,
        param_dtype: DTypeLike = jnp.float32,
    ):
        assert attn_dim % num_heads == 0, "attn_dim must be divisible by num_heads"
        self.pixel_dim = pixel_dim
        self.patch_size = patch_size
        self.attn_dim = attn_dim
        self.post_modulation = post_modulation
        self.n_mod = 4 if post_modulation else 6
        p2 = patch_size * patch_size

        # fp32 island; each feeds a compute Linear next (compress/mlp.fc1),
        # whose own promote_dtype performs the downcast.
        self.norm1 = nnx.RMSNorm(pixel_dim, epsilon=1e-6, rngs=rngs, dtype=jnp.float32, param_dtype=param_dtype)
        self.norm2 = nnx.RMSNorm(pixel_dim, epsilon=1e-6, rngs=rngs, dtype=jnp.float32, param_dtype=param_dtype)

        self.compress = nnx.Linear(
            p2 * pixel_dim, attn_dim, use_bias=True, rngs=rngs, dtype=dtype, param_dtype=param_dtype
        )
        self.expand = nnx.Linear(
            attn_dim, p2 * pixel_dim, use_bias=True, rngs=rngs, dtype=dtype, param_dtype=param_dtype
        )

        self.attn = SelfAttention(
            dim=attn_dim,
            num_heads=num_heads,
            qkv_features=attn_dim,
            qkv_bias=False,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
        )

        self.mlp = PixelMLP(pixel_dim, mlp_ratio, rngs=rngs, dtype=dtype, param_dtype=param_dtype)

        # Plain linear, no internal silu (c arrives pre-activated; see MMDiTBlock note).
        if zero_init:
            adaln_kwargs = dict(
                kernel_init=jax.nn.initializers.zeros,
                bias_init=jax.nn.initializers.zeros,
            )
        else:
            adaln_kwargs = {}
        self.adaLN = nnx.Linear(
            context_dim,
            self.n_mod * pixel_dim * p2,
            use_bias=True,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            **adaln_kwargs,
        )

    def __call__(self, x, s_cond, pe, batch):
        BL, P2, D = x.shape
        if D != self.pixel_dim:
            raise ValueError(
                f"PiTBlock expected pixel_dim={self.pixel_dim}, got {D}"
            )
        if BL % batch != 0:
            raise ValueError(
                f"BL={BL} is not divisible by batch={batch}"
            )
        L = BL // batch

        # (BL, n_mod*D_pix*p^2) -> (BL, p^2, n_mod*D_pix): each pixel gets its
        # own modulation parameters (ref lines 166-168).
        cond_params = self.adaLN(s_cond).reshape(BL, P2, self.n_mod * self.pixel_dim)

        if self.post_modulation:
            # Ref line 168: scale-before-shift in the post variant.
            scale1, shift1, scale2, shift2 = jnp.split(cond_params, 4, axis=-1)
            x_norm = self.norm1(x)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(
                cond_params, 6, axis=-1
            )
            x_norm = (1 + scale_msa) * self.norm1(x) + shift_msa

        # Token compaction: one attn token per patch, global attention over
        # the (B, L) patch grid, then expand back to pixels.
        x_comp = self.compress(x_norm.reshape(BL, P2 * self.pixel_dim))
        x_comp = x_comp.reshape(batch, L, self.attn_dim)
        attn_out = self.attn(x_comp, pe=pe)
        attn_exp = self.expand(attn_out.reshape(BL, self.attn_dim))
        attn_exp = attn_exp.reshape(BL, P2, self.pixel_dim)

        if self.post_modulation:
            x = x + attn_exp * (1 + scale1) + shift1
            x = x + self.mlp(self.norm2(x)) * (1 + scale2) + shift2
        else:
            x = x + gate_msa * attn_exp
            x = x + gate_mlp * self.mlp((1 + scale_mlp) * self.norm2(x) + shift_mlp)

        return x
