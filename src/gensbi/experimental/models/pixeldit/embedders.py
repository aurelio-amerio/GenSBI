"""Token embedders for the PixelDiT port.

Faithful port of ``reference/PixelDiT/pixdit_core/pixeldit_c2i.py`` embedders
plus the cond-token embedder from ``pixeldit_t2i.py``.  Operates channel-last
``(B, H, W, C)`` throughout.
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import DTypeLike

from gensbi.experimental.models.pixeldit.modules import Buffer, get_2d_sincos_pos_embed
from gensbi.models.embedding import FeatureEmbedder


# ---------------------------------------------------------------------------
# Module-level helpers: patchify / unpatchify
# ---------------------------------------------------------------------------


def patchify(x: jax.Array, p: int) -> jax.Array:
    """Fold a channel-last image into patch tokens, row-major.

    Parameters
    ----------
    x:
        ``(B, H, W, C)`` channel-last image.
    p:
        Patch size (both spatial dims).

    Returns
    -------
    jax.Array
        ``(B, L, p²·C)`` where ``L = Hs*Ws``, patches in row-major order
        ``(Hs, Ws)`` and pixels within each patch in row-major order ``(p, p)``.
    """
    B, H, W, C = x.shape
    Hs, Ws = H // p, W // p
    x = x.reshape(B, Hs, p, Ws, p, C)
    x = x.transpose(0, 1, 3, 2, 4, 5)   # (B, Hs, Ws, p, p, C)
    x = x.reshape(B, Hs * Ws, p * p * C)
    return x


def unpatchify(tokens: jax.Array, grid: tuple[int, int], p: int, C: int) -> jax.Array:
    """Exact inverse of :func:`patchify`.

    Parameters
    ----------
    tokens:
        ``(B, L, p²·C)`` patch tokens.
    grid:
        ``(Hs, Ws)`` — number of patches along each spatial axis.
    p:
        Patch size.
    C:
        Number of output channels.

    Returns
    -------
    jax.Array
        ``(B, H, W, C)`` channel-last image.
    """
    Hs, Ws = grid
    B = tokens.shape[0]
    x = tokens.reshape(B, Hs, Ws, p, p, C)
    x = x.transpose(0, 1, 3, 2, 4, 5)   # (B, Hs, p, Ws, p, C)
    x = x.reshape(B, Hs * p, Ws * p, C)
    return x


# ---------------------------------------------------------------------------
# PatchTokenEmbedder (ref pixeldit_c2i.py:21-38)
# ---------------------------------------------------------------------------


class PatchTokenEmbedder(nnx.Module):
    """Linear patch-token embedder (ref ``PatchTokenEmbedder``, pixeldit_c2i.py:21-38).

    ``Linear(in_features → hidden_size, bias=True)``; kernel xavier_uniform,
    bias zeros.  No norm layer (``norm_layer=None`` case).
    """

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        *,
        rngs: nnx.Rngs,
        dtype: DTypeLike = jnp.bfloat16,
        param_dtype: DTypeLike = jnp.float32,
    ):
        self.proj = nnx.Linear(
            in_features,
            hidden_size,
            use_bias=True,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.glorot_uniform(),
            bias_init=jax.nn.initializers.zeros,
        )

    def __call__(self, x):
        return self.proj(x)


# ---------------------------------------------------------------------------
# PixelTokenEmbedder (ref pixeldit_c2i.py:60-111)
# ---------------------------------------------------------------------------


class PixelTokenEmbedder(nnx.Module):
    """Per-pixel projection + optional sincos abs-pos + patch grouping.

    Faithful port of ``PixelTokenEmbedder.forward`` (pixeldit_c2i.py:93-111).
    Operates channel-last: input ``(B, H, W, C)``, output ``(B·L, p², D_pix)``.

    Parameters
    ----------
    in_channels:
        Number of input channels ``C``.
    pixel_hidden_size:
        Per-pixel hidden dimension ``D_pix``.
    field_shape:
        ``(H, W)`` — fixed spatial resolution; the sincos table is precomputed
        and stored as a non-trainable :class:`Buffer`.
    patch_size:
        Patch size ``p``.
    use_abs_pos:
        If ``True`` (default), add the sincos 2D positional embedding.
    """

    def __init__(
        self,
        in_channels: int,
        pixel_hidden_size: int,
        field_shape: tuple[int, int],
        patch_size: int,
        *,
        use_abs_pos: bool = True,
        rngs: nnx.Rngs,
        dtype: DTypeLike = jnp.bfloat16,
        param_dtype: DTypeLike = jnp.float32,
    ):
        self.pixel_hidden_size = pixel_hidden_size
        self.patch_size = patch_size
        self.use_abs_pos = use_abs_pos
        self.field_shape = field_shape

        self.proj = nnx.Linear(
            in_channels,
            pixel_hidden_size,
            use_bias=True,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
        )

        H, W = field_shape
        pos_np = get_2d_sincos_pos_embed(pixel_hidden_size, H, W)  # (H*W, D_pix) float32
        pos_np = pos_np.reshape(H, W, pixel_hidden_size)           # (H, W, D_pix)
        self.pos_embed = Buffer(jnp.array(pos_np))                  # non-trainable

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Parameters
        ----------
        x:
            ``(B, H, W, C)`` channel-last image.

        Returns
        -------
        jax.Array
            ``(B·L, p², D_pix)`` grouped pixel tokens.
        """
        B, H, W, C = x.shape
        exp_H, exp_W = self.field_shape
        if (H, W) != (exp_H, exp_W):
            raise ValueError(
                f"PixelTokenEmbedder was built for field_shape=({exp_H}, {exp_W}) "
                f"but received input with (H, W)=({H}, {W})."
            )
        p = self.patch_size
        Hs, Ws = H // p, W // p
        D_pix = self.pixel_hidden_size

        x = self.proj(x)  # (B, H, W, D_pix)

        if self.use_abs_pos:
            pos = self.pos_embed.get_value().astype(x.dtype)   # (H, W, D_pix)
            x = x + pos[None, :, :, :]

        # Group into (B·L, p², D_pix) — row-major patch order, row-major pixel order
        x = x.reshape(B, Hs, p, Ws, p, D_pix)
        x = x.transpose(0, 1, 3, 2, 4, 5)          # (B, Hs, Ws, p, p, D_pix)
        x = x.reshape(B * Hs * Ws, p * p, D_pix)
        return x


# ---------------------------------------------------------------------------
# CondTokenEmbedder (ref y_embedder + y_pos_embedding, pixeldit_t2i.py:179-180)
# ---------------------------------------------------------------------------


class CondTokenEmbedder(nnx.Module):
    """Condition token embedder: linear → RMSNorm → add id embedding.

    Faithful port of the t2i cond pipeline
    (``y_embedder`` + ``y_pos_embedding``, pixeldit_t2i.py:179-180, 267-268).

    Parameters
    ----------
    cond_in_channels:
        Dimension of each condition token ``D_c``.
    hidden_size:
        Output embedding dimension ``D``.
    n_tokens:
        Number of condition tokens ``K`` (used to build the id embedding table).
    id_embedding:
        How to embed token positions: ``"absolute"`` (learned), ``"pos1d"``
        (sinusoidal 1D), or ``"none"`` (no positional information added).
    """

    def __init__(
        self,
        cond_in_channels: int,
        hidden_size: int,
        n_tokens: int,
        *,
        id_embedding: str = "absolute",
        rngs: nnx.Rngs,
        dtype: DTypeLike = jnp.bfloat16,
        param_dtype: DTypeLike = jnp.float32,
    ):
        self.cond_in_channels = cond_in_channels

        self.proj = nnx.Linear(
            cond_in_channels,
            hidden_size,
            use_bias=True,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        # fp32 island: the norm's own output is fp32 regardless of the
        # compute-dtype knob. Unlike a norm that feeds straight into another
        # compute-dtype Linear (whose promote_dtype would downcast it for
        # free), this norm's output is returned/added directly further down
        # -- so __call__ below downcasts it back explicitly to avoid leaking
        # fp32 into the cond-token stream (same failure mode as the
        # flux1joint condition_embedding bug).
        self.norm = nnx.RMSNorm(
            hidden_size,
            epsilon=1e-6,
            rngs=rngs,
            dtype=jnp.float32,
            param_dtype=param_dtype,
        )

        self.id_embedding_kind = id_embedding
        if id_embedding != "none":
            # Reference (pixeldit_t2i.py:180) inits the learned id embedding as
            # ``torch.randn`` (std 1.0), co-equal with the RMS-normed value.
            # nnx.Embed's default init is variance_scaling -> std ~1/sqrt(hidden),
            # ~sqrt(hidden)x too small, which would leave token-identity ~20x
            # under-weighted at init.  Match the reference for the learned path.
            extra = {}
            if id_embedding == "absolute":
                extra["embedding_init"] = nnx.initializers.normal(stddev=1.0)
            self.id_embedder = FeatureEmbedder(
                num_embeddings=n_tokens,
                hidden_size=hidden_size,
                kind=id_embedding,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
                **extra,
            )
            self._n_tokens = n_tokens

    def __call__(self, cond: jax.Array) -> jax.Array:
        """
        Parameters
        ----------
        cond:
            ``(B, K, D_c)`` condition tokens; or ``(B, K)`` when
            ``cond_in_channels == 1`` (auto-expanded to ``(B, K, 1)``).

        Returns
        -------
        jax.Array
            ``(B, K, D)`` embedded condition tokens.
        """
        if cond.ndim == 2:
            if self.cond_in_channels != 1:
                raise ValueError(
                    f"2D cond (B, k) is only valid when cond_in_channels == 1; "
                    f"this embedder has cond_in_channels={self.cond_in_channels} — "
                    f"pass cond as (B, k, {self.cond_in_channels})"
                )
            cond = cond[..., None]

        x = self.proj(cond)   # (B, K, D), compute dtype
        # norm is an fp32 island; downcast its raw output back to the
        # compute dtype so it doesn't leak fp32 into the cond-token stream
        # (see the fp32-island note on ``self.norm`` in __init__).
        x = self.norm(x).astype(x.dtype)

        if self.id_embedding_kind != "none":
            # absolute: nnx.Embed requires (..., 1) index; pos1d: requires (...,) index.
            ids = jnp.arange(self._n_tokens)[None, :]             # (1, K)
            if self.id_embedding_kind == "absolute":
                ids = ids[..., None]                              # (1, K, 1)
            id_emb = self.id_embedder(ids)[0]                    # (K, D)
            x = x + id_emb[None, :, :].astype(x.dtype)

        return x
