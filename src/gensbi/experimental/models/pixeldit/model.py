"""PixelDiT config and assembly.

PixelDiT = dual-level pixel-space DiT: a patch-level MMDiT transformer over
patch tokens (with joint cond attention) feeds per-patch conditioning into a
pixel-level PiT stack that refines every pixel inside each patch, for
conditional pixel-space flow matching on 2D fields.

Faithful port of ``reference/PixelDiT/pixdit_core/pixeldit_t2i.py`` (cond enters
via tokens only; ``c = silu(t_emb)``), minus repa / attention-mask / ``s``
caching (YAGNI).
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import jax.numpy as jnp
from flax import nnx
from jax.typing import DTypeLike

from gensbi.experimental.models.pixeldit.modules import (
    Buffer,
    FinalLayer,
    TimestepConditioner,
)
from gensbi.experimental.models.pixeldit.embedders import (
    CondTokenEmbedder,
    PatchTokenEmbedder,
    PixelTokenEmbedder,
    patchify,
    unpatchify,
)
from gensbi.experimental.models.pixeldit.blocks import MMDiTBlock, PiTBlock
from gensbi.experimental.models.pixeldit.rope import (
    precompute_freqs_cis_1d,
    precompute_freqs_cis_2d,
)


@dataclass
class PixelDiTParams:
    """Configuration for :class:`PixelDiT` (mirrors the style of ``FieldDiTParams``).

    Note: ``rngs`` is a live ``nnx.Rngs`` stream (mirrors ``FieldDiTParams`` /
    ``Flux1Params``). Constructing two models from the *same* params object
    yields *different* weights, because the stream advances; build a fresh
    ``PixelDiTParams`` (or a fresh ``nnx.Rngs(seed)``) per model for
    reproducibility.
    """

    in_channels: int
    field_shape: Tuple[int, int]
    cond_dim: int
    rngs: nnx.Rngs
    cond_in_channels: int = 1
    patch_size: int = 4
    hidden_size: int = 384
    pixel_hidden_size: int = 16
    patch_depth: int = 6
    pixel_depth: int = 2
    num_heads: int = 6
    pixel_attn_hidden_size: Optional[int] = None   # None -> hidden_size
    pixel_num_heads: Optional[int] = None           # None -> num_heads
    mlp_ratio: float = 4.0
    cond_id_embedding: str = "absolute"             # {"absolute", "pos1d", "none"}
    use_cond_rope: bool = True                      # reference-faithful default; False for unordered theta
    use_pixel_abs_pos: bool = True
    pit_post_modulation: bool = False
    zero_init_blocks: bool = True                   # c2i recipe; False = t2i recipe (final layer still zero)
    rope_scale: float = 16.0
    theta: float = 10_000.0
    param_dtype: DTypeLike = jnp.bfloat16

    def __post_init__(self):
        H, W = self.field_shape
        p = self.patch_size
        assert H % p == 0, f"field_shape H={H} must be divisible by patch_size {p}"
        assert W % p == 0, f"field_shape W={W} must be divisible by patch_size {p}"

        assert self.hidden_size % self.num_heads == 0, (
            f"hidden_size {self.hidden_size} must be divisible by num_heads {self.num_heads}"
        )

        self.resolved_pixel_attn_hidden_size = (
            self.pixel_attn_hidden_size
            if self.pixel_attn_hidden_size is not None
            else self.hidden_size
        )
        self.resolved_pixel_num_heads = (
            self.pixel_num_heads if self.pixel_num_heads is not None else self.num_heads
        )
        assert self.resolved_pixel_attn_hidden_size % self.resolved_pixel_num_heads == 0, (
            f"pixel_attn_hidden_size {self.resolved_pixel_attn_hidden_size} must be "
            f"divisible by pixel_num_heads {self.resolved_pixel_num_heads}"
        )

        patch_head_dim = self.hidden_size // self.num_heads
        pixel_head_dim = (
            self.resolved_pixel_attn_hidden_size // self.resolved_pixel_num_heads
        )
        assert patch_head_dim % 4 == 0, (
            f"patch head dim {patch_head_dim} must be divisible by 4 (2D rope needs dim/4 freqs)"
        )
        assert pixel_head_dim % 4 == 0, (
            f"pixel head dim {pixel_head_dim} must be divisible by 4 (2D rope needs dim/4 freqs)"
        )

        self.token_grid = (H // p, W // p)
        self.n_obs_tokens = self.token_grid[0] * self.token_grid[1]


class PixelDiT(nnx.Module):
    """Dual-level pixel-space DiT for conditional flow matching on 2D fields.

    Forward: ``(t, obs=field, cond) -> velocity field`` of the same shape.
    Patch-level MMDiT blocks attend over patch tokens jointly with cond tokens,
    producing per-patch conditioning ``s_cond``; pixel-level PiT blocks then
    refine every pixel inside each patch under that conditioning.
    """

    def __init__(self, params: PixelDiTParams):
        p = params
        H, W = p.field_shape
        ps = p.patch_size
        Hs, Ws = p.token_grid
        D = p.hidden_size
        attn_dim = p.resolved_pixel_attn_hidden_size

        self.s_embedder = PatchTokenEmbedder(
            p.in_channels * ps * ps,
            D,
            rngs=p.rngs,
            param_dtype=p.param_dtype,
        )
        self.pixel_embedder = PixelTokenEmbedder(
            p.in_channels,
            p.pixel_hidden_size,
            p.field_shape,
            ps,
            use_abs_pos=p.use_pixel_abs_pos,
            rngs=p.rngs,
            param_dtype=p.param_dtype,
        )
        self.cond_embedder = CondTokenEmbedder(
            p.cond_in_channels,
            D,
            p.cond_dim,
            id_embedding=p.cond_id_embedding,
            rngs=p.rngs,
            param_dtype=p.param_dtype,
        )
        self.t_conditioner = TimestepConditioner(
            D, rngs=p.rngs, param_dtype=p.param_dtype
        )
        self.patch_blocks = nnx.List([
            MMDiTBlock(
                D,
                p.num_heads,
                p.mlp_ratio,
                zero_init=p.zero_init_blocks,
                rngs=p.rngs,
                param_dtype=p.param_dtype,
            )
            for _ in range(p.patch_depth)
        ])
        self.pixel_blocks = nnx.List([
            PiTBlock(
                p.pixel_hidden_size,
                D,
                ps,
                attn_dim,
                p.resolved_pixel_num_heads,
                p.mlp_ratio,
                post_modulation=p.pit_post_modulation,
                zero_init=p.zero_init_blocks,
                rngs=p.rngs,
                param_dtype=p.param_dtype,
            )
            for _ in range(p.pixel_depth)
        ])
        self.final_layer = FinalLayer(
            p.pixel_hidden_size, p.in_channels, rngs=p.rngs, param_dtype=p.param_dtype
        )

        # Rope tables built once, stored as non-trainable Buffers (excluded from
        # Param state, immune to blanket float casts over Params).
        pe_patch = precompute_freqs_cis_2d(
            D // p.num_heads, Hs, Ws, p.theta, p.rope_scale
        )
        pe_pit = precompute_freqs_cis_2d(
            attn_dim // p.resolved_pixel_num_heads, Hs, Ws, p.theta, p.rope_scale
        )
        self.pe_patch = Buffer(pe_patch)
        self.pe_pit = Buffer(pe_pit)
        if p.use_cond_rope:
            pe_cond = precompute_freqs_cis_1d(D // p.num_heads, p.cond_dim, p.theta)
            self.pe_cond = Buffer(pe_cond)
        else:
            self.pe_cond = None

        # static primitives needed at call time (the dataclass itself is NOT
        # stored: it holds Rngs/derived arrays and would poison the GraphDef)
        self.field_shape = tuple(p.field_shape)
        self.in_channels = p.in_channels
        self.cond_dim = p.cond_dim          # number of cond tokens K
        self.cond_in_channels = p.cond_in_channels
        self.patch_size = ps
        self.token_grid = tuple(p.token_grid)
        self.param_dtype = p.param_dtype

    def __call__(
        self,
        t,
        obs,
        cond,
        obs_ids=None,      # accepted & ignored (pipeline compatibility)
        cond_ids=None,     # accepted & ignored
        conditioned=True,  # only True supported (CFG/null-cond is deferred)
        guidance=None,
    ):
        if conditioned is not True:
            raise NotImplementedError(
                "PixelDiT has no unconditional path yet (CFG / null-conditioning "
                f"is deferred work); got conditioned={conditioned!r}"
            )
        if guidance is not None:
            raise ValueError(
                "PixelDiT has no guidance embedding; got "
                f"guidance={guidance!r}"
            )

        obs = jnp.asarray(obs, dtype=self.param_dtype)
        cond = jnp.asarray(cond, dtype=self.param_dtype)

        if obs.ndim != 4:
            raise ValueError(
                f"obs must be rank-4 (B, H, W, C), got rank {obs.ndim} with shape {obs.shape}"
            )
        if obs.shape[1:3] != self.field_shape:
            raise ValueError(
                f"obs spatial shape {obs.shape[1:3]} does not match "
                f"field_shape {self.field_shape}"
            )
        if obs.shape[-1] != self.in_channels:
            raise ValueError(
                f"obs has {obs.shape[-1]} channels, expected in_channels={self.in_channels}"
            )

        if cond.ndim not in (2, 3):
            raise ValueError(
                f"cond must be rank-2 (B, K) or rank-3 (B, K, C), "
                f"got rank {cond.ndim} with shape {cond.shape}"
            )
        cond_K = cond.shape[1]
        if cond_K != self.cond_dim:
            raise ValueError(
                f"cond has {cond_K} tokens (axis 1), expected cond_dim={self.cond_dim}; "
                f"cond.shape={cond.shape}"
            )
        if cond.ndim == 3 and cond.shape[-1] != self.cond_in_channels:
            raise ValueError(
                f"cond has {cond.shape[-1]} channels (axis 2), "
                f"expected cond_in_channels={self.cond_in_channels}; "
                f"cond.shape={cond.shape}"
            )

        t = jnp.asarray(t, dtype=jnp.float32)
        # FieldConditionalWrapper expands t from (B,) to (B,1) via _expand_time;
        # PixelDiT's _timestep_embedding expects a 1-D (B,) vector (unlike Flux1
        # whose timestep_embedding ravels internally).  Normalise at this boundary
        # so the faithful port stays untouched.
        t = t.ravel()

        B = obs.shape[0]
        p = self.patch_size
        C = self.in_channels
        Hs, Ws = self.token_grid
        L = Hs * Ws

        pe_patch = self.pe_patch.get_value()
        pe_pit = self.pe_pit.get_value()
        pe_cond = None if self.pe_cond is None else self.pe_cond.get_value()

        t_emb = self.t_conditioner(t)[:, None, :]          # (B, 1, D)
        cond_tokens = self.cond_embedder(cond)             # (B, K, D)
        c = nnx.silu(t_emb)                                # t2i: cond enters via tokens only
        s = self.s_embedder(patchify(obs, p))              # (B, L, D)
        for blk in self.patch_blocks:
            s, cond_tokens = blk(s, cond_tokens, c, pe_patch, pe_cond)
        s = nnx.silu(t_emb + s)                            # (B, L, D)

        D = s.shape[-1]
        s_cond = s.reshape(B * L, D)                       # row-major (B, L)
        x_pix = self.pixel_embedder(obs)                   # (B*L, p^2, D_pix)
        for blk in self.pixel_blocks:
            x_pix = blk(x_pix, s_cond, pe_pit, batch=B)
        x_pix = self.final_layer(x_pix)                    # (B*L, p^2, C)
        return unpatchify(x_pix.reshape(B, L, p * p * C), self.token_grid, p, C)
