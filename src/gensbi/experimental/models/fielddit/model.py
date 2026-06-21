"""FieldDiT config and assembly.

FieldDiT = conv U-Net (ObsEncoder/ObsDecoder) with an MMDiT transformer
bottleneck, for conditional pixel-space flow matching on 2D fields.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import DTypeLike

from gensbi.recipes.utils import init_ids_1d, init_ids_2d
from gensbi.models.flux1.layers import MLPEmbedder, timestep_embedding, Identity

from gensbi.experimental.models.fielddit.codec import (
    ObsEncoder,
    ObsDecoder,
    Tokenizer,
    Untokenizer,
)
from gensbi.experimental.models.fielddit.cond import ScalarCondEmbedder
from gensbi.experimental.models.fielddit.core import MMDiTCore


class RopeIds(nnx.Variable):
    """Integer rope/positional id buffers.

    A dedicated Variable type so the ids are (a) filterable with
    ``nnx.state(model, RopeIds)``, (b) excluded from ``nnx.Param`` state, and
    (c) safe from blanket float-dtype casts applied to the parameter state.
    """


@dataclass
class FieldDiTParams:
    """Configuration for :class:`FieldDiT` (mirrors the style of ``Flux1Params``).

    The meeting-grid token count is **derived** from ``field_shape``,
    ``encoder_widths`` (depth) and ``patch_size`` — it is not prescribed:
    ``tokens = (H / (2**D * p)) * (W / (2**D * p))`` with ``D = len(encoder_widths) - 1``.

    Note: ``rngs`` is a live ``nnx.Rngs`` stream (mirrors ``Flux1Params``).
    Constructing two models from the *same* params object yields *different*
    weights, because the stream advances; build a fresh ``FieldDiTParams``
    (or a fresh ``nnx.Rngs(seed)``) per model for reproducibility.
    """

    in_channels: int
    field_shape: Tuple[int, int]
    encoder_widths: Tuple[int, ...]
    cond_dim: int
    rngs: nnx.Rngs
    cond_in_channels: int = 1
    res_blocks_down: int = 2
    res_blocks_up: int = 2
    patch_size: int = 2
    num_heads: int = 12
    axes_dim: Optional[List[int]] = None
    mlp_ratio: float = 4.0
    depth: int = 2
    depth_single_blocks: int = 2
    qkv_bias: bool = False
    theta: Optional[int] = None  # None -> min(10 * (n_obs_tokens + cond_dim), 10_000)
    use_cond_summary_in_vec: bool = True
    cond_modulates_encoder: bool = False  # ON: encoder gets the full vec (symmetric with decoder); OFF: time-only encoder, shareable across CFG branches
    norm_groups: int = 8
    guidance_embed: bool = False
    vec_in_dim: Optional[int] = None  # input dim for guidance MLP (required iff guidance_embed=True)
    param_dtype: DTypeLike = jnp.bfloat16

    def __post_init__(self):
        if self.axes_dim is None:
            # (semantic, h, w) split of the rope head dims. In Phase 1 the
            # semantic axis is identical (unrotated) for every token: obs uses
            # semantic_id=0 and cond tokens use learned absolute embeddings
            # with zero rope ids — benign, the same situation as Flux1
            # txt/img. The semantic-dim budget and the 1D/2D id axis-order
            # unification are revisited wholesale in the Phase-2
            # co-tokenization design (see the 2026-06-10 phase-1.5 spec §2.6).
            self.axes_dim = [16, 24, 24]
        assert len(self.axes_dim) == 3, "axes_dim must have 3 entries (semantic, h, w)"
        for a in self.axes_dim:
            assert a % 2 == 0, f"each axes_dim entry must be even for rope, got {self.axes_dim}"
        self.axes_dim = tuple(self.axes_dim)

        self.hidden_size = int(sum(self.axes_dim) * self.num_heads)
        self.depth_levels = len(self.encoder_widths) - 1

        H, W = self.field_shape
        factor = 2 ** self.depth_levels
        assert H % factor == 0 and W % factor == 0, (
            f"field_shape {self.field_shape} must be divisible by 2**D={factor}"
        )
        self.feat_h = H // factor
        self.feat_w = W // factor

        p = self.patch_size
        assert self.feat_h % p == 0 and self.feat_w % p == 0, (
            f"meeting grid ({self.feat_h},{self.feat_w}) must be divisible by patch_size {p}"
        )
        self.token_grid = (self.feat_h // p, self.feat_w // p)
        self.n_obs_tokens = self.token_grid[0] * self.token_grid[1]

        if self.theta is None:
            # rule of thumb: 10x the joint token count, capped at the
            # literature default; rope frequency coverage then matches the
            # actual grid instead of assuming ~10k positions
            self.theta = min(10 * (self.n_obs_tokens + self.cond_dim), 10_000)


class FieldDiT(nnx.Module):
    """Conditional flow-matching network for 2D fields (pixel space).

    Forward: ``(t, obs=field, cond) -> velocity field`` of the same shape. The
    conv encoder is modulated by time only (or by the full ``vec`` when
    ``cond_modulates_encoder=True``); the decoder and the MMDiT core are
    modulated by ``vec = time (+ cond summary if flagged-C) (+ guidance)``. The
    meeting-grid rope2d obs ids and absolute cond ids are built internally.
    """

    def __init__(self, params: FieldDiTParams):
        p = params
        hid = p.hidden_size

        self.encoder = ObsEncoder(
            p.in_channels, p.encoder_widths, p.res_blocks_down,
            vec_dim=hid, norm_groups=p.norm_groups, rngs=p.rngs, param_dtype=p.param_dtype,
        )
        c_bottleneck = p.encoder_widths[-1]
        self.tokenizer = Tokenizer(
            c_bottleneck, p.patch_size, hid, rngs=p.rngs, param_dtype=p.param_dtype
        )
        self.cond_embedder = ScalarCondEmbedder(
            p.cond_in_channels, hid, rngs=p.rngs, param_dtype=p.param_dtype
        )
        self.core = MMDiTCore(
            hid, p.num_heads, p.mlp_ratio, p.depth, p.depth_single_blocks,
            axes_dim=p.axes_dim, theta=p.theta, n_cond_tokens=p.cond_dim,
            qkv_bias=p.qkv_bias, rngs=p.rngs, param_dtype=p.param_dtype,
        )
        self.untokenizer = Untokenizer(
            c_bottleneck, p.patch_size, hid, rngs=p.rngs, param_dtype=p.param_dtype
        )
        self.decoder = ObsDecoder(
            p.in_channels, p.encoder_widths, p.res_blocks_up,
            vec_dim=hid, norm_groups=p.norm_groups, rngs=p.rngs, param_dtype=p.param_dtype,
        )
        self.time_in = MLPEmbedder(
            in_dim=256, hidden_dim=hid, rngs=p.rngs, param_dtype=p.param_dtype
        )
        if p.guidance_embed:
            assert p.vec_in_dim is not None, "vec_in_dim required when guidance_embed=True"
            self.guidance_in = MLPEmbedder(
                p.vec_in_dim, hid, rngs=p.rngs, param_dtype=p.param_dtype
            )
        else:
            self.guidance_in = Identity()

        # static primitives needed at call time (the dataclass itself is NOT
        # stored: it holds Rngs/derived arrays and would poison the GraphDef)
        self.field_shape = tuple(p.field_shape)
        self.in_channels = p.in_channels
        self.cond_dim = p.cond_dim
        self.use_cond_summary_in_vec = p.use_cond_summary_in_vec
        self.cond_modulates_encoder = p.cond_modulates_encoder
        self.guidance_embed = p.guidance_embed
        self.param_dtype = p.param_dtype
        self.token_grid = tuple(p.token_grid)

        # rope id buffers (int32) — built here, kept out of Param state
        obs_ids, _ = init_ids_2d((p.feat_h, p.feat_w), semantic_id=0, size=p.patch_size)
        cond_ids, _ = init_ids_1d(p.cond_dim, semantic_id=None)
        self.obs_ids = RopeIds(obs_ids)
        self.cond_ids = RopeIds(cond_ids)

    def __call__(
        self,
        t,
        obs,
        cond,
        obs_ids=None,      # accepted & ignored (ids built internally)
        cond_ids=None,     # accepted & ignored
        conditioned=True,  # only True supported (CFG/null-cond is deferred)
        guidance=None,
    ):
        if conditioned is not True:
            raise NotImplementedError(
                "FieldDiT has no unconditional path yet (CFG / null-conditioning "
                f"is deferred work); got conditioned={conditioned!r}"
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

        # timestep sinusoid in f32 (bf16 t quantizes t*1000 to ~2.0 ulp);
        # cast only the finished embedding to the model dtype
        t = jnp.asarray(t, dtype=jnp.float32)
        time_vec = self.time_in(
            timestep_embedding(t, 256).astype(self.param_dtype)
        )  # (B, hidden)

        cond_tokens, summary = self.cond_embedder(cond)  # (B, k, hidden), (B, hidden)
        assert cond_tokens.shape[1] == self.cond_dim, (
            f"cond has {cond_tokens.shape[1]} tokens but cond_dim={self.cond_dim}"
        )

        vec = time_vec
        if self.use_cond_summary_in_vec:
            vec = vec + summary
        if self.guidance_embed:
            if guidance is None:
                raise ValueError("guidance required when guidance_embed=True")
            vec = vec + self.guidance_in(guidance)

        enc_vec = vec if self.cond_modulates_encoder else time_vec
        feat, pos_skips, neg_skips = self.encoder(obs, enc_vec)
        obs_tokens = self.tokenizer(feat)
        obs_tokens = self.core(
            obs_tokens, cond_tokens, vec, self.obs_ids[...], self.cond_ids[...]
        )
        feat = self.untokenizer(obs_tokens, self.token_grid)
        v = self.decoder(feat, vec, pos_skips, neg_skips)          # time + cond modulation
        return v
