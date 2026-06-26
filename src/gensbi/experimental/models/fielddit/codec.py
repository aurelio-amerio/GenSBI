"""Conv U-Net halves and the patch boundary for FieldDiT.

ObsEncoder (conv down, time-only modulation, captures SiD2 skips) and
ObsDecoder (conv up, residual skips, time+cond modulation, zero-init final
conv) sandwich the MMDiT bottleneck; Tokenizer/Untokenizer cross the
patchify boundary.
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import DTypeLike

from gensbi.models.core.patching import patchify_2d, depatchify_2d
from gensbi.experimental.models.fielddit.blocks import (
    ModulatedResBlock2D,
    _safe_groups,
)

# FIXME: this is a repetition of what we also have for the VAE, so we might want to move it somewhere common
class Downsample2D(nnx.Module):
    """Stride-2 conv that also changes channel count (asymmetric pad, AE-style)."""

    def __init__(self, in_channels: int, out_channels: int, rngs: nnx.Rngs, param_dtype: DTypeLike = jnp.bfloat16):
        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=(0, 0),
            rngs=rngs,
            param_dtype=param_dtype,
        )

    def __call__(self, x):
        x = jnp.pad(x, ((0, 0), (0, 1), (0, 1), (0, 0)), mode="constant", constant_values=0)
        return self.conv(x)

# same as downsample
class Upsample2D(nnx.Module):
    """Nearest-neighbour 2x upsample + conv that also changes channel count."""

    def __init__(self, in_channels: int, out_channels: int, rngs: nnx.Rngs, param_dtype: DTypeLike = jnp.bfloat16):
        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            rngs=rngs,
            param_dtype=param_dtype,
        )

    def __call__(self, x):
        b, h, w, c = x.shape
        x = jax.image.resize(x, (b, h * 2, w * 2, c), method="nearest")
        return self.conv(x)


class ObsEncoder(nnx.Module):
    """Conv encoder: down-sampling stages with time-only AdaGN-zero modulation.

    ``widths`` has length ``D + 1`` (one width per resolution incl. the
    bottleneck). Stage ``j`` (j = 0..D-1) runs ``res_blocks`` blocks at width
    ``widths[j]``, then downsamples ``widths[j] -> widths[j+1]``. Returns the
    bottleneck feature plus per-stage ``pos_skips`` (pre-downsample) and
    ``neg_skips`` (post-downsample) for the SiD2 residual decoder.
    """

    def __init__(
        self,
        in_channels: int,
        widths: tuple[int, ...],
        res_blocks: int,
        vec_dim: int,
        norm_groups: int,
        rngs: nnx.Rngs,
        param_dtype: DTypeLike = jnp.bfloat16,
    ):
        self.widths = tuple(widths)
        self.depth = len(self.widths) - 1
        self.conv_in = nnx.Conv(
            in_features=in_channels,
            out_features=self.widths[0],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.down = nnx.Sequential()
        for j in range(self.depth):
            stage = nnx.Module()
            stage.block = nnx.Sequential(
                *[
                    ModulatedResBlock2D(
                        self.widths[j], self.widths[j], vec_dim, norm_groups,
                        rngs=rngs, param_dtype=param_dtype,
                    )
                    for _ in range(res_blocks)
                ]
            )
            stage.downsample = Downsample2D(
                self.widths[j], self.widths[j + 1], rngs=rngs, param_dtype=param_dtype
            )
            self.down.layers.append(stage)

    def __call__(self, x, time_vec):
        h = self.conv_in(x)
        pos_skips = []
        neg_skips = []
        for j in range(self.depth):
            stage = self.down.layers[j]
            for blk in stage.block.layers:
                h = blk(h, time_vec)
            pos_skips.append(h)
            h = stage.downsample(h)
            neg_skips.append(h)
        return h, pos_skips, neg_skips


class ObsDecoder(nnx.Module):
    """Conv decoder: SiD2 residual skips + time+cond AdaGN-zero modulation.

    Mirrors ``ObsEncoder``. ``self.up.layers[i]`` corresponds to encoder stage
    ``j = depth - 1 - i``. Per stage: subtract the matching ``neg_skip``,
    upsample ``widths[j+1] -> widths[j]``, add the matching ``pos_skip``, then
    run the stage's blocks. The final conv is zero-initialized so the velocity
    field is exactly zero at initialization.

    ``out_channels`` is the channel count of the produced velocity field.
    """

    def __init__(
        self,
        out_channels: int,
        widths: tuple[int, ...],
        res_blocks: int,
        vec_dim: int,
        norm_groups: int,
        rngs: nnx.Rngs,
        param_dtype: DTypeLike = jnp.bfloat16,
    ):
        self.widths = tuple(widths)
        self.depth = len(self.widths) - 1
        self.up = nnx.Sequential()
        for j in reversed(range(self.depth)):
            stage = nnx.Module()
            stage.upsample = Upsample2D(
                self.widths[j + 1], self.widths[j], rngs=rngs, param_dtype=param_dtype
            )
            stage.block = nnx.Sequential(
                *[
                    ModulatedResBlock2D(
                        self.widths[j], self.widths[j], vec_dim, norm_groups,
                        rngs=rngs, param_dtype=param_dtype,
                    )
                    for _ in range(res_blocks)
                ]
            )
            self.up.layers.append(stage)

        self.norm_out = nnx.GroupNorm(
            num_groups=_safe_groups(self.widths[0], norm_groups),
            num_features=self.widths[0],
            epsilon=1e-6,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.conv_out = nnx.Conv(
            in_features=self.widths[0],
            out_features=out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            rngs=rngs,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.zeros,
        )

    def __call__(self, feat, vec, pos_skips, neg_skips):
        h = feat
        for i in range(self.depth):
            j = self.depth - 1 - i
            stage = self.up.layers[i]
            h = h - neg_skips[j]
            h = stage.upsample(h)
            h = h + pos_skips[j]
            for blk in stage.block.layers:
                h = blk(h, vec)
        h = nnx.silu(self.norm_out(h))
        return self.conv_out(h)


class Tokenizer(nnx.Module):
    """Patchify a conv feature map and project to ``hidden_size`` tokens."""

    def __init__(self, in_channels: int, patch_size: int, hidden_size: int, rngs: nnx.Rngs, param_dtype: DTypeLike = jnp.bfloat16):
        self.patch_size = patch_size
        self.proj = nnx.Linear(
            in_features=in_channels * patch_size * patch_size,
            out_features=hidden_size,
            use_bias=True,
            rngs=rngs,
            param_dtype=param_dtype,
        )

    def __call__(self, feat):
        x = patchify_2d(feat, size=self.patch_size)  # (B, N, C * p * p)
        return self.proj(x)                          # (B, N, hidden)


class Untokenizer(nnx.Module):
    """Project tokens back to patch pixels and depatchify to a conv feature map.

    ``grid`` is the ``(h, w)`` token grid (``feat_h // p``, ``feat_w // p``),
    passed to the (now grid-aware) ``depatchify_2d``.
    """

    def __init__(self, out_channels: int, patch_size: int, hidden_size: int, rngs: nnx.Rngs, param_dtype: DTypeLike = jnp.bfloat16):
        self.patch_size = patch_size
        self.out_channels = out_channels
        # bound the transformer residual stream before re-entering conv space
        # (Flux1 exits through LastLayer's norm for the same reason)
        self.norm = nnx.LayerNorm(
            num_features=hidden_size,
            epsilon=1e-6,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.proj = nnx.Linear(
            in_features=hidden_size,
            out_features=out_channels * patch_size * patch_size,
            use_bias=True,
            rngs=rngs,
            param_dtype=param_dtype,
        )

    def __call__(self, tokens, grid):
        x = self.proj(self.norm(tokens))  # (B, N, C * p * p)
        return depatchify_2d(x, size=self.patch_size, grid=tuple(grid))
