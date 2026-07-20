"""AdaGN-zero modulated residual conv block for the FieldDiT conv codec.

Ported (not imported) from the reference SiD2 ``ResidualBlock2D`` (Keras) and
GenSBI's ``ResnetBlock2D``: FiLM scale/shift over GroupNorm, plus a
conditioning-predicted *gate* (zero-initialized) so each block is identity at
initialization (``out = residual + gate * h``).
"""

import math

import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import DTypeLike


def _safe_groups(num_features: int, groups: int) -> int:
    """A common divisor of ``num_features`` and ``groups`` via gcd.

    Returns a value that always divides ``num_features`` and is <= ``groups``,
    so GroupNorm never breaks on small or unusual channel counts.  It is *not*
    guaranteed to be the largest such divisor.
    """
    return math.gcd(int(num_features), int(groups))


class ConvModulation(nnx.Module):
    """Project a global ``vec`` to (scale, shift, gate) for an NHWC feature map.

    Mirrors Flux1's ``Modulation`` (zero-init linear => neutral modulation /
    closed gate at init), but reshapes outputs to ``(B, 1, 1, C)`` so they
    broadcast over the spatial dims.
    """

    def __init__(
        self,
        vec_dim: int,
        channels: int,
        rngs: nnx.Rngs,
        dtype: DTypeLike = jnp.bfloat16,
        param_dtype: DTypeLike = jnp.float32,
    ):
        self.channels = channels
        self.lin = nnx.Linear(
            in_features=vec_dim,
            out_features=3 * channels,
            use_bias=True,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.zeros,
        )

    def __call__(self, vec):
        out = self.lin(nnx.silu(vec))
        scale, shift, gate = jnp.split(out, 3, axis=-1)
        reshape = lambda z: z[:, None, None, :]
        return reshape(scale), reshape(shift), reshape(gate)


class ModulatedResBlock2D(nnx.Module):
    """SiD2-style residual conv block with AdaGN-zero modulation (NHWC).

    Structure: ``norm1 -> silu -> conv1 -> norm2 -> FiLM(scale,shift) -> silu
    -> conv2``, returned as ``residual + gate * h``. ``norm2`` has its own
    affine disabled (the predicted scale/shift is the sole affine). The gate is
    zero-initialized, giving (a) identity at init and (b) condition-dependent
    block strength.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        vec_dim: int,
        norm_groups: int,
        rngs: nnx.Rngs,
        dtype: DTypeLike = jnp.bfloat16,
        param_dtype: DTypeLike = jnp.float32,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels

        # fp32 island: GroupNorm stats always computed/stored in fp32,
        # regardless of the model's compute-dtype knob. Its output feeds
        # directly into conv1/conv2 (compute-dtype Linears) below, which
        # self-heal the downcast via their own promote_dtype -- no explicit
        # ``.astype`` needed here.
        self.norm1 = nnx.GroupNorm(
            num_groups=_safe_groups(in_channels, norm_groups),
            num_features=in_channels,
            epsilon=1e-6,
            rngs=rngs,
            dtype=jnp.float32,
            param_dtype=param_dtype,
        )
        self.conv1 = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        # affine off: ConvModulation provides the sole scale/shift
        self.norm2 = nnx.GroupNorm(
            num_groups=_safe_groups(out_channels, norm_groups),
            num_features=out_channels,
            epsilon=1e-6,
            use_scale=False,
            use_bias=False,
            rngs=rngs,
            dtype=jnp.float32,
            param_dtype=param_dtype,
        )
        self.conv2 = nnx.Conv(
            in_features=out_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.mod = ConvModulation(
            vec_dim=vec_dim, channels=out_channels, rngs=rngs,
            dtype=dtype, param_dtype=param_dtype,
        )
        if in_channels != out_channels:
            self.nin_shortcut = nnx.Conv(
                in_features=in_channels,
                out_features=out_channels,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding=(0, 0),
                rngs=rngs,
                dtype=dtype,
                param_dtype=param_dtype,
            )
        else:
            self.nin_shortcut = None

    def __call__(self, x, vec):
        residual = x if self.nin_shortcut is None else self.nin_shortcut(x)
        h = self.conv1(nnx.silu(self.norm1(x)))
        scale, shift, gate = self.mod(vec)
        h = self.norm2(h)
        h = (1 + scale) * h + shift
        h = self.conv2(nnx.silu(h))
        return residual + gate * h
