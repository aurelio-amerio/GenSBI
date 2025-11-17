from dataclasses import dataclass

from flax import nnx
from jax.typing import DTypeLike

from jax import Array
import jax
import jax.numpy as jnp

import optax

@dataclass
class AutoEncoderParams:
    """
    Configuration parameters for the AutoEncoder models.

    Attributes:
        resolution (int):
            The input feature dimension (length for 1D, height/width for 2D).
        in_channels (int):
            Number of input channels (e.g., 1 for scalar features, >1 for multi-channel).
        ch (int):
            Base number of channels for the first convolutional layer.
        out_ch (int):
            Number of output channels produced by the decoder (matches input channels for reconstruction).
        ch_mult (list[int]):
            Multipliers for the number of channels at each resolution level (controls model width/depth).
        num_res_blocks (int):
            Number of residual blocks per resolution level.
        z_channels (int):
            Number of latent channels in the bottleneck (size of encoded representation).
        scale_factor (float):
            Scaling factor applied to the latent representation (for normalization or data scaling).
        shift_factor (float):
            Shift factor applied to the latent representation (for normalization or data centering).
        rngs (nnx.Rngs):
            Random number generators for parameter initialization and stochastic layers.
        param_dtype (DTypeLike):
            Data type for model parameters (e.g., jnp.float32, jnp.bfloat16).
    """

    resolution: int
    in_channels: int
    ch: int
    out_ch: int
    ch_mult: list[int]
    num_res_blocks: int
    z_channels: int
    scale_factor: float
    shift_factor: float
    rngs: nnx.Rngs
    param_dtype: DTypeLike


class Loss(nnx.Variable):
    pass


class DiagonalGaussian(nnx.Module):
    def __init__(
        self,
        sample: bool = True,
        chunk_dim: int = -1,
        key: Array = jax.random.PRNGKey(42),
    ):
        self.sample = sample
        self.chunk_dim = chunk_dim
        self.key = key

    def __call__(self, z: Array) -> Array:
        mean, logvar = jnp.split(z, 2, axis=self.chunk_dim)
        std = jnp.exp(0.5 * logvar)

        self.kl_loss = Loss(
            jnp.mean(0.5 * jnp.mean(-jnp.log(std**2) - 1.0 + std**2 + mean**2, axis=-1))
        )


        if self.sample:
            return mean + std * jax.random.normal(
                key=self.key, shape=mean.shape, dtype=z.dtype
            )
        else:
            return mean


def swish(x: Array) -> Array:
    return nnx.swish(x)


def vae_loss_fn(model: nnx.Module, x: jax.Array) -> jax.Array:
    logits = model(x)
    losses = nnx.pop(model, Loss)
    kl_loss = sum(jax.tree_util.tree_leaves(losses), 0.0)
    reconstruction_loss = jnp.mean(
      optax.sigmoid_binary_cross_entropy(logits, x)
    )
    loss = reconstruction_loss + 0.1 * kl_loss
    return loss