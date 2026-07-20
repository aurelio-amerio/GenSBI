"""
Common components for autoencoder architectures.

This module provides shared components used by 1D and 2D autoencoder implementations,
including parameter dataclasses, Gaussian latent space modules, and loss functions.
"""
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
            Data type for master-weight storage (parameters). Defaults to jnp.float32.
        dtype (DTypeLike):
            Compute dtype used by the encoder/decoder's Conv layers. GroupNorms
            and the decoder's final output Conv are fp32 islands regardless of
            this setting (see `Encoder2D`/`Decoder2D`/`DiagonalGaussian`).
            Defaults to jnp.bfloat16.
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
    param_dtype: DTypeLike = jnp.float32
    dtype: DTypeLike = jnp.bfloat16


class Loss(nnx.Variable):
    """
    Placeholder variable for storing loss values in the model.
    """

    pass


class DiagonalGaussian(nnx.Module):
    """
    Diagonal Gaussian distribution module for VAE latent space.

    Parameters
    ----------
        sample : bool
            Whether to sample from the distribution (default: True).
        chunk_dim : int
            Axis along which to split mean and logvar (default: -1).
    """

    def __init__(
        self,
        sample: bool = True,
        chunk_dim: int = -1,
    ) -> None:
        """
        Initialize the Diagonal Gaussian module.
        
        Parameters
        ----------
            sample: Whether to sample from the distribution. Defaults to True.
            chunk_dim: Axis along which to split mean and logvar. Defaults to -1.
        """
        self.sample = sample
        self.chunk_dim = chunk_dim

        self.kl_loss = Loss(jnp.array(1e10))

        self.update_KL = False

    def __call__(self, z: Array, key=None) -> Array:
        """
        Split input into mean and log-variance, compute KL loss, and sample if required.

        Parameters
        ----------
            z : Array
                Input tensor containing concatenated mean and logvar.
            key : Array, optional
                PRNG key for sampling. Required if sampling is enabled.

        Returns
        -------
            Array
                Sampled latent or mean, depending on self.sample.
        """
        # fp32 island: the exp/log math for the variance and the Gaussian
        # sample are numerically sensitive, so they always run in fp32
        # regardless of the compute dtype knob. The result is downcast back
        # to the incoming dtype below so this island self-heals instead of
        # leaking fp32 into the caller's scale/shift arithmetic, which (unlike
        # a Linear/Conv's `promote_dtype`) won't downcast on its own.
        orig_dtype = z.dtype
        z = jnp.asarray(z, dtype=jnp.float32)
        mean, logvar = jnp.split(z, 2, axis=self.chunk_dim)
        std = jnp.exp(0.5 * logvar)

        if self.update_KL:
            self.kl_loss = Loss(
                jnp.mean(
                    0.5 * jnp.mean(-jnp.log(std**2) - 1.0 + std**2 + mean**2, axis=-1)
                )
            )

        if self.sample:
            out = mean + std * jax.random.normal(
                key=key, shape=mean.shape, dtype=jnp.float32
            )
        else:
            out = mean

        return jnp.asarray(out, dtype=orig_dtype)


def vae_loss_fn(
    model: nnx.Module, x: jax.Array, key: jax.random.PRNGKey, kl_weight: float = 0.1
) -> jax.Array:
    """
    Compute the VAE loss as the sum of reconstruction and KL divergence losses.

    Parameters
    ----------
        model: The VAE model.
        x: Input data.
        key: PRNG key for stochastic operations.
        kl_weight: Weight for the KL divergence term. Defaults to 0.1.

    Returns
    -------
        Scalar loss value combining reconstruction and KL losses.
    """
    logits = model(x, key)
    losses = nnx.state(model, Loss)
    kl_loss = sum(jax.tree_util.tree_leaves(losses), 0.0)

    # Loss is always computed in fp32 regardless of the model's compute
    # dtype (defense-in-depth on top of the models-emit-fp32 contract).
    logits = jnp.asarray(logits, jnp.float32)
    target = jnp.asarray(x, jnp.float32)
    kl_loss = jnp.asarray(kl_loss, jnp.float32)

    reconstruction_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits, target))
    loss = reconstruction_loss + kl_weight * kl_loss
    return loss
