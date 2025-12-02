
"""
Autoencoders submodule.

This module provides 1D and 2D autoencoder architectures with Gaussian latent spaces,
including configuration dataclasses and VAE loss functions.
"""

from .autoencoder_1d import AutoEncoder1D
from .autoencoder_2d import AutoEncoder2D
from .commons import AutoEncoderParams, vae_loss_fn

__all__ = [
    "AutoEncoder1D",
    "AutoEncoder2D",
    "AutoEncoderParams",
    "vae_loss_fn",
]

