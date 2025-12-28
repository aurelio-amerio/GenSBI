"""
Experimental module for GenSBI.

This module contains experimental features that are not yet part of the main library,
including autoencoder models and VAE training pipelines.

These components may be moved to the main library in future releases after further
development and validation.
"""

from .autoencoders import AutoEncoder1D, AutoEncoder2D, AutoEncoderParams, vae_loss_fn
from .pipeline_vae import VAE1DPipeline, VAE2DPipeline, AbstractVAEPipeline

__all__ = [
    "AutoEncoder1D",
    "AutoEncoder2D",
    "AutoEncoderParams",
    "vae_loss_fn",
    "VAE1DPipeline",
    "VAE2DPipeline",
    "AbstractVAEPipeline",
]
