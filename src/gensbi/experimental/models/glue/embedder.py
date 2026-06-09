import jax
from jax import Array
from jax import numpy as jnp
from flax import nnx

from gensbi.recipes.utils import patchify_2d

from gensbi.experimental.models.autoencoders import AutoEncoder1D, AutoEncoder2D
from gensbi.models import Flux1


class Embedded2DModel(nnx.Module):
    """Wraps a VAE encoder and a Flux1 SBI model for 2D image conditioning.

    Encodes raw 2D conditioning images into latent space via the VAE, patchifies
    the result, and forwards everything to the underlying SBI model.

    Parameters
    ----------
    vae : AutoEncoder2D
        Variational autoencoder used to encode conditioning images into latents.
    sbi_model : Flux1
        Simulation-based inference model that receives the encoded conditioning.
    """

    def __init__(self, vae: AutoEncoder2D, sbi_model: Flux1):
        self.vae = vae
        self.sbi_model = sbi_model

    def __call__(
        self,
        t: Array,
        obs: Array,
        obs_ids: Array,
        cond: Array,
        cond_ids: Array,
        conditioned: bool | Array = True,
        guidance: Array | None = None,
        encoder_key=None,
    ) -> Array:
        """Encode conditioning images and run the SBI model forward pass.

        Parameters
        ----------
        t : Array
            Diffusion/flow timestep, shape ``(batch,)`` or scalar.
        obs : Array
            Observation tokens (noisy samples) passed directly to the SBI model.
        obs_ids : Array
            Positional IDs for the observation tokens.
        cond : Array
            Raw 2D conditioning images to be encoded by the VAE, shape
            ``(batch, H, W, C)``.
        cond_ids : Array
            Positional IDs for the conditioning tokens after patchification.
        conditioned : bool or Array, optional
            Whether to apply conditioning (classifier-free guidance mask).
            Defaults to ``True``.
        guidance : Array or None, optional
            Guidance scale for classifier-free guidance. ``None`` disables it.
        encoder_key : jax.random.PRNGKey or None, optional
            PRNG key forwarded to the VAE encoder for stochastic encoding.
            Pass ``None`` for deterministic encoding.

        Returns
        -------
        Array
            Output of the SBI model with the VAE-encoded conditioning.
        """
        cond_latent = self.vae.encode(cond, encoder_key)
        cond_latent = patchify_2d(cond_latent)

        return self.sbi_model(
            t=t,
            obs=obs,
            obs_ids=obs_ids,
            cond=cond_latent,
            cond_ids=cond_ids,
            conditioned=conditioned,
            guidance=guidance,
        )



class Embedded1DModel(nnx.Module):
    """Wraps a VAE encoder and a Flux1 SBI model for 1D sequence conditioning.

    Encodes raw 1D conditioning sequences into latent space via the VAE and
    forwards everything to the underlying SBI model. Unlike ``Embedded2DModel``,
    no patchification step is applied after encoding.

    Parameters
    ----------
    vae : AutoEncoder1D
        Variational autoencoder used to encode conditioning sequences into latents.
    sbi_model : Flux1
        Simulation-based inference model that receives the encoded conditioning.
    """

    def __init__(self, vae: AutoEncoder1D, sbi_model: Flux1):
        self.vae = vae
        self.sbi_model = sbi_model

    def __call__(
        self,
        t: Array,
        obs: Array,
        obs_ids: Array,
        cond: Array,
        cond_ids: Array,
        conditioned: bool | Array = True,
        guidance: Array | None = None,
        encoder_key=None,
    ) -> Array:
        """Encode conditioning sequences and run the SBI model forward pass.

        Parameters
        ----------
        t : Array
            Diffusion/flow timestep, shape ``(batch,)`` or scalar.
        obs : Array
            Observation tokens (noisy samples) passed directly to the SBI model.
        obs_ids : Array
            Positional IDs for the observation tokens.
        cond : Array
            Raw 1D conditioning sequences to be encoded by the VAE, shape
            ``(batch, L, C)``.
        cond_ids : Array
            Positional IDs for the conditioning tokens.
        conditioned : bool or Array, optional
            Whether to apply conditioning (classifier-free guidance mask).
            Defaults to ``True``.
        guidance : Array or None, optional
            Guidance scale for classifier-free guidance. ``None`` disables it.
        encoder_key : jax.random.PRNGKey or None, optional
            PRNG key forwarded to the VAE encoder for stochastic encoding.
            Pass ``None`` for deterministic encoding.

        Returns
        -------
        Array
            Output of the SBI model with the VAE-encoded conditioning.
        """
        cond_latent = self.vae.encode(cond, encoder_key)

        return self.sbi_model(
            t=t,
            obs=obs,
            obs_ids=obs_ids,
            cond=cond_latent,
            cond_ids=cond_ids,
            conditioned=conditioned,
            guidance=guidance,
        )