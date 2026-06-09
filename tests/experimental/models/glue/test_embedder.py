import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import pytest

from gensbi.experimental.models import Embedded1DModel, Embedded2DModel
from gensbi.experimental.models.glue import (
    Embedded1DModel as Embedded1DModelDirect,
    Embedded2DModel as Embedded2DModelDirect,
)


class MockVAE:
    """Encoder that returns a fixed latent shape regardless of input."""

    def __init__(self, latent_shape):
        self.latent_shape = latent_shape
        self.last_cond = None
        self.last_key = None

    def encode(self, cond, key=None):
        self.last_cond = cond
        self.last_key = key
        batch = cond.shape[0]
        return jnp.ones((batch, *self.latent_shape), dtype=cond.dtype)


class MockSBIModel:
    """SBI stand-in that records its kwargs and returns a sentinel array."""

    def __init__(self, output_shape=(4, 2, 1)):
        self.output_shape = output_shape
        self.last_kwargs = None

    def __call__(
        self,
        t,
        obs,
        obs_ids,
        cond,
        cond_ids,
        conditioned=True,
        guidance=None,
    ):
        self.last_kwargs = dict(
            t=t,
            obs=obs,
            obs_ids=obs_ids,
            cond=cond,
            cond_ids=cond_ids,
            conditioned=conditioned,
            guidance=guidance,
        )
        return jnp.zeros(self.output_shape, dtype=obs.dtype)


@pytest.mark.experimental
def test_embedded_2d_model_call():
    batch = 4
    h, w, c = 8, 8, 3
    latent_h, latent_w, latent_c = 4, 4, 2

    vae = MockVAE(latent_shape=(latent_h, latent_w, latent_c))
    sbi = MockSBIModel(output_shape=(batch, 2, 1))

    model = Embedded2DModel(vae=vae, sbi_model=sbi)

    t = jnp.zeros((batch,))
    obs = jnp.ones((batch, 2, 1))
    obs_ids = jnp.zeros((batch, 2, 3), dtype=jnp.int32)
    cond = jnp.ones((batch, h, w, c))
    # cond_ids shape isn't validated by the embedder, just forwarded.
    cond_ids = jnp.zeros((batch, (latent_h // 2) * (latent_w // 2), 3), dtype=jnp.int32)
    encoder_key = jax.random.PRNGKey(0)

    out = model(
        t=t,
        obs=obs,
        obs_ids=obs_ids,
        cond=cond,
        cond_ids=cond_ids,
        conditioned=True,
        guidance=None,
        encoder_key=encoder_key,
    )

    assert out.shape == (batch, 2, 1)

    # VAE saw the raw conditioning and the encoder key.
    assert vae.last_cond.shape == (batch, h, w, c)
    assert vae.last_key is encoder_key

    # SBI received patchified latents: (batch, (lh/2)*(lw/2), latent_c * 2 * 2).
    forwarded_cond = sbi.last_kwargs["cond"]
    assert forwarded_cond.shape == (
        batch,
        (latent_h // 2) * (latent_w // 2),
        latent_c * 4,
    )
    # Other kwargs passed through unchanged.
    assert sbi.last_kwargs["t"] is t
    assert sbi.last_kwargs["obs"] is obs
    assert sbi.last_kwargs["obs_ids"] is obs_ids
    assert sbi.last_kwargs["cond_ids"] is cond_ids
    assert sbi.last_kwargs["conditioned"] is True
    assert sbi.last_kwargs["guidance"] is None


@pytest.mark.experimental
def test_embedded_1d_model_call():
    batch = 4
    length, channels = 16, 2
    latent_len, latent_c = 8, 4

    vae = MockVAE(latent_shape=(latent_len, latent_c))
    sbi = MockSBIModel(output_shape=(batch, 2, 1))

    model = Embedded1DModel(vae=vae, sbi_model=sbi)

    t = jnp.zeros((batch,))
    obs = jnp.ones((batch, 2, 1))
    obs_ids = jnp.zeros((batch, 2, 3), dtype=jnp.int32)
    cond = jnp.ones((batch, length, channels))
    cond_ids = jnp.zeros((batch, latent_len, 3), dtype=jnp.int32)
    encoder_key = jax.random.PRNGKey(1)

    out = model(
        t=t,
        obs=obs,
        obs_ids=obs_ids,
        cond=cond,
        cond_ids=cond_ids,
        conditioned=True,
        guidance=None,
        encoder_key=encoder_key,
    )

    assert out.shape == (batch, 2, 1)

    assert vae.last_cond.shape == (batch, length, channels)
    assert vae.last_key is encoder_key

    # 1D embedder does NOT patchify — latent passes through unchanged.
    forwarded_cond = sbi.last_kwargs["cond"]
    assert forwarded_cond.shape == (batch, latent_len, latent_c)
    assert sbi.last_kwargs["t"] is t
    assert sbi.last_kwargs["obs"] is obs
    assert sbi.last_kwargs["obs_ids"] is obs_ids
    assert sbi.last_kwargs["cond_ids"] is cond_ids


@pytest.mark.experimental
def test_glue_exports_match_models_reexport():
    assert Embedded1DModel is Embedded1DModelDirect
    assert Embedded2DModel is Embedded2DModelDirect
