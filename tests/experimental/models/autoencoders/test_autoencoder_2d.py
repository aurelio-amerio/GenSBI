import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
from flax import nnx
import pytest
import jax 

from gensbi.experimental.models.autoencoders.autoencoder_2d import AutoEncoder2D
from gensbi.experimental.models.autoencoders import AutoEncoderParams, vae_loss_fn

def test_autoencoder_2d():
    resolution = 16
    data = jnp.ones((4, resolution, resolution, 3))  # (batch_size, height, width, channels)
    
    params = AutoEncoderParams(
        resolution=resolution,
        in_channels=3,
        ch=32,
        out_ch=3,
        ch_mult=[1, 1, 1],
        num_res_blocks=2,
        z_channels=8,
        scale_factor=1.0,
        shift_factor=0.0,
        rngs=nnx.Rngs(4),
        param_dtype=jnp.float32,
    )
    model = AutoEncoder2D(params)

    res = model(data)
    
    assert res.shape == data.shape, f"Expected shape {data.shape}, got {res.shape}"
    
    key = jax.random.PRNGKey(0)
    loss = vae_loss_fn(model, data, key)
    assert loss >= 0, f"Loss should be non-negative, got {loss}"
    return
    
    
    