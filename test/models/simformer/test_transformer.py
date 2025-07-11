import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
from flax import nnx
import jax

from gensbi.models.simformer.transformer import Transformer


def get_rngs():
    return nnx.Rngs(0)


def test_transformer_forward_shape():
    rngs = get_rngs()
    transformer = Transformer(
        din=8,
        dcontext=4,
        num_heads=2,
        num_layers=2,
        features=4,
        widening_factor=2,
        num_hidden_layers=1,
        act=jax.nn.gelu,
        skip_connection_attn=True,
        skip_connection_mlp=True,
        rngs=rngs,
    )
    x = jnp.ones((2, 3, 8))
    context = jnp.ones((2, 3, 4))
    out = transformer(x, context=context)
    assert out.shape == (2, 3, 8)
