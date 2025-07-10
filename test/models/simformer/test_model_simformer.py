import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
from flax import nnx

from gensbi.models.simformer.simformer import Simformer, SimformerParams, SimformerConditioner

def get_rngs():
    return nnx.Rngs(0)

def get_params():
    return SimformerParams(
        rngs=get_rngs(),
        dim_value=2,
        dim_id=2,
        dim_condition=2,
        dim_joint=4,
        fourier_features=8,
        num_heads=2,
        num_layers=2,
        widening_factor=2,
        qkv_features=4,
        num_hidden_layers=1,
        dropout_rate=0.0,
    )

def test_simformer_forward_shape():
    params = get_params()
    model = Simformer(params)
    x = jnp.ones((1, 4, 1))
    t = jnp.ones((1, 1, 1))
    node_ids = jnp.arange(4).reshape(1, 4)
    condition_mask = jnp.zeros((1, 4, 1))
    out = model(x, t, node_ids=node_ids, condition_mask=condition_mask)
    assert out.shape == (1, 4)

def test_simformer_conditioner_shapes():
    params = get_params()
    model = Simformer(params)
    conditioner = SimformerConditioner(model)
    obs = jnp.ones((1, 2, 1))
    obs_ids = jnp.array([0, 1])
    cond = jnp.ones((1, 2, 1))
    cond_ids = jnp.array([2, 3])
    t = jnp.ones((1, 1, 1))
    out = conditioner(obs, obs_ids, cond, cond_ids, t, conditioned=True)
    assert out.shape[0] == 1
