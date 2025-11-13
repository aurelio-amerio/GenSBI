import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
from flax import nnx

from gensbi.models.simformer.model import Simformer, SimformerParams
from gensbi.models.wrappers import JointWrapper

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
    )


def test_simformer_forward_shape():
    params = get_params()
    model = Simformer(params)
    x = jnp.ones((1, 4, 1))
    t = jnp.ones((1, 1))
    node_ids = jnp.arange(4).reshape(1, 4)
    condition_mask = jnp.zeros((1, 4, 1))
    out = model(t, x, node_ids=node_ids, condition_mask=condition_mask)
    assert out.shape == (1, 4, 1), f"Output shape is incorrect, got {out.shape}"


def test_simformer_wrapper():
    params = get_params()
    model = Simformer(params)
    wrapper = JointWrapper(model)

    obs = jnp.ones((12, 2, 1))
    cond = jnp.ones((12, 2, 1))
    obs_ids = jnp.arange(2).reshape(1,-1)
    cond_ids = jnp.arange(2).reshape(1,-1)
    t = jnp.ones((12,1))

    extra_args = {"cond": cond, "cond_ids": cond_ids, "obs_ids": obs_ids, "edge_mask": None, "conditioned": True}

    out = wrapper(
        t=t,
        obs=obs,
        **extra_args,
    )

    assert out.shape == (12, 2, 1), f"1 - Wrapper output shape is incorrect, got {out.shape}"

    vf = wrapper.get_vector_field(**extra_args)
    out = vf(t, obs, None)

    assert out.shape == (12, 2), f"2 - Vector field output shape is incorrect, got {out.shape}"

    vf = wrapper.get_vector_field()
    out = vf(t, obs, args=extra_args)

    assert out.shape == (12, 2), f"3 - Vector field output shape is incorrect, got {out.shape}"
