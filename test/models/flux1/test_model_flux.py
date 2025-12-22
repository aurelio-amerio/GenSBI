import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
from flax import nnx
import pytest

from gensbi.models.flux1.model import Flux1, Flux1Params
from gensbi.models.wrappers import ConditionalWrapper


def get_rngs():
    return nnx.Rngs(0)


def test_flux_params_instantiation():
    params = Flux1Params(
        in_channels=1,
        vec_in_dim=None,
        context_in_dim=1,
        mlp_ratio=4,
        num_heads=4,
        depth=1,
        depth_single_blocks=2,
        axes_dim=[
            4,
        ],
        obs_dim=2,
        cond_dim=2,
        qkv_bias=True,
        guidance_embed=False,
        rngs=get_rngs(),
        param_dtype=jnp.bfloat16,
    )
    hidden_size = int(
        jnp.sum(jnp.asarray(params.axes_dim, dtype=jnp.int32))
        * params.num_heads
    )
    qkv_features = params.hidden_size

    assert params.hidden_size == hidden_size
    assert params.qkv_features == qkv_features

def init_test_model():
    params = Flux1Params(
        in_channels=1,
        vec_in_dim=None,
        context_in_dim=1,
        mlp_ratio=4,
        num_heads=4,
        depth=1,
        depth_single_blocks=2,
        axes_dim=[
            4,
        ],
        obs_dim=2,
        cond_dim=2,
        qkv_bias=True,
        guidance_embed=False,
        rngs=get_rngs(),
        param_dtype=jnp.float32,
    )
    model = Flux1(params)
    return model


def test_flux_forward_shape_embed_rope():

    model = init_test_model()

    obs = jnp.ones((3, 2, 1))
    cond = jnp.ones((3, 2, 1))
    obs_ids = jnp.arange(2).reshape(1,-1,1)
    cond_ids = jnp.arange(2).reshape(1,-1,1)
    t = jnp.ones((3))

    out = model(
        t=t,
        obs=obs,
        obs_ids=obs_ids,
        cond=cond,
        cond_ids=cond_ids,
        conditioned=True,
    )

    assert out.shape == (3, 2, 1)

def test_flux_wrapper():

    model = init_test_model()
    wrapper = ConditionalWrapper(model)

    obs = jnp.ones((3, 2, 1))
    cond = jnp.ones((3, 2, 1))
    obs_ids = jnp.arange(2).reshape(1,-1,1)
    cond_ids = jnp.arange(2).reshape(1,-1,1)
    t = jnp.ones((3,1))

    extra_args = {"cond": cond, "cond_ids": cond_ids, "obs_ids": obs_ids, "conditioned": True}

    out = wrapper(
        t=t,
        obs=obs,
        **extra_args,
    )

    assert out.shape == (3, 2, 1), f"Wrapper output shape is incorrect, got {out.shape}"

    vf = wrapper.get_vector_field(**extra_args)
    out = vf(t, obs, None)

    assert out.shape == (3, 2, 1), f"Vector field output shape is incorrect, got {out.shape}"

    vf = wrapper.get_vector_field()
    out = vf(t, obs, args=extra_args)

    assert out.shape == (3, 2, 1), f"Vector field output shape is incorrect, got {out.shape}"


def test_flux_param_dtype_propagation():
    params = Flux1Params(
        in_channels=1,
        vec_in_dim=None,
        context_in_dim=1,
        mlp_ratio=4,
        num_heads=4,
        depth=1,
        depth_single_blocks=2,
        axes_dim=[4],
        obs_dim=2,
        cond_dim=2,
        qkv_bias=True,
        guidance_embed=False,
        rngs=get_rngs(),
        param_dtype=jnp.bfloat16,
    )

    model = Flux1(params)

    assert model.obs_in.kernel[...].dtype == jnp.bfloat16
    assert model.cond_in.kernel[...].dtype == jnp.bfloat16
    assert model.time_in.in_layer.kernel[...].dtype == jnp.bfloat16
    assert model.time_in.out_layer.kernel[...].dtype == jnp.bfloat16

    assert model.condition_embedding[...].dtype == jnp.bfloat16
    assert model.condition_null[...].dtype == jnp.bfloat16

    # Representative transformer weights
    assert (
        model.double_blocks.layers[0].obs_attn.qkv.kernel[...].dtype
        == jnp.bfloat16
    )
    assert model.final_layer.linear.kernel[...].dtype == jnp.bfloat16
