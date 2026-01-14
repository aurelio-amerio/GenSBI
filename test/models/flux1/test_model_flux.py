# %%
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
from flax import nnx
import pytest

from gensbi.models.flux1.model import Flux1, Flux1Params
from gensbi.models.wrappers import ConditionalWrapper


def get_rngs():
    return nnx.Rngs(0)


# %%
def test_flux_params_instantiation():
    # Test default (non-rope) embedding
    params = Flux1Params(
        in_channels=1,
        vec_in_dim=None,
        context_in_dim=1,
        mlp_ratio=4,
        num_heads=4,
        depth=1,
        depth_single_blocks=2,
        axes_dim=[4],
        dim_obs=2,
        dim_cond=2,
        qkv_bias=True,
        guidance_embed=False,
        rngs=get_rngs(),
        param_dtype=jnp.bfloat16,
        id_embedding_strategy=("absolute", "absolute"),
    )
    hidden_size = int(
        jnp.sum(jnp.asarray(params.axes_dim, dtype=jnp.int32)) * params.num_heads
    )
    qkv_features = params.hidden_size
    assert params.hidden_size == hidden_size
    assert params.qkv_features == qkv_features

    # Test rope embedding (valid)
    params_rope = Flux1Params(
        in_channels=1,
        vec_in_dim=None,
        context_in_dim=1,
        mlp_ratio=4,
        num_heads=4,
        depth=1,
        depth_single_blocks=2,
        axes_dim=[4, 0],
        dim_obs=2,
        dim_cond=2,
        qkv_bias=True,
        guidance_embed=False,
        rngs=get_rngs(),
        param_dtype=jnp.bfloat16,
        id_embedding_strategy=("rope1d", "rope1d"),
    )
    hidden_size_rope = int(
        jnp.sum(jnp.asarray(params_rope.axes_dim, dtype=jnp.int32))
        * params_rope.num_heads
    )
    assert params_rope.hidden_size == hidden_size_rope
    assert params_rope.qkv_features == hidden_size_rope

    # Test obs (absolute) cond (rope) embedding (valid)
    params_mixed = Flux1Params(
        in_channels=1,
        vec_in_dim=None,
        context_in_dim=1,
        mlp_ratio=4,
        num_heads=4,
        depth=1,
        depth_single_blocks=2,
        axes_dim=[4, 0],
        dim_obs=2,
        dim_cond=2,
        qkv_bias=True,
        guidance_embed=False,
        rngs=get_rngs(),
        param_dtype=jnp.bfloat16,
        id_embedding_strategy=("absolute", "rope1d"),
    )
    hidden_size_rope = int(
        jnp.sum(jnp.asarray(params_mixed.axes_dim, dtype=jnp.int32))
        * params_mixed.num_heads
    )
    assert params_mixed.hidden_size == hidden_size_rope
    assert params_mixed.qkv_features == hidden_size_rope


# %%


def init_test_model_rope():
    params = Flux1Params(
        in_channels=1,
        vec_in_dim=None,
        context_in_dim=1,
        mlp_ratio=4,
        num_heads=4,
        depth=1,
        depth_single_blocks=2,
        axes_dim=[4, 2],
        dim_obs=3,
        dim_cond=5,
        qkv_bias=True,
        id_embedding_strategy=("rope1d", "rope1d"),
        guidance_embed=False,
        rngs=get_rngs(),
        param_dtype=jnp.bfloat16,
    )
    model = Flux1(params)
    return model


def init_test_model_standard():
    params = Flux1Params(
        in_channels=1,
        vec_in_dim=None,
        context_in_dim=1,
        mlp_ratio=4,
        num_heads=4,
        depth=1,
        depth_single_blocks=2,
        axes_dim=[4],
        dim_obs=3,
        dim_cond=5,
        qkv_bias=True,
        id_embedding_strategy=("absolute", "absolute"),
        guidance_embed=False,
        rngs=get_rngs(),
        param_dtype=jnp.bfloat16,
    )
    model = Flux1(params)
    return model


def init_test_model_mixed():
    params = Flux1Params(
        in_channels=1,
        vec_in_dim=None,
        context_in_dim=1,
        mlp_ratio=4,
        num_heads=4,
        depth=1,
        depth_single_blocks=2,
        axes_dim=[4, 2],
        dim_obs=3,
        dim_cond=5,
        qkv_bias=True,
        id_embedding_strategy=("absolute", "rope1d"),
        guidance_embed=False,
        rngs=get_rngs(),
        param_dtype=jnp.bfloat16,
    )
    model = Flux1(params)
    return model


# %%


@pytest.mark.parametrize(
    "model_fn",
    [
        init_test_model_rope,
        init_test_model_standard,
        init_test_model_mixed,
    ],
)
def test_flux_forward_shape_embed(model_fn):

    model = model_fn()

    obs = jnp.ones((4, 3, 1))
    cond = jnp.ones((4, 5, 1))

    obs_ids = jnp.zeros((1, 3, 2), dtype=jnp.int32)
    obs_ids = obs_ids.at[0, :, 0].set(jnp.arange(3))
    cond_ids = jnp.zeros((1, 5, 2), dtype=jnp.int32)
    cond_ids = cond_ids.at[0, :, 0].set(jnp.arange(5))
    cond_ids = cond_ids.at[0, :, 1].set(1)

    t = jnp.ones((4))

    out = model(
        t=t,
        obs=obs,
        obs_ids=obs_ids,
        cond=cond,
        cond_ids=cond_ids,
        conditioned=True,
    )

    assert out.shape == (4, 3, 1), f"Output shape is incorrect, got {out.shape}"


# %%


@pytest.mark.parametrize(
    "model_fn",
    [
        init_test_model_rope,
        init_test_model_standard,
        init_test_model_mixed,
    ],
)
def test_flux_wrapper(model_fn):

    model = model_fn()
    wrapper = ConditionalWrapper(model)

    obs = jnp.ones((4, 3, 1))
    cond = jnp.ones((4, 5, 1))

    obs_ids = jnp.zeros((1, 3, 2), dtype=jnp.int32)
    obs_ids = obs_ids.at[0, :, 0].set(jnp.arange(3))
    cond_ids = jnp.zeros((1, 5, 2), dtype=jnp.int32)
    cond_ids = cond_ids.at[0, :, 0].set(jnp.arange(5))
    cond_ids = cond_ids.at[0, :, 1].set(1)

    t = jnp.ones((4, 1))

    extra_args = {
        "cond": cond,
        "cond_ids": cond_ids,
        "obs_ids": obs_ids,
        "conditioned": True,
    }

    out = wrapper(
        t=t,
        obs=obs,
        **extra_args,
    )

    assert out.shape == (4, 3, 1), f"Wrapper output shape is incorrect, got {out.shape}"

    vf = wrapper.get_vector_field(**extra_args)
    out = vf(t, obs, None)

    assert out.shape == (
        4,
        3,
        1,
    ), f"Vector field output shape is incorrect, got {out.shape}"

    vf = wrapper.get_vector_field()
    out = vf(t, obs, args=extra_args)

    assert out.shape == (
        4,
        3,
        1,
    ), f"Vector field output shape is incorrect, got {out.shape}"


@pytest.mark.parametrize(
    "model_fn",
    [
        init_test_model_rope,
        init_test_model_standard,
        init_test_model_mixed,
    ],
)
def test_flux_param_dtype_propagation(model_fn):
    model = model_fn()

    assert (
        model.obs_in.kernel[...].dtype == jnp.bfloat16
    ), "obs_in layer dtype not propagated correctly."
    assert (
        model.cond_in.kernel[...].dtype == jnp.bfloat16
    ), "cond_in layer dtype not propagated correctly."
    assert (
        model.time_in.in_layer.kernel[...].dtype == jnp.bfloat16
    ), "time_in layer dtype not propagated correctly."
    assert (
        model.time_in.out_layer.kernel[...].dtype == jnp.bfloat16
    ), "time_in layer dtype not propagated correctly."

    # assert (
    #     model.condition_embedding[...].dtype == jnp.bfloat16
    # ), "condition_embedding dtype not propagated correctly."
    # assert (
    #     model.condition_null[...].dtype == jnp.bfloat16
    # ), "condition_null dtype not propagated correctly."

    # Representative transformer weights
    assert (
        model.double_blocks.layers[0].obs_attn.qkv.kernel[...].dtype == jnp.bfloat16
    ), "double_blocks obs_attn qkv layer dtype not propagated correctly."
    assert (
        model.final_layer.linear.kernel[...].dtype == jnp.bfloat16
    ), "final_layer linear layer dtype not propagated correctly."
