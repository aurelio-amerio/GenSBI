# %%
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
from flax import nnx

from gensbi.models.flux1joint.model import Flux1Joint, Flux1JointParams
from gensbi.models.wrappers import JointWrapper

import pytest

# %%


def get_params(id_embedding_strategy="absolute", param_dtype=jnp.float32):
    return Flux1JointParams(
        in_channels=1,
        vec_in_dim=None,
        mlp_ratio=3.0,
        num_heads=2,
        depth_single_blocks=2,
        axes_dim=[4],
        condition_dim=[2],
        qkv_bias=True,
        rngs=nnx.Rngs(0),
        dim_joint=4,
        theta=16,
        id_embedding_strategy=id_embedding_strategy,
        guidance_embed=False,
        param_dtype=param_dtype,
    )


def init_test_model_joint_rope():
    params = get_params(id_embedding_strategy="rope", param_dtype=jnp.bfloat16)
    model = Flux1Joint(params)
    return model


def init_test_model_joint_absolute():
    params = get_params(id_embedding_strategy="absolute", param_dtype=jnp.bfloat16)
    model = Flux1Joint(params)
    return model


# %%
# test that calling with rope issues a warning
def test_flux1joint_rope_warning():
    with pytest.warns(
        UserWarning,
        match="Using RoPE embedding for joint density estimation is not recommended. Consider using 'absolute' embeddings instead.",
    ):
        _ = init_test_model_joint_rope()


@pytest.mark.parametrize(
    "model_fn",
    [init_test_model_joint_rope, init_test_model_joint_absolute],
)
def test_flux1joint_forward_shape(model_fn):
    model = model_fn()
    x = jnp.ones((1, 4, 1))
    t = jnp.ones((1, 1))
    node_ids = jnp.arange(4).reshape(1, -1, 1)
    condition_mask = jnp.zeros((1, 4, 1))
    out = model(t=t, obs=x, node_ids=node_ids, condition_mask=condition_mask)
    assert out.shape == (1, 4, 1), f"Output shape is incorrect, got {out.shape}"


@pytest.mark.parametrize(
    "model_fn",
    [init_test_model_joint_rope, init_test_model_joint_absolute],
)
def test_flux1joint_wrapper(model_fn):
    model = model_fn()
    wrapper = JointWrapper(model)

    obs = jnp.ones((12, 2, 1))
    cond = jnp.ones((12, 2, 1))
    obs_ids = jnp.arange(2).reshape(1, -1, 1)
    cond_ids = jnp.arange(2).reshape(1, -1, 1)
    t = jnp.ones((12, 1))

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

    assert out.shape == (
        12,
        2,
        1,
    ), f"1 - Wrapper output shape is incorrect, got {out.shape}"

    vf = wrapper.get_vector_field(**extra_args)
    out = vf(t, obs, None)

    assert out.shape == (
        12,
        2,
        1,
    ), f"2 - Vector field output shape is incorrect, got {out.shape}"

    vf = wrapper.get_vector_field()
    out = vf(t, obs, args=extra_args)

    assert out.shape == (
        12,
        2,
        1,
    ), f"3 - Vector field output shape is incorrect, got {out.shape}"


@pytest.mark.parametrize(
    "model_fn",
    [init_test_model_joint_rope, init_test_model_joint_absolute],
)
def test_flux1joint_param_dtype_propagation(model_fn):
    model = model_fn()

    assert model.obs_in.kernel[...].dtype == jnp.bfloat16
    assert model.time_in.in_layer.kernel[...].dtype == jnp.bfloat16
    assert model.time_in.out_layer.kernel[...].dtype == jnp.bfloat16
    assert model.condition_embedding[...].dtype == jnp.bfloat16

    # Representative block + head weights
    assert model.single_blocks.layers[0].linear1.kernel[...].dtype == jnp.bfloat16
    assert model.final_layer.linear.kernel[...].dtype == jnp.bfloat16


# %%
