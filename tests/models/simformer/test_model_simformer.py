import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
from flax import nnx

import pytest

from gensbi.models.simformer.model import Simformer, SimformerParams
from gensbi.models.wrappers import JointWrapper


def get_rngs():
    return nnx.Rngs(0)


def get_params():
    return SimformerParams(
        rngs=get_rngs(),
        in_channels=1,
        val_emb_dim=2,
        id_emb_dim=2,
        cond_emb_dim=2,
        dim_joint=4,
        fourier_features=8,
        num_heads=2,
        depth=2,
        mlp_ratio=2,
        qkv_features=4,
        num_hidden_layers=1,
        param_dtype=jnp.bfloat16,
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

    edge_mask = jnp.ones((1, 4, 4))
    out = model(
        t, x, node_ids=node_ids, condition_mask=condition_mask, edge_mask=edge_mask
    )
    assert out.shape == (1, 4, 1), f"Output shape is incorrect, got {out.shape}"

    edge_mask = jnp.ones((4, 4))
    out = model(
        t, x, node_ids=node_ids, condition_mask=condition_mask, edge_mask=edge_mask
    )
    assert out.shape == (1, 4, 1), f"Output shape is incorrect, got {out.shape}"

    # test shape error
    edge_mask = jnp.zeros(4)
    with pytest.raises(ValueError):
        out = model(
            t, x, node_ids=node_ids, condition_mask=condition_mask, edge_mask=edge_mask
        )


def test_simformer_wrapper():
    params = get_params()
    model = Simformer(params)
    wrapper = JointWrapper(model)

    obs = jnp.ones((12, 2, 1))
    cond = jnp.ones((12, 2, 1))
    obs_ids = jnp.arange(2).reshape(1, -1)
    cond_ids = jnp.arange(2).reshape(1, -1)
    t = jnp.ones((12, 1))

    extra_args = {
        "cond": cond,
        "cond_ids": cond_ids,
        "obs_ids": obs_ids,
        "edge_mask": None,
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


def test_simformer_param_dtype_propagation():
    params = SimformerParams(
        rngs=get_rngs(),
        in_channels=1,
        val_emb_dim=2,
        id_emb_dim=2,
        cond_emb_dim=2,
        dim_joint=4,
        fourier_features=8,
        num_heads=2,
        depth=2,
        mlp_ratio=2,
        qkv_features=4,
        num_hidden_layers=1,
        param_dtype=jnp.bfloat16,
    )
    model = Simformer(params)

    assert model.condition_embedding[...].dtype == jnp.bfloat16
    assert model.embedding_time.B[...].dtype == jnp.bfloat16
    assert model.embedding_net_value.p_skip[...].dtype == jnp.bfloat16
    assert model.embedding_net_id.embedding[...].dtype == jnp.bfloat16
    assert model.output_fn.kernel[...].dtype == jnp.bfloat16

    # Transformer internals (one representative parameter is enough)
    assert model.transformer.layer_norm.scale[...].dtype == jnp.bfloat16
    assert (
        model.transformer.attention_blocks[0].attn.query.kernel[...].dtype
        == jnp.bfloat16
    )


# Coverage improvement tests


def test_simformer_default_qkv_features():
    """Test SimformerParams with qkv_features=None (default calculated)."""
    params = SimformerParams(
        rngs=get_rngs(),
        in_channels=1,
        val_emb_dim=2,
        id_emb_dim=2,
        cond_emb_dim=2,
        dim_joint=4,
        fourier_features=8,
        num_heads=2,
        depth=2,
        mlp_ratio=2,
        qkv_features=None,  # should default to val+id+cond = 6
        num_hidden_layers=1,
    )
    assert params.qkv_features == 6  # 2 + 2 + 2


def test_simformer_custom_embedding_net():
    """Test Simformer with a custom embedding_net_value."""
    from gensbi.models.embedding import MLPEmbedder

    params = get_params()
    custom_emb = MLPEmbedder(
        in_dim=1,
        hidden_dim=params.val_emb_dim,
        rngs=get_rngs(),
        param_dtype=params.param_dtype,
    )
    model = Simformer(params, embedding_net_value=custom_emb)

    x = jnp.ones((1, 4, 1))
    t = jnp.ones((1, 1))
    node_ids = jnp.arange(4).reshape(1, 4)
    condition_mask = jnp.zeros((1, 4, 1))
    out = model(t, x, node_ids=node_ids, condition_mask=condition_mask)
    assert out.shape == (1, 4, 1)


def test_simformer_node_ids_1d():
    """Test Simformer with 1D node_ids (should be auto-reshaped)."""
    params = get_params()
    model = Simformer(params)

    x = jnp.ones((1, 4, 1))
    t = jnp.ones((1, 1))
    node_ids = jnp.arange(4)  # 1D
    condition_mask = jnp.zeros((1, 4, 1))
    out = model(t, x, node_ids=node_ids, condition_mask=condition_mask)
    assert out.shape == (1, 4, 1)


def test_simformer_node_ids_3d():
    """Test Simformer with 3D node_ids (shape (-1, seq_len, 1))."""
    params = get_params()
    model = Simformer(params)

    x = jnp.ones((1, 4, 1))
    t = jnp.ones((1, 1))
    node_ids = jnp.arange(4).reshape(1, 4, 1)  # 3D
    condition_mask = jnp.zeros((1, 4, 1))
    out = model(t, x, node_ids=node_ids, condition_mask=condition_mask)
    assert out.shape == (1, 4, 1)


def test_simformer_node_ids_invalid_ndim():
    """Test Simformer with 4D node_ids should raise ValueError."""
    params = get_params()
    model = Simformer(params)

    x = jnp.ones((1, 4, 1))
    t = jnp.ones((1, 1))
    node_ids = jnp.arange(4).reshape(1, 1, 4, 1)  # 4D — invalid
    condition_mask = jnp.zeros((1, 4, 1))
    with pytest.raises(ValueError, match="ndim"):
        model(t, x, node_ids=node_ids, condition_mask=condition_mask)

