import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from gensbi.experimental.models.pixeldit.model import PixelDiT, PixelDiTParams


def _params(**overrides):
    base = dict(
        in_channels=1,
        field_shape=(16, 16),
        cond_dim=2,
        rngs=nnx.Rngs(0),
        hidden_size=64,
        num_heads=4,
        patch_depth=2,
        pixel_depth=2,
        patch_size=4,
        pixel_hidden_size=8,
        param_dtype=jnp.float32,
    )
    base.update(overrides)
    return PixelDiTParams(**base)


def _model(seed=0, **overrides):
    overrides.setdefault("rngs", nnx.Rngs(seed))
    return PixelDiT(_params(**overrides))


def _inputs(B=2, H=16, W=16, C=1, K=2, cond_channels=1):
    obs = jax.random.normal(jax.random.PRNGKey(1), (B, H, W, C))
    if cond_channels == 1:
        cond = jax.random.normal(jax.random.PRNGKey(2), (B, K, 1))
    else:
        cond = jax.random.normal(jax.random.PRNGKey(2), (B, K, cond_channels))
    t = jnp.ones((B,))
    return t, obs, cond


# --------------------------------------------------------------------------
# Params __post_init__
# --------------------------------------------------------------------------


def test_params_derive_grid_and_token_count():
    p = _params()
    assert p.token_grid == (4, 4)        # 16 / 4
    assert p.n_obs_tokens == 16
    assert p.resolved_pixel_attn_hidden_size == 64
    assert p.resolved_pixel_num_heads == 4


def test_params_reject_indivisible_field_shape():
    with pytest.raises(AssertionError, match="divisible by patch_size"):
        _params(field_shape=(14, 16))


def test_params_reject_hidden_not_divisible_by_heads():
    with pytest.raises(AssertionError, match="num_heads"):
        _params(hidden_size=65)


def test_params_reject_head_dim_not_mult_of_4():
    # hidden 64 / num_heads 32 -> head_dim 2, not a multiple of 4
    with pytest.raises(AssertionError, match="divisible by 4"):
        _params(num_heads=32)


def test_params_reject_pixel_attn_indivisible():
    with pytest.raises(AssertionError, match="pixel_num_heads"):
        _params(pixel_attn_hidden_size=10, pixel_num_heads=4)


def test_params_pixel_defaults_fall_back_to_patch():
    p = _params(pixel_attn_hidden_size=None, pixel_num_heads=None)
    assert p.resolved_pixel_attn_hidden_size == p.hidden_size
    assert p.resolved_pixel_num_heads == p.num_heads


# --------------------------------------------------------------------------
# Forward shape + zero at init
# --------------------------------------------------------------------------


def test_forward_shape_and_zero_at_init():
    model = _model()
    t, obs, cond = _inputs()
    v = model(t, obs, cond)
    assert v.shape == obs.shape
    assert jnp.all(jnp.isfinite(v))
    # zero adaLN + zero final layer => exactly zero velocity at init
    assert jnp.array_equal(v, jnp.zeros_like(v))


def test_zero_init_blocks_false_still_zero_but_blocks_nonzero():
    model = _model(zero_init_blocks=False)
    t, obs, cond = _inputs()
    v = model(t, obs, cond)
    # final layer alone gates the output to zero
    assert jnp.array_equal(v, jnp.zeros_like(v))
    # but some block adaLN kernel is non-zero in state now
    params = nnx.state(model, nnx.Param)
    adaln_x = params["patch_blocks"][0]["adaLN_x"]["kernel"][...]
    assert jnp.any(jnp.abs(adaln_x) > 0)


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------


def test_rejects_rank3_obs():
    model = _model()
    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 16, 16))
    cond = jax.random.normal(jax.random.PRNGKey(2), (2, 2, 1))
    with pytest.raises(ValueError, match="rank"):
        model(jnp.ones((2,)), obs, cond)


def test_rejects_wrong_spatial_shape():
    model = _model()  # field_shape (16, 16)
    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 8, 8, 1))
    cond = jax.random.normal(jax.random.PRNGKey(2), (2, 2, 1))
    with pytest.raises(ValueError, match="field_shape"):
        model(jnp.ones((2,)), obs, cond)


def test_rejects_wrong_channel_count():
    model = _model()  # in_channels 1
    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 16, 16, 3))
    cond = jax.random.normal(jax.random.PRNGKey(2), (2, 2, 1))
    with pytest.raises(ValueError, match="in_channels"):
        model(jnp.ones((2,)), obs, cond)


def test_conditioned_false_raises():
    model = _model()
    t, obs, cond = _inputs()
    with pytest.raises(NotImplementedError, match="unconditional"):
        model(t, obs, cond, conditioned=False)


def test_guidance_raises():
    model = _model()
    t, obs, cond = _inputs()
    with pytest.raises(ValueError, match="guidance"):
        model(t, obs, cond, guidance=1.0)


def test_cond_2d_accepted_when_cond_in_channels_one():
    model = _model()  # cond_in_channels 1
    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 16, 16, 1))
    cond = jax.random.normal(jax.random.PRNGKey(2), (2, 2))  # (B, K)
    v = model(jnp.ones((2,)), obs, cond)
    assert v.shape == obs.shape


# --------------------------------------------------------------------------
# Ignored ids
# --------------------------------------------------------------------------


def test_ids_ignored():
    model = _model()
    t, obs, cond = _inputs()
    v_clean = model(t, obs, cond)
    v_garbage = model(t, obs, cond, obs_ids="garbage", cond_ids=12345)
    assert jnp.array_equal(v_clean, v_garbage)


# --------------------------------------------------------------------------
# Differentiable
# --------------------------------------------------------------------------


def test_differentiable_wrt_obs():
    model = _model()
    t, obs, cond = _inputs()

    def loss_fn(obs):
        return jnp.sum(model(t, obs, cond) ** 2)

    g = jax.grad(loss_fn)(obs)
    assert g.shape == obs.shape
    assert jnp.all(jnp.isfinite(g))


# --------------------------------------------------------------------------
# Config variants construct + run
# --------------------------------------------------------------------------


def test_use_cond_rope_false_runs():
    model = _model(use_cond_rope=False)
    assert model.pe_cond is None
    t, obs, cond = _inputs()
    v = model(t, obs, cond)
    assert v.shape == obs.shape
    assert jnp.all(jnp.isfinite(v))


def test_cond_id_embedding_none_runs():
    model = _model(cond_id_embedding="none")
    t, obs, cond = _inputs()
    v = model(t, obs, cond)
    assert v.shape == obs.shape
    assert jnp.all(jnp.isfinite(v))


def test_cond_id_embedding_pos1d_runs():
    model = _model(cond_id_embedding="pos1d")
    t, obs, cond = _inputs()
    v = model(t, obs, cond)
    assert v.shape == obs.shape
    assert jnp.all(jnp.isfinite(v))


# --------------------------------------------------------------------------
# Buffers excluded from Param state
# --------------------------------------------------------------------------


def test_rope_buffers_excluded_from_params():
    model = _model()
    param_leaves = jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))
    # no (..., 2, 2) rope rotation tables in Param state
    assert all(l.shape[-2:] != (2, 2) for l in param_leaves)


def test_model_does_not_store_params_dataclass():
    model = _model()
    assert not hasattr(model, "params")


# --------------------------------------------------------------------------
# cond shape guards
# --------------------------------------------------------------------------


def test_rejects_rank1_cond():
    model = _model()
    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 16, 16, 1))
    cond = jax.random.normal(jax.random.PRNGKey(2), (2,))   # rank-1: neither (B,K) nor (B,K,C)
    with pytest.raises(ValueError, match="rank"):
        model(jnp.ones((2,)), obs, cond)


def test_rejects_wrong_k_cond_none_embedding_no_rope():
    """Silent path: cond_id_embedding='none' + use_cond_rope=False; wrong K must raise."""
    model = _model(cond_id_embedding="none", use_cond_rope=False)
    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 16, 16, 1))
    cond_bad = jax.random.normal(jax.random.PRNGKey(2), (2, 99, 1))  # K=99, expected K=2
    with pytest.raises(ValueError, match="cond_dim"):
        model(jnp.ones((2,)), obs, cond_bad)


def test_rejects_wrong_k_cond_standard_config():
    """Wrong K is caught even with the default (absolute + rope) config."""
    model = _model()  # cond_dim=2
    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 16, 16, 1))
    cond_bad = jax.random.normal(jax.random.PRNGKey(2), (2, 5, 1))   # K=5, expected K=2
    with pytest.raises(ValueError, match="cond_dim"):
        model(jnp.ones((2,)), obs, cond_bad)


def test_rejects_rank3_cond_wrong_channel_dim():
    """Rank-3 cond with correct K but wrong last dim raises ValueError."""
    model = _model()  # cond_in_channels=1, cond_dim=2
    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 16, 16, 1))
    cond_bad = jax.random.normal(jax.random.PRNGKey(2), (2, 2, 5))  # K=2 ok, C=5 wrong
    with pytest.raises(ValueError, match="cond_in_channels"):
        model(jnp.ones((2,)), obs, cond_bad)


# --------------------------------------------------------------------------
# Non-square field_shape
# --------------------------------------------------------------------------


def test_nonsquare_field_shape_forward_shape_and_zero_at_init():
    """(16, 8) field with patch_size=4 → token_grid (4, 2); output shape matches."""
    model = _model(field_shape=(16, 8), patch_size=4)
    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 16, 8, 1))
    cond = jax.random.normal(jax.random.PRNGKey(2), (2, 2, 1))
    t = jnp.ones((2,))
    v = model(t, obs, cond)
    assert v.shape == obs.shape
    assert jnp.all(jnp.isfinite(v))
    assert jnp.array_equal(v, jnp.zeros_like(v))


# --------------------------------------------------------------------------
# Multi-channel cond
# --------------------------------------------------------------------------


def test_cond_in_channels_3_forward():
    """Full-model forward with cond_in_channels=3."""
    model = _model(cond_in_channels=3)
    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 16, 16, 1))
    cond = jax.random.normal(jax.random.PRNGKey(2), (2, 2, 3))   # K=2, C=3
    t = jnp.ones((2,))
    v = model(t, obs, cond)
    assert v.shape == obs.shape
    assert jnp.all(jnp.isfinite(v))


# --------------------------------------------------------------------------
# t shape normalisation — (B,1) must equal (B,)
# --------------------------------------------------------------------------


def test_t_shape_b1_equals_b():
    """Forward with t of shape (B,1) must produce identical output to (B,).

    FieldConditionalWrapper passes t as (B,1) via _expand_time; PixelDiT
    normalises to (B,) at the model boundary so the faithful port's
    _timestep_embedding is not affected.
    """
    model = _model(zero_init_blocks=False)
    t_1d, obs, cond = _inputs()
    t_2d = t_1d[:, None]  # (B,) -> (B,1)

    v_1d = model(t_1d, obs, cond)
    v_2d = model(t_2d, obs, cond)

    assert v_1d.shape == obs.shape
    assert v_2d.shape == obs.shape
    assert jnp.array_equal(v_1d, v_2d)


# --------------------------------------------------------------------------
# Package-level exports
# --------------------------------------------------------------------------


def test_package_exports_pixeldit_and_params():
    from gensbi.experimental.models import PixelDiT as Exp_PixelDiT
    from gensbi.experimental.models import PixelDiTParams as Exp_PixelDiTParams
    from gensbi.experimental.models.pixeldit import PixelDiT as Pkg_PixelDiT
    from gensbi.experimental.models.pixeldit import PixelDiTParams as Pkg_PixelDiTParams
    from gensbi.experimental.models.pixeldit import MMDiTBlock, PiTBlock

    assert Exp_PixelDiT is Pkg_PixelDiT
    assert Exp_PixelDiTParams is Pkg_PixelDiTParams
    assert MMDiTBlock is not None
    assert PiTBlock is not None
