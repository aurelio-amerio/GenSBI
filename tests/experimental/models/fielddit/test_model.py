import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from gensbi.experimental.models.fielddit.model import FieldDiTParams


def _params(**overrides):
    base = dict(
        in_channels=1,
        field_shape=(32, 32),
        encoder_widths=(8, 16, 32),  # D = 2
        cond_dim=3,
        rngs=nnx.Rngs(0),
        num_heads=2,
        axes_dim=[2, 2, 4],          # sum 8 -> hidden 16
        patch_size=2,
        param_dtype=jnp.float32,
    )
    base.update(overrides)
    return FieldDiTParams(**base)


def test_params_derive_hidden_and_grid():
    p = _params()
    assert p.hidden_size == 16           # sum([2,2,4]) * num_heads(2)
    assert p.depth_levels == 2           # len(encoder_widths) - 1
    assert (p.feat_h, p.feat_w) == (8, 8)  # 32 / 2**2
    assert p.token_grid == (4, 4)        # feat / patch_size
    assert p.n_obs_tokens == 16


def test_model_builds_rope_ids():
    model = _small_model()
    assert model.obs_ids[...].shape == (1, 16, 3)
    assert model.cond_ids[...].shape == (1, 3, 1)


def test_params_reject_indivisible_field_shape():
    with pytest.raises(AssertionError, match="divisible"):
        _params(field_shape=(30, 32))  # 30 not divisible by 2**2


def test_params_reject_indivisible_patch_size():
    # field 32, D=2 -> feat 8; 8 not divisible by patch_size 3
    with pytest.raises(AssertionError, match="patch_size"):
        _params(patch_size=3)


def test_params_reject_odd_axes_dim():
    with pytest.raises(AssertionError, match="even"):
        _params(axes_dim=[3, 2, 3])  # odd entries invalid for rope


def test_params_reject_wrong_axes_len():
    with pytest.raises(AssertionError, match="3 entries"):
        _params(axes_dim=[4, 4])


from gensbi.experimental.models.fielddit.model import FieldDiT


def _small_model(seed=0):
    return FieldDiT(_params(rngs=nnx.Rngs(seed)))


def test_fielddit_forward_shape_and_zero_init():
    model = _small_model()
    B = 2
    obs = jax.random.normal(jax.random.PRNGKey(1), (B, 32, 32, 1))
    cond = jax.random.normal(jax.random.PRNGKey(2), (B, 3, 1))
    t = jnp.ones((B,))
    v = model(t, obs, cond)
    assert v.shape == obs.shape
    # zero-init decoder conv_out => exactly-zero velocity at init (also proves
    # no NaN anywhere upstream, since 0 * NaN == NaN).
    assert jnp.allclose(v, 0.0)


def test_fielddit_handles_batch_sizes():
    model = _small_model()
    for B in (1, 4):
        obs = jax.random.normal(jax.random.PRNGKey(B), (B, 32, 32, 1))
        cond = jax.random.normal(jax.random.PRNGKey(B + 100), (B, 3, 1))
        t = jnp.ones((B,))
        v = model(t, obs, cond)
        assert v.shape == (B, 32, 32, 1)


def test_fielddit_ignores_extra_kwargs():
    """Accepts (and ignores) obs_ids/cond_ids/conditioned for wrapper compat."""
    model = _small_model()
    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 32, 32, 1))
    cond = jax.random.normal(jax.random.PRNGKey(2), (2, 3, 1))
    t = jnp.ones((2,))
    v = model(t, obs, cond, obs_ids="ignored", cond_ids="ignored", conditioned=True)
    assert v.shape == obs.shape


def test_fielddit_is_differentiable():
    model = _small_model()
    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 32, 32, 1))
    cond = jax.random.normal(jax.random.PRNGKey(2), (2, 3, 1))
    t = jnp.ones((2,))

    # NOTE: use a non-zero target so the loss does not vanish at v == 0. With
    # mean(v**2) the gradient is identically zero at init (v == 0), and the
    # zero-init output conv blocks gradient to pre-final params at step 0 — both
    # are expected for a zero-init output layer, so we only assert finiteness +
    # a non-zero gradient on the final conv (the output path is connected).
    def loss_fn(model):
        return jnp.mean((model(t, obs, cond) - 1.0) ** 2)

    grads = nnx.grad(loss_fn)(model)
    leaves = jax.tree_util.tree_leaves(nnx.state(grads, nnx.Param))
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)
    conv_out_grad = nnx.state(grads, nnx.Param)["decoder"]["conv_out"]["kernel"][...]
    assert jnp.any(jnp.abs(conv_out_grad) > 0)


def test_fielddit_param_dtype_propagates():
    params = _params(rngs=nnx.Rngs(0), param_dtype=jnp.bfloat16)
    model = FieldDiT(params)
    assert model.time_in.in_layer.kernel[...].dtype == jnp.bfloat16
    assert model.decoder.conv_out.kernel[...].dtype == jnp.bfloat16


def test_fielddit_bfloat16_forward_runs():
    """Smoke-test the full assembly in the bfloat16 default compute config;
    the conv codec path (GroupNorm/conv + depatchify->decoder) is not
    otherwise forward-tested in bf16. models-emit-fp32 contract: the output
    is always fp32 regardless of the compute-dtype knob (zero-init conv_out
    is constructed with dtype=jnp.float32)."""
    model = FieldDiT(_params(rngs=nnx.Rngs(0), dtype=jnp.bfloat16))
    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 32, 32, 1))
    cond = jax.random.normal(jax.random.PRNGKey(2), (2, 3, 1))
    t = jnp.ones((2,))
    v = model(t, obs, cond)
    assert v.shape == obs.shape
    assert v.dtype == jnp.float32
    assert jnp.all(jnp.isfinite(v))


def test_fielddit_guidance_embed_path():
    """The optional guidance plumbing hook: vec += guidance_in(guidance)."""
    model = FieldDiT(_params(rngs=nnx.Rngs(0), guidance_embed=True, vec_in_dim=4))
    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 32, 32, 1))
    cond = jax.random.normal(jax.random.PRNGKey(2), (2, 3, 1))
    t = jnp.ones((2,))
    guidance = jnp.ones((2, 4))
    v = model(t, obs, cond, guidance=guidance)
    assert v.shape == obs.shape
    assert jnp.all(jnp.isfinite(v))


def test_fielddit_guidance_embed_requires_guidance():
    """guidance_embed=True with guidance=None must raise."""
    model = FieldDiT(_params(rngs=nnx.Rngs(0), guidance_embed=True, vec_in_dim=4))
    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 32, 32, 1))
    cond = jax.random.normal(jax.random.PRNGKey(2), (2, 3, 1))
    t = jnp.ones((2,))
    with pytest.raises(ValueError, match="guidance"):
        model(t, obs, cond, guidance=None)


def test_public_exports():
    """FieldDiT and FieldDiTParams must be reachable from both import paths."""
    from gensbi.experimental.models.fielddit import FieldDiT as FD, FieldDiTParams as FDP
    from gensbi.experimental.models import FieldDiT as FD2, FieldDiTParams as FDP2

    assert FD is FD2
    assert FDP is FDP2


def test_fielddit_rejects_cond_token_count_mismatch():
    """cond_dim=1 but cond carries 3 tokens -> guard fires (would silently broadcast otherwise)."""
    model = FieldDiT(_params(rngs=nnx.Rngs(0), cond_dim=1))
    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 32, 32, 1))
    cond = jax.random.normal(jax.random.PRNGKey(2), (2, 3, 1))  # 3 tokens != cond_dim 1
    t = jnp.ones((2,))
    with pytest.raises(AssertionError, match="cond_dim"):
        model(t, obs, cond)


def test_fielddit_conditioned_false_raises():
    """No unconditional path exists yet (CFG deferred): must fail loudly."""
    model = _small_model()
    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 32, 32, 1))
    cond = jax.random.normal(jax.random.PRNGKey(2), (2, 3, 1))
    t = jnp.ones((2,))
    with pytest.raises(NotImplementedError, match="unconditional"):
        model(t, obs, cond, conditioned=False)


def test_fielddit_rejects_wrong_spatial_shape():
    """obs spatial dims must match field_shape; fail at the door, not in attention."""
    model = _small_model()  # field_shape (32, 32)
    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 16, 16, 1))
    cond = jax.random.normal(jax.random.PRNGKey(2), (2, 3, 1))
    t = jnp.ones((2,))
    with pytest.raises(ValueError, match="field_shape"):
        model(t, obs, cond)


def test_fielddit_rejects_wrong_channel_count():
    """obs channel count must equal in_channels; fail at the door, not in the conv."""
    model = _small_model()  # in_channels 1
    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 32, 32, 3))
    cond = jax.random.normal(jax.random.PRNGKey(2), (2, 3, 1))
    t = jnp.ones((2,))
    with pytest.raises(ValueError, match="in_channels"):
        model(t, obs, cond)


def test_fielddit_rejects_wrong_rank():
    """obs missing the channel axis must fail at the door, not in the conv."""
    model = _small_model()
    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 32, 32))  # rank 3, no channel axis
    cond = jax.random.normal(jax.random.PRNGKey(2), (2, 3, 1))
    t = jnp.ones((2,))
    with pytest.raises(ValueError, match="rank"):
        model(t, obs, cond)


def test_timestep_embedding_receives_f32(monkeypatch):
    """The sinusoidal embedding must be computed in f32 even for a bf16 model
    (bf16 t quantizes ~0.0005 differences in t away before the sinusoid)."""
    import gensbi.experimental.models.fielddit.model as fielddit_model

    seen = {}
    orig = fielddit_model.timestep_embedding

    def spy(t, dim, **kwargs):
        seen["dtype"] = t.dtype
        return orig(t, dim, **kwargs)

    monkeypatch.setattr(fielddit_model, "timestep_embedding", spy)

    model = FieldDiT(_params(rngs=nnx.Rngs(0), dtype=jnp.bfloat16))
    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 32, 32, 1))
    cond = jax.random.normal(jax.random.PRNGKey(2), (2, 3, 1))
    v = model(jnp.ones((2,)), obs, cond)
    assert seen["dtype"] == jnp.float32
    assert v.dtype == jnp.float32  # models-emit-fp32 contract: output always fp32


def test_theta_default_derives_from_token_count():
    # default test config: 16 obs tokens + 3 cond tokens -> theta = 190
    p = _params()
    assert p.theta == 10 * (p.n_obs_tokens + p.cond_dim) == 190


def test_theta_explicit_override_wins():
    p = _params(theta=777)
    assert p.theta == 777


def test_theta_default_capped_at_10k():
    # 64x64 meeting grid (field 256, D=1, p=2 -> 128x128 feat -> 64x64 grid = 4096 tokens)
    p = _params(field_shape=(256, 256), encoder_widths=(8, 16))
    assert p.theta == 10_000


def test_graphdef_hashable_and_equal_across_instances():
    """Two identically-configured models must share a hashable, equal GraphDef
    (otherwise nnx.jit retraces per instance — EMA/eval patterns pay twice)."""
    m1 = FieldDiT(_params(rngs=nnx.Rngs(0)))
    m2 = FieldDiT(_params(rngs=nnx.Rngs(1)))
    g1, _ = nnx.split(m1)
    g2, _ = nnx.split(m2)
    hash(g1)  # must not raise
    assert g1 == g2


def test_rope_ids_are_filterable_variables():
    """obs/cond ids live in a dedicated Variable type: excluded from Param
    state and immune to blanket float casts over Params."""
    from gensbi.experimental.models.fielddit import RopeIds

    model = _small_model()
    ids_state = nnx.state(model, RopeIds)
    leaves = jax.tree_util.tree_leaves(ids_state)
    assert len(leaves) == 2  # obs_ids, cond_ids
    assert all(l.dtype == jnp.int32 for l in leaves)
    # and they are NOT in the Param state
    param_leaves = jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))
    assert all(jnp.issubdtype(l.dtype, jnp.floating) for l in param_leaves)


def test_model_does_not_store_params_dataclass():
    model = _small_model()
    assert not hasattr(model, "params")


def _open_gates(model):
    """Surgery so cond can reach the output at all: open the zero-init output
    conv and one encoder-stage modulation (everything is gated shut at init)."""
    k = model.decoder.conv_out.kernel
    k[...] = jnp.ones_like(k[...])
    mod = model.encoder.down.layers[0].block.layers[0].mod.lin
    mod.kernel[...] = 0.01 * jnp.ones_like(mod.kernel[...])


def test_cond_modulates_encoder_routes_cond_through_encoder():
    model = FieldDiT(_params(rngs=nnx.Rngs(0), cond_modulates_encoder=True))
    _open_gates(model)
    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 32, 32, 1))
    t = jnp.ones((2,))
    cond_a = jnp.zeros((2, 3, 1))
    cond_b = jnp.ones((2, 3, 1))
    v_a = model(t, obs, cond_a)
    v_b = model(t, obs, cond_b)
    # encoder modulation sees vec (incl. cond summary) -> output must differ
    assert not jnp.allclose(v_a, v_b)


def test_encoder_is_cond_free_by_default():
    model = FieldDiT(_params(rngs=nnx.Rngs(0)))  # flag off
    _open_gates(model)
    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 32, 32, 1))
    t = jnp.ones((2,))
    v_a = model(t, obs, jnp.zeros((2, 3, 1)))
    v_b = model(t, obs, jnp.ones((2, 3, 1)))
    # only the encoder path is opened; with a cond-free encoder (and all other
    # gates still zero-init) the cond cannot reach the output
    assert jnp.allclose(v_a, v_b)


def test_cond_modulates_encoder_preserves_zero_at_init():
    model = FieldDiT(_params(rngs=nnx.Rngs(0), cond_modulates_encoder=True))
    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 32, 32, 1))
    cond = jax.random.normal(jax.random.PRNGKey(2), (2, 3, 1))
    v = model(jnp.ones((2,)), obs, cond)
    assert jnp.allclose(v, 0.0)


def test_fielddit_non_square_field():
    model = FieldDiT(_params(rngs=nnx.Rngs(0), field_shape=(16, 32)))
    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 16, 32, 1))
    cond = jax.random.normal(jax.random.PRNGKey(2), (2, 3, 1))
    v = model(jnp.ones((2,)), obs, cond)
    assert v.shape == (2, 16, 32, 1)
    assert jnp.allclose(v, 0.0)  # zero-at-init must survive non-square grids


def test_fielddit_patch_size_one():
    model = FieldDiT(_params(rngs=nnx.Rngs(0), patch_size=1))
    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 32, 32, 1))
    cond = jax.random.normal(jax.random.PRNGKey(2), (2, 3, 1))
    v = model(jnp.ones((2,)), obs, cond)
    assert v.shape == obs.shape


def test_fielddit_single_level_encoder():
    model = FieldDiT(_params(rngs=nnx.Rngs(0), encoder_widths=(8, 16)))  # D = 1
    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 32, 32, 1))
    cond = jax.random.normal(jax.random.PRNGKey(2), (2, 3, 1))
    v = model(jnp.ones((2,)), obs, cond)
    assert v.shape == obs.shape


def test_fielddit_split_merge_jit_roundtrip():
    """The bare-nnx.Module-container idiom in the codec must survive
    split/merge and nnx.jit (checkpointing + compiled training rely on it)."""
    model = _small_model()
    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 32, 32, 1))
    cond = jax.random.normal(jax.random.PRNGKey(2), (2, 3, 1))
    t = jnp.ones((2,))

    v_eager = model(t, obs, cond)

    graphdef, state = nnx.split(model)
    model2 = nnx.merge(graphdef, state)
    v_merged = model2(t, obs, cond)
    assert jnp.array_equal(v_eager, v_merged)

    @nnx.jit
    def fwd(m, t, obs, cond):
        return m(t, obs, cond)

    v_jit = fwd(model, t, obs, cond)
    assert v_jit.shape == v_eager.shape
    assert jnp.allclose(v_jit, v_eager, atol=1e-5)
