import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import grain
import pytest

from gensbi.models import MAFlow, MAFlowParams
from gensbi.recipes.flow_pipeline import ConditionalFlowPipeline
from gensbi.recipes.utils import _require_channel, _single_obs

DIM_OBS = 2
DIM_COND = 3
N = 1024

_key = jax.random.PRNGKey(0)
_kth, _kx = jax.random.split(_key)
_theta = jax.random.normal(_kth, (N, DIM_OBS))
_W = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])          # (DIM_COND, DIM_OBS)
_x = _theta @ _W.T + 0.1 * jax.random.normal(_kx, (N, DIM_COND))
DATA = jnp.concatenate([_theta[..., None], _x[..., None]], axis=1)  # (N, 5, 1)


def split_obs_cond(d):
    return d[:, :DIM_OBS], d[:, DIM_OBS:]


def _make_ds(arr, bs=128):
    return (grain.MapDataset.source(np.array(arr)).shuffle(0).repeat()
            .to_iter_dataset().batch(bs).map(split_obs_cond))


def build_pipeline(**cfg):
    flow = MAFlow(MAFlowParams(rngs=nnx.Rngs(0), dim=DIM_OBS, cond_dim=DIM_COND,
                               n_layers=4, nn_width=32, nn_depth=2, standardize=True))
    train_ds = _make_ds(DATA[:800])
    val_ds = _make_ds(DATA[800:])
    training_config = ConditionalFlowPipeline.get_default_training_config()
    training_config["val_every"] = 1
    training_config.update(cfg)
    return ConditionalFlowPipeline(
        flow, train_ds, val_ds, DIM_OBS, DIM_COND,
        ch_obs=1, ch_cond=1, training_config=training_config)


def test_require_channel_rejects_bare_2d():
    assert _require_channel(jnp.zeros((4, DIM_OBS, 1)), "obs").shape == (4, DIM_OBS, 1)
    with pytest.raises(ValueError):
        _require_channel(jnp.zeros((4, DIM_OBS)), "obs")


def test_single_obs_require_keeps_batch_and_channel():
    out = _single_obs(jnp.zeros((1, DIM_COND, 1)), channel="require")
    assert out.shape == (1, DIM_COND, 1)


def test_single_obs_none_keeps_structured_shape():
    img = jnp.arange(1 * 1 * 4 * 2).reshape(1, 1, 4, 2)
    assert _single_obs(img, channel="none").shape == (1, 1, 4, 2)


def test_single_obs_batched_raises():
    x_o = jnp.arange(3 * DIM_COND).reshape(3, DIM_COND, 1)
    with pytest.raises(ValueError, match="single observation"):
        _single_obs(x_o, channel="require")
    with pytest.raises(ValueError, match="single observation"):
        _single_obs(x_o, channel="none")


def test_single_obs_require_rejects_channelless():
    # (1, dim): documented contract violation -> the class-docstring ValueError
    with pytest.raises(ValueError, match="channel axis"):
        _single_obs(jnp.zeros((1, DIM_COND)), channel="require")
    # (dim, C): must NOT be misread as `dim` observations (review Finding 3)
    with pytest.raises(ValueError, match="channel axis"):
        _single_obs(jnp.zeros((DIM_COND, 2)), channel="require")


def test_single_obs_promote_1d_and_2d():
    assert _single_obs(jnp.zeros((DIM_COND,)), channel="promote").shape == (1, DIM_COND, 1)
    assert _single_obs(jnp.zeros((1, DIM_COND)), channel="promote").shape == (1, DIM_COND, 1)


def test_single_obs_none_rejects_rank_lt_2():
    with pytest.raises(ValueError):
        _single_obs(jnp.zeros((DIM_COND,)), channel="none")


def test_get_sampler_rejects_channelless_xo():
    pipe = build_pipeline()
    with pytest.raises(ValueError, match="channel axis"):
        pipe.get_sampler(jnp.zeros((1, DIM_COND)))
    with pytest.raises(ValueError, match="channel axis"):
        pipe.get_sampler(jnp.zeros((DIM_COND, 2)))


def test_get_log_prob_fn_rejects_channelless_xo():
    pipe = build_pipeline()
    with pytest.raises(ValueError, match="channel axis"):
        pipe.get_log_prob_fn(jnp.zeros((1, DIM_COND)))


def test_get_sampler_batched_xo_raises():
    pipe = build_pipeline()
    with pytest.raises(ValueError, match="single observation"):
        pipe.get_sampler(jnp.zeros((5, DIM_COND, 1)))


def test_init_and_wrap():
    pipe = build_pipeline()
    assert isinstance(pipe, ConditionalFlowPipeline)
    assert pipe.ema_model is not None
    assert pipe.model_wrapped is None            # not wrapped yet
    pipe._wrap_model()
    assert pipe.model_wrapped is pipe.model      # identity, no ConditionalWrapper
    assert pipe.ema_model_wrapped is pipe.ema_model


def test_stubs_raise():
    with pytest.raises(NotImplementedError):
        ConditionalFlowPipeline.get_default_params(2, 3, 1, 1)


def test_loss_fn_scalar_and_finite():
    pipe = build_pipeline()
    loss_fn = pipe.get_loss_fn()
    obs = jnp.asarray(DATA[:32, :DIM_OBS])      # (32, DIM_OBS, 1)
    cond = jnp.asarray(DATA[:32, DIM_OBS:])     # (32, DIM_COND, 1)
    loss = loss_fn(pipe.model, (obs, cond), key=jax.random.PRNGKey(0))
    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_loss_fn_has_param_gradients():
    # The flow's MaskedLinear kernels are nnx.Param; grads must flow to them.
    pipe = build_pipeline()
    loss_fn = pipe.get_loss_fn()
    obs = jnp.asarray(DATA[:32, :DIM_OBS])
    cond = jnp.asarray(DATA[:32, DIM_OBS:])
    # zero_init=True zeroes the MADE OUTPUT layer, so at init the OUTPUT-layer
    # weights carry the gradient (hidden/input grads are 0 until the output
    # moves off zero). It suffices that SOME Param leaf has a non-zero gradient.
    grads = nnx.grad(loss_fn)(pipe.model, (obs, cond), jax.random.PRNGKey(0))
    leaves = jax.tree_util.tree_leaves(grads)
    assert len(leaves) > 0
    assert any(jnp.any(jnp.abs(g) > 0) for g in leaves)


from gensbi.normalizing_flows.bijections.standardize import Standardize


def _get_std(flow):
    return [b for b in flow.chain.bijections if isinstance(b, Standardize)][0]


def test_fit_standardization_sets_both_models():
    pipe = build_pipeline()
    theta = DATA[:800, :DIM_OBS]                 # (800, DIM_OBS, 1)
    pipe.fit_standardization(theta)

    expected_mean = jnp.mean(theta[..., 0], axis=0)
    expected_std = jnp.std(theta[..., 0], axis=0)
    for flow in (pipe.model, pipe.ema_model):
        sb = _get_std(flow)
        assert jnp.allclose(sb.mean[...], expected_mean, atol=1e-4)
        assert jnp.allclose(sb.std[...], expected_std, atol=1e-4)
    assert pipe._standardized is True


def test_train_warns_without_standardization(tmp_path):
    pipe = build_pipeline(checkpoint_dir=str(tmp_path))
    with pytest.warns(UserWarning, match="fit_standardization"):
        pipe.train(nnx.Rngs(0), nsteps=1, save_model=False)


def test_sample_shape(tmp_path):
    pipe = build_pipeline(checkpoint_dir=str(tmp_path))
    pipe.fit_standardization(DATA[:800, :DIM_OBS])
    pipe.train(nnx.Rngs(0), nsteps=2, save_model=False)

    x_o = jnp.zeros((1, DIM_COND, 1))
    s = pipe.sample(jax.random.PRNGKey(1), x_o, nsamples=64, use_ema=False)
    assert s.shape == (64, DIM_OBS, 1)
    assert jnp.all(jnp.isfinite(s))

    s_ema = pipe.sample(jax.random.PRNGKey(1), x_o, nsamples=64, use_ema=True)
    assert s_ema.shape == (64, DIM_OBS, 1)


def test_log_prob_shape(tmp_path):
    pipe = build_pipeline(checkpoint_dir=str(tmp_path))
    pipe.fit_standardization(DATA[:800, :DIM_OBS])
    pipe.train(nnx.Rngs(0), nsteps=2, save_model=False)

    x_1 = jnp.zeros((5, DIM_OBS, 1))
    x_o = jnp.zeros((1, DIM_COND, 1))
    lp = pipe.log_prob(x_1, x_o, use_ema=False)
    assert lp.shape == (5,)
    assert jnp.all(jnp.isfinite(lp))


def test_log_prob_depends_on_condition(tmp_path):
    # Test the property on a LIVE flow (zero_init=False) so cond-dependence is
    # present immediately and does not rely on training dynamics. Phase-0
    # conditioning is concat-at-rank −1, so every output dim depends on cond.
    flow = MAFlow(MAFlowParams(rngs=nnx.Rngs(0), dim=DIM_OBS, cond_dim=DIM_COND,
                               n_layers=4, nn_width=32, nn_depth=2,
                               standardize=True, zero_init=False))
    cfg = ConditionalFlowPipeline.get_default_training_config()
    cfg["checkpoint_dir"] = str(tmp_path)
    pipe = ConditionalFlowPipeline(
        flow, _make_ds(DATA[:800]), _make_ds(DATA[800:]),
        DIM_OBS, DIM_COND, ch_obs=1, ch_cond=1, training_config=cfg)

    x_1 = jnp.zeros((5, DIM_OBS, 1))
    lp_a = pipe.log_prob(x_1, jnp.zeros((1, DIM_COND, 1)), use_ema=False)
    lp_b = pipe.log_prob(x_1, jnp.ones((1, DIM_COND, 1)), use_ema=False)
    assert not jnp.allclose(lp_a, lp_b)



def test_exported_from_recipes():
    from gensbi.recipes import ConditionalFlowPipeline as CFP
    assert CFP is ConditionalFlowPipeline


def test_sample_batched_shape(tmp_path):
    pipe = build_pipeline(checkpoint_dir=str(tmp_path))
    pipe.fit_standardization(DATA[:800, :DIM_OBS])
    pipe.train(nnx.Rngs(0), nsteps=2, save_model=False)

    B = 3
    x_o = jnp.zeros((B, DIM_COND, 1))
    s = pipe.sample_batched(jax.random.PRNGKey(2), x_o, nsamples=16, use_ema=False)
    assert s.shape == (16, B, DIM_OBS, 1)
    assert jnp.all(jnp.isfinite(s))

    # each per-condition slice equals the single-observation sampler for that cond
    s0 = pipe.sample(jax.random.PRNGKey(2), x_o[0:1], nsamples=16, use_ema=False)
    # not asserting equality of RNG streams across the two call paths; just shapes
    assert s0.shape == (16, DIM_OBS, 1)


class _EchoFlow:
    """Stub flow whose sample() echoes its condition — makes the
    condition->column routing of sample_batched exactly checkable."""
    def sample(self, key, cond):
        return cond


def test_sample_batched_routes_each_condition_to_its_column():
    pipe = build_pipeline()
    pipe.ema_model = _EchoFlow()
    B, nsamples = 3, 5
    x_o = jnp.stack([jnp.full((DIM_COND, 1), float(i)) for i in range(B)])
    out = pipe.sample_batched(jax.random.PRNGKey(0), x_o, nsamples)
    assert out.shape == (nsamples, B, DIM_COND, 1)
    for i in range(B):
        assert jnp.all(out[:, i] == float(i))


def test_sample_batched_shape_with_real_flow():
    pipe = build_pipeline()
    out = pipe.sample_batched(jax.random.PRNGKey(0), jnp.zeros((2, DIM_COND, 1)), 7)
    assert out.shape == (7, 2, DIM_OBS, 1)


def test_sample_batched_rejects_channelless_xo():
    pipe = build_pipeline()
    with pytest.raises(ValueError, match="channel axis"):
        pipe.sample_batched(jax.random.PRNGKey(0), jnp.zeros((2, DIM_COND)), 4)


def test_get_sampler_warns_on_unknown_kwarg():
    pipe = build_pipeline()
    with pytest.warns(UserWarning, match="ignores unsupported keyword"):
        pipe.get_sampler(jnp.zeros((1, DIM_COND, 1)), step_size=0.1)


def test_get_log_prob_fn_warns_on_unknown_kwarg():
    pipe = build_pipeline()
    with pytest.warns(UserWarning, match="ignores unsupported keyword"):
        pipe.get_log_prob_fn(jnp.zeros((1, DIM_COND, 1)), nsteps=10)


def test_sample_batched_warns_on_unknown_kwarg():
    pipe = build_pipeline()
    with pytest.warns(UserWarning, match="ignores unsupported keyword"):
        pipe.sample_batched(jax.random.PRNGKey(0), jnp.zeros((2, DIM_COND, 1)), 4,
                            solver="dopri5")


def test_known_calls_do_not_warn(recwarn):
    pipe = build_pipeline()
    pipe.get_sampler(jnp.zeros((1, DIM_COND, 1)))
    assert not any(
        "ignores unsupported" in str(w.message) for w in recwarn.list)


# ---------------------------------------------------------------------------
# Multichannel passthrough tests (Task 5)
# ---------------------------------------------------------------------------

def _build_multichannel_pipeline():
    CH = 2
    flow = MAFlow(MAFlowParams(rngs=nnx.Rngs(0), dim=DIM_OBS, cond_dim=DIM_COND,
                               channels=CH, n_layers=4, nn_width=32, nn_depth=2,
                               standardize=True))
    # obs carries a channel axis (N, DIM_OBS, CH); cond stays tabular (N, DIM_COND, 1)
    theta_c = jnp.broadcast_to(_theta[:, :, None], (N, DIM_OBS, CH))
    data = (theta_c, jnp.broadcast_to(_x[:, :, None], (N, DIM_COND, 1)))

    def gen(arr_obs, arr_cond, bs=128):
        idx = grain.MapDataset.source(np.arange(arr_obs.shape[0]))
        return (idx.shuffle(0).repeat().to_iter_dataset().batch(bs)
                .map(lambda i: (np.array(arr_obs)[i], np.array(arr_cond)[i])))

    train_ds = gen(data[0][:800], data[1][:800])
    val_ds = gen(data[0][800:], data[1][800:])
    tc = ConditionalFlowPipeline.get_default_training_config()
    tc["val_every"] = 1
    return ConditionalFlowPipeline(
        flow, train_ds, val_ds, DIM_OBS, DIM_COND,
        ch_obs=CH, ch_cond=1, training_config=tc)


def test_multichannel_prep_obs_passthrough():
    pipe = _build_multichannel_pipeline()
    x = jnp.zeros((5, DIM_OBS, 2))
    assert pipe._prep_obs(x).shape == (5, DIM_OBS, 2)   # NOT squeezed


def test_multichannel_sample_and_logprob_shapes():
    pipe = _build_multichannel_pipeline()
    x_o = jnp.zeros((1, DIM_COND, 1))
    s = pipe.sample(jax.random.PRNGKey(0), x_o, nsamples=7)
    assert s.shape == (7, DIM_OBS, 2)                   # channel axis preserved
    lp = pipe.log_prob(jnp.zeros((7, DIM_OBS, 2)), x_o)
    assert lp.shape == (7,)                             # one scalar per sample


def test_fit_standardization_per_channel_axis():
    pipe = _build_multichannel_pipeline()
    obs = jax.random.normal(jax.random.PRNGKey(3), (64, DIM_OBS, 2))
    pipe.fit_standardization(obs, axis=(0, 1))          # per-channel stats
    assert pipe._standardized


def _build_multichannel_both_pipeline():
    """Pipeline with ch_obs=2 AND ch_cond=2 — both passthrough paths active."""
    CH = 2
    flow = MAFlow(MAFlowParams(rngs=nnx.Rngs(0), dim=DIM_OBS, cond_dim=DIM_COND,
                               channels=CH, cond_channels=CH,
                               n_layers=4, nn_width=32, nn_depth=2,
                               standardize=True))
    # obs and cond both carry a channel axis (N, dim, CH)
    theta_c = jnp.broadcast_to(_theta[:, :, None], (N, DIM_OBS, CH))
    x_c = jnp.broadcast_to(_x[:, :, None], (N, DIM_COND, CH))

    def gen(arr_obs, arr_cond, bs=128):
        idx = grain.MapDataset.source(np.arange(arr_obs.shape[0]))
        return (idx.shuffle(0).repeat().to_iter_dataset().batch(bs)
                .map(lambda i: (np.array(arr_obs)[i], np.array(arr_cond)[i])))

    train_ds = gen(theta_c[:800], x_c[:800])
    val_ds = gen(theta_c[800:], x_c[800:])
    tc = ConditionalFlowPipeline.get_default_training_config()
    tc["val_every"] = 1
    return ConditionalFlowPipeline(
        flow, train_ds, val_ds, DIM_OBS, DIM_COND,
        ch_obs=CH, ch_cond=CH, training_config=tc)


def test_multichannel_both_sample_and_logprob_shapes():
    """Both ch_obs=ch_cond=2 active."""
    pipe = _build_multichannel_both_pipeline()

    # single cond must carry a leading batch axis + channel axis
    x_o = jnp.zeros((1, DIM_COND, 2))
    s = pipe.sample(jax.random.PRNGKey(0), x_o, nsamples=7)
    assert s.shape == (7, DIM_OBS, 2)           # channel axis preserved

    lp = pipe.log_prob(jnp.zeros((7, DIM_OBS, 2)), x_o)
    assert lp.shape == (7,)                     # one scalar per sample
