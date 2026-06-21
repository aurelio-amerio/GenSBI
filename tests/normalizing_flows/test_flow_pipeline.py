import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import grain
import pytest

from gensbi.normalizing_flows import make_maf
from gensbi.recipes.flow_pipeline import (
    ConditionalFlowPipeline, _squeeze_ch, _single_cond,
)

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
    flow = make_maf(nnx.Rngs(0), dim=DIM_OBS, cond_dim=DIM_COND,
                    n_layers=4, nn_width=32, nn_depth=2, standardize=True)
    train_ds = _make_ds(DATA[:800])
    val_ds = _make_ds(DATA[800:])
    training_config = ConditionalFlowPipeline.get_default_training_config()
    training_config["val_every"] = 1
    training_config.update(cfg)
    return ConditionalFlowPipeline(
        flow, train_ds, val_ds, DIM_OBS, DIM_COND,
        ch_obs=1, ch_cond=1, training_config=training_config)


def test_squeeze_ch():
    x = jnp.zeros((4, DIM_OBS, 1))
    assert _squeeze_ch(x).shape == (4, DIM_OBS)
    assert _squeeze_ch(jnp.zeros((4, DIM_OBS))).shape == (4, DIM_OBS)
    with pytest.raises(ValueError):
        _squeeze_ch(jnp.zeros((4, DIM_OBS, 2)))


def test_single_cond():
    assert _single_cond(jnp.zeros((1, DIM_COND, 1))).shape == (DIM_COND,)
    assert _single_cond(jnp.zeros((DIM_COND,))).shape == (DIM_COND,)


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
