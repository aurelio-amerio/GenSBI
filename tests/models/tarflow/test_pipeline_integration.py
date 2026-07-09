# tests/models/tarflow/test_pipeline_integration.py
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import grain

from gensbi.models import TarFlow, TarFlowParams
from gensbi.recipes.flow_pipeline import ConditionalFlowPipeline
from gensbi.inference import NLEPosterior
from gensbi.core.prior import make_gaussian_prior

# NLE convention: obs = x (M-dim), cond = theta (D-dim)
M, D, N = 3, 2, 1024
_k = jax.random.PRNGKey(0)
_kth, _kx = jax.random.split(_k)
_theta = jax.random.normal(_kth, (N, D))
_W = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])     # (M, D)
_x = _theta @ _W.T + 0.1 * jax.random.normal(_kx, (N, M))
DATA = jnp.concatenate([_x[..., None], _theta[..., None]], axis=1)  # (N, M+D, 1)


def _split(d):
    return d[:, :M], d[:, M:]            # (obs=x, cond=theta)


def _ds(arr, bs=128):
    return (grain.MapDataset.source(np.array(arr)).shuffle(0).repeat()
            .to_iter_dataset().batch(bs).map(_split))


def _pipe(tmp_path):
    flow = TarFlow(TarFlowParams(rngs=nnx.Rngs(0), dim=M, cond_dim=D, head_dim=8,
                                 num_heads=2, num_blocks=4, layers_per_block=2,
                                 standardize=True))
    cfg = ConditionalFlowPipeline.get_default_training_config()
    cfg.update(dict(val_every=1, checkpoint_dir=str(tmp_path)))
    return ConditionalFlowPipeline(flow, _ds(DATA[:800]), _ds(DATA[800:]),
                                   M, D, ch_obs=1, ch_cond=1,
                                   training_config=cfg)


def test_loss_scalar_and_finite(tmp_path):
    pipe = _pipe(tmp_path)
    loss_fn = pipe.get_loss_fn()
    obs = jnp.asarray(DATA[:32, :M])
    cond = jnp.asarray(DATA[:32, M:])
    loss = loss_fn(pipe.model, (obs, cond), key=jax.random.PRNGKey(0))
    assert loss.shape == () and jnp.isfinite(loss)


def test_grads_flow_to_params(tmp_path):
    pipe = _pipe(tmp_path)
    loss_fn = pipe.get_loss_fn()
    obs, cond = jnp.asarray(DATA[:32, :M]), jnp.asarray(DATA[:32, M:])
    grads = nnx.grad(loss_fn)(pipe.model, (obs, cond), jax.random.PRNGKey(0))
    leaves = jax.tree_util.tree_leaves(grads)
    assert any(jnp.any(jnp.abs(g) > 0) for g in leaves)


def test_fit_standardization_sets_both_models(tmp_path):
    pipe = _pipe(tmp_path)
    pipe.fit_standardization(DATA[:800, :M])     # standardize x
    exp_mean = jnp.mean(DATA[:800, :M], axis=0)  # (M, 1)
    exp_std = jnp.std(DATA[:800, :M], axis=0)    # (M, 1)
    for flow in (pipe.model, pipe.ema_model):
        assert jnp.allclose(flow.mean[...], exp_mean, atol=1e-4)
        assert jnp.allclose(flow.std[...], exp_std, atol=1e-4)
    assert pipe._standardized is True


def test_train_smoke_and_log_prob(tmp_path):
    pipe = _pipe(tmp_path)
    pipe.fit_standardization(DATA[:800, :M])
    pipe.train(nnx.Rngs(0), nsteps=2, save_model=False)
    x_1 = jnp.zeros((5, M, 1))
    x_o = jnp.zeros((1, D, 1))
    lp = pipe.log_prob(x_1, x_o, use_ema=False)
    assert lp.shape == (5,) and jnp.all(jnp.isfinite(lp))


def test_nle_log_posterior_value_and_grad(tmp_path):
    flow = TarFlow(TarFlowParams(rngs=nnx.Rngs(0), dim=M, cond_dim=D, head_dim=8,
                                 num_heads=2, num_blocks=3, layers_per_block=1,
                                 zero_init=False))
    prior = make_gaussian_prior((D,))
    post = NLEPosterior(flow, prior)
    target = post.build_target(jnp.array([0.5, -0.5, 0.2]))
    theta = jnp.array([0.1, 0.2])
    val = target.log_posterior(theta)
    grad = jax.grad(target.log_posterior)(theta)
    assert val.shape == () and jnp.isfinite(val)
    assert grad.shape == (D,) and jnp.all(jnp.isfinite(grad))
