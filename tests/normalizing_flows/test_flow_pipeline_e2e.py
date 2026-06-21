import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import grain
import pytest

from gensbi.normalizing_flows import make_maf
from gensbi.recipes.flow_pipeline import ConditionalFlowPipeline

D = 2          # dim theta
M = 3          # dim x
SIGMA = 0.5
G = jnp.array([[1.0, 0.5], [0.0, 1.0], [0.5, -1.0]])   # (M, D)


def _simulate(key, n):
    kth, ke = jax.random.split(key)
    theta = jax.random.normal(kth, (n, D))
    x = theta @ G.T + SIGMA * jax.random.normal(ke, (n, M))
    return theta, x


def _analytic_posterior(x_o):
    prec = jnp.eye(D) + (G.T @ G) / SIGMA**2
    cov = jnp.linalg.inv(prec)
    mean = cov @ (G.T @ x_o) / SIGMA**2
    return mean, cov


def split_obs_cond(d):
    return d[:, :D], d[:, D:]


@pytest.mark.slow
def test_npe_recovers_linear_gaussian(tmp_path):
    key = jax.random.PRNGKey(0)
    theta, x = _simulate(key, 20_000)
    data = jnp.concatenate([theta[..., None], x[..., None]], axis=1)  # (N, D+M, 1)

    def make_ds(arr):
        return (grain.MapDataset.source(np.array(arr)).shuffle(0).repeat()
                .to_iter_dataset().batch(256).map(split_obs_cond))

    flow = make_maf(nnx.Rngs(0), dim=D, cond_dim=M,
                    n_layers=6, nn_width=64, nn_depth=2, standardize=True)
    cfg = ConditionalFlowPipeline.get_default_training_config()
    cfg.update(dict(nsteps=4000, val_every=200, max_lr=3e-4,
                    checkpoint_dir=str(tmp_path), early_stopping=False))
    pipe = ConditionalFlowPipeline(flow, make_ds(data[:18_000]),
                                   make_ds(data[18_000:]), D, M,
                                   ch_obs=1, ch_cond=1, training_config=cfg)
    pipe.fit_standardization(data[:18_000, :D])
    pipe.train(nnx.Rngs(0), nsteps=4000, save_model=False)

    x_o = jnp.array([1.0, -0.5, 0.3])
    mean_a, cov_a = _analytic_posterior(x_o)

    s = pipe.sample(jax.random.PRNGKey(7), x_o[None, :, None],
                    nsamples=20_000, use_ema=True)[..., 0]   # (n, D)
    mean_s = jnp.mean(s, axis=0)
    cov_s = jnp.cov(s.T)

    assert jnp.allclose(mean_s, mean_a, atol=0.1), (mean_s, mean_a)
    assert jnp.allclose(cov_s, cov_a, atol=0.1), (cov_s, cov_a)
