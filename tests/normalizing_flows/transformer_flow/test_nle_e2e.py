import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import grain
import pytest

from gensbi.core.prior import make_gaussian_prior
from gensbi.normalizing_flows import make_tarflow
from gensbi.recipes.flow_pipeline import ConditionalFlowPipeline
from gensbi.inference import NLEPosterior

D, M, SIGMA = 2, 3, 0.5
G = jnp.array([[1.0, 0.5], [0.0, 1.0], [0.5, -1.0]])   # (M, D)


def _simulate(key, n):
    kth, ke = jax.random.split(key)
    theta = jax.random.normal(kth, (n, D))
    x = theta @ G.T + SIGMA * jax.random.normal(ke, (n, M))
    return theta, x


def _analytic_posterior(x_o):
    prec = jnp.eye(D) + (G.T @ G) / SIGMA ** 2
    cov = jnp.linalg.inv(prec)
    mean = cov @ (G.T @ x_o) / SIGMA ** 2
    return mean, cov


@pytest.mark.slow
def test_tarflow_nle_recovers_linear_gaussian(tmp_path):
    theta, x = _simulate(jax.random.PRNGKey(0), 20_000)
    data = jnp.concatenate([x[..., None], theta[..., None]], axis=1)  # x FIRST

    def split(d):
        return d[:, :M], d[:, M:]            # (obs=x, cond=theta)

    def make_ds(arr):
        return (grain.MapDataset.source(np.array(arr)).shuffle(0).repeat()
                .to_iter_dataset().batch(256).map(split))

    flow = make_tarflow(nnx.Rngs(0), dim=M, cond_dim=D, channels=64,
                        num_blocks=6, layers_per_block=2, head_dim=16,
                        standardize=True)
    cfg = ConditionalFlowPipeline.get_default_training_config()
    cfg.update(dict(nsteps=4000, val_every=200, max_lr=3e-4,
                    checkpoint_dir=str(tmp_path), early_stopping=False))
    pipe = ConditionalFlowPipeline(flow, make_ds(data[:18_000]),
                                   make_ds(data[18_000:]), M, D,
                                   ch_obs=1, ch_cond=1, training_config=cfg)
    pipe.fit_standardization(data[:18_000, :M])     # standardize x
    pipe.train(nnx.Rngs(0), nsteps=4000, save_model=False)

    x_o = jnp.array([1.0, -0.5, 0.3])
    mean_a, cov_a = _analytic_posterior(x_o)
    prior = make_gaussian_prior((D,))
    post = NLEPosterior(pipe.ema_model, prior, num_warmup=500, num_samples=4000)
    s = post.sample(jax.random.PRNGKey(7), x_o)[..., 0]   # (n, D)

    assert jnp.allclose(jnp.mean(s, axis=0), mean_a, atol=0.2), (jnp.mean(s, 0), mean_a)
    assert jnp.allclose(jnp.cov(s.T), cov_a, atol=0.2)
