# %%
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from flax import nnx

import pytest

from gensbi.diagnostics import plot_lc2st, LC2ST


# %%
class MockPipeline:
    def __init__(
        self,
    ):

        self.dim_obs = 3
        self.ch_obs = 2

        self.dim_cond = 4
        self.ch_cond = 3

    def sample(self, key, cond, nsamples, *args, **kwargs):
        shape = (nsamples, self.dim_obs, self.ch_obs)
        return jax.random.normal(key, shape) + 1

    def sample_batched(
        self,
        key,
        cond,
        nsamples,
        *args,
        **kwargs,
    ):
        shape = (nsamples, cond.shape[0], self.dim_obs, self.ch_obs)
        return jax.random.normal(key, shape) + 1


def test_lc2st():
    pipeline = MockPipeline()
    # LC2ST diagnostic

    xs_ = jax.random.normal(jax.random.PRNGKey(0), (10_000, pipeline.dim_cond, pipeline.ch_cond)) 
    thetas_ = jax.random.normal(jax.random.PRNGKey(1), (10_000, pipeline.dim_obs, pipeline.ch_obs)) + 1.05 

    num_posterior_samples = 1

    posterior_samples_ = pipeline.sample(jax.random.PRNGKey(42), xs_, nsamples=xs_.shape[0])

    thetas = thetas_.reshape(thetas_.shape[0], -1)  
    xs = xs_.reshape(xs_.shape[0], -1)  
    posterior_samples = posterior_samples_.reshape(posterior_samples_.shape[0], -1)  

    # Train the L-C2ST classifier.
    lc2st = LC2ST(
        thetas=thetas[:-1],
        xs=xs[:-1],
        posterior_samples=posterior_samples[:-1],
        classifier="mlp",
        num_ensemble=1,
        num_trials_null=2,
    )

    _ = lc2st.train_under_null_hypothesis()
    _ = lc2st.train_on_observed_data()

    x_o = xs_[-1 : ]  # Take the last observation as observed data.
    theta_o = thetas_[-1 : ]  # True parameter for the observed data.

    post_samples_star = pipeline.sample(jax.random.PRNGKey(42), x_o, nsamples=10_000) 

    x_o = x_o.reshape(1,-1)  
    post_samples_star = np.array(post_samples_star.reshape(post_samples_star.shape[0], -1))  

    fig,ax = plot_lc2st(
        lc2st,
        post_samples_star,
        x_o,
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
# %%
