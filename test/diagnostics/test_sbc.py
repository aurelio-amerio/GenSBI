# %%
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from flax import nnx

import pytest

from gensbi.diagnostics import run_sbc, sbc_rank_plot, check_sbc


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
        return jax.random.normal(key, shape) + 1.05

    def sample_batched(
        self,
        key,
        cond,
        nsamples,
        *args,
        **kwargs,
    ):
        shape = (nsamples, cond.shape[0], self.dim_obs, self.ch_obs)
        return jax.random.normal(key, shape) + 1.05


def test_sbc():
    pipeline = MockPipeline()
    # SBC diagnostic

    xs = jax.random.normal(
        jax.random.PRNGKey(0), (200, pipeline.dim_cond, pipeline.ch_cond)
    )
    thetas = (
        jax.random.normal(
            jax.random.PRNGKey(1), (200, pipeline.dim_obs, pipeline.ch_obs)
        )
        + 1.05
    )

    num_posterior_samples = 1_000

    posterior_samples = pipeline.sample_batched(
        jax.random.PRNGKey(12345), xs, num_posterior_samples, use_ema=True
    )

    # reshape
    xs = xs.reshape((xs.shape[0], -1))
    thetas = thetas.reshape((thetas.shape[0], -1))
    posterior_samples = posterior_samples.reshape(
        (posterior_samples.shape[0], posterior_samples.shape[1], -1)
    )

    ranks, dap_samples = run_sbc(thetas, xs, posterior_samples)

    res_sbc = check_sbc(ranks, thetas, dap_samples, num_posterior_samples)
    assert "ks_pvals" in res_sbc.keys(), "ks_pvals not in results"
    assert "c2st_ranks" in res_sbc.keys(), "c2st_ranks not in results"
    assert "c2st_dap" in res_sbc.keys(), "c2st_dap not in results"

    fig, ax = sbc_rank_plot(ranks, num_posterior_samples, plot_type="hist", num_bins=20)

    assert isinstance(fig, plt.Figure), f"fig is not a matplotlib Figure, got {type(fig)}"
    
    fig, ax = sbc_rank_plot(ranks, num_posterior_samples, plot_type="cdf", num_bins=20)

    assert isinstance(fig, plt.Figure), f"fig is not a matplotlib Figure, got {type(fig)}"