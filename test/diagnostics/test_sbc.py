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
from gensbi.diagnostics.sbc import _validate_sbc_inputs

def get_sbc_data():
    class DummyPipeline:
        def __init__(self):
            self.dim_obs = 3
            self.ch_obs = 2
            self.dim_cond = 4
            self.ch_cond = 3
    
    pipeline = DummyPipeline()
    # Create small dataset for testing
    num_sbc_samples = 20 # small for speed
    num_posterior_samples = 100 # min for warnings check usually, but we check warnings explicitly
    
    xs = jax.random.normal(
        jax.random.PRNGKey(0), (num_sbc_samples, pipeline.dim_cond * pipeline.ch_cond)
    )
    thetas = (
        jax.random.normal(
            jax.random.PRNGKey(1), (num_sbc_samples, pipeline.dim_obs * pipeline.ch_obs)
        )
        + 1.05
    )
    
    posterior_samples = jax.random.normal(
        jax.random.PRNGKey(12345),
        (num_posterior_samples, num_sbc_samples, pipeline.dim_obs * pipeline.ch_obs)
    ) + 1.05
    
    return thetas, xs, posterior_samples

def test_sbc_basic():
    thetas, xs, posterior_samples = get_sbc_data()
    num_posterior_samples = posterior_samples.shape[0]

    ranks, dap_samples = run_sbc(thetas, xs, posterior_samples)

    res_sbc = check_sbc(ranks, thetas, dap_samples, num_posterior_samples)
    assert "ks_pvals" in res_sbc.keys()
    assert "c2st_ranks" in res_sbc.keys()
    assert "c2st_dap" in res_sbc.keys()

    # Basic plot checks
    fig, ax = sbc_rank_plot(ranks, num_posterior_samples, plot_type="hist", num_bins=None)
    assert isinstance(fig, plt.Figure)
    fig, ax = sbc_rank_plot(ranks, num_posterior_samples, plot_type="cdf", num_bins=None)
    assert isinstance(fig, plt.Figure)

def test_sbc_input_validation():
    thetas, xs, posterior_samples = get_sbc_data()
    num_posterior_samples = posterior_samples.shape[0]
    num_sbc_samples = thetas.shape[0]
    
    # Test _validate_sbc_inputs explicitly
    # 1. Warning on low samples
    with pytest.warns(UserWarning, match="Number of SBC samples should be on the order"):
        _validate_sbc_inputs(thetas, xs, num_sbc_samples=10, num_posterior_samples=1000)
        
    with pytest.warns(UserWarning, match="Number of posterior samples for ranking"):
        _validate_sbc_inputs(thetas, xs, num_sbc_samples=1000, num_posterior_samples=10)

    # 2. Error on shape mismatch
    xs_bad = xs[:-1]
    with pytest.raises(ValueError, match="Unequal number of parameters and observations"):
        _validate_sbc_inputs(thetas, xs_bad, num_sbc_samples, num_posterior_samples)

    # Test run_sbc assertions
    bad_posterior = posterior_samples[:, :-1, :]
    with pytest.raises(AssertionError, match="Wrong posterior samples shape"):
        run_sbc(thetas, xs, bad_posterior)


def test_sbc_reduce_fns():
    thetas, xs, posterior_samples = get_sbc_data()
    
    # 1. Custom callable
    def custom_reduce(theta, x):
        # Simplest reduction: just take first dim
        return theta[:, 0]
        
    ranks_custom, _ = run_sbc(thetas, xs, posterior_samples, reduce_fns=custom_reduce)
    assert ranks_custom.shape == (thetas.shape[0], 1)
    
    # 2. List of callables
    ranks_list, _ = run_sbc(thetas, xs, posterior_samples, reduce_fns=[custom_reduce, custom_reduce])
    assert ranks_list.shape == (thetas.shape[0], 2)
    
    # 3. Invalid string
    with pytest.raises(ValueError, match="must either be the string"):
        run_sbc(thetas, xs, posterior_samples, reduce_fns="invalid_option")


def test_check_sbc_warnings():
    thetas, xs, posterior_samples = get_sbc_data()
    # reduce samples to trigger warning in check_sbc
    ranks, dap_samples = run_sbc(thetas, xs, posterior_samples)
    ranks_small = ranks[:10] 
    
    with pytest.warns(UserWarning, match="computing SBC checks with less than 100 samples"):
        check_sbc(ranks_small, thetas[:10], dap_samples[:10], num_posterior_samples=100)
        
    # Test check_prior_vs_dap mismatch
    with pytest.raises(ValueError, match="Prior and DAP samples must have the same shape"):
         from gensbi.diagnostics.sbc import check_prior_vs_dap
         check_prior_vs_dap(thetas, dap_samples[:-1])


def test_sbc_plotting_detailed():
    thetas, xs, posterior_samples = get_sbc_data()
    ranks, _ = run_sbc(thetas, xs, posterior_samples)
    num_posterior_samples = posterior_samples.shape[0]
    
    # Test list of ranks (comparison mode)
    fig, ax = sbc_rank_plot([ranks, ranks], num_posterior_samples)
    assert isinstance(fig, plt.Figure)
    
    # Test invalid plot type
    with pytest.raises(AssertionError, match="plot type invalid not implemented"):
        sbc_rank_plot(ranks, num_posterior_samples, plot_type="invalid")
        
    # Test plotting options
    fig, ax = sbc_rank_plot(
        ranks, num_posterior_samples, 
        plot_type="hist", 
        sharey=True, 
        params_in_subplots=True,
        show_ylabel=True,
        colors=["red"] * ranks.shape[1]
    )
    plt.close(fig)
    
    # Test 1D rank (not in list) to trigger else block in plot logic if possible
    # Actually run_sbc returns 2D array (N, dim).
    # If we pass a single rank array for 1 param? 
    ranks_1d = ranks[:, 0:1] # shape (N, 1)
    fig, ax = sbc_rank_plot(ranks_1d, num_posterior_samples)
    plt.close(fig)