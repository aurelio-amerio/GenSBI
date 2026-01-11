# %%
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from flax import nnx

import pytest


from gensbi.diagnostics import check_tarp, run_tarp, plot_tarp
from gensbi.diagnostics.metrics import l1

def get_tarp_data():
    class DummyPipeline:
        def __init__(self):
            self.dim_obs = 3
            self.ch_obs = 2
            self.dim_cond = 4
            self.ch_cond = 3
    
    pipeline = DummyPipeline()
    # Create small dataset for testing
    num_tarp_samples = 20
    num_posterior_samples = 50
    
    xs = jax.random.normal(
        jax.random.PRNGKey(0), (num_tarp_samples, pipeline.dim_cond * pipeline.ch_cond)
    )
    thetas = (
        jax.random.normal(
            jax.random.PRNGKey(1), (num_tarp_samples, pipeline.dim_obs * pipeline.ch_obs)
        )
        + 1.05
    )
    
    posterior_samples = jax.random.normal(
        jax.random.PRNGKey(12345), 
        (num_posterior_samples, num_tarp_samples, pipeline.dim_obs * pipeline.ch_obs)
    ) + 1.05
    
    # ensure shapes match what run_tarp expects
    # run_tarp expects:
    # thetas: (num_tarp_samples, dim_theta)
    # posterior_samples: (num_posterior_samples, num_tarp_samples, dim_theta)
    
    return thetas, posterior_samples


def test_tarp_basic():
    thetas, posterior_samples = get_tarp_data()

    ecp, alpha = run_tarp(
        thetas,
        posterior_samples,
        references=None,  # will be calculated automatically.
    )

    fig, ax = plot_tarp(ecp, alpha)

    assert isinstance(fig, plt.Figure), f"fig is not a matplotlib Figure, got {type(fig)}"


def test_tarp_input_validation():
    thetas, posterior_samples = get_tarp_data()
    
    # Test wrong posterior samples shape
    bad_posterior = posterior_samples[:, :-1, :] # modify num_tarp_samples dimension
    with pytest.raises(AssertionError, match="Wrong posterior samples shape"):
        run_tarp(thetas, bad_posterior)
        
    # Test wrong references shape (passed via _run_tarp implicitly if we pass references)
    references_bad = jax.random.normal(jax.random.PRNGKey(99), (thetas.shape[0]+1, thetas.shape[1]))
    with pytest.raises(AssertionError, match="references must have the same shape"):
         run_tarp(thetas, posterior_samples, references=references_bad)

def test_tarp_options():
    thetas, posterior_samples = get_tarp_data()
    
    # Test z_score options
    ecp1, _ = run_tarp(thetas, posterior_samples, z_score_theta=True)
    ecp2, _ = run_tarp(thetas, posterior_samples, z_score_theta=False)
    # Just check they run and return correct shapes, values might differ
    assert ecp1.shape == ecp2.shape
    
    # Test explicit num_bins
    ecp_bins, alpha_bins = run_tarp(thetas, posterior_samples, num_bins=10)
    assert len(alpha_bins) == 11 # histogram edges = bins + 1
    
    # Test explicit references
    refs = jax.random.normal(jax.random.PRNGKey(42), thetas.shape)
    ecp_refs, _ = run_tarp(thetas, posterior_samples, references=refs)
    
    # Test l1 distance
    ecp_l1, _ = run_tarp(thetas, posterior_samples, distance=l1)
    
    
def test_check_tarp():
    thetas, posterior_samples = get_tarp_data()
    ecp, alpha = run_tarp(thetas, posterior_samples)
    
    atc, ks_prob = check_tarp(ecp, alpha)
    
    assert isinstance(atc, float)
    assert isinstance(ks_prob, float)
    assert 0 <= ks_prob <= 1


def test_plot_tarp_options():
    thetas, posterior_samples = get_tarp_data()
    ecp, alpha = run_tarp(thetas, posterior_samples)
    
    fig, ax = plot_tarp(ecp, alpha, title="My TARP Plot")
    assert ax.get_title() == "My TARP Plot"

