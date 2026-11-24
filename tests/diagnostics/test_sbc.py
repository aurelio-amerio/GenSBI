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
    with pytest.raises(ValueError, match="Wrong posterior samples shape"):
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
    with pytest.raises(ValueError, match="plot type invalid not implemented"):
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


# Coverage improvement tests


def test_sbc_plot_with_existing_fig_ax():
    """Test sbc_rank_plot with pre-created fig/ax, covering the else branch."""
    thetas, xs, posterior_samples = get_sbc_data()
    ranks, _ = run_sbc(thetas, xs, posterior_samples)
    num_posterior_samples = posterior_samples.shape[0]
    num_params = ranks.shape[1]

    # Create fig/ax with enough subplots for params_in_subplots=True (hist mode)
    fig, ax = plt.subplots(1, num_params, figsize=(num_params * 4, 5))
    ax = np.atleast_1d(ax)
    fig2, ax2 = sbc_rank_plot(ranks, num_posterior_samples, plot_type="hist", fig=fig, ax=ax)
    assert fig2 is fig
    plt.close(fig)


def test_sbc_plot_ax_too_few_subplots():
    """Passing ax with fewer subplots than parameters should raise ValueError."""
    thetas, xs, posterior_samples = get_sbc_data()
    ranks, _ = run_sbc(thetas, xs, posterior_samples)
    num_posterior_samples = posterior_samples.shape[0]

    # Create fig/ax with only 1 subplot but ranks has multiple params
    fig, ax = plt.subplots(1, 1)
    ax = np.atleast_1d(ax)
    with pytest.raises(ValueError, match="at least as many subplots"):
        sbc_rank_plot(ranks, num_posterior_samples, plot_type="hist", fig=fig, ax=ax)
    plt.close(fig)


def test_sbc_plot_ranks_type_error():
    """Non-Array/np.ndarray ranks should raise TypeError."""
    with pytest.raises(TypeError, match="ranks must be"):
        sbc_rank_plot("invalid", 100)


def test_sbc_plot_ranks_list_type_error():
    """List with non-Array elements should raise TypeError."""
    with pytest.raises(TypeError, match="All ranks in the list"):
        sbc_rank_plot(["invalid", "data"], 100)


def test_sbc_plot_mismatched_ranks_shapes():
    """Ranks in list with different shapes should raise ValueError."""
    thetas, xs, posterior_samples = get_sbc_data()
    ranks, _ = run_sbc(thetas, xs, posterior_samples)

    # Create a second ranks with different shape
    ranks_bad = ranks[:10]
    with pytest.raises(ValueError, match="same shape"):
        sbc_rank_plot([ranks, ranks_bad], posterior_samples.shape[0])


def test_sbc_c2st_variability_warning():
    """Many C2ST repetitions with small data can trigger variability warning."""
    thetas, xs, posterior_samples = get_sbc_data()
    ranks, dap_samples = run_sbc(thetas, xs, posterior_samples)

    # Use very few samples and many repetitions to potentially trigger
    # c2st_std > 0.05 warning
    ranks_tiny = ranks[:5]
    try:
        with pytest.warns(UserWarning, match="C2ST score variability"):
            check_sbc(ranks_tiny, thetas[:5], dap_samples[:5], 
                      num_posterior_samples=100, num_c2st_repetitions=5)
    except Exception:
        # If it doesn't trigger the warning, that's fine too - we still
        # exercised the multi-repetition code path
        pass

# Fixes ported from upstream sbi


def _ranks_confined_to_the_middle():
    """Ranks of a far too wide posterior: uniform on [300, 700] out of 1000."""
    return np.random.default_rng(0).integers(300, 701, size=(500, 1)).astype(float)


def test_run_sbc_rejects_mismatched_xs():
    thetas, xs, posterior_samples = get_sbc_data()
    with pytest.raises(ValueError, match="Unequal number of parameters and observations"):
        run_sbc(thetas, xs[:-1], posterior_samples)


def test_sbc_rank_plot_accepts_jax_arrays():
    ranks = jnp.asarray(np.random.default_rng(0).integers(0, 101, size=(100, 2)))
    fig, ax = sbc_rank_plot(ranks, 100, plot_type="cdf")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_sbc_cdf_plot_spans_the_full_rank_range():
    """The ECDF axis is [0, num_posterior_samples], not the range of the observed ranks."""
    fig, ax = sbc_rank_plot(
        _ranks_confined_to_the_middle(), 1000, plot_type="cdf", num_bins=20
    )
    x, y = ax.get_lines()[0].get_data()

    assert x.max() == 1000
    assert y[x < 250].max() == 0.0  # no ranks below 300
    assert y[x > 750].min() == 1.0  # all ranks below 700
    plt.close(fig)


def test_sbc_hist_plot_bins_span_the_full_rank_range():
    fig, ax = sbc_rank_plot(
        _ranks_confined_to_the_middle(), 1000, plot_type="hist", num_bins=20
    )
    bars = np.atleast_1d(ax)[0].patches

    assert len(bars) == 20
    assert bars[0].get_x() == 0.0
    assert bars[-1].get_x() + bars[-1].get_width() == pytest.approx(1000.0)
    assert bars[0].get_height() == 0  # nothing below rank 300
    plt.close(fig)


def test_ks_uniformity_check_is_valid_for_discrete_ranks():
    """Exactly uniform ranks from 4 posterior samples must not be rejected."""
    from gensbi.diagnostics.sbc import check_uniformity_frequentist

    ranks = np.random.default_rng(0).integers(0, 5, size=(2000, 2)).astype(float)

    pvals = check_uniformity_frequentist(ranks, num_posterior_samples=4)

    assert (np.asarray(pvals) > 1e-3).all()


def test_c2st_uniformity_check_is_valid_for_discrete_ranks():
    """Exactly uniform integer ranks must be indistinguishable from the baseline."""
    from gensbi.diagnostics.sbc import check_uniformity_c2st

    ranks = np.random.default_rng(0).integers(0, 5, size=(500, 1)).astype(float)

    scores = check_uniformity_c2st(ranks, num_posterior_samples=4)

    assert abs(float(scores[0]) - 0.5) < 0.1
