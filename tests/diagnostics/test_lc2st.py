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

def get_dummy_data():
    class DummyPipeline:
        def __init__(self):
            self.dim_obs = 3
            self.ch_obs = 2
            self.dim_cond = 4
            self.ch_cond = 3
    
    pipeline = DummyPipeline()
    # Create small dataset for testing
    num_samples = 20
    xs = jax.random.normal(jax.random.PRNGKey(0), (num_samples, pipeline.dim_cond * pipeline.ch_cond))
    thetas = jax.random.normal(jax.random.PRNGKey(1), (num_samples, pipeline.dim_obs * pipeline.ch_obs))
    posterior_samples = jax.random.normal(jax.random.PRNGKey(2), (num_samples, pipeline.dim_obs * pipeline.ch_obs))
    
    return thetas, xs, posterior_samples


def test_lc2st():
    thetas, xs, posterior_samples = get_dummy_data()

    # Train the L-C2ST classifier.
    lc2st = LC2ST(
        thetas=thetas[:-1],
        xs=xs[:-1],
        posterior_samples=posterior_samples[:-1],
        classifier="mlp",
        num_ensemble=1,
        num_trials_null=2,
    )

    _ = lc2st.train_under_null_hypothesis(verbosity=0)
    _ = lc2st.train_on_observed_data(verbosity=0)

    x_o = xs[-1 : ]  # Take the last observation as observed data.
    theta_o = thetas[-1 : ]  # True parameter for the observed data.
    
    # In original test, post_samples_star was generated from pipeline given x_o
    # Here we can just use the last posterior sample as a proxy for "samples from posterior"
    # or just use a slice of the posterior samples we have.
    post_samples_star = posterior_samples[-1:]
    
    # Note: plot_lc2st expects theta_o to be samples (N, D) and x_o to be (1, Dx)
    # posterior_samples is already (N, D).
    # Let's use more than 1 sample for post_samples_star to be realistic for plotting distributions
    post_samples_star = posterior_samples  # Use all samples for plotting to have some density

    fig,ax = plot_lc2st(
        lc2st,
        post_samples_star,
        x_o[0], # Pass single observation as array
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

def test_lc2st_init_validation():
    thetas, xs, posterior_samples = get_dummy_data()
    
    # Test mismatched shapes
    with pytest.raises(AssertionError, match="Number of samples must match"):
        LC2ST(thetas[:-1], xs, posterior_samples)
        
    # Test invalid classifier string
    with pytest.raises(ValueError, match="Invalid classifier"):
        LC2ST(thetas, xs, posterior_samples, classifier="invalid_clf")
        
    # Test validation of classifier class
    with pytest.raises(AssertionError, match="classier must either be a string or a subclass"):
        LC2ST(thetas, xs, posterior_samples, classifier=dict) # type: ignore


def test_lc2st_training_logic():
    thetas, xs, posterior_samples = get_dummy_data()
    
    # Test z_score=True, num_folds > 1, verbosity=1
    lc2st_cv = LC2ST(
        thetas, xs, posterior_samples, 
        z_score=True, 
        num_folds=2, 
        classifier="mlp",
        num_trials_null=2 # fast
    )
    # Check that it runs without error
    lc2st_cv.train_on_observed_data(verbosity=1)
    assert len(lc2st_cv.trained_clfs) == 2
    
    # Test ensemble classifier
    lc2st_ensemble = LC2ST(
        thetas, xs, posterior_samples, 
        num_ensemble=2, 
        classifier="random_forest",
        num_trials_null=2
    )
    lc2st_ensemble.train_on_observed_data(verbosity=0)
    assert len(lc2st_ensemble.trained_clfs) == 1 # 1-fold (default)
    # Check if wrapped in EnsembleClassifier
    # Note: Ensembling happens inside _train, returning a list of trained classifiers.
    # The trained classifier should be an instance of EnsembleClassifier
    from gensbi.diagnostics.lc2st import EnsembleClassifier
    assert isinstance(lc2st_ensemble.trained_clfs[0], EnsembleClassifier)


def test_null_hypothesis_logic():
    thetas, xs, posterior_samples = get_dummy_data()
    
    lc2st = LC2ST(thetas, xs, posterior_samples, num_trials_null=2)
    
    # Error: get stats before training null
    with pytest.raises(ValueError, match="You need to train the classifiers under"):
         lc2st.get_statistics_under_null_hypothesis(thetas, xs)

    # Train null
    lc2st.train_under_null_hypothesis(verbosity=0)
    assert len(lc2st.trained_clfs_null) == 2
    
    # Now getter should work
    stats = lc2st.get_statistics_under_null_hypothesis(thetas, xs[0], return_probs=False)
    assert len(stats) == 2
    
    # Test permutation=False strategy (requires null distribution)
    class DummyNullDist:
        def sample(self, key, shape):
            return jax.random.normal(key, shape + (thetas.shape[-1],))
            
    lc2st_noperm = LC2ST(
        thetas, xs, posterior_samples, 
        permutation=False, 
        num_trials_null=2,
        z_score=True
    )
    lc2st_noperm.null_distribution = DummyNullDist()
    
    lc2st_noperm.train_under_null_hypothesis(verbosity=0)
    stats_noperm = lc2st_noperm.get_statistics_under_null_hypothesis(thetas, xs[0])
    assert len(stats_noperm) == 2


def test_plotting_edge_cases():
    thetas, xs, posterior_samples = get_dummy_data()
    lc2st = LC2ST(thetas, xs, posterior_samples, num_trials_null=2, num_folds=2)
    
    lc2st.train_on_observed_data(verbosity=0)
    lc2st.train_under_null_hypothesis(verbosity=0)
    
    # Test plotting with existing axis
    fig, ax = plt.subplots()
    plot_lc2st(lc2st, thetas, xs[0], fig=fig, ax=ax)
    plt.close(fig)
    
    # Test plotting without existing axis (creates new)
    fig2, ax2 = plot_lc2st(lc2st, thetas, xs[0])
    assert fig2 is not None
    plt.close(fig2)


def test_reject_test_and_pvalue():
    thetas, xs, posterior_samples = get_dummy_data()
    lc2st = LC2ST(thetas, xs, posterior_samples, num_trials_null=2)
    lc2st.train_on_observed_data()
    lc2st.train_under_null_hypothesis()
    
    # Just verify they run and return correct types
    pval = lc2st.p_value(thetas, xs[0])
    assert isinstance(pval, float)
    assert 0 <= pval <= 1
    
    reject = lc2st.reject_test(thetas, xs[0], alpha=0.5)
    assert isinstance(reject, bool)
