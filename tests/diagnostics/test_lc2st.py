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
    with pytest.raises(AssertionError, match="classifier must either be a string or a subclass"):
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


# Fixes ported from upstream sbi

_FAST_MLP = {"hidden_layer_sizes": (8,), "max_iter": 60, "random_state": 0}


def _gaussian_toy(num_samples=300, x_mean=10.0, x_std=0.3):
    """theta ~ N(0, 1), x an affine map of theta + noise, exact posterior samples."""
    rng = np.random.default_rng(0)
    thetas = rng.normal(size=(num_samples, 2))
    x_raw = thetas + rng.normal(size=(num_samples, 2))
    posterior_samples = x_raw / 2 + np.sqrt(0.5) * rng.normal(size=(num_samples, 2))
    xs = x_mean + x_std * x_raw
    return tuple(jnp.asarray(a, dtype=jnp.float32) for a in (thetas, xs, posterior_samples))


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_z_score_matches_normalising_the_inputs_by_hand():
    """z_score=True must be the same test as z_score=False on pre-normalised data."""
    thetas, xs, posterior_samples = _gaussian_toy()
    theta_o, x_o = posterior_samples[:100], xs[0]

    t_mean, t_std = posterior_samples.mean(0), posterior_samples.std(0)
    x_mean, x_std = xs.mean(0), xs.std(0)

    def norm_t(a):
        return (a - t_mean) / t_std

    def norm_x(a):
        return (a - x_mean) / x_std

    kwargs = dict(classifier="mlp", classifier_kwargs=dict(_FAST_MLP), num_trials_null=3)
    auto = LC2ST(thetas, xs, posterior_samples, z_score=True, **kwargs)
    manual = LC2ST(
        norm_t(thetas), norm_x(xs), norm_t(posterior_samples), z_score=False, **kwargs
    )
    for lc2st in (auto, manual):
        lc2st.train_under_null_hypothesis(verbosity=0)
        lc2st.train_on_observed_data(verbosity=0)

    np.testing.assert_allclose(
        auto.get_statistics_under_null_hypothesis(theta_o, x_o),
        manual.get_statistics_under_null_hypothesis(norm_t(theta_o), norm_x(x_o)),
        rtol=1e-3,
    )
    np.testing.assert_allclose(
        auto.get_statistic_on_observed_data(theta_o, x_o),
        manual.get_statistic_on_observed_data(norm_t(theta_o), norm_x(x_o)),
        rtol=1e-3,
    )


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_z_score_handles_constant_features():
    thetas, xs, posterior_samples = _gaussian_toy(num_samples=100)
    xs = xs.at[:, 1].set(3.0)  # zero standard deviation

    lc2st = LC2ST(
        thetas, xs, posterior_samples, z_score=True, classifier_kwargs=dict(_FAST_MLP)
    )
    lc2st.train_on_observed_data(verbosity=0)

    assert np.isfinite(lc2st.get_statistic_on_observed_data(posterior_samples, xs[0]))


def test_permute_data_swaps_labels_within_pairs():
    """Row i of P and row i of Q share an observation, so they may only swap with each other."""
    from gensbi.diagnostics.lc2st import permute_data

    theta_p = jnp.arange(40.0).reshape(20, 2)
    theta_q = -theta_p - 1.0

    perm_p, perm_q = permute_data(theta_p, theta_q, seed=0)

    kept = np.all(np.asarray(perm_p == theta_p) & np.asarray(perm_q == theta_q), axis=1)
    swapped = np.all(np.asarray(perm_p == theta_q) & np.asarray(perm_q == theta_p), axis=1)
    assert np.all(kept | swapped)
    assert kept.any() and swapped.any()


def test_p_value_counts_the_observed_statistic_among_the_null_ones():
    """p = (1 + #{T_null >= T_obs}) / (1 + num_trials), so it is never 0."""
    thetas, xs, posterior_samples = get_dummy_data()
    lc2st = LC2ST(thetas, xs, posterior_samples, num_trials_null=4)
    null_stats = np.array([0.1, 0.2, 0.3, 0.5])

    def fake_null(theta_o, x_o, return_probs=False, verbosity=0):
        return (None, null_stats) if return_probs else null_stats

    lc2st.get_statistics_under_null_hypothesis = fake_null

    lc2st.get_statistic_on_observed_data = lambda theta_o, x_o: 0.5
    assert lc2st.p_value(thetas, xs[0]) == pytest.approx(2 / 5)

    lc2st.get_statistic_on_observed_data = lambda theta_o, x_o: 0.9
    assert lc2st.p_value(thetas, xs[0]) == pytest.approx(1 / 5)


def test_ensemble_classifier_supports_estimators_without_random_state():
    from sklearn.naive_bayes import GaussianNB
    from gensbi.diagnostics.lc2st import EnsembleClassifier

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    y = (X[:, 0] > 0).astype(float)

    ensemble = EnsembleClassifier(GaussianNB(), num_ensemble=2, verbosity=0)

    assert ensemble.fit(X, y) is ensemble
    assert ensemble.predict_proba(X).shape == (40, 2)
