# tests/inference/test_nested.py
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import pytest

from gensbi.core.prior import make_gaussian_prior
from gensbi.inference import NLEPosterior, NestedSampler, NestedSamplerInfo


class GaussianMock:
    def log_prob(self, x, cond):
        diff = (x - cond).reshape(x.shape[0], -1)     # flatten all non-batch dims
        return -0.5 * jnp.sum(diff ** 2, axis=-1)     # (B,)


class BimodalMock:
    """log q(x | theta) = mixture of N(theta; +mu, 0.5 I) and N(theta; -mu, 0.5 I).

    Independent of the observation x; posterior under a broad prior is bimodal at +/-mu.
    """
    def __init__(self, mu=3.0, sigma=0.5):
        self.mu, self.sigma = mu, sigma

    def log_prob(self, x, cond):
        cf = cond.reshape(cond.shape[0], -1)           # flatten all non-batch dims
        a = -0.5 * jnp.sum(((cf - self.mu) / self.sigma) ** 2, axis=-1)
        b = -0.5 * jnp.sum(((cf + self.mu) / self.sigma) ** 2, axis=-1)
        return jax.scipy.special.logsumexp(jnp.stack([a, b], axis=-1), axis=-1)


def test_constructor_defaults():
    s = NestedSampler()
    assert s.num_live == 500
    assert s.num_delete == 50                    # num_live // 10
    assert s.num_inner_steps is None             # resolved per-target at run time
    assert s.num_samples == 1000
    assert s.dlogz == -3.0
    assert s.max_iterations == 100_000


def test_num_inner_steps_auto_resolution():
    s = NestedSampler()
    assert s._resolve_num_inner_steps(2) == 5    # max(5, 2*2)
    assert s._resolve_num_inner_steps(10) == 20  # max(5, 2*10)
    assert NestedSampler(num_inner_steps=7)._resolve_num_inner_steps(10) == 7


def test_num_delete_floor_is_one():
    assert NestedSampler(num_live=5).num_delete == 1   # max(1, 5 // 10)


def test_constructor_validation():
    with pytest.raises(ValueError):
        NestedSampler(num_live=0)
    with pytest.raises(ValueError):
        NestedSampler(num_live=10, num_delete=10)      # must be < num_live
    with pytest.raises(ValueError):
        NestedSampler(num_live=10, num_delete=0)
    with pytest.raises(ValueError):
        NestedSampler(num_inner_steps=0)
    with pytest.raises(ValueError):
        NestedSampler(num_inner_steps=-3)


def test_info_dataclass_is_frozen():
    info = NestedSamplerInfo(log_evidence=0.0, log_evidence_err=0.1,
                             ess=100.0, num_dead=500, dead=None)
    with pytest.raises(Exception):   # dataclasses.FrozenInstanceError
        info.log_evidence = 1.0


@pytest.fixture(scope="module")
def gaussian_ns_run():
    """One shared NS run on the analytic 2D Gaussian target."""
    dim = 2
    x_o = jnp.array([1.0, -1.0])
    post = NLEPosterior(GaussianMock(), make_gaussian_prior((dim,)))
    target = post.build_target(x_o)
    samples, info = NestedSampler().run(jax.random.PRNGKey(0), target)
    return dim, x_o, samples, info


def test_gaussian_recovery_and_shape(gaussian_ns_run):
    dim, x_o, samples, info = gaussian_ns_run
    assert samples.shape == (1000, dim)          # num_samples default
    # prior N(0, I), likelihood N(x_o; theta, I) -> posterior mean x_o / 2
    assert jnp.allclose(jnp.mean(samples, axis=0), x_o / 2, atol=0.2)


def test_evidence_matches_analytic(gaussian_ns_run):
    dim, x_o, samples, info = gaussian_ns_run
    # GaussianMock omits the Gaussian normalisation constant, so
    # log Z = -||x_o||^2 / 4 - (dim / 2) * log 2   (see spec, Testing #3)
    logZ_true = -jnp.sum(x_o ** 2) / 4 - dim / 2 * jnp.log(2.0)
    tol = max(3.0 * info.log_evidence_err, 0.3)  # floor guards tiny stochastic-volume err
    assert abs(info.log_evidence - logZ_true) < tol


def test_info_contract(gaussian_ns_run):
    dim, x_o, samples, info = gaussian_ns_run
    assert isinstance(info, NestedSamplerInfo)
    assert jnp.isfinite(info.log_evidence)
    assert info.log_evidence_err > 0
    assert info.ess > 0
    assert info.num_dead > 0
    assert info.dead is not None                 # raw finalised NSInfo retained
    assert jnp.all(jnp.isfinite(samples))


def test_max_iterations_guard_raises():
    dim = 2
    post = NLEPosterior(GaussianMock(), make_gaussian_prior((dim,)))
    target = post.build_target(jnp.array([1.0, -1.0]))
    with pytest.raises(RuntimeError, match="max_iterations"):
        NestedSampler(max_iterations=2).run(jax.random.PRNGKey(0), target)


def test_bimodal_recovers_both_modes():
    dim = 2
    post = NLEPosterior(BimodalMock(mu=3.0), make_gaussian_prior((dim,), sigma=5.0))
    s = post.sample(jax.random.PRNGKey(1), jnp.zeros(dim),
                    sampler=NestedSampler(num_samples=2000))[..., 0]
    frac_pos = jnp.mean(jnp.all(s > 0, axis=1).astype(float))
    frac_neg = jnp.mean(jnp.all(s < 0, axis=1).astype(float))
    # both modes populated (a single MCMC chain would capture only one)
    assert frac_pos > 0.3 and frac_neg > 0.3


def test_constructor_rejuvenation_default_and_validation():
    assert NestedSampler().num_rejuvenation_steps == 0
    assert NestedSampler(num_rejuvenation_steps=5).num_rejuvenation_steps == 5
    with pytest.raises(ValueError):
        NestedSampler(num_rejuvenation_steps=-1)


def test_rejuvenation_breaks_duplicates_and_preserves_posterior():
    """Equal-weight resampling duplicates draws when num_samples >> run ESS;
    posterior-invariant slice moves must break every atom without shifting
    the posterior (mean x_o / 2, cov I / 2 for the analytic Gaussian)."""
    dim = 2
    x_o = jnp.array([1.0, -1.0])
    post = NLEPosterior(GaussianMock(), make_gaussian_prior((dim,)))
    target = post.build_target(x_o)
    key = jax.random.PRNGKey(3)

    plain, _ = NestedSampler(num_live=100, num_samples=1000).run(key, target)
    n_unique_plain = jnp.unique(plain, axis=0).shape[0]
    assert n_unique_plain < 900                  # the artifact is present

    samples, _ = NestedSampler(num_live=100, num_samples=1000,
                               num_rejuvenation_steps=5).run(key, target)
    assert samples.shape == (1000, dim)
    assert jnp.unique(samples, axis=0).shape[0] == 1000   # every atom broken
    assert jnp.allclose(jnp.mean(samples, axis=0), x_o / 2, atol=0.2)
    assert jnp.allclose(jnp.std(samples, axis=0), jnp.sqrt(0.5), atol=0.15)


def test_rejuvenation_preserves_bimodality():
    dim = 2
    post = NLEPosterior(BimodalMock(mu=3.0), make_gaussian_prior((dim,), sigma=5.0))
    s = post.sample(jax.random.PRNGKey(1), jnp.zeros(dim),
                    sampler=NestedSampler(num_samples=2000,
                                          num_rejuvenation_steps=5))[..., 0]
    frac_pos = jnp.mean(jnp.all(s > 0, axis=1).astype(float))
    frac_neg = jnp.mean(jnp.all(s < 0, axis=1).astype(float))
    # rejuvenation moves are local: both modes must survive with their weights
    assert frac_pos > 0.3 and frac_neg > 0.3


def test_pipeline_wiring_shapes_and_info():
    dim = 2
    post = NLEPosterior(GaussianMock(), make_gaussian_prior((dim,)))
    samples, info = post.sample(jax.random.PRNGKey(2), jnp.array([1.0, -1.0]),
                                sampler=NestedSampler(num_samples=200),
                                return_info=True)
    assert samples.shape == (200, dim, 1)        # NLEPosterior expands (n, dim) -> (n, dim, 1)
    assert isinstance(info, NestedSamplerInfo)
