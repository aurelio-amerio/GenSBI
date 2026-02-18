import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import pytest
from gensbi.diffusion.path.scheduler.sm_sde import VPSmScheduler, VESmScheduler


# =========================================================
# VPSmScheduler Tests
# =========================================================


class TestVPSmScheduler:
    def test_initialization(self):
        sde = VPSmScheduler()
        assert sde.name == "SM-VP"
        assert sde.T == 1.0
        assert sde.beta_min == 0.001
        assert sde.beta_max == 3.0

    def test_custom_params(self):
        sde = VPSmScheduler(beta_min=0.01, beta_max=10.0, e_s=1e-4)
        assert sde.beta_min == 0.01
        assert sde.beta_max == 10.0
        assert sde.e_s == 1e-4

    def test_beta_t(self):
        sde = VPSmScheduler()
        t = jnp.array([0.0, 0.5, 1.0])
        beta = sde.beta_t(t)
        assert beta.shape == (3,)
        assert jnp.allclose(beta[0], sde.beta_min)
        assert jnp.allclose(beta[2], sde.beta_max)

    def test_drift_shape(self):
        sde = VPSmScheduler()
        x = jnp.ones((4, 3))
        t = jnp.ones((4, 1)) * 0.5
        drift = sde.drift(x, t)
        assert drift.shape == x.shape

    def test_drift_direction(self):
        """VP drift should push towards zero."""
        sde = VPSmScheduler()
        x = jnp.ones((1, 2))
        t = jnp.ones((1, 1)) * 0.5
        drift = sde.drift(x, t)
        assert jnp.all(drift < 0)  # drift pulls x towards 0

    def test_diffusion_shape(self):
        sde = VPSmScheduler()
        t = jnp.array([0.1, 0.5, 0.9])
        g = sde.diffusion(t)
        assert g.shape == (3,)
        assert jnp.all(g > 0)

    def test_marginal_mean_coeff(self):
        sde = VPSmScheduler()
        t = jnp.array([0.0, 1.0])
        coeff = sde.marginal_mean_coeff(t)
        assert jnp.allclose(coeff[0], 1.0, atol=1e-3)
        assert coeff[1] < 1.0  # decays over time

    def test_marginal_std(self):
        sde = VPSmScheduler()
        t = jnp.array([0.0, 1.0])
        std = sde.marginal_std(t)
        assert jnp.allclose(std[0], 0.0, atol=1e-3)
        assert std[1] > 0.0

    def test_sample_t(self):
        sde = VPSmScheduler()
        key = jax.random.PRNGKey(0)
        t = sde.sample_t(key, (100,))
        assert t.shape == (100,)
        assert jnp.all(t >= sde.e_s)
        assert jnp.all(t <= 1.0)

    def test_sample_prior(self):
        sde = VPSmScheduler()
        key = jax.random.PRNGKey(0)
        samples = sde.sample_prior(key, (10, 5))
        assert samples.shape == (10, 5)

    def test_weight(self):
        sde = VPSmScheduler()
        t = jnp.array([0.5])
        w = sde.weight(t)
        expected = sde.diffusion(t) ** 2
        assert jnp.allclose(w, expected)


# =========================================================
# VESmScheduler Tests
# =========================================================


class TestVESmScheduler:
    def test_initialization(self):
        sde = VESmScheduler()
        assert sde.name == "SM-VE"
        assert sde.T == 1.0
        assert sde.sigma_min == 1e-3
        assert sde.sigma_max == 15.0

    def test_sigma(self):
        sde = VESmScheduler()
        t = jnp.array([0.0, 1.0])
        sigma = sde.sigma(t)
        assert jnp.allclose(sigma[0], sde.sigma_min, rtol=1e-3)
        assert jnp.allclose(sigma[1], sde.sigma_max, rtol=1e-3)

    def test_drift_is_zero(self):
        sde = VESmScheduler()
        x = jnp.ones((4, 3))
        t = jnp.ones((4, 1)) * 0.5
        drift = sde.drift(x, t)
        assert jnp.allclose(drift, 0.0)

    def test_diffusion_shape(self):
        sde = VESmScheduler()
        t = jnp.array([0.1, 0.5, 0.9])
        g = sde.diffusion(t)
        assert g.shape == (3,)
        assert jnp.all(g > 0)

    def test_marginal_mean_coeff_is_one(self):
        sde = VESmScheduler()
        t = jnp.array([0.1, 0.5, 0.9])
        coeff = sde.marginal_mean_coeff(t)
        assert jnp.allclose(coeff, 1.0)

    def test_marginal_std_equals_sigma(self):
        sde = VESmScheduler()
        t = jnp.array([0.1, 0.5, 0.9])
        std = sde.marginal_std(t)
        sigma = sde.sigma(t)
        assert jnp.allclose(std, sigma)

    def test_sample_t(self):
        sde = VESmScheduler()
        key = jax.random.PRNGKey(0)
        t = sde.sample_t(key, (100,))
        assert t.shape == (100,)
        assert jnp.all(t >= sde.e_s)
        assert jnp.all(t <= 1.0)

    def test_sample_prior(self):
        sde = VESmScheduler()
        key = jax.random.PRNGKey(0)
        samples = sde.sample_prior(key, (10, 5))
        assert samples.shape == (10, 5)

    def test_weight(self):
        sde = VESmScheduler()
        t = jnp.array([0.5])
        w = sde.weight(t)
        expected = sde.diffusion(t) ** 2
        assert jnp.allclose(w, expected)
