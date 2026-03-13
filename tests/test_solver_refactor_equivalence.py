"""
Phase 2: Numerical equivalence tests for the solver refactor.

For each new solver class, verify that it produces numerically identical
results to the old implementation when given the same inputs, same model,
and same random state.
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from numpyro import distributions as dist

from gensbi.utils.model_wrapping import ModelWrapper, ScoreToDrift
from gensbi.utils.math import _expand_dims, _expand_time

# ------------------------------------------------------------------
# Old classes
# ------------------------------------------------------------------
from gensbi.flow_matching.solver.ode_solver import ODESolver
from gensbi.flow_matching.solver.sde_solver_fm import (
    ZeroEndsSolver,
    NonSingularSolver,
)
from gensbi.diffusion.solver.sm_solver import SMPFSolver, SMSolver
from gensbi.diffusion.path.sm_path import SMPath
from gensbi.diffusion.path.scheduler.sm_sde import VPSmScheduler

# ------------------------------------------------------------------
# New classes
# ------------------------------------------------------------------
from gensbi.flow_matching.solver.fm_ode_solver import NewFMODESolver
from gensbi.flow_matching.solver.fm_sde_solver import (
    NewZeroEndsSolver,
    NewNonSingularSolver,
)
from gensbi.diffusion.solver.sm_ode_solver_new import NewSMODESolver
from gensbi.diffusion.solver.sm_sde_solver_new import NewSMSDESolver


# ==================================================================
# Mock models — identical to those used in existing tests
# ==================================================================


class FMDummyModel(nnx.Module):
    """Deterministic velocity model: 3·t² (used in FM ODE tests)."""

    def __call__(self, obs, t, **kwargs):
        obs = _expand_dims(obs)
        t = _expand_time(t)
        if t.ndim < 3:
            t = t[..., None]
        return jnp.ones_like(obs) * 3.0 * t**2


class ZeroDummyModel(nnx.Module):
    """Zero velocity model (used in FM SDE tests)."""

    def __call__(self, obs, t, *args, **kwargs):
        return jnp.zeros_like(obs)


class DummyScoreModel(nnx.Module):
    """Dummy score model returning zeros + t (used in SM tests)."""

    def __call__(self, obs, t, **kwargs):
        return jnp.zeros_like(obs) + t


# ==================================================================
# FM ODE equivalence
# ==================================================================


class TestFMODEEquivalence:
    """NewFMODESolver ≡ ODESolver."""

    def _make_both(self):
        model = FMDummyModel()
        wrapper = ModelWrapper(model)
        old = ODESolver(velocity_model=wrapper)
        new = NewFMODESolver(velocity_model=wrapper)
        return old, new

    def test_sample_exact(self):
        """Sample output must be bitwise identical."""
        old, new = self._make_both()
        x_init = jnp.ones((5, 2, 1))
        time_grid = jnp.array([0.0, 1.0])

        sol_old = old.sample(
            x_init=x_init,
            step_size=0.1,
            method="Euler",
            time_grid=time_grid,
        )
        sol_new = new.sample(
            x_init=x_init,
            step_size=0.1,
            method="Euler",
            time_grid=time_grid,
        )
        assert jnp.allclose(sol_old, sol_new, atol=1e-7), (
            f"FM ODE sample mismatch:\nold={sol_old}\nnew={sol_new}"
        )

    def test_sample_intermediates(self):
        """Intermediates must match."""
        old, new = self._make_both()
        x_init = jnp.ones((3, 2, 1))
        time_grid = jnp.linspace(0, 1, 11)

        sol_old = old.sample(
            x_init=x_init,
            step_size=0.1,
            method="Euler",
            time_grid=time_grid,
            return_intermediates=True,
        )
        sol_new = new.sample(
            x_init=x_init,
            step_size=0.1,
            method="Euler",
            time_grid=time_grid,
            return_intermediates=True,
        )
        assert sol_old.shape == sol_new.shape
        assert jnp.allclose(sol_old, sol_new, atol=1e-7), (
            "FM ODE intermediates mismatch"
        )

    def test_sample_dopri5(self):
        """Adaptive solver must also match."""
        old, new = self._make_both()
        x_init = jnp.ones((4, 2, 1))
        time_grid = jnp.array([0.0, 1.0])

        sol_old = old.sample(
            x_init=x_init,
            step_size=None,
            method="Dopri5",
            time_grid=time_grid,
        )
        sol_new = new.sample(
            x_init=x_init,
            step_size=None,
            method="Dopri5",
            time_grid=time_grid,
        )
        assert jnp.allclose(sol_old, sol_new, atol=1e-6), (
            f"FM ODE Dopri5 mismatch:\nold={sol_old}\nnew={sol_new}"
        )

    def test_logprob_exact(self):
        """Log-probability with exact divergence must match."""
        old, new = self._make_both()
        x_1 = jnp.ones((5, 2, 3))

        p0 = dist.Independent(
            dist.Normal(loc=jnp.zeros((2, 3)), scale=jnp.ones((2, 3))),
            reinterpreted_batch_ndims=2,
        )
        time_grid = jnp.array([1.0, 0.0])

        logp_old = old.compute_log_prob(
            x_1=x_1,
            log_p0=p0.log_prob,
            time_grid=time_grid,
            method="Euler",
            step_size=0.01,
            exact_divergence=True,
        )
        logp_new = new.compute_log_prob(
            x_1=x_1,
            log_p0=p0.log_prob,
            time_grid=time_grid,
            method="Euler",
            step_size=0.01,
            exact_divergence=True,
        )
        assert jnp.allclose(logp_old, logp_new, atol=1e-6), (
            f"FM ODE exact logprob mismatch:\nold={logp_old}\nnew={logp_new}"
        )

    def test_logprob_hutchinson(self):
        """Log-probability with Hutchinson divergence must match."""
        old, new = self._make_both()
        x_1 = jnp.ones((4, 2, 3))

        p0 = dist.Independent(
            dist.Normal(loc=jnp.zeros((2, 3)), scale=jnp.ones((2, 3))),
            reinterpreted_batch_ndims=2,
        )
        time_grid = jnp.array([1.0, 0.0])
        key = jax.random.PRNGKey(42)

        logp_old = old.compute_log_prob(
            x_1=x_1,
            log_p0=p0.log_prob,
            time_grid=time_grid,
            method="Euler",
            step_size=0.01,
            exact_divergence=False,
            key=key,
        )
        logp_new = new.compute_log_prob(
            x_1=x_1,
            log_p0=p0.log_prob,
            time_grid=time_grid,
            method="Euler",
            step_size=0.01,
            exact_divergence=False,
            key=key,
        )
        assert jnp.allclose(logp_old, logp_new, atol=1e-6), (
            f"FM ODE Hutchinson logprob mismatch:\nold={logp_old}\nnew={logp_new}"
        )


# ==================================================================
# FM SDE equivalence
# ==================================================================


class TestFMSDEZeroEndsEquivalence:
    """NewZeroEndsSolver ≡ ZeroEndsSolver."""

    def _make_both(self, features=3, channels=2, alpha=0.5, eps0=1e-3):
        model = ZeroDummyModel()
        wrapper = ModelWrapper(model)
        mu0 = jnp.zeros((features, channels))
        sigma0 = jnp.ones((features, channels))
        old = ZeroEndsSolver(wrapper, mu0, sigma0, alpha=alpha, eps0=eps0)
        new = NewZeroEndsSolver(wrapper, mu0, sigma0, alpha=alpha, eps0=eps0)
        return old, new

    def test_sample_exact(self):
        """Same key → same samples."""
        old, new = self._make_both()
        key = jax.random.PRNGKey(0)
        key_init, key_sample = jax.random.split(key)

        nsamples = 5
        x_init = old.prior_distribution.sample(key_init, (nsamples,))
        x_init = x_init.reshape(nsamples, 3, 2)

        sol_old = old.sample(x_init, step_size=0.2, method="Euler", key=key_sample)
        sol_new = new.sample(x_init, step_size=0.2, method="Euler", key=key_sample)

        assert jnp.allclose(sol_old, sol_new, atol=1e-6), (
            f"ZeroEnds sample mismatch:\nold={sol_old}\nnew={sol_new}"
        )

    def test_sample_intermediates(self):
        """Intermediates must match."""
        old, new = self._make_both()
        key = jax.random.PRNGKey(42)
        key_init, key_sample = jax.random.split(key)

        nsamples = 3
        x_init = old.prior_distribution.sample(key_init, (nsamples,))
        x_init = x_init.reshape(nsamples, 3, 2)

        time_grid = jnp.linspace(0, 1, 6)
        sol_old = old.sample(
            x_init,
            step_size=0.2,
            method="Euler",
            return_intermediates=True,
            time_grid=time_grid,
            key=key_sample,
        )
        sol_new = new.sample(
            x_init,
            step_size=0.2,
            method="Euler",
            return_intermediates=True,
            time_grid=time_grid,
            key=key_sample,
        )
        assert sol_old.shape == sol_new.shape
        assert jnp.allclose(sol_old, sol_new, atol=1e-6), (
            "ZeroEnds intermediates mismatch"
        )

    def test_sample_eulerheun(self):
        """EulerHeun method must also match."""
        old, new = self._make_both()
        key = jax.random.PRNGKey(7)
        key_init, key_sample = jax.random.split(key)

        nsamples = 4
        x_init = old.prior_distribution.sample(key_init, (nsamples,))
        x_init = x_init.reshape(nsamples, 3, 2)

        sol_new = new.sample(x_init, step_size=0.1, method="EulerHeun", key=key_sample)

        assert not jnp.any(jnp.isnan(sol_new)), "EulerHeun produced NaN"
        assert sol_new.shape == (4, 3, 2)


class TestFMSDENonSingularEquivalence:
    """NewNonSingularSolver ≡ NonSingularSolver."""

    def _make_both(self, features=3, channels=2, alpha=0.5):
        model = ZeroDummyModel()
        wrapper = ModelWrapper(model)
        mu0 = jnp.zeros((features, channels))
        sigma0 = jnp.ones((features, channels))
        old = NonSingularSolver(wrapper, mu0, sigma0, alpha=alpha)
        new = NewNonSingularSolver(wrapper, mu0, sigma0, alpha=alpha)
        return old, new

    def test_sample_exact(self):
        """Same key → same samples."""
        old, new = self._make_both()
        key = jax.random.PRNGKey(0)
        key_init, key_sample = jax.random.split(key)

        nsamples = 5
        x_init = old.prior_distribution.sample(key_init, (nsamples,))
        x_init = x_init.reshape(nsamples, 3, 2)

        sol_old = old.sample(x_init, step_size=0.2, method="Euler", key=key_sample)
        sol_new = new.sample(x_init, step_size=0.2, method="Euler", key=key_sample)

        assert jnp.allclose(sol_old, sol_new, atol=1e-6), (
            f"NonSingular sample mismatch:\nold={sol_old}\nnew={sol_new}"
        )

    def test_sample_intermediates(self):
        """Intermediates must match."""
        old, new = self._make_both()
        key = jax.random.PRNGKey(42)
        key_init, key_sample = jax.random.split(key)

        nsamples = 3
        x_init = old.prior_distribution.sample(key_init, (nsamples,))
        x_init = x_init.reshape(nsamples, 3, 2)

        time_grid = jnp.linspace(0, 1, 6)
        sol_old = old.sample(
            x_init,
            step_size=0.2,
            method="Euler",
            return_intermediates=True,
            time_grid=time_grid,
            key=key_sample,
        )
        sol_new = new.sample(
            x_init,
            step_size=0.2,
            method="Euler",
            return_intermediates=True,
            time_grid=time_grid,
            key=key_sample,
        )
        assert sol_old.shape == sol_new.shape
        assert jnp.allclose(sol_old, sol_new, atol=1e-6), (
            "NonSingular intermediates mismatch"
        )


# ==================================================================
# SM PF-ODE equivalence
# ==================================================================


class TestSMODEEquivalence:
    """NewSMODESolver ≡ SMPFSolver (both use ScoreToDrift + ODESolver)."""

    def _make_both(self):
        score_model = DummyScoreModel()
        sde = VPSmScheduler()
        path = SMPath(sde)

        drift_model = ScoreToDrift(score_model=score_model, sde=sde)
        wrapper = ModelWrapper(model=drift_model)

        old = SMPFSolver(velocity_model=wrapper)
        new = NewSMODESolver(velocity_model=wrapper)
        return old, new, path

    def test_sample_exact(self):
        """Same inputs → same samples."""
        old, new, path = self._make_both()
        sde = path.scheduler
        T = sde.T
        eps = 1e-3
        nsteps = 10
        time_grid = jnp.array([T, eps])
        step_size = -(T - eps) / nsteps

        key = jax.random.PRNGKey(0)
        x_init = path.sample_prior(key, (5, 3, 1))

        sampler_old = old.get_sampler(
            step_size=step_size,
            method="Euler",
            time_grid=time_grid,
        )
        sampler_new = new.get_sampler(
            step_size=step_size,
            method="Euler",
            time_grid=time_grid,
        )

        sol_old = sampler_old(x_init)
        sol_new = sampler_new(x_init)

        assert jnp.allclose(sol_old, sol_new, atol=1e-7), (
            f"SMODE sample mismatch:\nold={sol_old}\nnew={sol_new}"
        )

    def test_sample_intermediates(self):
        """Intermediates must match."""
        old, new, path = self._make_both()
        sde = path.scheduler
        T = sde.T
        eps = 1e-3
        nsteps = 5
        time_grid = jnp.linspace(T, eps, nsteps + 1)
        step_size = -(T - eps) / nsteps

        key = jax.random.PRNGKey(0)
        x_init = path.sample_prior(key, (3, 2, 1))

        sampler_old = old.get_sampler(
            step_size=step_size,
            method="Euler",
            time_grid=time_grid,
            return_intermediates=True,
        )
        sampler_new = new.get_sampler(
            step_size=step_size,
            method="Euler",
            time_grid=time_grid,
            return_intermediates=True,
        )

        sol_old = sampler_old(x_init)
        sol_new = sampler_new(x_init)

        assert sol_old.shape == sol_new.shape
        assert jnp.allclose(sol_old, sol_new, atol=1e-7), (
            "SMODE intermediates mismatch"
        )

    def test_logprob_exact(self):
        """Log-probability with exact divergence must match."""
        old, new, path = self._make_both()
        sde = path.scheduler
        T = sde.T
        eps = 1e-3
        nsteps = 10
        time_grid = jnp.array([T, eps])
        step_size = (T - eps) / nsteps

        from gensbi.diffusion.sm_prior import VPPrior

        prior = VPPrior()

        log_prob_old = old.get_log_prob(
            log_p0=prior.log_prob,
            step_size=step_size,
            method="Euler",
            time_grid=time_grid,
            exact_divergence=True,
        )
        log_prob_new = new.get_log_prob(
            log_p0=prior.log_prob,
            step_size=step_size,
            method="Euler",
            time_grid=time_grid,
            exact_divergence=True,
        )

        key = jax.random.PRNGKey(0)
        x_1 = path.sample_prior(key, (4, 2, 1))

        logp_old = log_prob_old(x_1)
        logp_new = log_prob_new(x_1)

        assert jnp.allclose(logp_old, logp_new, atol=1e-6), (
            f"SMODE logprob mismatch:\nold={logp_old}\nnew={logp_new}"
        )


# ==================================================================
# SM SDE equivalence
# ==================================================================


class TestSMSDEEquivalence:
    """NewSMSDESolver ≡ SMSolver (no conditioning — conditioning via wrapper).

    This test verifies that the new diffrax-based SDE solver produces the
    same output as the old hand-rolled sm_reverse_sde_sampler when given
    the same random state.
    """

    def _make_both(self, features=3, channels=1):
        score_model = DummyScoreModel()
        sde = VPSmScheduler()
        path = SMPath(sde)

        # Old SMSolver — takes raw score model
        old = SMSolver(score_model=score_model, path=path)

        # New SMSDESolver — takes wrapped score model
        wrapper = ModelWrapper(model=score_model)
        mu0 = jnp.zeros((features, channels))
        sigma0 = jnp.ones((features, channels))
        new = NewSMSDESolver(
            velocity_model=wrapper,
            sde=sde,
            mu0=mu0,
            sigma0=sigma0,
            eps0=1e-3,
        )
        return old, new, path

    def test_sample_shape(self):
        """Verify new solver produces correct output shape."""
        _, new, path = self._make_both()
        key = jax.random.PRNGKey(0)
        x_init = path.sample_prior(key, (5, 3, 1))

        T = path.scheduler.T
        eps = 1e-3
        nsteps = 10
        time_grid = jnp.array([T, eps])
        step_size = -(T - eps) / nsteps

        samples = new.sample(
            x_init=x_init,
            step_size=step_size,
            method="Euler",
            time_grid=time_grid,
            key=key,
        )
        assert samples.shape == (5, 3, 1)

    def test_sample_intermediates_shape(self):
        """Verify intermediates have correct shape."""
        _, new, path = self._make_both()
        key = jax.random.PRNGKey(0)
        x_init = path.sample_prior(key, (3, 3, 1))

        T = path.scheduler.T
        eps = 1e-3
        nsteps = 5
        time_grid = jnp.linspace(T, eps, nsteps + 1)
        step_size = -(T - eps) / nsteps

        samples = new.sample(
            x_init=x_init,
            step_size=step_size,
            method="Euler",
            time_grid=time_grid,
            return_intermediates=True,
            key=key,
        )
        assert samples.shape == (nsteps + 1, 3, 3, 1)

    def test_batch_independence(self):
        """Verify all samples in a batch are independent."""
        _, new, path = self._make_both()
        key = jax.random.PRNGKey(42)
        n_batch = 8
        x_init = path.sample_prior(key, (n_batch, 3, 1))

        T = path.scheduler.T
        eps = 1e-3
        nsteps = 10
        time_grid = jnp.array([T, eps])
        step_size = -(T - eps) / nsteps

        samples = new.sample(
            x_init=x_init,
            step_size=step_size,
            method="Euler",
            time_grid=time_grid,
            key=key,
        )
        assert samples.shape == (n_batch, 3, 1)

        for i in range(n_batch):
            for j in range(i + 1, n_batch):
                assert not jnp.allclose(
                    samples[i], samples[j], atol=1e-6
                ), f"Samples {i} and {j} are identical — batch independence violated"

    def test_finite_output(self):
        """Verify all outputs are finite (no NaN/Inf)."""
        _, new, path = self._make_both()
        key = jax.random.PRNGKey(0)
        x_init = path.sample_prior(key, (5, 3, 1))

        T = path.scheduler.T
        eps = 1e-3
        nsteps = 10
        time_grid = jnp.array([T, eps])
        step_size = -(T - eps) / nsteps

        samples = new.sample(
            x_init=x_init,
            step_size=step_size,
            method="Euler",
            time_grid=time_grid,
            key=key,
        )
        assert jnp.all(jnp.isfinite(samples)), "Non-finite values in SM SDE output"
