import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from gensbi.diffusion.solver.sm_ode_solver import SMODESolver
from gensbi.diffusion.solver.sm_sde_solver import SMSDESolver
from gensbi.utils.model_wrapping import ModelWrapper, ScoreToODEDrift
from gensbi.diffusion.path.sm_path import SMPath
from gensbi.diffusion.path.scheduler.sm_sde import VPSmScheduler, VESmScheduler


class DummyScoreModel(nnx.Module):
    def __call__(self, obs, t, **kwargs):
        return jnp.zeros_like(obs) + t


# =========================================================
# Helpers
# =========================================================


def _build_ode_solver(score_model, path):
    """Construct SMODESolver via ScoreToODEDrift + ModelWrapper."""
    sde = path.scheduler
    drift_model = ScoreToODEDrift(score_model=score_model, sde=sde)
    wrapper = ModelWrapper(model=drift_model)
    return SMODESolver(velocity_model=wrapper)


def _build_sde_solver(score_model, path):
    """Construct SMSDESolver with ModelWrapper + SDE scheduler."""
    wrapper = ModelWrapper(model=score_model)
    sde = path.scheduler
    return SMSDESolver(velocity_model=wrapper, sde=sde)


def _ode_sampler(score_model, path, nsteps, return_intermediates=False):
    """Build a sampler via SMODESolver, matching SM time conventions."""
    solver = _build_ode_solver(score_model, path)
    sde = path.scheduler
    T = sde.T
    eps = 1e-3

    if return_intermediates:
        time_grid = jnp.linspace(T, eps, nsteps + 1)
    else:
        time_grid = jnp.array([T, eps])

    step_size = -(T - eps) / nsteps

    return solver.get_sampler(
        step_size=step_size,
        method="Euler",
        time_grid=time_grid,
        return_intermediates=return_intermediates,
    )


def _sde_sampler(score_model, path, nsteps, method="Euler", return_intermediates=False):
    """Build a sampler via SMSDESolver, matching SM time conventions."""
    solver = _build_sde_solver(score_model, path)
    sde = path.scheduler
    T = sde.T
    eps = 1e-3

    if return_intermediates:
        time_grid = jnp.linspace(T, eps, nsteps + 1)
    else:
        time_grid = jnp.array([T, eps])

    step_size = -(T - eps) / nsteps

    return solver.get_sampler(
        step_size=step_size,
        method=method,
        time_grid=time_grid,
        return_intermediates=return_intermediates,
    )


# =========================================================
# Initialization tests
# =========================================================


@pytest.mark.parametrize("sde_cls", [VPSmScheduler, VESmScheduler])
def test_sm_ode_solver_initialization(sde_cls):
    """SMODESolver constructs without error."""
    sde = sde_cls()
    path = SMPath(sde)
    solver = _build_ode_solver(DummyScoreModel(), path)
    assert isinstance(solver, SMODESolver)


@pytest.mark.parametrize("sde_cls", [VPSmScheduler, VESmScheduler])
def test_sm_sde_solver_initialization(sde_cls):
    """SMSDESolver constructs without error and stores the SDE."""
    sde = sde_cls()
    path = SMPath(sde)
    solver = _build_sde_solver(DummyScoreModel(), path)
    assert isinstance(solver, SMSDESolver)
    assert solver.sde is sde


# =========================================================
# ODE solver (SMODESolver) shape tests
# =========================================================


@pytest.mark.parametrize("sde_cls", [VPSmScheduler, VESmScheduler])
def test_sm_ode_solver_sample_3d(sde_cls):
    """ODE sampling with input shape (batch, features, channels)."""
    score_model = DummyScoreModel()
    sde = sde_cls()
    path = SMPath(sde)
    sampler = _ode_sampler(score_model, path, nsteps=5)

    key = jax.random.PRNGKey(0)
    x_init = path.sample_prior(key, (5, 3, 4))

    samples = sampler(x_init)
    assert samples.shape == (5, 3, 4)


@pytest.mark.parametrize("sde_cls", [VPSmScheduler, VESmScheduler])
def test_sm_ode_solver_sample_3d_intermediates(sde_cls):
    """ODE sampling with intermediates for 3D input."""
    score_model = DummyScoreModel()
    sde = sde_cls()
    path = SMPath(sde)
    sampler = _ode_sampler(score_model, path, nsteps=5, return_intermediates=True)

    key = jax.random.PRNGKey(0)
    x_init = path.sample_prior(key, (5, 3, 4))

    samples = sampler(x_init)
    assert samples.shape == (6, 5, 3, 4)  # (nsteps+1, batch, features, channels)


@pytest.mark.parametrize("sde_cls", [VPSmScheduler, VESmScheduler])
def test_sm_ode_solver_sample_channel1(sde_cls):
    """ODE sampling with channel=1."""
    score_model = DummyScoreModel()
    sde = sde_cls()
    path = SMPath(sde)
    sampler = _ode_sampler(score_model, path, nsteps=5)

    key = jax.random.PRNGKey(0)
    x_init = path.sample_prior(key, (5, 3, 1))

    samples = sampler(x_init)
    assert samples.shape == (5, 3, 1)


@pytest.mark.parametrize("sde_cls", [VPSmScheduler, VESmScheduler])
def test_sm_ode_solver_dopri5(sde_cls):
    """ODE sampling with Dopri5 adaptive step sizing."""
    score_model = DummyScoreModel()
    sde = sde_cls()
    path = SMPath(sde)

    solver = _build_ode_solver(score_model, path)
    T = sde.T
    eps = 1e-3
    time_grid = jnp.array([T, eps])

    sampler = solver.get_sampler(
        step_size=None,
        method="Dopri5",
        time_grid=time_grid,
    )

    key = jax.random.PRNGKey(0)
    x_init = path.sample_prior(key, (5, 3, 4))

    samples = sampler(x_init)
    assert samples.shape == (5, 3, 4)


# =========================================================
# SDE solver (SMSDESolver) shape tests
# =========================================================


@pytest.mark.parametrize("sde_cls", [VPSmScheduler, VESmScheduler])
def test_sm_sde_solver_sample_3d(sde_cls):
    """SDE sampling with input shape (batch, features, channels)."""
    score_model = DummyScoreModel()
    sde = sde_cls()
    path = SMPath(sde)
    sampler = _sde_sampler(score_model, path, nsteps=5)

    key = jax.random.PRNGKey(0)
    key_init, key_sample = jax.random.split(key)
    x_init = path.sample_prior(key_init, (5, 3, 4))

    samples = sampler(x_init, key_sample)
    assert samples.shape == (5, 3, 4)


@pytest.mark.parametrize("sde_cls", [VPSmScheduler, VESmScheduler])
def test_sm_sde_solver_sample_3d_intermediates(sde_cls):
    """SDE sampling with intermediates for 3D input."""
    score_model = DummyScoreModel()
    sde = sde_cls()
    path = SMPath(sde)
    sampler = _sde_sampler(score_model, path, nsteps=5, return_intermediates=True)

    key = jax.random.PRNGKey(0)
    key_init, key_sample = jax.random.split(key)
    x_init = path.sample_prior(key_init, (5, 3, 4))

    samples = sampler(x_init, key_sample)
    assert samples.shape == (6, 5, 3, 4)  # (nsteps+1, batch, features, channels)


@pytest.mark.parametrize("sde_cls", [VPSmScheduler, VESmScheduler])
def test_sm_sde_solver_sample_channel1(sde_cls):
    """SDE sampling with channel=1."""
    score_model = DummyScoreModel()
    sde = sde_cls()
    path = SMPath(sde)
    sampler = _sde_sampler(score_model, path, nsteps=5)

    key = jax.random.PRNGKey(0)
    key_init, key_sample = jax.random.split(key)
    x_init = path.sample_prior(key_init, (5, 3, 1))

    samples = sampler(x_init, key_sample)
    assert samples.shape == (5, 3, 1)


@pytest.mark.parametrize("sde_cls", [VPSmScheduler, VESmScheduler])
def test_sm_sde_solver_batch_independence(sde_cls):
    """Verify all SDE samples in a batch are independent (not identical)."""
    score_model = DummyScoreModel()
    sde = sde_cls()
    path = SMPath(sde)
    sampler = _sde_sampler(score_model, path, nsteps=10)

    key = jax.random.PRNGKey(42)
    key_init, key_sample = jax.random.split(key)
    n_batch = 8
    x_init = path.sample_prior(key_init, (n_batch, 3, 4))

    samples = sampler(x_init, key_sample)
    assert samples.shape == (n_batch, 3, 4)

    for i in range(n_batch):
        for j in range(i + 1, n_batch):
            assert not jnp.allclose(
                samples[i], samples[j], atol=1e-6
            ), f"Samples {i} and {j} are identical — batch independence violated"


@pytest.mark.parametrize("sde_cls", [VPSmScheduler, VESmScheduler])
def test_sm_sde_solver_shark(sde_cls):
    """SDE sampling with ShARK adaptive step sizing."""
    score_model = DummyScoreModel()
    sde = sde_cls()
    path = SMPath(sde)
    sampler = _sde_sampler(score_model, path, nsteps=5, method="ShARK")

    key = jax.random.PRNGKey(0)
    key_init, key_sample = jax.random.split(key)
    x_init = path.sample_prior(key_init, (5, 3, 4))

    samples = sampler(x_init, key_sample)
    assert samples.shape == (5, 3, 4)


@pytest.mark.parametrize("sde_cls", [VPSmScheduler, VESmScheduler])
def test_sm_sde_solver_key_none_error(sde_cls):
    """SDE sampling with key=None should raise ValueError."""
    score_model = DummyScoreModel()
    sde = sde_cls()
    path = SMPath(sde)
    solver = _build_sde_solver(score_model, path)

    key = jax.random.PRNGKey(0)
    x_init = path.sample_prior(key, (2, 3, 1))

    sde_inst = path.scheduler
    T = sde_inst.T
    eps = 1e-3
    time_grid = jnp.array([T, eps])
    step_size = -(T - eps) / 5

    with pytest.raises(ValueError, match="key is required"):
        solver.sample(x_init, step_size=step_size, method="Euler",
                      time_grid=time_grid, key=None)


# =========================================================
# Log-probability tests (SMODESolver)
# =========================================================


def _build_ode_log_prob(score_model, path, nsteps=100, exact_divergence=True, dim=3, ch=1):
    """Build a log_prob callable via SMODESolver, matching SM time conventions."""
    solver = _build_ode_solver(score_model, path)
    sde = path.scheduler
    T = sde.T
    eps = 1e-3
    time_grid = jnp.array([T, eps])
    step_size = (T - eps) / nsteps

    from gensbi.core.prior import make_gaussian_prior

    if path.name == "SM-VP":
        prior = make_gaussian_prior(dim, ch)
    else:
        prior = make_gaussian_prior(dim, ch, sigma=sde.sigma_max)

    return solver.get_log_prob(
        log_p0=prior.log_prob,
        step_size=step_size,
        method="Euler",
        time_grid=time_grid,
        exact_divergence=exact_divergence,
    )


@pytest.mark.parametrize("sde_cls", [VPSmScheduler, VESmScheduler])
def test_sm_ode_log_prob_shape(sde_cls):
    """SMODESolver.get_log_prob produces log-prob with shape (batch,)."""
    score_model = DummyScoreModel()
    sde = sde_cls()
    path = SMPath(sde)

    log_prob_fn = _build_ode_log_prob(score_model, path, nsteps=10)

    key = jax.random.PRNGKey(0)
    x_1 = path.sample_prior(key, (5, 3, 1))

    logp = log_prob_fn(x_1)
    assert logp.shape == (5,)
    assert jnp.all(jnp.isfinite(logp))


@pytest.mark.parametrize("sde_cls", [VPSmScheduler, VESmScheduler])
def test_sm_ode_log_prob_exact_vs_hutchinson(sde_cls):
    """Hutchinson log-prob should match exact log-prob for the dummy score model."""
    score_model = DummyScoreModel()
    sde = sde_cls()
    path = SMPath(sde)

    key = jax.random.PRNGKey(0)
    x_1 = path.sample_prior(key, (4, 2, 1))

    # Exact divergence
    logp_exact_fn = _build_ode_log_prob(
        score_model, path, nsteps=10, exact_divergence=True, dim=2,
    )
    logp_exact = logp_exact_fn(x_1)

    # Hutchinson divergence
    hutch_key = jax.random.PRNGKey(42)
    logp_hutch_fn = _build_ode_log_prob(
        score_model, path, nsteps=10, exact_divergence=False, dim=2,
    )
    logp_hutch = logp_hutch_fn(x_1, key=hutch_key)

    assert logp_hutch.shape == logp_exact.shape
    assert jnp.allclose(logp_exact, logp_hutch, rtol=1e-3), (
        f"Exact vs Hutchinson mismatch: exact={logp_exact}, hutch={logp_hutch}"
    )




# =========================================================
# Pipeline integration test
# =========================================================


@pytest.mark.parametrize("sde_cls", [VPSmScheduler, VESmScheduler])
def test_score_matching_method_smpf_solver(sde_cls):
    """ScoreMatchingMethod.build_sampler_fn with SMODESolver produces valid samples."""
    from gensbi.core.score_matching import ScoreMatchingMethod

    sde_type = "VP" if sde_cls is VPSmScheduler else "VE"
    method = ScoreMatchingMethod(sde_type=sde_type)

    config = method.get_extra_training_config()
    path = method.build_path(config, event_shape=(3, 1))

    score_model = DummyScoreModel()

    sampler_fn = method.build_sampler_fn(
        model_wrapped=score_model,
        path=path,
        model_extras={},
        nsteps=10,
        solver=(SMODESolver, {}),
    )

    key = jax.random.PRNGKey(0)
    x_init = path.sample_prior(key, (4, 3, 1))
    result = sampler_fn(key, x_init)

    assert result.shape == (4, 3, 1)
    assert jnp.all(jnp.isfinite(result))

    # Verify prior was set correctly
    assert method.prior is not None
    lp = method.prior.log_prob(x_init)
    assert lp.shape == (4,)


@pytest.mark.parametrize("sde_cls", [VPSmScheduler, VESmScheduler])
def test_score_matching_method_sde_solver(sde_cls):
    """ScoreMatchingMethod.build_sampler_fn with SMSDESolver produces valid samples."""
    from gensbi.core.score_matching import ScoreMatchingMethod

    sde_type = "VP" if sde_cls is VPSmScheduler else "VE"
    method = ScoreMatchingMethod(sde_type=sde_type)

    config = method.get_extra_training_config()
    path = method.build_path(config, event_shape=(3, 1))

    score_model = DummyScoreModel()

    sampler_fn = method.build_sampler_fn(
        model_wrapped=score_model,
        path=path,
        model_extras={},
        nsteps=10,
        solver=(SMSDESolver, {}),
    )

    key = jax.random.PRNGKey(0)
    x_init = path.sample_prior(key, (4, 3, 1))
    result = sampler_fn(key, x_init)

    assert result.shape == (4, 3, 1)
    assert jnp.all(jnp.isfinite(result))


# =========================================================
# Pipeline-level log-prob tests
# =========================================================


@pytest.mark.parametrize("sde_cls", [VPSmScheduler, VESmScheduler])
def test_score_matching_method_build_log_prob_fn(sde_cls):
    """ScoreMatchingMethod.build_log_prob_fn with SMODESolver produces valid log-prob."""
    from gensbi.core.score_matching import ScoreMatchingMethod

    sde_type = "VP" if sde_cls is VPSmScheduler else "VE"
    method = ScoreMatchingMethod(sde_type=sde_type)

    config = method.get_extra_training_config()
    path = method.build_path(config, event_shape=(3, 1))

    score_model = DummyScoreModel()

    log_prob_fn = method.build_log_prob_fn(
        model_wrapped=score_model,
        path=path,
        model_extras={},
        nsteps=10,
        method="Euler",
        solver=(SMODESolver, {}),
    )

    key = jax.random.PRNGKey(0)
    x_1 = path.sample_prior(key, (4, 3, 1))
    logp = log_prob_fn(x_1)

    assert logp.shape == (4,)
    assert jnp.all(jnp.isfinite(logp))


def test_score_matching_method_log_prob_rejects_sde_solver():
    """build_log_prob_fn should reject SMSDESolver (SDE, not ODE)."""
    from gensbi.core.score_matching import ScoreMatchingMethod

    method = ScoreMatchingMethod(sde_type="VP")
    config = method.get_extra_training_config()
    path = method.build_path(config, event_shape=(3, 1))

    with pytest.raises(NotImplementedError, match="SMODESolver"):
        method.build_log_prob_fn(
            model_wrapped=DummyScoreModel(),
            path=path,
            model_extras={},
            solver=(SMSDESolver, {}),
        )
